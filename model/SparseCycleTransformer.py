"""
SparseCycleTransformer: 双通路稀疏周期Transformer模型

核心思想: 设计一个双通路Transformer，其中一条通路采用动态稀疏门控机制，
仅对序列中具有显著信息变化的"关键周期点"进行计算；另一条通路则负责
解析和注入全局周期性语义，引导稀疏通路更智能地选择关键点。

主要创新点:
1. 语义通路 (Semantic Pathway): 通过傅里叶变换和自适应池化发现潜在周期
2. 稀疏门控通路 (Sparse Gated Pathway): 只在关键时间点进行密集注意力计算
3. 双通路协同: 交叉注意力融合确保局部计算在全局语义指导下进行
4. 三部分联合损失: 预测损失 + 稀疏正则化 + 周期一致性

Paper target: 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from layers.SparseCycleEmbed import (
    FourierPeriodicTokenGenerator,
    SparseDataEmbedding,
    VariateAwareEmbedding
)
from layers.SparseCycleAttention import (
    SparseGatePredictor,
    SparseGatedAttention,
    DualPathwayFusion,
    SemanticEncoder,
    SparseEncoderLayer,
    PeriodicConsistencyModule
)


class Model(nn.Module):
    """
    SparseCycleTransformer: 将结构化稀疏性与周期性语义理解深度融合的创新模型
    
    核心优势:
    - 不依赖于预定义的周期长度，通过数据驱动发现潜在周期
    - 实现计算资源的精准投放，复杂度从O(L²)降低到O(kL)
    - 双通路协同机制确保效率与性能的平衡
    
    Args:
        configs: 配置对象，包含以下属性:
            - seq_len: 输入序列长度
            - pred_len: 预测序列长度
            - enc_in: 输入变量数量
            - d_model: 模型维度
            - n_heads: 注意力头数量
            - e_layers: 编码器层数
            - d_ff: 前馈网络维度
            - dropout: dropout比率
            - activation: 激活函数
            - num_semantic_tokens: 语义令牌数量 (默认8)
            - sparsity_ratio: 目标稀疏率 (默认0.5)
            - lambda_sparse: 稀疏正则化权重 (默认0.01)
            - lambda_period: 周期一致性损失权重 (默认0.01)
    """
    
    def __init__(self, configs):
        super(Model, self).__init__()
        
        # 基础配置
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = getattr(configs, 'enc_in', 7)
        self.d_model = configs.d_model
        self.n_heads = configs.n_heads
        self.e_layers = configs.e_layers
        self.d_ff = configs.d_ff
        self.dropout = configs.dropout
        self.activation = configs.activation
        self.output_attention = configs.output_attention
        self.use_norm = getattr(configs, 'use_norm', True)
        
        # SparseCycleTransformer特有配置
        self.num_semantic_tokens = getattr(configs, 'num_semantic_tokens', 8)
        self.sparsity_ratio = getattr(configs, 'sparsity_ratio', 0.5)
        self.lambda_sparse = getattr(configs, 'lambda_sparse', 0.01)
        self.lambda_period = getattr(configs, 'lambda_period', 0.01)
        
        # ==================== 语义通路 (Semantic Pathway) ====================
        # 傅里叶周期令牌生成器
        self.periodic_token_generator = FourierPeriodicTokenGenerator(
            seq_len=self.seq_len,
            d_model=self.d_model,
            num_tokens=self.num_semantic_tokens,
            enc_in=self.enc_in,
            dropout=self.dropout
        )
        
        # 语义编码器 (轻量级)
        self.semantic_encoder = SemanticEncoder(
            d_model=self.d_model,
            n_heads=min(4, self.n_heads),  # 使用较少的头
            d_ff=self.d_model * 2,
            n_layers=1,  # 单层足够
            dropout=self.dropout
        )
        
        # ==================== 稀疏门控通路 (Sparse Gated Pathway) ====================
        # 数据嵌入 (iTransformer风格: 变量作为token)
        self.sparse_embedding = VariateAwareEmbedding(
            seq_len=self.seq_len,
            d_model=self.d_model,
            dropout=self.dropout
        )
        
        # 门控预测器
        self.gate_predictor = SparseGatePredictor(
            d_model=self.d_model,
            d_semantic=self.d_model,
            num_semantic_tokens=self.num_semantic_tokens,
            dropout=self.dropout
        )
        
        # 稀疏编码器层
        self.sparse_encoder_layers = nn.ModuleList([
            SparseEncoderLayer(
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_ff=self.d_ff,
                dropout=self.dropout,
                activation=self.activation
            ) for _ in range(self.e_layers)
        ])
        
        # ==================== 双通路融合 ====================
        self.dual_pathway_fusion = DualPathwayFusion(
            d_model=self.d_model,
            n_heads=self.n_heads,
            dropout=self.dropout
        )
        
        # ==================== 周期一致性模块 ====================
        self.periodic_consistency = PeriodicConsistencyModule(
            d_model=self.d_model,
            num_tokens=self.num_semantic_tokens
        )
        
        # ==================== 输出层 ====================
        self.norm = nn.LayerNorm(self.d_model)
        self.projector = nn.Linear(self.d_model, self.pred_len, bias=True)
        
        # 存储中间结果用于损失计算
        self.gate_scores = None
        self.gate_mask = None
        self.freq_info = None
        self.consistency_score = None
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        前向传播
        
        Args:
            x_enc: 编码器输入 [B, L, N]
            x_mark_enc: 编码器时间戳 (暂未使用)
            x_dec: 解码器输入 (暂未使用，encoder-only架构)
            x_mark_dec: 解码器时间戳 (暂未使用)
            mask: 掩码 (暂未使用)
            
        Returns:
            dec_out: 预测输出 [B, S, N]
            (auxiliary_outputs): 辅助输出，用于损失计算
        """
        # 归一化 (Non-stationary Transformer style)
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        
        B, L, N = x_enc.shape
        
        # ==================== 语义通路 ====================
        # 生成周期语义令牌
        semantic_tokens, freq_info = self.periodic_token_generator(x_enc)  # [B, K, d_model]
        self.freq_info = freq_info
        
        # 编码语义令牌
        semantic_tokens = self.semantic_encoder(semantic_tokens)  # [B, K, d_model]
        
        # ==================== 稀疏门控通路 ====================
        # 数据嵌入
        sparse_emb, temporal_features = self.sparse_embedding(x_enc)  # [B, N, d_model]
        
        # 门控预测
        gate_scores, gate_mask, threshold = self.gate_predictor(
            sparse_emb, 
            semantic_tokens, 
            sparsity_ratio=self.sparsity_ratio
        )
        self.gate_scores = gate_scores
        self.gate_mask = gate_mask
        
        # 稀疏编码器
        enc_out = sparse_emb
        attns = []
        for layer in self.sparse_encoder_layers:
            enc_out, attn = layer(enc_out, gate_mask)
            attns.append(attn)
        
        # ==================== 双通路融合 ====================
        fused_out = self.dual_pathway_fusion(enc_out, semantic_tokens)
        
        # ==================== 周期一致性计算 ====================
        _, _, consistency_score = self.periodic_consistency(
            semantic_tokens, gate_scores, freq_info
        )
        self.consistency_score = consistency_score
        
        # ==================== 输出投影 ====================
        enc_out = self.norm(fused_out)
        
        # [B, N, d_model] -> [B, N, pred_len] -> [B, pred_len, N]
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]
        
        # 反归一化
        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]
    
    def compute_auxiliary_loss(self):
        """
        计算辅助损失 (稀疏正则化 + 周期一致性)
        
        Returns:
            sparse_loss: 稀疏正则化损失
            period_loss: 周期一致性损失
        """
        # 稀疏正则化损失: 鼓励用更少的关键点
        if self.gate_scores is not None:
            sparse_loss = self.gate_scores.mean()
        else:
            sparse_loss = torch.tensor(0.0)
        
        # 周期一致性损失: 负余弦相似度
        if self.consistency_score is not None:
            period_loss = 1 - self.consistency_score.mean()
        else:
            period_loss = torch.tensor(0.0)
        
        return sparse_loss, period_loss
    
    def get_total_loss(self, pred_loss):
        """
        计算总损失
        
        Args:
            pred_loss: 主预测损失 (MSE)
            
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的字典
        """
        sparse_loss, period_loss = self.compute_auxiliary_loss()
        
        total_loss = (pred_loss + 
                      self.lambda_sparse * sparse_loss + 
                      self.lambda_period * period_loss)
        
        loss_dict = {
            'pred_loss': pred_loss.item() if hasattr(pred_loss, 'item') else pred_loss,
            'sparse_loss': sparse_loss.item() if hasattr(sparse_loss, 'item') else sparse_loss,
            'period_loss': period_loss.item() if hasattr(period_loss, 'item') else period_loss,
            'total_loss': total_loss.item() if hasattr(total_loss, 'item') else total_loss
        }
        
        return total_loss, loss_dict
    
    def get_sparsity_stats(self):
        """
        获取稀疏统计信息
        
        Returns:
            stats: 稀疏统计字典
        """
        if self.gate_mask is None:
            return {}
        
        actual_sparsity = 1 - self.gate_mask.mean().item()
        active_ratio = self.gate_mask.mean().item()
        
        stats = {
            'actual_sparsity': actual_sparsity,
            'active_ratio': active_ratio,
            'target_sparsity': 1 - self.sparsity_ratio,
            'num_active_points': self.gate_mask.sum().item() / self.gate_mask.size(0)
        }
        
        return stats
