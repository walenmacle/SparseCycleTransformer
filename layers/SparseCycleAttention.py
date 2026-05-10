import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
import numpy as np


class SparseGatePredictor(nn.Module):
    """
    基于周期语义的稀疏门控预测器
    
    接收语义通路的周期编码，预测每个变量/时间点的重要性分数，
    用于决定哪些点需要进行完整的注意力计算。
    
    Args:
        d_model: 模型维度
        d_semantic: 语义编码维度
        num_semantic_tokens: 语义令牌数量
        dropout: dropout比率
    """
    
    def __init__(self, d_model, d_semantic=None, num_semantic_tokens=8, dropout=0.1):
        super(SparseGatePredictor, self).__init__()
        
        d_semantic = d_semantic or d_model
        
        # 语义信息聚合
        self.semantic_aggregator = nn.Sequential(
            nn.Linear(num_semantic_tokens * d_semantic, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 门控预测网络
        self.gate_network = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # 动态阈值网络
        self.threshold_network = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
        
        self.d_model = d_model
        
    def forward(self, x, semantic_tokens, sparsity_ratio=0.5, return_threshold=True):
        """
        Args:
            x: 输入嵌入 [B, N, d_model] (N是变量数量)
            semantic_tokens: 语义令牌 [B, K, d_model] (K是语义令牌数量)
            sparsity_ratio: 目标稀疏率 (选择top多少比例的关键点)
            return_threshold: 是否返回阈值信息
            
        Returns:
            gate_scores: 门控分数 [B, N]
            gate_mask: 二值门控掩码 [B, N]
            threshold: 动态阈值 (如果return_threshold=True)
        """
        B, N, _ = x.shape
        _, K, D = semantic_tokens.shape
        
        # 聚合语义信息
        semantic_flat = semantic_tokens.reshape(B, -1)  # [B, K*D]
        semantic_context = self.semantic_aggregator(semantic_flat)  # [B, d_model]
        
        # 扩展语义上下文到每个变量
        semantic_context = semantic_context.unsqueeze(1).expand(-1, N, -1)  # [B, N, d_model]
        
        # 拼接输入和语义上下文
        combined = torch.cat([x, semantic_context], dim=-1)  # [B, N, d_model*2]
        
        # 预测门控分数
        gate_scores = self.gate_network(combined).squeeze(-1)  # [B, N]
        
        # 计算动态阈值
        global_context = x.mean(dim=1)  # [B, d_model]
        base_threshold = self.threshold_network(global_context).squeeze(-1)  # [B]
        
        # 根据目标稀疏率调整阈值
        # 使用top-k选择来确保稀疏率
        k = max(1, int(N * sparsity_ratio))
        
        # 获取top-k分数的最小值作为阈值
        topk_values, _ = torch.topk(gate_scores, k, dim=-1)
        adaptive_threshold = topk_values[:, -1]  # [B]
        
        # 混合基础阈值和自适应阈值
        final_threshold = 0.5 * base_threshold + 0.5 * adaptive_threshold
        
        # 生成二值掩码 (使用straight-through estimator保持可微性)
        gate_mask = (gate_scores >= final_threshold.unsqueeze(-1)).float()
        
        # Straight-through estimator: 前向传播用硬掩码，反向传播用软分数
        gate_mask = gate_mask - gate_scores.detach() + gate_scores
        
        if return_threshold:
            return gate_scores, gate_mask, final_threshold
        return gate_scores, gate_mask


class SparseGatedAttention(nn.Module):
    """
    稀疏门控注意力层
    
    仅对关键点进行完整的注意力计算，非关键点使用极简的线性投影。
    这是稀疏通路的核心计算模块。
    
    Args:
        d_model: 模型维度
        n_heads: 注意力头数量
        d_keys: Key维度 (默认为d_model // n_heads)
        d_values: Value维度 (默认为d_model // n_heads)
        attention_dropout: 注意力dropout比率
        output_attention: 是否输出注意力权重
    """
    
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, 
                 attention_dropout=0.1, output_attention=False):
        super(SparseGatedAttention, self).__init__()
        
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        
        self.n_heads = n_heads
        self.d_keys = d_keys
        self.d_values = d_values
        self.scale = 1. / sqrt(d_keys)
        self.output_attention = output_attention
        
        # 完整注意力的投影层
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        
        # 非关键点的极简线性投影
        self.skip_projection = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values, gate_mask, attn_mask=None):
        """
        Args:
            queries: Query张量 [B, N, d_model]
            keys: Key张量 [B, N, d_model]
            values: Value张量 [B, N, d_model]
            gate_mask: 门控掩码 [B, N], 1表示关键点
            attn_mask: 注意力掩码 (可选)
            
        Returns:
            output: 输出张量 [B, N, d_model]
            attn_weights: 注意力权重 (如果output_attention=True)
        """
        B, N, D = queries.shape
        H = self.n_heads
        
        # 分离关键点和非关键点
        # gate_mask: [B, N], 1表示关键点
        
        # 对所有点进行投影 (为了保持梯度流)
        Q = self.query_projection(queries).view(B, N, H, self.d_keys)
        K = self.key_projection(keys).view(B, N, H, self.d_keys)
        V = self.value_projection(values).view(B, N, H, self.d_values)
        
        # 计算完整注意力
        # [B, N, H, d_k] x [B, N, H, d_k]^T -> [B, H, N, N]
        scores = torch.einsum("bnhd,bmhd->bhnm", Q, K) * self.scale
        
        # 应用注意力掩码 (如果存在)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, -np.inf)
        
        # Softmax
        attn_weights = self.dropout(torch.softmax(scores, dim=-1))
        
        # 注意力输出
        # [B, H, N, N] x [B, N, H, d_v] -> [B, N, H, d_v]
        attn_output = torch.einsum("bhnm,bmhd->bnhd", attn_weights, V)
        attn_output = attn_output.reshape(B, N, -1)  # [B, N, H*d_v]
        attn_output = self.out_projection(attn_output)  # [B, N, d_model]
        
        # 非关键点的线性投影
        skip_output = self.skip_projection(values)  # [B, N, d_model]
        
        # 根据门控掩码混合输出
        # 关键点使用注意力输出，非关键点使用线性投影
        gate_mask_expanded = gate_mask.unsqueeze(-1)  # [B, N, 1]
        output = gate_mask_expanded * attn_output + (1 - gate_mask_expanded) * skip_output
        
        if self.output_attention:
            return output, attn_weights
        return output, None


class DualPathwayFusion(nn.Module):
    """
    双通路融合模块
    
    通过交叉注意力将语义通路的周期编码融合到稀疏通路的输出中，
    确保局部计算始终在全局语义的指导下进行。
    
    Args:
        d_model: 模型维度
        n_heads: 注意力头数量
        dropout: dropout比率
    """
    
    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super(DualPathwayFusion, self).__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_keys = d_model // n_heads
        
        # Query来自稀疏通路
        self.query_proj = nn.Linear(d_model, d_model)
        # Key和Value来自语义通路
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        
        self.scale = 1. / sqrt(self.d_keys)
        
        self.out_proj = nn.Linear(d_model, d_model)
        
        # 门控融合
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, sparse_output, semantic_tokens):
        """
        Args:
            sparse_output: 稀疏通路输出 [B, N, d_model]
            semantic_tokens: 语义令牌 [B, K, d_model]
            
        Returns:
            fused_output: 融合后的输出 [B, N, d_model]
        """
        B, N, D = sparse_output.shape
        _, K, _ = semantic_tokens.shape
        H = self.n_heads
        
        # 投影
        Q = self.query_proj(sparse_output).view(B, N, H, self.d_keys)
        K_proj = self.key_proj(semantic_tokens).view(B, K, H, self.d_keys)
        V = self.value_proj(semantic_tokens).view(B, K, H, self.d_keys)
        
        # 交叉注意力
        # [B, N, H, d_k] x [B, K, H, d_k]^T -> [B, H, N, K]
        scores = torch.einsum("bnhd,bkhd->bhnk", Q, K_proj) * self.scale
        attn_weights = self.dropout(torch.softmax(scores, dim=-1))
        
        # [B, H, N, K] x [B, K, H, d_k] -> [B, N, H, d_k]
        cross_attn = torch.einsum("bhnk,bkhd->bnhd", attn_weights, V)
        cross_attn = cross_attn.reshape(B, N, -1)  # [B, N, d_model]
        cross_attn = self.out_proj(cross_attn)
        
        # 门控融合
        combined = torch.cat([sparse_output, cross_attn], dim=-1)
        gate = self.fusion_gate(combined)
        
        fused_output = gate * sparse_output + (1 - gate) * cross_attn
        fused_output = self.norm(sparse_output + fused_output)
        
        return fused_output


class SemanticEncoder(nn.Module):
    """
    语义通路编码器
    
    处理语义令牌的轻量级Transformer编码器，用于捕获周期间的相互依赖关系。
    
    Args:
        d_model: 模型维度
        n_heads: 注意力头数量
        d_ff: 前馈网络维度
        n_layers: 编码器层数
        dropout: dropout比率
    """
    
    def __init__(self, d_model, n_heads=4, d_ff=None, n_layers=1, dropout=0.1):
        super(SemanticEncoder, self).__init__()
        
        d_ff = d_ff or d_model * 2
        
        self.layers = nn.ModuleList([
            SemanticEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, semantic_tokens):
        """
        Args:
            semantic_tokens: 语义令牌 [B, K, d_model]
            
        Returns:
            encoded_tokens: 编码后的语义令牌 [B, K, d_model]
        """
        x = semantic_tokens
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class SemanticEncoderLayer(nn.Module):
    """
    语义编码器层
    
    标准Transformer编码器层，用于语义令牌的自注意力处理。
    """
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(SemanticEncoderLayer, self).__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: 输入 [B, K, d_model]
            
        Returns:
            output: 输出 [B, K, d_model]
        """
        # 自注意力
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # 前馈网络
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


class SparseEncoderLayer(nn.Module):
    """
    稀疏编码器层
    
    结合稀疏门控注意力和前馈网络的编码器层。
    
    Args:
        d_model: 模型维度
        n_heads: 注意力头数量
        d_ff: 前馈网络维度
        dropout: dropout比率
        activation: 激活函数类型
    """
    
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1, activation='gelu'):
        super(SparseEncoderLayer, self).__init__()
        
        d_ff = d_ff or 4 * d_model
        
        self.sparse_attention = SparseGatedAttention(
            d_model, n_heads, attention_dropout=dropout
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, gate_mask, attn_mask=None):
        """
        Args:
            x: 输入 [B, N, d_model]
            gate_mask: 门控掩码 [B, N]
            attn_mask: 注意力掩码 (可选)
            
        Returns:
            output: 输出 [B, N, d_model]
            attn_weights: 注意力权重
        """
        # 稀疏注意力
        attn_out, attn_weights = self.sparse_attention(x, x, x, gate_mask, attn_mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # 前馈网络
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x, attn_weights


class PeriodicConsistencyModule(nn.Module):
    """
    周期一致性模块
    
    计算语义通路识别的周期模式与稀疏通路选择的跨周期关键点之间的一致性，
    用于周期一致性损失的计算。
    
    Args:
        d_model: 模型维度
        num_tokens: 语义令牌数量
    """
    
    def __init__(self, d_model, num_tokens=8):
        super(PeriodicConsistencyModule, self).__init__()
        
        # 周期模式编码器
        self.semantic_pattern_encoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model // 4)
        )
        
        # 门控模式编码器
        self.sparse_pattern_encoder = nn.Sequential(
            nn.Linear(num_tokens, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model // 4)
        )
        
    def forward(self, semantic_tokens, gate_scores, freq_info=None):
        """
        Args:
            semantic_tokens: 语义令牌 [B, K, d_model]
            gate_scores: 门控分数 [B, N]
            freq_info: 频率信息字典 (可选)
            
        Returns:
            semantic_pattern: 语义周期模式 [B, d_model//4]
            sparse_pattern: 稀疏选择模式 [B, d_model//4]
            consistency_score: 一致性分数 [B]
        """
        B, K, D = semantic_tokens.shape
        
        # 编码语义周期模式
        semantic_pooled = semantic_tokens.mean(dim=1)  # [B, d_model]
        semantic_pattern = self.semantic_pattern_encoder(semantic_pooled)  # [B, d_model//4]
        
        # 编码稀疏选择模式
        # 将门控分数重塑为与语义令牌数量相同的维度
        N = gate_scores.shape[1]
        gate_reshaped = F.adaptive_avg_pool1d(
            gate_scores.unsqueeze(1), K
        ).squeeze(1)  # [B, K]
        sparse_pattern = self.sparse_pattern_encoder(gate_reshaped)  # [B, d_model//4]
        
        # 计算余弦相似度
        consistency_score = F.cosine_similarity(semantic_pattern, sparse_pattern, dim=-1)
        
        return semantic_pattern, sparse_pattern, consistency_score
