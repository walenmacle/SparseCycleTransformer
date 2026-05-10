import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class FourierPeriodicTokenGenerator(nn.Module):
    """
    可学习的傅里叶周期令牌生成器
    
    通过傅里叶变换提取序列的频率成分，并生成刻画周期、振幅和相位的语义令牌。
    这是语义通路的核心组件，作为模型的"宏观大脑"理解全局周期规律。
    
    Args:
        seq_len: 输入序列长度
        d_model: 模型维度
        num_tokens: 生成的语义令牌数量
        dropout: dropout比率
    """
    
    def __init__(self, seq_len, d_model, num_tokens=8, enc_in=7, dropout=0.1):
        super(FourierPeriodicTokenGenerator, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.num_tokens = num_tokens
        self.enc_in = enc_in
        
        # 选择top-k频率成分的数量（与输出token数量一致）
        self.num_freqs = min(seq_len // 2, num_tokens)
        
        # 可学习的变量权重向量（用于计算频率加权能量和幅度/相位加权）
        self.freq_weight = nn.Parameter(torch.ones(1, 5000, 1))
        
        # 令牌生成MLP: [B, K, 2N] → [B, K, D]
        # 输入为每个频率的幅度+相位交错排列 (2N维)
        self.token_mlp = nn.Sequential(
            nn.Linear(enc_in * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
        # 输出归一化
        self.token_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        Args:
            x: 输入序列 [B, L, N] 或 [B, N, L] (variate维度)
            
        Returns:
            semantic_tokens: 语义令牌 [B, num_tokens, d_model]
            freq_info: 频率信息字典 (用于周期一致性损失)
        """
        B, L, N = x.shape
        
        # 对每个变量进行傅里叶变换
        # [B, L, N] -> [B, N, L]
        x_permuted = x.permute(0, 2, 1)
        
        # 执行FFT
        fft_result = torch.fft.rfft(x_permuted, dim=-1)
        
        # 提取振幅和相位
        amplitudes = torch.abs(fft_result)  # [B, N, L//2+1]
        phases = torch.angle(fft_result)     # [B, N, L//2+1]
        
        # 排除DC分量（频率0），保留所有其他频率
        amp_no_dc = amplitudes[:, :, 1:]    # [B, N, F]  F = L//2
        phase_no_dc = phases[:, :, 1:]       # [B, N, F]
        F_dim = amp_no_dc.size(-1)
        
        # 可学习权重（softmax归一化，确保权重和为1）
        w = torch.softmax(self.freq_weight[:, :N, :], dim=1)  # [1, N, 1]
        
        # 计算每个频率的加权能量：各变量幅度乘以对应权重后求和
        energy = (amp_no_dc * w).sum(dim=1)  # [B, F]
        
        # 取能量前K大的频率
        num_select = min(self.num_freqs, F_dim)
        topk_energy, topk_indices = torch.topk(energy, num_select, dim=-1)  # [B, num_select]
        
        # 用topk_indices从amp_no_dc和phase_no_dc中gather出选中频率
        idx_expanded = topk_indices.unsqueeze(1).expand(-1, N, -1)  # [B, N, num_select]
        selected_amp = torch.gather(amp_no_dc, 2, idx_expanded)      # [B, N, num_select]
        selected_phase = torch.gather(phase_no_dc, 2, idx_expanded)  # [B, N, num_select]
        
        # 如果选中频率数不足num_freqs，补零保持维度一致
        if num_select < self.num_freqs:
            pad_size = self.num_freqs - num_select
            selected_amp = F.pad(selected_amp, (0, pad_size))      # [B, N, K]
            selected_phase = F.pad(selected_phase, (0, pad_size))  # [B, N, K]
        
        # ==================== freq_weight 逐元素加权 ====================
        # 使用 freq_weight 的 softmax 归一化权重对幅度和相位逐元素加权
        # w 形状 [1, N, 1]，广播到 [B, N, K]，每个变量的所有频率共享同一权重
        weighted_amp = selected_amp * w      # [B, N, K]
        weighted_phase = selected_phase * w   # [B, N, K]
        
        # ==================== 变换形状 + 拼接 + Shuffle ====================
        # 变换形状: [B, N, K] → [B, K, N]，每个频率拥有N个变量的信息
        weighted_amp = weighted_amp.permute(0, 2, 1)    # [B, K, N]
        weighted_phase = weighted_phase.permute(0, 2, 1)  # [B, K, N]
        
        # 拼接幅度和相位，并交错排列 (shuffle)
        # stack → [B, K, N, 2]，每个位置为 [amp_i, phase_i]
        # reshape → [B, K, 2N]，顺序: [amp_0, phase_0, amp_1, phase_1, ..., amp_{N-1}, phase_{N-1}]
        combined = torch.stack([weighted_amp, weighted_phase], dim=-1)  # [B, K, N, 2]
        combined = combined.reshape(B, self.num_freqs, 2 * N)  # [B, K, 2N] 幅度相位交错
        
        # ==================== MLP ====================
        # MLP 映射: [B, K, 2N] → [B, K, D]
        semantic_tokens = self.token_mlp(combined)  # [B, K, D]
        semantic_tokens = self.token_norm(semantic_tokens)  # [B, K, D]
        
        # 返回频率信息用于周期一致性损失
        freq_info = {
            'amplitudes': weighted_amp.mean(dim=1),   # [B, K] 加权后幅度的变量均值
            'phases': weighted_phase.mean(dim=1),      # [B, K] 加权后相位的变量均值
            'dominant_freqs': torch.topk(topk_energy, k=min(self.num_tokens, topk_energy.size(-1)), dim=-1),
            'freq_energy': topk_energy,           # 选中频率的加权能量 [B, K]
            'freq_indices': topk_indices,          # 选中频率的索引（相对于非DC频率） [B, K]
            'freq_weight': torch.softmax(self.freq_weight[:, :N, :], dim=1).squeeze(-1)  # 归一化后的变量权重 [1, N]
        }
        
        return semantic_tokens, freq_info


class SparseDataEmbedding(nn.Module):
    """
    稀疏场景下的数据嵌入层
    
    将原始时间序列嵌入到高维空间，同时保留位置信息用于稀疏选择。
    
    Args:
        c_in: 输入序列长度
        d_model: 模型维度
        dropout: dropout比率
    """
    
    def __init__(self, c_in, d_model, dropout=0.1):
        super(SparseDataEmbedding, self).__init__()
        
        # 值嵌入 (线性投影)
        self.value_embedding = nn.Linear(c_in, d_model)
        
        # 可学习的位置嵌入
        self.position_embedding = nn.Parameter(torch.randn(1, 5000, d_model) * 0.02)
        
        # 时间步编码器 (用于门控预测)
        self.time_encoder = nn.Linear(1, d_model)
        
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        
    def forward(self, x, return_positions=True):
        """
        Args:
            x: 输入序列 [B, L, N]
            return_positions: 是否返回位置编码
            
        Returns:
            embedded: 嵌入后的序列 [B, N, d_model] (iTransformer风格，变量作为token)
            positions: 位置编码 [B, N, d_model]
        """
        B, L, N = x.shape
        
        # iTransformer风格: 变量作为token
        # [B, L, N] -> [B, N, L]
        x = x.permute(0, 2, 1)
        
        # 值嵌入
        embedded = self.value_embedding(x)  # [B, N, d_model]
        
        # 添加位置嵌入 (对变量维度)
        positions = self.position_embedding[:, :N, :]  # [1, N, d_model]
        embedded = embedded + positions
        
        embedded = self.dropout(embedded)
        
        if return_positions:
            return embedded, positions.expand(B, -1, -1)
        return embedded


class TemporalPositionEncoding(nn.Module):
    """
    时序位置编码
    
    为时间序列的每个时间步生成位置编码，用于门控预测和稀疏选择。
    
    Args:
        d_model: 模型维度
        max_len: 最大序列长度
    """
    
    def __init__(self, d_model, max_len=5000):
        super(TemporalPositionEncoding, self).__init__()
        
        # 正弦位置编码
        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * 
                    -(math.log(10000.0) / d_model)).exp()
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
        
    def forward(self, seq_len):
        """
        Args:
            seq_len: 序列长度
            
        Returns:
            position_encoding: [1, seq_len, d_model]
        """
        return self.pe[:, :seq_len, :]


class VariateAwareEmbedding(nn.Module):
    """
    变量感知嵌入层
    
    结合iTransformer的倒置嵌入思想，同时为每个变量生成可用于门控的表示。
    
    Args:
        seq_len: 序列长度
        d_model: 模型维度
        num_variates: 变量数量 (可选，用于可学习变量嵌入)
        dropout: dropout比率
    """
    
    def __init__(self, seq_len, d_model, num_variates=None, dropout=0.1):
        super(VariateAwareEmbedding, self).__init__()
        
        self.seq_len = seq_len
        self.d_model = d_model
        
        # 时序值嵌入
        self.value_embedding = nn.Linear(seq_len, d_model)
        
        # 可学习的变量嵌入 (如果指定了变量数量)
        if num_variates is not None:
            self.variate_embedding = nn.Embedding(num_variates, d_model)
        else:
            self.variate_embedding = None
            
        # 时序特征嵌入 (用于捕获时序模式)
        self.temporal_conv = nn.Conv1d(
            in_channels=1, 
            out_channels=d_model // 4,
            kernel_size=3,
            padding=1
        )
        
        self.temporal_proj = nn.Linear(d_model // 4, d_model)
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: 输入序列 [B, L, N]
            
        Returns:
            embedded: 变量嵌入 [B, N, d_model]
            temporal_features: 时序特征 [B, N, d_model]
        """
        B, L, N = x.shape
        
        # [B, L, N] -> [B, N, L]
        x_permuted = x.permute(0, 2, 1)
        
        # 值嵌入
        value_emb = self.value_embedding(x_permuted)  # [B, N, d_model]
        
        # 时序卷积特征
        # [B, N, L] -> [B*N, 1, L]
        x_reshaped = x_permuted.reshape(B * N, 1, L)
        temporal_conv = self.temporal_conv(x_reshaped)  # [B*N, d_model//4, L]
        temporal_conv = temporal_conv.mean(dim=-1)  # [B*N, d_model//4]
        temporal_conv = temporal_conv.reshape(B, N, -1)  # [B, N, d_model//4]
        temporal_features = self.temporal_proj(temporal_conv)  # [B, N, d_model]
        
        # 融合值嵌入和时序特征
        combined = torch.cat([value_emb, temporal_features], dim=-1)
        embedded = self.fusion(combined)
        
        # 添加变量嵌入 (如果存在)
        if self.variate_embedding is not None:
            variate_ids = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
            embedded = embedded + self.variate_embedding(variate_ids)
        
        embedded = self.dropout(embedded)
        
        return embedded, temporal_features
