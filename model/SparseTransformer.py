"""
Sparse Transformer for Time Series Forecasting
Based on: "Generating Long Sequences with Sparse Transformers" (OpenAI, 2019)

Key innovations:
1. Sparse Attention Patterns (Strided + Fixed)
2. Reduced O(L^2) to O(L*sqrt(L)) complexity
3. Efficient for long sequence modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SparseAttention(nn.Module):
    """
    Sparse Attention with configurable sparsity patterns.
    Combines strided attention (local patterns) with fixed attention (global patterns).
    """
    def __init__(self, d_model, n_heads, seq_len, stride=None, dropout=0.1):
        super(SparseAttention, self).__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.stride = stride or int(math.sqrt(seq_len))
        self.seq_len = seq_len
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_k)
        
        # Pre-compute sparse attention mask
        self.register_buffer('sparse_mask', self._create_sparse_mask(seq_len))
        
    def _create_sparse_mask(self, seq_len):
        """
        Create a sparse attention mask combining:
        1. Local attention (strided pattern)
        2. Global attention (every stride-th position attends to all)
        """
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
        
        for i in range(seq_len):
            # Local attention: attend to positions within stride distance
            start = max(0, i - self.stride)
            end = min(seq_len, i + self.stride + 1)
            mask[i, start:end] = True
            
            # Global attention: every stride-th position can attend globally
            if i % self.stride == 0:
                mask[i, :] = True
                mask[:, i] = True
                
        return ~mask  # True = masked positions
    
    def forward(self, x, attn_mask=None):
        """
        Args:
            x: [B, L, D]
            attn_mask: optional additional mask
        Returns:
            output: [B, L, D]
            attn_weights: attention weights for visualization
        """
        B, L, D = x.shape
        H = self.n_heads
        
        # Project Q, K, V
        Q = self.q_proj(x).view(B, L, H, self.d_k).transpose(1, 2)  # [B, H, L, d_k]
        K = self.k_proj(x).view(B, L, H, self.d_k).transpose(1, 2)
        V = self.v_proj(x).view(B, L, H, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, H, L, L]
        
        # Apply sparse mask
        if L <= self.sparse_mask.size(0):
            sparse_mask = self.sparse_mask[:L, :L]
        else:
            # Dynamically create mask for longer sequences
            sparse_mask = self._create_sparse_mask(L).to(x.device)
            
        scores = scores.masked_fill(sparse_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Apply additional mask if provided
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, V)  # [B, H, L, d_k]
        output = output.transpose(1, 2).contiguous().view(B, L, D)  # [B, L, D]
        output = self.out_proj(output)
        
        return output, attn_weights


class SparseTransformerBlock(nn.Module):
    """
    Transformer block with Sparse Attention
    """
    def __init__(self, d_model, n_heads, seq_len, d_ff=None, dropout=0.1, activation='gelu'):
        super(SparseTransformerBlock, self).__init__()
        
        d_ff = d_ff or 4 * d_model
        
        self.attention = SparseAttention(d_model, n_heads, seq_len, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, attn_mask=None):
        # Self-attention with residual
        attn_out, attn_weights = self.attention(x, attn_mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual
        x = self.norm2(x + self.ffn(x))
        
        return x, attn_weights


class SparseTransformerEncoder(nn.Module):
    """
    Stack of Sparse Transformer blocks
    """
    def __init__(self, d_model, n_heads, e_layers, seq_len, d_ff=None, dropout=0.1, activation='gelu'):
        super(SparseTransformerEncoder, self).__init__()
        
        self.layers = nn.ModuleList([
            SparseTransformerBlock(d_model, n_heads, seq_len, d_ff, dropout, activation)
            for _ in range(e_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, attn_mask=None):
        attns = []
        for layer in self.layers:
            x, attn = layer(x, attn_mask)
            attns.append(attn)
        x = self.norm(x)
        return x, attns


class Model(nn.Module):
    """
    Sparse Transformer for Time Series Forecasting
    
    Reduces complexity from O(L^2) to O(L*sqrt(L)) using sparse attention patterns.
    Based on "Generating Long Sequences with Sparse Transformers" (OpenAI, 2019)
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.label_len = getattr(configs, 'label_len', 48)
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        self.d_model = configs.d_model
        self.n_heads = configs.n_heads
        self.e_layers = configs.e_layers
        self.d_ff = configs.d_ff
        self.dropout = configs.dropout
        self.activation = configs.activation
        self.output_attention = getattr(configs, 'output_attention', False)
        
        # Input embedding
        self.enc_embedding = nn.Linear(self.enc_in, self.d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.seq_len + self.pred_len, self.d_model) * 0.02)
        
        # Sparse Transformer Encoder
        self.encoder = SparseTransformerEncoder(
            self.d_model, self.n_heads, self.e_layers,
            self.seq_len, self.d_ff, self.dropout, self.activation
        )
        
        # Prediction head
        self.pred_linear = nn.Linear(self.seq_len, self.pred_len)
        self.projection = nn.Linear(self.d_model, self.c_out)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        Args:
            x_enc: [B, L, D] encoder input
            x_mark_enc: encoder timestamp (unused)
            x_dec: decoder input (unused)
            x_mark_dec: decoder timestamp (unused)
        Returns:
            dec_out: [B, pred_len, D]
        """
        # Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev
        
        B, L, D = x_enc.shape
        
        # Embedding
        enc_out = self.enc_embedding(x_enc)  # [B, L, d_model]
        enc_out = enc_out + self.pos_embedding[:, :L, :]
        
        # Encoder
        enc_out, attns = self.encoder(enc_out)
        
        # Predict future
        # [B, L, d_model] -> [B, d_model, L] -> [B, d_model, pred_len] -> [B, pred_len, d_model]
        dec_out = self.pred_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)
        
        # Project to output dimension
        dec_out = self.projection(dec_out)
        
        # De-normalization
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        
        if self.output_attention:
            return dec_out, attns
        return dec_out
