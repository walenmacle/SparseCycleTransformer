"""
Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting
Paper: https://openreview.net/forum?id=vSVLM2j9eie (ICLR 2023)

Key components:
1. Dimension-Segment-Wise (DSW) Embedding
2. Two-Stage Attention (TSA) Layer: Cross-Time + Cross-Dimension
3. Hierarchical Encoder-Decoder (HED)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat


class DSWEmbedding(nn.Module):
    """
    Dimension-Segment-Wise Embedding
    Embeds data points in each dimension by forming segments over time.
    """
    def __init__(self, seg_len, d_model):
        super(DSWEmbedding, self).__init__()
        self.seg_len = seg_len
        self.linear = nn.Linear(seg_len, d_model)

    def forward(self, x):
        # x: [B, L, D] -> segment and embed
        B, L, D = x.shape
        # Pad if necessary
        if L % self.seg_len != 0:
            pad_len = self.seg_len - (L % self.seg_len)
            x = F.pad(x, (0, 0, 0, pad_len), mode='constant', value=0)
            L = L + pad_len
        
        n_seg = L // self.seg_len
        # [B, L, D] -> [B, n_seg, seg_len, D] -> [B, D, n_seg, seg_len]
        x = x.view(B, n_seg, self.seg_len, D).permute(0, 3, 1, 2)
        # [B, D, n_seg, seg_len] -> [B, D, n_seg, d_model]
        x = self.linear(x)
        return x  # [B, D, n_seg, d_model]


class TwoStageAttentionLayer(nn.Module):
    """
    Two-Stage Attention (TSA) Layer
    Stage 1: Cross-Time Attention (within each dimension)
    Stage 2: Cross-Dimension Attention (across dimensions at each time segment)
    """
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1):
        super(TwoStageAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        
        # Cross-Time Self-Attention
        self.time_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.time_norm1 = nn.LayerNorm(d_model)
        self.time_norm2 = nn.LayerNorm(d_model)
        self.time_ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Cross-Dimension Self-Attention
        self.dim_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dim_norm1 = nn.LayerNorm(d_model)
        self.dim_norm2 = nn.LayerNorm(d_model)
        self.dim_ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        """
        Args:
            x: [B, D, n_seg, d_model]
        Returns:
            x: [B, D, n_seg, d_model]
        """
        B, D, n_seg, d_model = x.shape
        
        # Stage 1: Cross-Time Attention (within each dimension)
        # Reshape to [B*D, n_seg, d_model] for attention over time segments
        x_time = x.reshape(B * D, n_seg, d_model)
        attn_out, _ = self.time_attention(x_time, x_time, x_time)
        x_time = self.time_norm1(x_time + attn_out)
        x_time = self.time_norm2(x_time + self.time_ffn(x_time))
        x_time = x_time.reshape(B, D, n_seg, d_model)
        
        # Stage 2: Cross-Dimension Attention (at each time segment)
        # [B, D, n_seg, d_model] -> [B, n_seg, D, d_model] -> [B*n_seg, D, d_model]
        x_dim = x_time.permute(0, 2, 1, 3).reshape(B * n_seg, D, d_model)
        attn_out, _ = self.dim_attention(x_dim, x_dim, x_dim)
        x_dim = self.dim_norm1(x_dim + attn_out)
        x_dim = self.dim_norm2(x_dim + self.dim_ffn(x_dim))
        
        # Reshape back to [B, D, n_seg, d_model]
        x_out = x_dim.reshape(B, n_seg, D, d_model).permute(0, 2, 1, 3)
        
        return x_out


class SegmentMerging(nn.Module):
    """
    Segment Merging Layer for Hierarchical Encoder
    Reduces the number of segments by 2x
    """
    def __init__(self, d_model):
        super(SegmentMerging, self).__init__()
        self.linear = nn.Linear(2 * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        Args:
            x: [B, D, n_seg, d_model]
        Returns:
            x: [B, D, n_seg//2, d_model]
        """
        B, D, n_seg, d_model = x.shape
        
        # Handle odd number of segments by padding
        if n_seg % 2 != 0:
            x = F.pad(x, (0, 0, 0, 1), mode='constant', value=0)
            n_seg = n_seg + 1
            
        # [B, D, n_seg, d_model] -> [B, D, n_seg//2, 2*d_model]
        x = x.reshape(B, D, n_seg // 2, 2 * d_model)
        x = self.linear(x)
        x = self.norm(x)
        return x


class CrossformerEncoder(nn.Module):
    """
    Hierarchical Encoder with TSA layers and Segment Merging
    """
    def __init__(self, d_model, n_heads, e_layers, d_ff=None, dropout=0.1):
        super(CrossformerEncoder, self).__init__()
        
        self.layers = nn.ModuleList()
        self.merge_layers = nn.ModuleList()
        
        for i in range(e_layers):
            self.layers.append(TwoStageAttentionLayer(d_model, n_heads, d_ff, dropout))
            if i < e_layers - 1:  # No merging after the last layer
                self.merge_layers.append(SegmentMerging(d_model))
                
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        Args:
            x: [B, D, n_seg, d_model]
        Returns:
            enc_outs: list of encoder outputs at each scale
        """
        enc_outs = []
        
        for i, layer in enumerate(self.layers):
            x = layer(x)
            enc_outs.append(x)
            if i < len(self.merge_layers):
                x = self.merge_layers[i](x)
                
        return enc_outs


class CrossformerDecoder(nn.Module):
    """
    Decoder that makes predictions at each scale and sums them
    """
    def __init__(self, d_model, n_heads, d_layers, seg_len, pred_len, d_ff=None, dropout=0.1):
        super(CrossformerDecoder, self).__init__()
        
        self.pred_len = pred_len
        self.seg_len = seg_len
        
        # Prediction heads for each scale
        self.pred_heads = nn.ModuleList()
        for i in range(d_layers):
            scale_factor = 2 ** i
            self.pred_heads.append(nn.Linear(d_model, seg_len * scale_factor))
            
    def forward(self, enc_outs, D):
        """
        Args:
            enc_outs: list of encoder outputs at each scale
            D: number of dimensions
        Returns:
            pred: [B, pred_len, D]
        """
        B = enc_outs[0].shape[0]
        
        # Accumulate predictions from each scale
        predictions = []
        for i, (enc_out, pred_head) in enumerate(zip(enc_outs, self.pred_heads)):
            # enc_out: [B, D, n_seg, d_model]
            B, D_enc, n_seg, d_model = enc_out.shape
            
            # Predict for each scale
            # [B, D, n_seg, d_model] -> [B, D, n_seg, seg_len * scale]
            pred = pred_head(enc_out)
            
            # Reshape to [B, D, pred_len_at_scale]
            pred = pred.reshape(B, D_enc, -1)
            
            # Interpolate to target pred_len
            pred = F.interpolate(pred, size=self.pred_len, mode='linear', align_corners=False)
            
            predictions.append(pred)
        
        # Sum predictions from all scales
        final_pred = sum(predictions) / len(predictions)
        
        # [B, D, pred_len] -> [B, pred_len, D]
        final_pred = final_pred.permute(0, 2, 1)
        
        return final_pred


class Model(nn.Module):
    """
    Crossformer: Transformer Utilizing Cross-Dimension Dependency
    for Multivariate Time Series Forecasting
    
    Paper: https://openreview.net/forum?id=vSVLM2j9eie
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.n_heads = configs.n_heads
        self.e_layers = configs.e_layers
        self.d_layers = getattr(configs, 'd_layers', configs.e_layers)
        self.d_ff = configs.d_ff
        self.dropout = configs.dropout
        
        # Segment length (hyperparameter)
        self.seg_len = getattr(configs, 'seg_len', 12)
        
        # DSW Embedding
        self.embedding = DSWEmbedding(self.seg_len, self.d_model)
        
        # Encoder
        self.encoder = CrossformerEncoder(
            self.d_model, self.n_heads, self.e_layers, self.d_ff, self.dropout
        )
        
        # Decoder
        self.decoder = CrossformerDecoder(
            self.d_model, self.n_heads, self.e_layers, 
            self.seg_len, self.pred_len, self.d_ff, self.dropout
        )
        
        # Output projection
        self.out_proj = nn.Linear(self.enc_in, configs.c_out)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        Args:
            x_enc: [B, L, D] encoder input
            x_mark_enc: encoder timestamp (unused)
            x_dec: decoder input (unused, encoder-only)
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
        
        # Embedding: [B, L, D] -> [B, D, n_seg, d_model]
        enc_emb = self.embedding(x_enc)
        
        # Encoder: get multi-scale representations
        enc_outs = self.encoder(enc_emb)
        
        # Decoder: predict from multi-scale outputs
        dec_out = self.decoder(enc_outs, D)
        
        # Project to output dimension
        if D != self.out_proj.out_features:
            dec_out = self.out_proj(dec_out)
        
        # De-normalization
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        
        return dec_out
