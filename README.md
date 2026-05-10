## Usage 

1. Install Pytorch and the necessary dependencies.

```
pip install -r requirements.txt
```



1. Train and evaluate the model. We provide all the above tasks under the folder ./scripts/. You can reproduce the results as the following examples:

```
# Multivariate forecasting with SparseCycleTransformer
bash ./scripts/benchmark/Traffic/SparseCycleTransformer.sh

```


## SparseCycleTransformer Implementation

This section documents the complete implementation of the **SparseCycleTransformer**, an innovative model fusing **structural sparsity** with **periodic semantic understanding** for multivariate long-term time series forecasting.

> **核心问题**: 如何高效地从长序列中捕获周期性模式和变量间依赖关系  
> **我们的方案**: 双通路架构 = 语义通路(全局周期理解) + 稀疏门控通路(高效局部计算)

---

### 🎯 模型核心思想

SparseCycleTransformer 设计了一个**双通路 Transformer 架构**：
- **语义通路 (Semantic Pathway)**: 作为模型的"宏观大脑"，通过傅里叶变换发现全局周期规律
- **稀疏门控通路 (Sparse Gated Pathway)**: 在语义指导下，仅对"关键变量"进行密集计算，实现计算资源的精准投放

---


### 🔑 Key Components Summary

#### [SparseCycleEmbed.py](layers/SparseCycleEmbed.py)

| Class                           | Function                         | 输入 → 输出                 |
| ------------------------------- | -------------------------------- | --------------------------- |
| `FourierPeriodicTokenGenerator` | FFT 分析生成周期语义令牌         | `[B,L,N]` → `[B,K,d_model]` |
| `SparseDataEmbedding`           | 稀疏场景数据嵌入                 | `[B,L,N]` → `[B,N,d_model]` |
| `TemporalPositionEncoding`      | 正弦位置编码                     | `seq_len` → `[1,L,d_model]` |
| `VariateAwareEmbedding`         | 变量感知嵌入 (iTransformer 风格) | `[B,L,N]` → `[B,N,d_model]` |

#### [SparseCycleAttention.py](layers/SparseCycleAttention.py)

| Class                       | Function             | 核心机制                            |
| --------------------------- | -------------------- | ----------------------------------- |
| `SparseGatePredictor`       | 语义引导的门控预测   | Semantic-guided + STE               |
| `SparseGatedAttention`      | 稀疏门控注意力       | Active: Attention, Inactive: Linear |
| `DualPathwayFusion`         | 双通路交叉注意力融合 | Cross-Attention + Gated Fusion      |
| `SemanticEncoder`           | 轻量级语义编码器     | 1-layer Transformer                 |
| `SparseEncoderLayer`        | 稀疏编码器层         | Sparse Attention + FFN              |
| `PeriodicConsistencyModule` | 周期一致性损失计算   | Cosine Similarity                   |

#### [SparseCycleTransformer.py](model/SparseCycleTransformer.py)

主模型类整合所有组件：

```python
class Model(nn.Module):
    # 语义通路
    self.periodic_token_generator  # FourierPeriodicTokenGenerator
    self.semantic_encoder          # SemanticEncoder (1 layer)
    
    # 稀疏门控通路
    self.sparse_embedding          # VariateAwareEmbedding
    self.gate_predictor            # SparseGatePredictor
    self.sparse_encoder_layers     # SparseEncoderLayer × L
    
    # 双通路融合
    self.dual_pathway_fusion       # DualPathwayFusion
    
    # 辅助模块
    self.periodic_consistency      # PeriodicConsistencyModule
    
    # 输出层
    self.norm                      # LayerNorm
    self.projector                 # Linear(d_model, pred_len)
```

**辅助方法**:
- `compute_auxiliary_loss()`: 计算稀疏和周期一致性损失
- `get_total_loss(pred_loss)`: 计算联合损失
- `get_sparsity_stats()`: 获取稀疏统计信息



### 💻 Usage

#### Basic Usage

```python
from model.SparseCycleTransformer import Model

class Args:
    seq_len = 96
    pred_len = 96
    enc_in = 7
    d_model = 256
    n_heads = 8
    d_ff = 256
    e_layers = 2
    dropout = 0.1
    activation = 'gelu'
    output_attention = False
    use_norm = True
    # SparseCycleTransformer specific
    num_semantic_tokens = 8
    sparsity_ratio = 0.5
    lambda_sparse = 0.01
    lambda_period = 0.01

model = Model(Args())
output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
```

#### Training with Joint Loss

```python
# Forward pass
output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

# Calculate prediction loss
pred_loss = criterion(output, y_true)

# Get total loss with auxiliary losses
total_loss, loss_dict = model.get_total_loss(pred_loss)
# loss_dict = {
#     'pred_loss': ...,
#     'sparse_loss': ...,
#     'period_loss': ...,
#     'total_loss': ...
# }

# Backward and optimize
total_loss.backward()
optimizer.step()

# Optional: Monitor sparsity statistics
stats = model.get_sparsity_stats()
# stats = {
#     'actual_sparsity': ...,
#     'active_ratio': ...,
#     'target_sparsity': ...,
#     'num_active_points': ...
# }
```

#### Command Line Interface

```bash
python -u run.py \
  --is_training 1 \
  --model SparseCycleTransformer \
  --data ETTh1 \
  --seq_len 96 \
  --pred_len 96 \
  --d_model 256 \
  --n_heads 8 \
  --e_layers 2 \
  --num_semantic_tokens 8 \
  --sparsity_ratio 0.5 \
  --lambda_sparse 0.01 \
  --lambda_period 0.01
```

Email:1948673480@qq.com





