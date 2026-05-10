export CUDA_VISIBLE_DEVICES=0

model_name=SparseCycleTransformer

# Ablation: Without Sparse Gating on ETTm2

echo "=== Ablation: Without Sparse Gating (ETTm2) ==="

# 96 -> 96
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_96_full_attn \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Ablation_FullAttn' \
  --d_model 256 \
  --d_ff 256 \
  --num_semantic_tokens 8 \
  --sparsity_ratio 1.0 \
  --lambda_sparse 0.01 \
  --lambda_period 0.01 \
  --itr 1

# 96 -> 192
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_192_full_attn \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Ablation_FullAttn' \
  --d_model 256 \
  --d_ff 256 \
  --num_semantic_tokens 8 \
  --sparsity_ratio 1.0 \
  --lambda_sparse 0.01 \
  --lambda_period 0.01 \
  --itr 1

# 96 -> 336
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_336_full_attn \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Ablation_FullAttn' \
  --d_model 256 \
  --d_ff 256 \
  --num_semantic_tokens 16 \
  --sparsity_ratio 1.0 \
  --lambda_sparse 0.01 \
  --lambda_period 0.01 \
  --itr 1

# 96 -> 720
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_720_full_attn \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Ablation_FullAttn' \
  --d_model 256 \
  --d_ff 256 \
  --num_semantic_tokens 24 \
  --sparsity_ratio 1.0 \
  --lambda_sparse 0.01 \
  --lambda_period 0.01 \
  --itr 1
