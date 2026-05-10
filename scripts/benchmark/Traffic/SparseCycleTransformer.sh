export CUDA_VISIBLE_DEVICES=0

model_name=SparseCycleTransformer

# SparseCycleTransformer on Traffic dataset (862 variates)
# 超高维数据集 - 稀疏门控的计算效率优势最为显著

# Traffic 96 -> 96
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id Traffic_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 32 \
  --learning_rate 0.0005 \
  --num_semantic_tokens 24 \
  --sparsity_ratio 0.3 \
  --lambda_sparse 0.02 \
  --lambda_period 0.01 \
  --itr 1

# Traffic 96 -> 192
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id Traffic_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 32 \
  --learning_rate 0.0005 \
  --num_semantic_tokens 24 \
  --sparsity_ratio 0.3 \
  --lambda_sparse 0.02 \
  --lambda_period 0.01 \
  --itr 1

# Traffic 96 -> 336
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id Traffic_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 32 \
  --learning_rate 0.0005 \
  --num_semantic_tokens 32 \
  --sparsity_ratio 0.25 \
  --lambda_sparse 0.02 \
  --lambda_period 0.01 \
  --itr 1

# Traffic 96 -> 720
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id Traffic_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 32 \
  --learning_rate 0.0005 \
  --num_semantic_tokens 32 \
  --sparsity_ratio 0.2 \
  --lambda_sparse 0.02 \
  --lambda_period 0.01 \
  --itr 1
