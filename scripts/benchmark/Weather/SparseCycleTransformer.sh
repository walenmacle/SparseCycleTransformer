export CUDA_VISIBLE_DEVICES=0

model_name=SparseCycleTransformer

# SparseCycleTransformer on Weather dataset (21 variates)
# 中等规模数据集 - 周期性特征明显

# Weather 96 -> 96
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id Weather_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --num_semantic_tokens 8 \
  --sparsity_ratio 0.5 \
  --lambda_sparse 0.01 \
  --lambda_period 0.02 \
  --itr 1

# Weather 96 -> 192
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id Weather_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 2 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --num_semantic_tokens 8 \
  --sparsity_ratio 0.5 \
  --lambda_sparse 0.01 \
  --lambda_period 0.02 \
  --itr 1

# Weather 96 -> 336
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id Weather_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 2 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --num_semantic_tokens 12 \
  --sparsity_ratio 0.4 \
  --lambda_sparse 0.01 \
  --lambda_period 0.02 \
  --itr 1

# Weather 96 -> 720
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id Weather_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 2 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --num_semantic_tokens 16 \
  --sparsity_ratio 0.3 \
  --lambda_sparse 0.01 \
  --lambda_period 0.02 \
  --itr 1
