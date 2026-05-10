export CUDA_VISIBLE_DEVICES=0

model_name=SparseCycleTransformer

# SparseCycleTransformer on ETTh2 dataset

# ETTh2 96 -> 96
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_96 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --num_semantic_tokens 8 \
  --sparsity_ratio 0.5 \
  --lambda_sparse 0.01 \
  --lambda_period 0.01 \
  --itr 1

# ETTh2 96 -> 192
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_192 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --num_semantic_tokens 8 \
  --sparsity_ratio 0.5 \
  --lambda_sparse 0.01 \
  --lambda_period 0.01 \
  --itr 1

# ETTh2 96 -> 336
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_336 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --num_semantic_tokens 12 \
  --sparsity_ratio 0.4 \
  --lambda_sparse 0.01 \
  --lambda_period 0.01 \
  --itr 1

# ETTh2 96 -> 720
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_720 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --num_semantic_tokens 16 \
  --sparsity_ratio 0.3 \
  --lambda_sparse 0.01 \
  --lambda_period 0.01 \
  --itr 1
