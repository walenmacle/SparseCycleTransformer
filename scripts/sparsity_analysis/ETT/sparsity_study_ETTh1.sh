export CUDA_VISIBLE_DEVICES=0

model_name=SparseCycleTransformer

# Sparsity Ratio Study on ETTh1 dataset
# 测试不同稀疏率对预测性能的影响 (pred_len 96 & 336)

echo "=== Sparsity Ratio Study on ETTh1 ==="
echo "Testing sparsity ratios: 0.2, 0.3, 0.4, 0.5, 0.7, 1.0"

for ratio in 0.2 0.3 0.4 0.5 0.7 1.0; do
  
  # Special handling for ratio 1.0 (lambda_sparse = 0)
  if [ "$ratio" == "1.0" ]; then
    lambda_s=0.0
  else
    lambda_s=0.01
  fi
  
  # pred_len 96
  echo ">>> Testing sparsity_ratio=$ratio, pred_len=96"
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_96_96_spa${ratio} \
    --model $model_name \
    --data ETTh1 \
    --features M \
    --seq_len 96 \
    --pred_len 96 \
    --e_layers 2 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'SparsityStudy' \
    --d_model 256 \
    --d_ff 256 \
    --num_semantic_tokens 8 \
    --sparsity_ratio $ratio \
    --lambda_sparse $lambda_s \
    --lambda_period 0.01 \
    --itr 1

  # pred_len 336
  echo ">>> Testing sparsity_ratio=$ratio, pred_len=336"
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_96_336_spa${ratio} \
    --model $model_name \
    --data ETTh1 \
    --features M \
    --seq_len 96 \
    --pred_len 336 \
    --e_layers 2 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'SparsityStudy' \
    --d_model 256 \
    --d_ff 256 \
    --num_semantic_tokens 8 \
    --sparsity_ratio $ratio \
    --lambda_sparse $lambda_s \
    --lambda_period 0.01 \
    --itr 1

done

echo "=== Sparsity Ratio Study Complete ==="
