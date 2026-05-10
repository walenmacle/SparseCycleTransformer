export CUDA_VISIBLE_DEVICES=0

model_name=SparseCycleTransformer

# Semantic Token Study on ETTm2 dataset
# 测试不同语义令牌数量对预测性能的影响
# Fixed sparsity_ratio = 0.5, pred_len = 96

echo "=== Semantic Token Study on ETTm2 ==="
echo "Testing num_semantic_tokens: 4, 8, 16"

for n_tokens in 4 8 16; do
  
  echo ">>> Testing num_semantic_tokens=$n_tokens"
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm2.csv \
    --model_id ETTm2_96_96_tok${n_tokens} \
    --model $model_name \
    --data ETTm2 \
    --features M \
    --seq_len 96 \
    --pred_len 96 \
    --e_layers 2 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'TokenStudy' \
    --d_model 256 \
    --d_ff 256 \
    --num_semantic_tokens $n_tokens \
    --sparsity_ratio 0.5 \
    --lambda_sparse 0.01 \
    --lambda_period 0.01 \
    --itr 1

done

echo "=== Semantic Token Study Complete ==="
