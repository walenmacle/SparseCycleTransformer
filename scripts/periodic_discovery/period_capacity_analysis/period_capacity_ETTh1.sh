export CUDA_VISIBLE_DEVICES=0

# Period Capacity Analysis on ETTh1

# Hypothesis: '32' is optimal for ETTh1 (Daily, Weekly, etc.), going too low (2) misses periods, too high (16) adds noise.

echo "=== Period Capacity Analysis: ETTh1 ==="


for n_tokens in 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48
do
  echo "Running with num_semantic_tokens=$n_tokens..."
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_96_96_Period_${n_tokens} \
    --model SparseCycleTransformer \
    --data ETTh1 \
    --features M \
    --seq_len 96 \
    --pred_len 96 \
    --e_layers 2 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Period_Discovery_Capacity' \
    --d_model 256 \
    --d_ff 256 \
    --num_semantic_tokens $n_tokens \
    --sparsity_ratio 0.5 \
    --itr 1
done
