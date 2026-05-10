export CUDA_VISIBLE_DEVICES=0

# Memory Footprint Analysis on ETTh1
# Models: Transformer, Informer, SparseCycleTransformer
# Varied Input Lengths: 96, 192, 336, 720, 1440 (Extended for stress test)
# Fixed Output Length: 96
# Goal: Observe peak memory usage (requires external monitoring or parsing 'nvidia-smi' during run)
# Note: This script runs the training; users can monitor gpu/cpu memory.

echo "=== Memory Footprint Analysis: ETTh1 ==="

# 1. Transformer
echo "--- Transformer ---"
for seq_len in 96 336 720 1440
do
  echo "Running Transformer with seq_len=$seq_len..."
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_${seq_len}_96_Transformer_Mem \
    --model Transformer \
    --data ETTh1 \
    --features M \
    --seq_len $seq_len \
    --pred_len 96 \
    --e_layers 2 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Eff_Mem_Transformer' \
    --d_model 256 \
    --d_ff 256 \
    --batch_size 32 \
    --itr 1
done

# 2. Informer
echo "--- Informer ---"
for seq_len in 96 336 720 1440
do
  echo "Running Informer with seq_len=$seq_len..."
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_${seq_len}_96_Informer_Mem \
    --model Informer \
    --data ETTh1 \
    --features M \
    --seq_len $seq_len \
    --pred_len 96 \
    --e_layers 2 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Eff_Mem_Informer' \
    --d_model 256 \
    --d_ff 256 \
    --batch_size 32 \
    --itr 1
done

# 3. SparseCycleTransformer
echo "--- SparseCycleTransformer ---"
for seq_len in 96 336 720 1440
do
  echo "Running SparseCycleTransformer with seq_len=$seq_len..."
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_${seq_len}_96_SparseCycle_Mem \
    --model SparseCycleTransformer \
    --data ETTh1 \
    --features M \
    --seq_len $seq_len \
    --pred_len 96 \
    --e_layers 2 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Eff_Mem_SparseCycle' \
    --d_model 256 \
    --d_ff 256 \
    --num_semantic_tokens 8 \
    --sparsity_ratio 0.5 \
    --batch_size 32 \
    --itr 1
done
