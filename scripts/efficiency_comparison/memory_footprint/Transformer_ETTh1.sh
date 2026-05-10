export CUDA_VISIBLE_DEVICES=0

# Memory Footprint Analysis: Transformer on ETTh1
# Stress Test: Large Batch, Long Sequences
# Varied Input Lengths: 96, 336, 720, 1440

echo "=== Memory Analysis: Transformer (ETTh1) ==="

for seq_len in 96 336 720 1440
do
  echo "Running seq_len=$seq_len..."
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
