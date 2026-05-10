export CUDA_VISIBLE_DEVICES=0

# Time Complexity Analysis: Transformer on ETTh1
# O(L^2) Complexity Test
# Varied Input Lengths: 96, 192, 336, 720

echo "=== Time Complexity: Transformer (ETTh1) ==="

for seq_len in 96 192 336 720
do
  echo "Running seq_len=$seq_len..."
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_${seq_len}_96_Transformer \
    --model Transformer \
    --data ETTh1 \
    --features M \
    --seq_len $seq_len \
    --pred_len 96 \
    --e_layers 2 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Eff_Time_Transformer' \
    --d_model 256 \
    --d_ff 256 \
    --itr 1
done
