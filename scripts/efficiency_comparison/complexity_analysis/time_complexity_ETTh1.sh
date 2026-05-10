export CUDA_VISIBLE_DEVICES=0

# Time Complexity Analysis on ETTh1
# Models: Transformer, Informer, iTransformer, PatchTST, SparseTransformer, SparseCycleTransformer
# Varied Input Lengths: 96, 192, 336, 720
# Fixed Output Length: 96
# Goal: Observe 'cost time' in logs to analyze complexity (O(L^2) vs O(L) etc.)

echo "=== Time Complexity Analysis: ETTh1 ==="

# 1. Transformer (Baseline, O(L^2))
echo "--- Transformer ---"
for seq_len in 96 192 336 720
do
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

# 2. Informer (Baseline, O(L log L))
echo "--- Informer ---"
for seq_len in 96 192 336 720
do
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_${seq_len}_96_Informer \
    --model Informer \
    --data ETTh1 \
    --features M \
    --seq_len $seq_len \
    --pred_len 96 \
    --e_layers 2 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Eff_Time_Informer' \
    --d_model 256 \
    --d_ff 256 \
    --itr 1
done

# 3. iTransformer (Inverted Attention, O(N^2) where N=variates)
echo "--- iTransformer ---"
for seq_len in 96 192 336 720
do
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_${seq_len}_96_iTransformer \
    --model iTransformer \
    --data ETTh1 \
    --features M \
    --seq_len $seq_len \
    --pred_len 96 \
    --e_layers 2 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Eff_Time_iTransformer' \
    --d_model 256 \
    --d_ff 256 \
    --itr 1
done

# 4. PatchTST (Patch-based, O(L/P)^2 where P=patch_size)
echo "--- PatchTST ---"
for seq_len in 96 192 336 720
do
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_${seq_len}_96_PatchTST \
    --model PatchTST \
    --data ETTh1 \
    --features M \
    --seq_len $seq_len \
    --pred_len 96 \
    --e_layers 2 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Eff_Time_PatchTST' \
    --d_model 256 \
    --d_ff 256 \
    --itr 1
done

# 5. SparseTransformer (Sparse Attention, O(L * sqrt(L)))
echo "--- SparseTransformer ---"
for seq_len in 96 192 336 720
do
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_${seq_len}_96_SparseTransformer \
    --model SparseTransformer \
    --data ETTh1 \
    --features M \
    --seq_len $seq_len \
    --pred_len 96 \
    --e_layers 2 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Eff_Time_SparseTransformer' \
    --d_model 256 \
    --d_ff 256 \
    --itr 1
done

# 6. SparseCycleTransformer (Ours, O(kL) where k=active points)
echo "--- SparseCycleTransformer ---"
for seq_len in 96 192 336 720
do
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_${seq_len}_96_SparseCycle \
    --model SparseCycleTransformer \
    --data ETTh1 \
    --features M \
    --seq_len $seq_len \
    --pred_len 96 \
    --e_layers 2 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Eff_Time_SparseCycle' \
    --d_model 256 \
    --d_ff 256 \
    --num_semantic_tokens 8 \
    --sparsity_ratio 0.5 \
    --itr 1
done

echo "=== Analysis Complete ==="
