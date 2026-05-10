export CUDA_VISIBLE_DEVICES=0

model_name=Crossformer

# Crossformer on SolarEnergy dataset (137 variates)
# Medium-high dimensional dataset

# SolarEnergy 96 -> 96
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/Solar/ \
  --data_path solar_AL.txt \
  --model_id Solar_96_96 \
  --model $model_name \
  --data Solar \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 512 \
  --batch_size 256 \
  --learning_rate 0.0005 \
  --itr 1

# SolarEnergy 96 -> 192
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/Solar/ \
  --data_path solar_AL.txt \
  --model_id Solar_96_192 \
  --model $model_name \
  --data Solar \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 512 \
  --batch_size 256 \
  --learning_rate 0.0005 \
  --itr 1

# SolarEnergy 96 -> 336
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/Solar/ \
  --data_path solar_AL.txt \
  --model_id Solar_96_336 \
  --model $model_name \
  --data Solar \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 512 \
  --batch_size 256 \
  --learning_rate 0.0005 \
  --itr 1

# SolarEnergy 96 -> 720
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/Solar/ \
  --data_path solar_AL.txt \
  --model_id Solar_96_720 \
  --model $model_name \
  --data Solar \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 512 \
  --batch_size 256 \
  --learning_rate 0.0005 \
  --itr 1
