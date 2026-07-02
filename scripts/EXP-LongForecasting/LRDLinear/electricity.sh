if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=LRDLinear

for pred_len in 96 192 336 720
do
uv run python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path electricity.csv \
  --model_id Electricity_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 321 \
  --revin_mode std \
  --lr_rank 32 \
  --des 'Exp' \
  --itr 1 --batch_size 16 --learning_rate 0.001 >logs/LongForecasting/$model_name'_'electricity_$seq_len'_'$pred_len.log
done
