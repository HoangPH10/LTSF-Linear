if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=MLPLinear

for pred_len in 96 192 336 720
do
uv run python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path weather.csv \
  --model_id weather_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 21 \
  --revin_mode std \
  --mlp_hidden 32 \
  --mlp_dropout 0.2 \
  --mlp_branches seasonal \
  --des 'Exp' \
  --itr 1 --batch_size 16 >logs/LongForecasting/$model_name'_'Weather_$seq_len'_'$pred_len.log
done
