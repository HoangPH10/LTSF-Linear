if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

if [ ! -d "./logs/LongForecasting/univariate" ]; then
    mkdir ./logs/LongForecasting/univariate
fi
seq_len=336
model_name=HRDLinear

# ETTh2, univariate results, pred_len= 24 48 96 192 336 720
for pred_len in 24 48 96 192 336 720
do
uv run python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTh2 \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 1 \
  --revin_mode std \
  --fourier_b 6 \
  --poly_deg 2 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.005 --feature S >logs/LongForecasting/$model_name'_'fS_ETTh2_$seq_len'_'$pred_len.log
done
