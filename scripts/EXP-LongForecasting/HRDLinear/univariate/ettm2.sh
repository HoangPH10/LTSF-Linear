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

# ETTm2, univariate results, pred_len= 96 192 336 720
for pred_len in 96 192
do
uv run python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTm2 \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 1 \
  --revin_mode std \
  --fourier_b 8 \
  --poly_deg 2 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.001 --feature S >logs/LongForecasting/$model_name'_'fS_ETTm2_$seq_len'_'$pred_len.log
done

for pred_len in 336 720
do
uv run python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTm2 \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 1 \
  --revin_mode std \
  --fourier_b 8 \
  --poly_deg 2 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.01 --feature S >logs/LongForecasting/$model_name'_'fS_ETTm2_$seq_len'_'$pred_len.log
done
