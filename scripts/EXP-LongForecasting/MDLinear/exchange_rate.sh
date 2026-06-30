if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=MDLinear

for pred_len in 96 192
do
uv run python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 8 \
  --decomp_kernels 9 25 49 \
  --use_revin \
  --revin_mode std \
  --des 'Exp' \
  --itr 1 --batch_size 8 --learning_rate 0.0005 >logs/LongForecasting/$model_name'_'Exchange_$seq_len'_'$pred_len.log
done

for pred_len in 336 720
do
uv run python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 8 \
  --decomp_kernels 9 25 49 \
  --use_revin \
  --revin_mode std \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.0005 >logs/LongForecasting/$model_name'_'Exchange_$seq_len'_'$pred_len.log
done
