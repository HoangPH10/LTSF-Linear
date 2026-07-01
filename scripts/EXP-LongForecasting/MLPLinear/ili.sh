if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=104
model_name=MLPLinear

for pred_len in 24 36 48 60
do
uv run python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path national_illness.csv \
  --model_id national_illness_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 18 \
  --pred_len $pred_len \
  --enc_in 7 \
  --revin_mode std \
  --mlp_hidden 16 \
  --mlp_dropout 0.2 \
  --mlp_branches seasonal \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.01 >logs/LongForecasting/$model_name'_'ili_$seq_len'_'$pred_len.log
done
