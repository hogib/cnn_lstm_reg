#!/usr/bin/env bash
# SVRC winters, LSTM and transformer side by side (scripts/train_winters_arch.sh), then the summary
# (runs/winters/winters_summary.txt, winters_transfer.csv, winters_direct.csv).
set -u
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs runs/winters
scripts/train_winters_arch.sh lstm > logs/winters_lstm.log 2>&1 &
scripts/train_winters_arch.sh transformer > logs/winters_transformer.log 2>&1 &
wait
uv run python -m cnn_lstm_reg.winters runs/winters > runs/winters/winters_summary.tmp 2>&1; s=$?
mv runs/winters/winters_summary.tmp runs/winters/winters_summary.txt
echo "[$(date '+%F %T')] WINTERS DONE exit=$s"
