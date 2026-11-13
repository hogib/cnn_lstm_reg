#!/usr/bin/env bash
# Pseudo-event placebo, LSTM and transformer side by side (scripts/train_placebo_arch.sh), then the ranking
# of the real event among the pseudo-events (runs/placebo/placebo_summary.txt, placebo_events.csv).
set -u
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs runs/placebo
scripts/train_placebo_arch.sh lstm > logs/placebo_lstm.log 2>&1 &
scripts/train_placebo_arch.sh transformer > logs/placebo_transformer.log 2>&1 &
wait
uv run python -m cnn_lstm_reg.placebo runs/placebo > runs/placebo/placebo_summary.tmp 2>&1; s=$?
mv runs/placebo/placebo_summary.tmp runs/placebo/placebo_summary.txt
echo "[$(date '+%F %T')] PLACEBO DONE exit=$s"
