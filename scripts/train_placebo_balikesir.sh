#!/usr/bin/env bash
# Pseudo-event placebo at Balıkesir, LSTM and transformer side by side, then the ranking of the mainshock
# among the pseudo-events (runs/placebo_balikesir/placebo_summary.txt, placebo_events.csv).
set -u
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs runs/placebo_balikesir
scripts/train_placebo_balikesir_arch.sh lstm > logs/placebo_balikesir_lstm.log 2>&1 &
scripts/train_placebo_balikesir_arch.sh transformer > logs/placebo_balikesir_transformer.log 2>&1 &
wait
uv run python -m cnn_lstm_reg.placebo runs/placebo_balikesir 2025-08-10 > runs/placebo_balikesir/placebo_summary.tmp 2>&1; s=$?
mv runs/placebo_balikesir/placebo_summary.tmp runs/placebo_balikesir/placebo_summary.txt
echo "[$(date '+%F %T')] PLACEBO DONE exit=$s"
