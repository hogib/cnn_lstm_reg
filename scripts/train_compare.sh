#!/usr/bin/env bash
# LSTM and transformer side by side (scripts/train_archs.sh), then the out-of-fold summary of both.
# usage: scripts/train_compare.sh [cnn|patch]
set -u
cd "$(dirname "$0")/.." || exit 1
TOK=${1:-cnn}; mkdir -p logs
suffix=""; [ "$TOK" = cnn ] || suffix=_$TOK
scripts/train_archs.sh lstm $TOK > logs/train_lstm$suffix.log 2>&1 &
scripts/train_archs.sh transformer $TOK > logs/train_transformer$suffix.log 2>&1 &
wait
names=""; for arch in lstm transformer; do for k in features pca raw_station raw_day; do names="$names ${k}_${arch}$suffix"; done; done
uv run python -m cnn_lstm_reg.summarize runs $names > runs/compare_summary$suffix.txt 2>&1
echo "[$(date '+%F %T')] COMPARE DONE exit=$?"
