#!/usr/bin/env bash
# All four inputs on all four folds, one run after the other (40 epochs each), then the out-of-fold summary.
set -u
cd "$(dirname "$0")/.." || exit 1
status=0
run() { name=$1; shift; for fold in 0 1 2 3; do
  echo "[$(date '+%F %T')] $name fold $fold"
  uv run python -u -m cnn_lstm_reg.train --fold $fold --out runs/${name}_f$fold "$@" || status=1
done; }
run features --kind features
run pca --kind pca
run raw_station --kind raw --norm station --workers 8
run raw_day --kind raw --norm day --workers 8
echo "[$(date '+%F %T')] TRAINING DONE status=$status"
uv run python -m cnn_lstm_reg.summarize runs features pca raw_station raw_day > runs/folds_summary.txt 2>&1 || status=1
echo "[$(date '+%F %T')] ALL DONE status=$status"
exit $status
