#!/usr/bin/env bash
# The two feature inputs, one after the other, with the defaults of train.py (40 epochs, split seed 0).
set -u
cd "$(dirname "$0")/.." || exit 1
status=0
for kind in features pca; do
  echo "[$(date '+%F %T')] $kind"
  uv run python -u -m cnn_lstm_reg.train --kind $kind --out runs/$kind || status=1
done
echo "[$(date '+%F %T')] ALL DONE status=$status"
exit $status
