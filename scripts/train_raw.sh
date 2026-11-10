#!/usr/bin/env bash
# The raw 5 Hz input with both normalisations, one after the other (40 epochs, split seed 0).
set -u
cd "$(dirname "$0")/.." || exit 1
status=0
for norm in station day; do
  echo "[$(date '+%F %T')] raw, norm $norm"
  uv run python -u -m cnn_lstm_reg.train --kind raw --norm $norm --workers 8 --out runs/raw_$norm || status=1
done
echo "[$(date '+%F %T')] ALL DONE status=$status"
exit $status
