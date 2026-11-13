#!/usr/bin/env bash
# Raw 5 Hz cache of the four Balıkesir stations up to the Sındırgı mainshock of 10 August 2025, 16:53 UTC.
# usage: scripts/build_cache_balikesir.sh SPLIT_DATA_DIR [JOBS]     (SPLIT_DATA_DIR/<STATION>/days/*.mseed)
set -u
cd "$(dirname "$0")/.." || exit 1
SRC=$1; JOBS=${2:-4}; status=0
for s in PURS STEP DURS BKES; do
  echo "[$(date '+%F %T')] $s"
  uv run python -u -m cnn_lstm_reg.build_cache "$SRC/$s/days" data/raw5hz --station $s --jobs "$JOBS" \
    --event-time 2025-08-10T16:53:00 || status=1
done
echo "[$(date '+%F %T')] ALL DONE status=$status"
exit $status
