#!/usr/bin/env bash
# Raw 5 Hz cache of the five Elazığ stations from the day files of seismic-features-classification.
# usage: scripts/build_cache_elazig.sh SPLIT_DATA_DIR [JOBS]     (SPLIT_DATA_DIR/<STATION>/days/*.mseed)
set -u
cd "$(dirname "$0")/.." || exit 1
SRC=$1; JOBS=${2:-4}; status=0
for s in FRT KOVA MAYA NARI SVRC; do
  echo "[$(date '+%F %T')] $s"
  uv run python -u -m cnn_lstm_reg.build_cache "$SRC/$s/days" data/raw5hz --station $s --jobs "$JOBS" || status=1
done
echo "[$(date '+%F %T')] ALL DONE status=$status"
exit $status
