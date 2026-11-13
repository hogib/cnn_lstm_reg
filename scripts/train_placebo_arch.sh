#!/usr/bin/env bash
# Pseudo-event placebo for one architecture: the raw input with both scalings, trained towards the real
# event day and towards 13 pseudo-events 14, 28, ... days before it, each with the 182 days before it,
# four folds, one seed, stations FRT KOVA MAYA NARI (SVRC has no data from February to September 2019).
# Runs that are finished are skipped. usage: scripts/train_placebo_arch.sh lstm|transformer
set -u
cd "$(dirname "$0")/.." || exit 1
ARCH=$1; REAL=2020-01-24; EVENTS=13; HISTORY=182; status=0
for k in $(seq 0 $EVENTS); do
  day=$(date -u -d "$REAL - $((14 * k)) days" +%F)
  for norm in station day; do for fold in 0 1 2 3; do
    out=runs/placebo/e${day}_raw_${norm}_${ARCH}_f$fold
    [ -e $out/DONE ] && continue
    echo "[$(date '+%F %T')] $out"
    uv run python -u -m cnn_lstm_reg.train --kind raw --norm $norm --arch $ARCH --stations FRT,KOVA,MAYA,NARI \
      --event-day $day --history $HISTORY --fold $fold --workers 6 --out $out || status=1
  done; done
done
echo "[$(date '+%F %T')] ALL DONE $ARCH status=$status"
exit $status
