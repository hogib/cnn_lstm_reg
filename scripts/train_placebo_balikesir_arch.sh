#!/usr/bin/env bash
# Pseudo-event placebo at Balıkesir for one architecture: the raw input with both scalings, trained towards
# the mainshock of 10 August 2025 and towards 7 pseudo-events 14, 28, ... days before it, each with the 84
# days before it (the recordings start in February 2025, which leaves no room for a longer history), four
# folds, three seeds, stations PURS STEP DURS BKES. Runs that are finished are skipped.
# usage: scripts/train_placebo_balikesir_arch.sh lstm|transformer
set -u
cd "$(dirname "$0")/.." || exit 1
ARCH=$1; REAL=2025-08-10; EVENTS=7; HISTORY=84; status=0
for seed in 0 1 2; do for k in $(seq 0 $EVENTS); do
  day=$(date -u -d "$REAL - $((14 * k)) days" +%F)
  for norm in station day; do for fold in 0 1 2 3; do
    out=runs/placebo_balikesir/e${day}_raw_${norm}_${ARCH}_s${seed}_f$fold
    [ -e $out/DONE ] && continue
    echo "[$(date '+%F %T')] $out"
    uv run python -u -m cnn_lstm_reg.train --event balikesir_2025 --kind raw --norm $norm --arch $ARCH --seed $seed \
      --event-day $day --history $HISTORY --fold $fold --workers 6 --out $out || status=1
  done; done
done; done
echo "[$(date '+%F %T')] ALL DONE $ARCH status=$status"
exit $status
