#!/usr/bin/env bash
# SVRC winters for one architecture (raw input, both scalings, five seeds). SVRC has three winters with
# continuous data: 2015-16 and 2018-19 without an earthquake, and 2019-20, which ends with it.
#   transfer  train on the 115 days up to 24 January of one or two winters, predict the other winter(s),
#             each day counted to the 24 January of its own winter; the last epoch is scored
#   direct    the placebo of train_placebo_arch.sh at SVRC alone: 98 days of history, four folds, towards
#             the earthquake, 11 January 2019 and 15 January 2016
# Runs that are finished are skipped. usage: scripts/train_winters_arch.sh lstm|transformer
set -u
cd "$(dirname "$0")/.." || exit 1
ARCH=$1; status=0
W16=2016-01-24; W19=2019-01-24; W20=2020-01-24
train() { out=$1; shift
  [ -e $out/DONE ] && return
  echo "[$(date '+%F %T')] $out"
  uv run python -u -m cnn_lstm_reg.train --kind raw --arch $ARCH --stations SVRC --workers 4 --out $out "$@" || status=1
}
for seed in 0 1 2 3 4; do for norm in station day; do
  for pair in "2016 $W16 $W19,$W20" "2019 $W19 $W16,$W20" "2020 $W20 $W16,$W19" \
              "2016+2019 $W16,$W19 $W20" "2016+2020 $W16,$W20 $W19" "2019+2020 $W19,$W20 $W16"; do
    set -- $pair
    train runs/winters/transfer_$1_raw_${norm}_${ARCH}_s$seed --norm $norm --seed $seed \
      --train-events $2 --val-events $3 --history 115
  done
  for day in 2020-01-24 2019-01-11 2016-01-15; do for fold in 0 1 2 3; do
    train runs/winters/direct_${day}_raw_${norm}_${ARCH}_s${seed}_f$fold --norm $norm --seed $seed \
      --event-day $day --history 98 --fold $fold
  done; done
done; done
echo "[$(date '+%F %T')] ALL DONE $ARCH status=$status"
exit $status
