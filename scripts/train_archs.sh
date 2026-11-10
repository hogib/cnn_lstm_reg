#!/usr/bin/env bash
# One architecture on all four inputs, three seeds, four folds (40 epochs each); runs that are finished are
# skipped. usage: scripts/train_archs.sh lstm|transformer [cnn|patch]
set -u
cd "$(dirname "$0")/.." || exit 1
ARCH=$1; TOK=${2:-cnn}; status=0
tag=$ARCH; [ "$TOK" = cnn ] || tag=${ARCH}_$TOK
run() { name=$1; shift; for seed in 0 1 2; do for fold in 0 1 2 3; do
  out=runs/${name}_${tag}_s${seed}_f$fold
  [ -e $out/DONE ] && continue
  echo "[$(date '+%F %T')] $out"
  uv run python -u -m cnn_lstm_reg.train --arch $ARCH --tokenizer $TOK --seed $seed --fold $fold --out $out "$@" || status=1
done; done; }
run features --kind features
run pca --kind pca
run raw_station --kind raw --norm station --workers 6
run raw_day --kind raw --norm day --workers 6
echo "[$(date '+%F %T')] ALL DONE $tag status=$status"
exit $status
