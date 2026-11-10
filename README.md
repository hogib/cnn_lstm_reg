# cnn_lstm_reg

CNN-LSTM regression of the time to an earthquake from one full day of a station's recording.
One sample is one station-day; the target is `log1p(days to the event)`. Event: Elazığ, 24 January 2020,
17:55 UTC (Mw 6.8); stations FRT, KOVA, MAYA, NARI, SVRC, N component.

Three inputs are compared with the same model:

| `--kind` | Input | Shape per day |
|----------|-------|---------------|
| `raw` | the day's signal, 0.1–2 Hz, 5 Hz, plus a data mask | 2 × 432 000 |
| `features` | the 12 window features of seismic-features-classification (200 s windows), plus a mask | 13 × 432 |
| `pca` | principal components of those features (fitted on the training days), plus a mask | 8 × 432 |

A strided 1-D CNN turns the day into tokens (one per minute for `raw`, one per 400 s for the features), a
bidirectional LSTM reads them, and the mean of its outputs gives the prediction (`model.py`).

## Steps

```
uv sync

# 1. raw cache from the daily MiniSEED files (one component per file, as `sfc split-days` writes them)
scripts/build_cache_elazig.sh /path/to/split_data          # -> data/raw5hz/<STATION>/

# 2. feature tensors from the sfc feature tables
uv run python -m cnn_lstm_reg.import_features data/features --station FRT /path/to/FRT_N_features_labeled.csv ...

# 3. training (validation blocks only; logs loss, learning rate, speed and ETA every 10 steps)
uv run python -u -m cnn_lstm_reg.train --kind raw --out runs/raw_station

# 4. all four inputs on all four folds, then the out-of-fold summary (runs/folds_summary.txt)
scripts/train_folds.sh
```

`data/`, `runs/` and `logs/` are not in the repository.

## Split

The calendar is cut into 14-day blocks counted back from the event day, the same at every station; each block
is train, validation or test as a whole, and a day within 3 days of a day of another role is not used. The
blocks of the last year (where the data are dense) and the earlier blocks are dealt out separately: 20 % of
each are test blocks (`--split-seed`), the rest form four folds, and `--fold` chooses the fold that validates,
so over the four folds every block that is not a test block is validated once.
**The test blocks are held out**: `train.py` never reads them.

## What a score means here

With one event, the days to the event are the calendar date counted backwards, so a model can lower its error
from anything that changes slowly with time (season, noise level, instrument changes) and has nothing to do
with the earthquake. `train.py` therefore prints two baselines on the same validation days: the mean of the
training targets, and a linear fit on the day's level alone. A model has to be read against those, and a low
error on blocks that lie between training blocks is interpolation in time, not prediction.

The filtering in `build_cache.py` and the feature scaling in `data.py` follow seismic-features-classification
(commit d274fda); the differences are listed at the top of `build_cache.py`.
