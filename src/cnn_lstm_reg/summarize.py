"""Out-of-fold summary of the four folds of one input: every validation day predicted by the model that did
not train on its block.

    uv run python -m cnn_lstm_reg.summarize RUNS_DIR NAME [NAME ...]      (reads RUNS_DIR/NAME_f<fold>/)

Per input and for the two baselines of train.py: MAE of log1p(days) and Spearman correlation over all
validation days, and over the days of the last year only, where the stations have continuous data; then the
Spearman correlation per station, and the median prediction per range of the true time to the event.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

RANGES = [(0, 14), (15, 30), (31, 90), (91, 180), (181, 365), (366, 10**6)]


def rho(y, p):
    return float(spearmanr(y, p).statistic) if len(y) > 5 and np.ptp(p) > 1e-9 else float("nan")


def main(runs, *names):
    for name in names:
        folds = sorted(Path(runs).glob(f"{name}_f[0-9]"))
        if not folds:
            sys.exit(f"no folds of {name} in {runs}")
        d = pd.concat([pd.read_csv(f / "val_predictions.csv").assign(fold=f.name[-1]) for f in folds], ignore_index=True)
        d["y"] = np.log1p(d["days_to_event"])
        year = d[d["days_to_event"] <= 365]
        print(f"\n== {name}: {len(folds)} folds, {len(d)} validation station-days ({len(year)} in the last year)")
        print(f"{'':12s} {'MAE(log) all':>13s} {'Spearman all':>13s} {'MAE last year':>14s} {'Spearman last year':>19s}")
        for label, col in (("model", "predicted_log"), ("level only", "level_only_log"), ("train mean", "train_mean_log")):
            print(f"{label:12s} {np.abs(d.y - d[col]).mean():13.3f} {rho(d.y, d[col]):13.2f} "
                  f"{np.abs(year.y - year[col]).mean():14.3f} {rho(year.y, year[col]):19.2f}")
        print("Spearman per station, last year (model / level only; days):  " + "  ".join(
            f"{s} {rho(g.y, g.predicted_log):.2f} / {rho(g.y, g.level_only_log):.2f} ({len(g)})"
            for s, g in year.groupby("station")))
        print("true days to event -> median predicted days (model / level only; days):  " + "  ".join(
            f"{a}-{b if b < 10**6 else ''}: {np.expm1(g.predicted_log).median():.0f} / {np.expm1(g.level_only_log).median():.0f} ({len(g)})"
            for a, b in RANGES for g in [d[d.days_to_event.between(a, b)]] if len(g)))


if __name__ == "__main__":
    main(*sys.argv[1:])
