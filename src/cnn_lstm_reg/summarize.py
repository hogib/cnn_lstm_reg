"""Out-of-fold summary: every validation day predicted by the model that did not train on its block.

    uv run python -m cnn_lstm_reg.summarize RUNS_DIR NAME [NAME ...]

Reads RUNS_DIR/NAME_s<seed>_f<fold>/val_predictions.csv (or NAME_f<fold> for runs without a seed in the name).
Per NAME: MAE of log1p(days) and Spearman correlation over all validation days and over the days of the last
year (where the stations have continuous data), each as mean +- standard deviation over the seeds, with the
two baselines of train.py; then, from the predictions averaged over the seeds, the Spearman correlation per
station in the last year and the median prediction per range of the true time to the event.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

RANGES = [(0, 14), (15, 30), (31, 90), (91, 180), (181, 365), (366, 10**6)]
COLS = [("model", "predicted_log"), ("level only", "level_only_log"), ("train mean", "train_mean_log")]


def rho(y, p):
    return float(spearmanr(y, p).statistic) if len(y) > 5 and np.ptp(p) > 1e-9 else float("nan")


def metrics(d, col):
    year = d[d["days_to_event"] <= 365]
    return [np.abs(d.y - d[col]).mean(), rho(d.y, d[col]), np.abs(year.y - year[col]).mean(), rho(year.y, year[col])]


def main(runs, *names):
    rows = []
    for name in names:
        seeds = {}
        for f in sorted(Path(runs).glob(f"{name}_*")):
            m = re.fullmatch(re.escape(name) + r"(?:_s(\d+))?_f(\d)", f.name)
            if m and (f / "val_predictions.csv").exists() and (f / "DONE").exists():
                seeds.setdefault(m.group(1) or "0", []).append(pd.read_csv(f / "val_predictions.csv"))
        if not seeds:
            sys.exit(f"no finished folds of {name} in {runs}")
        per_seed = {s: pd.concat(v, ignore_index=True).assign(y=lambda t: np.log1p(t["days_to_event"])) for s, v in seeds.items()}
        d = pd.concat(per_seed.values()).groupby(["station", "date", "days_to_event"], as_index=False).mean(numeric_only=True)
        year = d[d["days_to_event"] <= 365]
        print(f"\n== {name}: {len(seeds)} seed(s), folds per seed {sorted({len(v) for v in seeds.values()})}, "
              f"{len(d)} validation station-days ({len(year)} in the last year)")
        print(f"{'':12s} {'MAE(log) all':>15s} {'Spearman all':>15s} {'MAE last year':>15s} {'Spearman last year':>19s}")
        for label, col in COLS:
            m = np.array([metrics(x, col) for x in per_seed.values()])
            mean, sd = m.mean(axis=0), m.std(axis=0)
            print(f"{label:12s} " + " ".join(f"{a:8.3f} ± {b:.3f}" + ("    " if i == 3 else "") for i, (a, b) in enumerate(zip(mean, sd))))
            if label == "model":
                rows.append(dict(name=name, seeds=len(seeds), mae_all=mean[0], rho_all=mean[1], mae_year=mean[2],
                                 mae_year_sd=sd[2], rho_year=mean[3], rho_year_sd=sd[3]))
        print("Spearman per station, last year (model / level only; days):  " + "  ".join(
            f"{s} {rho(g.y, g.predicted_log):.2f} / {rho(g.y, g.level_only_log):.2f} ({len(g)})"
            for s, g in year.groupby("station")))
        print("true days to event -> median predicted days (model / level only; days):  " + "  ".join(
            f"{a}-{b if b < 10**6 else ''}: {np.expm1(g.predicted_log).median():.0f} / {np.expm1(g.level_only_log).median():.0f} ({len(g)})"
            for a, b in RANGES for g in [d[d.days_to_event.between(a, b)]] if len(g)))
    if len(rows) > 1:
        print("\n== overview (mean over seeds)\n" + pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:])
