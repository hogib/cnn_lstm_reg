"""Winters with and without the earthquake at one station: summary of the runs of scripts/train_winters.sh.

    uv run python -m cnn_lstm_reg.winters RUNS_DIR/winters

transfer_<train winters>_<input>_<arch>_s<seed>: a model trained on the days up to 24 January of one or two
winters predicts the days of the other winter(s), every day counted to the 24 January of its own winter. Only
the winter 2019-20 ends with the earthquake. If the time to the earthquake can be read from a recording
because of the season, a model that has only seen a winter without an earthquake orders the days of the
earthquake winter as well as it orders the days of another winter without one.

direct_<event day>_<input>_<arch>_s<seed>_f<fold>: the pseudo-event placebo (placebo.py) at this station
alone, towards the earthquake and towards a day in January of the two other winters.

Scores as in placebo.py (mae, rho, auc_month, month_pred), per seed, given as mean and standard deviation
over the seeds; `*_level` is the level-only baseline of train.py, which does not depend on the seed.
Writes winters_transfer.csv and winters_direct.csv and prints the tables.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .data import FOLDS
from .placebo import MONTH, auc_month, event_scores, rho

SCORES = ["mae", "rho", "auc_month", "month_pred"]


def winter(event):
    """'2019-20' for the event day 2020-01-24"""
    y = int(event[:4])
    return f"{y - 1}-{y % 100:02d}"


def transfer_scores(d):
    d = d.assign(y=np.log1p(d["days_to_event"]))
    month = d[d.days_to_event <= MONTH]
    return dict(days=len(d), month_days=len(month), nearest=int(d.days_to_event.min()),
                mae=float(np.abs(d.y - d.predicted_log).mean()), mae_level=float(np.abs(d.y - d.level_only_log).mean()),
                rho=rho(d.y, d.predicted_log), rho_level=rho(d.y, d.level_only_log),
                auc_month=auc_month(d), auc_month_level=auc_month(d, "level_only_log"),
                month_pred=float(np.expm1(month.predicted_log).median()) if len(month) else float("nan"))


def over_seeds(rows, keys):
    """One line per group: mean of every score over the seeds, the standard deviation of SCORES, the seeds."""
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    g = t.groupby(keys, sort=False)
    out = g.mean(numeric_only=True).drop(columns="seed")
    for s in SCORES:
        out[f"{s}_sd"] = g[s].std(ddof=0)
    out["seeds"] = g["seed"].nunique()
    return out.reset_index()


def transfer_table(runs):
    rows = []
    for f in sorted(Path(runs).glob("transfer_*")):
        m = re.fullmatch(r"transfer_([\d+]+)_(.+)_(lstm|transformer)_s(\d+)", f.name)
        if not m or not (f / "DONE").exists():
            continue
        p = pd.read_csv(f / "val_predictions.csv")
        for event, d in p.groupby("event"):
            rows.append(dict(input=m.group(2), arch=m.group(3), trained_on=" + ".join(winter(f"{y}-01-24") for y in m.group(1).split("+")),
                             predicts=winter(event), seed=int(m.group(4)), **transfer_scores(d)))
    return over_seeds(rows, ["input", "arch", "trained_on", "predicts"])


def direct_table(runs):
    found = {}
    for f in sorted(Path(runs).glob("direct_*")):
        m = re.fullmatch(r"direct_(\d{4}-\d\d-\d\d)_(.+)_(lstm|transformer)_s(\d+)_f(\d)", f.name)
        if m and (f / "DONE").exists():
            found.setdefault((m.group(2), m.group(3), m.group(1), int(m.group(4))), []).append(
                pd.read_csv(f / "val_predictions.csv").assign(fold=int(m.group(5))))
    rows = [dict(input=k[0], arch=k[1], event=k[2], seed=k[3], folds=len(v), **event_scores(pd.concat(v, ignore_index=True)))
            for k, v in found.items() if len(v) == FOLDS]        # a seed counts when all its folds are finished
    return over_seeds(rows, ["input", "arch", "event"])


def main(runs):
    tr, di = transfer_table(runs), direct_table(runs)
    if tr.empty and di.empty:
        sys.exit(f"no finished runs in {runs}")
    pd.set_option("display.width", 250)
    for name, t, cols in (("transfer", tr, ["trained_on", "predicts", "seeds", "days", "month_days", "nearest", "mae", "mae_sd",
                                            "mae_level", "rho", "rho_sd", "rho_level", "auc_month", "auc_month_sd",
                                            "auc_month_level", "month_pred"]),
                          ("direct", di, ["event", "seeds", "folds", "days", "month_days", "mae", "mae_sd", "mae_level", "gain",
                                          "rho", "rho_sd", "rho_folds", "rho_level", "auc_month", "auc_month_sd",
                                          "auc_month_level", "month_pred"])):
        if t.empty:
            continue
        tmp = Path(runs) / f"winters_{name}.tmp"
        t.to_csv(tmp, index=False)
        tmp.replace(Path(runs) / f"winters_{name}.csv")
        for (kind, arch), g in t.groupby(["input", "arch"], sort=False):
            print(f"\n== {name}: {kind}, {arch}")
            print(g[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:])
