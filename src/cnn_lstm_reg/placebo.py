"""Pseudo-event placebo: is the real event scored better than dates on which no earthquake happened?

    uv run python -m cnn_lstm_reg.placebo RUNS_DIR/placebo [REAL_EVENT_DAY]

Runs with _s<seed> before _f<fold> in their name are averaged over the seeds, day by day, before scoring.
REAL_EVENT_DAY (YYYY-MM-DD) is the day of the earthquake when it is not Elazığ's.

Reads the runs of scripts/train_placebo.sh, RUNS_DIR/placebo/e<event day>_<input>_<arch>_f<fold>: the same
model trained with the days counted to <event day>, which is the real event day or a pseudo-event 14, 28, ...
days before it, each with the same length of history and the same geometry of blocks (data.event_roles).
Per input and architecture, every event gets the out-of-fold scores of its validation days, and the real
event is ranked among all events (rank 1 = best). With E events the smallest share a rank can give is 1/E,
and the events share most of their days, so they are not independent draws.

Scores per event (y = log1p(days to that event), all validation days of the four folds together):
  mae           mean |error| of y
  gain          MAE of the level-only baseline of train.py minus the model's MAE (> 0: better than loudness)
  rho           Spearman correlation of prediction and y
  rho_folds     the same per fold, averaged (not affected by offsets between the folds)
  auc_month     share of pairs (a day 0-30 days before the event, an earlier day) in which the model predicts
                the shorter time for the closer day; 0.5 = chance
  month_pred    median predicted days for the days 0-30 days before the event
Writes placebo_events.csv (one line per input, architecture and event) and prints the tables.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

from .build_cache import EVENT_TIME
from .data import FOLDS

MONTH = 30
RANKED = [("mae", False), ("gain", True), ("rho", True), ("rho_folds", True), ("auc_month", True)]   # (score, higher is better)


def rho(y, p):
    return float(spearmanr(y, p).statistic) if len(y) > 5 and np.ptp(p) > 1e-9 and np.ptp(y) > 0 else float("nan")


def auc_month(d, col="predicted_log"):
    near, far = d.loc[d.days_to_event <= MONTH, col], d.loc[d.days_to_event > MONTH, col]
    if len(near) < 5 or len(far) < 5:
        return float("nan")
    return float(mannwhitneyu(far, near).statistic / (len(near) * len(far)))     # P(far predicted later than near)


def event_scores(d):
    d = d.assign(y=np.log1p(d["days_to_event"]))
    month = d[d.days_to_event <= MONTH]
    with np.errstate(invalid="ignore"):
        per_fold = [rho(g.y, g.predicted_log) for _, g in d.groupby("fold")]
    return dict(days=len(d), month_days=len(month),
                mae=float(np.abs(d.y - d.predicted_log).mean()),
                mae_level=float(np.abs(d.y - d.level_only_log).mean()),
                gain=float(np.abs(d.y - d.level_only_log).mean() - np.abs(d.y - d.predicted_log).mean()),
                rho=rho(d.y, d.predicted_log), rho_folds=float(np.nanmean(per_fold)),
                rho_level=rho(d.y, d.level_only_log),
                auc_month=auc_month(d), auc_month_level=auc_month(d, "level_only_log"),
                month_pred=float(np.expm1(month.predicted_log).median()) if len(month) else float("nan"))


def collect(runs):
    """{(input, arch): {event day: predictions of all folds}} for the events of which every fold is finished."""
    found = {}
    for f in sorted(Path(runs).glob("e*_f?")):
        m = re.fullmatch(r"e(\d{4}-\d\d-\d\d)_(.+)_(lstm|transformer)(?:_s(\d+))?_f(\d)", f.name)
        if m and (f / "DONE").exists():
            found.setdefault((m.group(2), m.group(3)), {}).setdefault(m.group(1), {}).setdefault(int(m.group(5)), []).append(
                pd.read_csv(f / "val_predictions.csv").assign(fold=int(m.group(5))))
    seeds = {len(v) for events in found.values() for folds in events.values() for v in folds.values()}
    if len(seeds) > 1:
        print(f"warning: the folds do not all have the same number of seeds ({sorted(seeds)})", file=sys.stderr)
    mean = lambda v: (pd.concat(v).groupby(["station", "date", "days_to_event", "fold"], as_index=False)  # noqa: E731
                      .mean(numeric_only=True))
    return {k: {e: pd.concat([mean(v) for v in folds.values()], ignore_index=True)
                for e, folds in sorted(events.items(), reverse=True) if len(folds) == FOLDS} for k, events in found.items()}


def table(runs, real=EVENT_TIME.date().isoformat()):
    rows = []
    for (kind, arch), events in collect(runs).items():
        if not events:
            continue
        t = pd.DataFrame([dict(input=kind, arch=arch, event=e, real=e == real, **event_scores(d)) for e, d in events.items()])
        for score, higher in RANKED:            # rank 1 = best; an event without the score is not ranked
            t[f"rank_{score}"] = t[score].rank(ascending=not higher, method="max")
        rows.append(t)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main(runs, real=EVENT_TIME.date().isoformat()):
    t = table(runs, real)
    if t.empty:
        sys.exit(f"no event with all {FOLDS} folds finished in {runs}")
    t.to_csv(Path(runs) / "placebo_events.tmp", index=False)
    (Path(runs) / "placebo_events.tmp").replace(Path(runs) / "placebo_events.csv")
    pd.set_option("display.width", 250)
    for (kind, arch), g in t.groupby(["input", "arch"], sort=False):
        print(f"\n== {kind}, {arch}: {len(g)} events ({int(g.real.sum())} real)")
        print(g[["event", "real", "days", "month_days", "mae", "mae_level", "gain", "rho", "rho_folds", "rho_level",
                 "auc_month", "auc_month_level", "month_pred"]].round(3).to_string(index=False))
        r = g[g.real]
        if len(r):
            print("real event, rank among the events that have the score (1 = best): " + ", ".join(
                f"{s} {int(r[f'rank_{s}'].iloc[0])} of {int(g[s].notna().sum())}" if np.isfinite(r[s].iloc[0]) else f"{s} –"
                for s, _ in RANKED))


if __name__ == "__main__":
    main(*sys.argv[1:])
