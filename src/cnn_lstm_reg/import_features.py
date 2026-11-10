"""Day tensors of the window features that seismic-features-classification (sfc) extracted.

    uv run python -m cnn_lstm_reg.import_features OUT_DIR --station FRT TABLE [TABLE ...]

TABLE: <STATION>_N_features_labeled.csv files of sfc (the real project and, if wanted, its placebo projects,
whose days are ordinary days of the same station). Every day becomes a (432, 12) array: one row per 200 s
window at its place in the day, NaN where sfc has no window or no value. Days after the event are left out;
a day present in several tables is taken from the first one.

  OUT_DIR/<STATION>.npz             x float32 (days, 432, 12), dates, days_to_event, features (the column names)
  OUT_DIR/<STATION>_index.csv       date, days_to_event, coverage (share of windows with all features), table
  OUT_DIR/<STATION>_provenance.json command line, parameters, commit, versions, sha256 of every table
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .build_cache import DAY, EVENT_TIME, commit

FEATURES = ["raw_mean", "raw_std", "raw_min", "raw_max", "norm_min", "norm_max", "mean_freq", "mean_period",
            "wolf_lye", "ros_lye", "samp_ent", "corr_dim"]
WIN_SEC = 200
WINDOWS = DAY // WIN_SEC


def read_table(path):
    df = pd.read_csv(path, sep=";", decimal=",", parse_dates=["Date", "Window_Start_UTC"])
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--station", required=True)
    ap.add_argument("tables", nargs="+", type=Path)
    args = ap.parse_args()

    days, hashes = {}, {}
    for table in args.tables:
        hashes[str(table)] = hashlib.sha256(table.read_bytes()).hexdigest()
        df = read_table(table)
        if set(df["Station"]) != {args.station}:
            sys.exit(f"{table}: stations {sorted(set(df['Station']))}, expected {args.station}")
        for date, g in df.groupby(df["Date"].dt.date):
            if date in days or date > EVENT_TIME.date():
                continue
            slot = ((g["Window_Start_UTC"] - pd.Timestamp(date)).dt.total_seconds() // WIN_SEC).astype(int)
            x = np.full((WINDOWS, len(FEATURES)), np.nan, dtype=np.float32)
            x[slot.to_numpy()] = g[FEATURES].to_numpy(dtype=np.float32)
            days[date] = (x, str(table))
    if not days:
        sys.exit(f"{args.station}: no days in {len(args.tables)} table(s)")

    dates = sorted(days)
    x = np.stack([days[d][0] for d in dates])
    to_event = np.array([(EVENT_TIME.date() - d).days for d in dates])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tmp = args.out_dir / f"{args.station}.tmp.npz"
    np.savez_compressed(tmp, x=x, dates=np.array([d.isoformat() for d in dates]), days_to_event=to_event,
                        features=np.array(FEATURES))
    os.replace(tmp, args.out_dir / f"{args.station}.npz")
    pd.DataFrame(dict(date=[d.isoformat() for d in dates], days_to_event=to_event,
                      coverage=np.isfinite(x).all(axis=2).mean(axis=1).round(5),
                      table=[days[d][1] for d in dates])).to_csv(args.out_dir / f"{args.station}_index.csv", index=False)
    (args.out_dir / f"{args.station}_provenance.json").write_text(json.dumps(dict(
        argv=sys.argv, written=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        parameters=dict(event_time=EVENT_TIME.isoformat(), features=FEATURES, win_sec=WIN_SEC),
        commit=commit(), versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__),
        days=len(dates), input_hashes=hashes), indent=1))
    print(f"{args.station}: {len(dates)} days ({dates[0]} .. {dates[-1]}), "
          f"{np.isfinite(x).all(axis=2).mean():.3f} of the windows complete")


if __name__ == "__main__":
    main()
