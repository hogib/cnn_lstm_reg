"""Raw-signal cache: one band-passed, downsampled array per station and day.

    uv run python -m cnn_lstm_reg.build_cache DAYS_DIR OUT_DIR --station FRT [--jobs 4]

DAYS_DIR holds one MiniSEED file per day of one component (the files `sfc split-days` writes, named
<NET>_<STATION>_<DDMMYYYY>_000000_<DDMMYYYY>_000000_<CHANNEL>.mseed). For each day up to the event day:

  OUT_DIR/<STATION>/<YYYY-MM-DD>.npy   float32, 86400 * RATE samples on the day's time grid, NaN where there
                                       is no data
  OUT_DIR/<STATION>/index.csv          date, days_to_event, coverage (share of samples that are not NaN),
                                       input sampling rate, number of gaps, input file and its sha256
  OUT_DIR/<STATION>/provenance.json    command line, parameters, commit, package versions

Days already in the cache are kept (their index rows are reused) unless --overwrite is given. The event day
is cut at the start of the hour of the mainshock, as in sfc, so the mainshock is never in the cache; later days
are left out.

The filtering follows preprocess.py of seismic-features-classification (commit d274fda): gaps shorter than
GAP_THRESHOLD seconds are interpolated, every longer gap splits the day into segments that are demeaned,
detrended and filtered on their own (zero-phase Butterworth band-pass), then every FACTOR-th sample is kept.
Differences: sfc filters hour by hour with 60 s of padding and detrends each hour again after downsampling;
here a segment is filtered in one piece, the interpolation is linear, and there is no padding from the
neighbouring days, so the first and last seconds of a segment carry the filter's edge effect.
"""
import argparse
import datetime as dt
import hashlib
import io
import json
import os
import platform
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.signal import butter, detrend, sosfiltfilt

EVENT_TIME = dt.datetime(2020, 1, 24, 17, 55, 11)   # Elazığ mainshock, UTC
RATE = 5.0             # Hz, output sampling rate
FREQMIN, FREQMAX = 0.1, 2.0
FILTER_ORDER = 4
GAP_THRESHOLD = 2.0    # s; shorter gaps are interpolated
MIN_SEGMENT = 30.0     # s; shorter segments between gaps are dropped (3 periods of FREQMIN)
DAY = 86400


def file_date(name):
    m = re.search(r"_(\d{2})(\d{2})(\d{4})_000000_", name)
    return dt.date(int(m.group(3)), int(m.group(2)), int(m.group(1))) if m else None


def day_signal(raw_bytes, date, event_time=EVENT_TIME):
    """The day's samples on its native grid (NaN where no data), the sampling rate and the number of gaps."""
    from obspy import UTCDateTime, read
    st = read(io.BytesIO(raw_bytes))
    rates = {tr.stats.sampling_rate for tr in st}
    if len(rates) != 1:
        raise ValueError(f"several sampling rates in one day: {sorted(rates)}")
    fs = rates.pop()
    n = int(round(DAY * fs))
    x = np.full(n, np.nan)
    start = UTCDateTime(date.year, date.month, date.day)
    for tr in st:
        i = int(round((tr.stats.starttime - start) * fs))
        d = tr.data.astype(float)
        lo, hi = max(i, 0), min(i + len(d), n)
        if hi > lo:
            x[lo:hi] = d[lo - i:hi - i]
    if date == event_time.date():                       # nothing from the hour of the mainshock on
        x[int(event_time.hour * 3600 * fs):] = np.nan
    return x, fs


def segments(mask):
    """(start, end) of every run of True."""
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.where(edges == 1)[0], np.where(edges == -1)[0]))


def process(x, fs):
    """Band-pass and downsample one day. Returns the output array and the number of gaps kept as NaN."""
    factor = fs / RATE
    if abs(factor - round(factor)) > 1e-9:
        raise ValueError(f"sampling rate {fs} Hz is not a multiple of {RATE} Hz")
    factor = int(round(factor))
    valid = ~np.isnan(x)
    cut = len(x) if valid.all() else 0
    if valid.any():
        cut = np.where(valid)[0][-1] + 1                # trailing NaN (event day, late start of the next file)
    for a, b in segments(~valid[:cut]):                 # short gaps inside the data: linear interpolation
        if a > 0 and (b - a) < GAP_THRESHOLD * fs:
            x[a:b] = np.interp(np.arange(a, b), [a - 1, b], [x[a - 1], x[b]])
    valid = ~np.isnan(x)
    out = np.full(int(round(DAY * RATE)), np.nan, dtype=np.float32)
    sos = butter(FILTER_ORDER, [FREQMIN, FREQMAX], btype="bandpass", fs=fs, output="sos")
    runs = segments(valid)
    for a, b in runs:
        if (b - a) < MIN_SEGMENT * fs:
            continue
        y = sosfiltfilt(sos, detrend(x[a:b] - x[a:b].mean(), type="linear"))
        first = -(-a // factor) * factor                # first sample of the output grid inside the segment
        out[first // factor:first // factor + len(y[first - a::factor])] = y[first - a::factor]
    return out, max(len(runs) - 1, 0)


def one_day(job):
    path, date, out_file = job
    raw = Path(path).read_bytes()
    x, fs = day_signal(raw, date)
    out, gaps = process(x, fs)
    tmp = out_file.with_suffix(".tmp.npy")
    np.save(tmp, out)
    os.replace(tmp, out_file)
    return dict(date=date.isoformat(), days_to_event=(EVENT_TIME.date() - date).days,
                coverage=round(float(np.isfinite(out).mean()), 5), input_rate=fs, gaps=gaps,
                file=Path(path).name, sha256=hashlib.sha256(raw).hexdigest())


def commit():
    try:
        root = Path(__file__).resolve().parent
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True).stdout
        return rev + (" (uncommitted changes)" if dirty.strip() else "")
    except OSError:
        return "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("days_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--station", required=True)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out = args.out_dir / args.station
    out.mkdir(parents=True, exist_ok=True)
    index_file = out / "index.csv"
    old = pd.read_csv(index_file).set_index("date").to_dict("index") if index_file.exists() and not args.overwrite else {}
    files = {}
    for f in sorted(args.days_dir.glob("*.mseed")):
        d = file_date(f.name)
        if d is not None and d <= EVENT_TIME.date():
            files[d] = f
    jobs, rows = [], []
    for d, f in sorted(files.items()):
        target = out / f"{d.isoformat()}.npy"
        if d.isoformat() in old and target.exists():
            rows.append(dict(date=d.isoformat(), **old[d.isoformat()]))
        else:
            jobs.append((str(f), d, target))
    print(f"{args.station}: {len(files)} day files up to the event, {len(rows)} already cached, {len(jobs)} to do",
          flush=True)

    t0, failed = time.time(), []
    with ProcessPoolExecutor(args.jobs) as pool:
        for i, (job, res) in enumerate(zip(jobs, pool.map(_safe, jobs)), 1):
            if isinstance(res, str):
                failed.append((job[0], res))
                print(f"[FAILED] {Path(job[0]).name}: {res}", flush=True)
            else:
                rows.append(res)
            if i % 20 == 0 or i == len(jobs):
                rate = i / (time.time() - t0)
                print(f"{args.station} {i}/{len(jobs)} days | {rate:.1f} days/s | ETA {(len(jobs) - i) / rate / 60:.1f} min",
                      flush=True)

    if not rows:
        sys.exit(f"{args.station}: nothing was written")
    index = pd.DataFrame(rows).sort_values("date")
    index.to_csv(index_file.with_suffix(".tmp"), index=False)
    os.replace(index_file.with_suffix(".tmp"), index_file)
    import obspy
    (out / "provenance.json").write_text(json.dumps(dict(
        argv=sys.argv, written=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        parameters=dict(event_time=EVENT_TIME.isoformat(), rate=RATE, freqmin=FREQMIN, freqmax=FREQMAX,
                        filter_order=FILTER_ORDER, gap_threshold=GAP_THRESHOLD, min_segment=MIN_SEGMENT),
        commit=commit(), versions=dict(python=platform.python_version(), numpy=np.__version__,
                                       scipy=scipy.__version__, obspy=obspy.__version__, pandas=pd.__version__),
        days=len(index), failed=failed, input_hashes="sha256 column of index.csv"), indent=1))
    print(f"{args.station}: DONE, {len(index)} days in {out}, {len(failed)} failed, "
          f"mean coverage {index['coverage'].mean():.3f}", flush=True)
    if failed:
        sys.exit(1)


def _safe(job):
    try:
        return one_day(job)
    except Exception as error:   # one unreadable day must not stop the station
        return f"{type(error).__name__}: {error}"


if __name__ == "__main__":
    main()
