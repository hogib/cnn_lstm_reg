"""Days, their split and the three kinds of model input.

One sample is one station-day; the target is log1p(days to the event).

Split (blocked in time): the calendar is cut into blocks of BLOCK_DAYS days counted back from the event day,
the same blocks at every station, and each block is train, val or test as a whole (see block_roles). A day closer than GAP_DAYS
to a day of another role is not used at all, so no sample has a neighbouring day in another set. The test
blocks are held out: nothing in this package reads them until `roles(..., with_test=True)` is asked for.

Inputs (`kind`):
  raw       the 5 Hz day signal of build_cache.py, (2, 432000): the scaled signal and a 0/1 data mask
  features  the 12 window features of import_features.py, (13, 432): robust-scaled, clipped, plus the mask
  pca       the same after a PCA fitted on the training windows, (components + 1, 432)
Everything that is fitted (scales, PCA) is fitted on the training days only.
"""
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

from .build_cache import EVENT_TIME

BLOCK_DAYS = 14
GAP_DAYS = 3
CLIP = 5.0           # limit of the robust-scaled features, as in sfc
TEST_SHARE = 0.2     # of the blocks, held out
FOLDS = 4            # the other blocks: one fold validates, three train (60 / 20 / 20 % of all blocks)
RECENT_DAYS = 365    # the last year is dealt out on its own
FIRST_DAY = dt.date(2015, 1, 24)   # five years before the event; fixed, so a block has the same role in every run


def block_roles(seed, fold=0, first_day=FIRST_DAY, last_day=EVENT_TIME.date()):
    """{date: 'train' | 'val' | 'test'} for every calendar day that is not within GAP_DAYS of another role.

    The blocks of the last year before the event (where the data are dense) and the earlier blocks are dealt
    out separately, so each period has its share of every role: TEST_SHARE of the blocks are test, fixed by
    the seed; the others are dealt into FOLDS folds, and fold `fold` is the validation set. Over the FOLDS
    values of `fold` every block that is not a test block is validated once."""
    n_days = (last_day - first_day).days + 1
    n_blocks = -(-n_days // BLOCK_DAYS)
    rng = np.random.default_rng(seed)
    role_of_block = np.empty(n_blocks, dtype=object)
    recent = -(-RECENT_DAYS // BLOCK_DAYS)
    for stratum in (np.arange(min(recent, n_blocks)), np.arange(min(recent, n_blocks), n_blocks)):
        order = rng.permutation(stratum)
        n_test = round(TEST_SHARE * len(order))
        role_of_block[order[:n_test]] = "test"
        for i, block in enumerate(order[n_test:]):
            role_of_block[block] = "val" if i % FOLDS == fold else "train"
    by_age = [role_of_block[age // BLOCK_DAYS] for age in range(n_days)]      # age = days before the event
    roles = {}
    for age in range(n_days):
        near = by_age[max(age - GAP_DAYS, 0):age + GAP_DAYS + 1]
        if all(r == by_age[age] for r in near):
            roles[last_day - dt.timedelta(days=age)] = by_age[age]
    return roles


def list_days(kind, data_dir, stations, min_coverage):
    """Table of the usable station-days: station, date, days_to_event, coverage (and row in the feature file)."""
    frames = []
    for s in stations:
        f = Path(data_dir) / ("raw5hz" if kind == "raw" else "features") / (f"{s}/index.csv" if kind == "raw" else f"{s}_index.csv")
        df = pd.read_csv(f, parse_dates=["date"])
        df["row"] = np.arange(len(df))
        df["station"] = s
        frames.append(df[["station", "date", "days_to_event", "coverage", "row"]])
    days = pd.concat(frames, ignore_index=True)
    days["date"] = days["date"].dt.date
    return days[(days["coverage"] >= min_coverage) & (days["days_to_event"] >= 0)].reset_index(drop=True)


def split(days, seed, fold=0, with_test=False):
    """Adds the 'role' column; days in a gap are dropped, and so are the test days unless with_test."""
    roles = block_roles(seed, fold)
    days = days.assign(role=days["date"].map(roles)).dropna(subset=["role"])
    return (days if with_test else days[days["role"] != "test"]).reset_index(drop=True)


def target(days_to_event):
    return np.log1p(np.asarray(days_to_event, dtype=np.float32))


class RawDays(torch.utils.data.Dataset):
    """norm 'station': divided by the station's median daily standard deviation of the training days, so the
    level of a day (loud or quiet) stays in the input. norm 'day': divided by the day's own standard deviation,
    which removes the level. Then asinh, which is linear for small values and logarithmic for transients."""

    def __init__(self, days, data_dir, norm, scales=None):
        self.days, self.dir, self.norm = days.reset_index(drop=True), Path(data_dir) / "raw5hz", norm
        self.scales = scales
        if norm == "station" and scales is None:
            self.scales = {}
            for s, g in self.days[self.days["role"] == "train"].groupby("station"):
                self.scales[s] = float(np.median([np.nanstd(self._load(r)) for r in g.itertuples()]))

    def _load(self, r):
        return np.load(self.dir / r.station / f"{r.date.isoformat()}.npy")

    def __len__(self):
        return len(self.days)

    def __getitem__(self, i):
        r = self.days.iloc[i]
        x = self._load(r)
        mask = np.isfinite(x)
        scale = self.scales[r.station] if self.norm == "station" else float(np.nanstd(x))
        x = np.arcsinh(np.where(mask, x, 0.0) / max(scale, 1e-12)).astype(np.float32)
        return torch.from_numpy(np.stack([x, mask.astype(np.float32)])), torch.tensor(target(r.days_to_event))


class FeatureDays(torch.utils.data.Dataset):
    """Per station: robust scaling and clipping fitted on the training windows, then optionally a PCA
    (pca = share of variance or number of components). Windows with a missing feature are zero with mask 0."""

    def __init__(self, days, data_dir, pca=None, fitted=None):
        self.days = days.reset_index(drop=True)
        self.fitted = {} if fitted is None else fitted
        self.x = {}
        for s, g in self.days.groupby("station"):
            x = np.load(Path(data_dir) / "features" / f"{s}.npz")["x"]
            ok = np.isfinite(x).all(axis=2)
            if s not in self.fitted:
                train = x[g.loc[g["role"] == "train", "row"].to_numpy()]
                train = train[np.isfinite(train).all(axis=2)]
                scaler = RobustScaler().fit(train)
                self.fitted[s] = (scaler, None if pca is None else
                                  PCA(n_components=pca, svd_solver="full").fit(self._scale(scaler, train)))
            scaler, p = self.fitted[s]
            flat = np.where(ok[..., None], x, 0.0).reshape(-1, x.shape[2])
            z = self._scale(scaler, flat)
            z = (z if p is None else p.transform(z)).reshape(x.shape[0], x.shape[1], -1)
            z = np.where(ok[..., None], z, 0.0)
            self.x[s] = np.concatenate([z, ok[..., None]], axis=2).transpose(0, 2, 1).astype(np.float32)
        self.channels = next(iter(self.x.values())).shape[1]
        if len({v.shape[1] for v in self.x.values()}) != 1:
            raise ValueError("the PCA keeps a different number of components per station; give pca as a number")

    @staticmethod
    def _scale(scaler, x):
        return np.clip(scaler.transform(x), -CLIP, CLIP)

    def __len__(self):
        return len(self.days)

    def __getitem__(self, i):
        r = self.days.iloc[i]
        return torch.from_numpy(self.x[r.station][r.row]), torch.tensor(target(r.days_to_event))
