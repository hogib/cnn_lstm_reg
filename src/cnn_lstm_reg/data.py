"""Days, their split and the three kinds of model input.

One sample is one station-day; the target is log1p(days to the event).

Split (blocked in time): the calendar is cut into blocks of BLOCK_DAYS days counted back from the event day,
the same blocks at every station, and each block is train, val or test as a whole (see block_roles). A day closer than GAP_DAYS
to a day of another role is not used at all, so no sample has a neighbouring day in another set. The test
blocks are held out: nothing in this package reads them until `roles(..., with_test=True)` is asked for.

Pseudo-events (placebo): `event_day` moves the target to a date a whole number of blocks before the real
event, where no earthquake happened, and `history` keeps only that many days before it (see event_roles).

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
EVENTS = {   # name: (mainshock, UTC; first day of the split; stations). Elazığ is the default everywhere.
    "elazig_2020": (EVENT_TIME, FIRST_DAY, ["FRT", "KOVA", "MAYA", "NARI", "SVRC"]),
    # time as in the sfc project of the stations; the day files start on 9 February 2025 (13 whole blocks from the 10th)
    "balikesir_2025": (dt.datetime(2025, 8, 10, 16, 53, 0), dt.date(2025, 2, 10), ["PURS", "STEP", "DURS", "BKES"]),
}


def block_roles(seed, fold=0, first_day=FIRST_DAY, last_day=EVENT_TIME.date()):
    """{date: 'train' | 'val' | 'test'} for every calendar day that is not within GAP_DAYS of another role.

    The blocks of the last year before the event (where the data are dense) and the earlier blocks are dealt
    out separately, so each period has its share of every role: TEST_SHARE of the blocks are test, fixed by
    the seed; the others are dealt into FOLDS folds, and fold `fold` is the validation set. Over the FOLDS
    values of `fold` every block that is not a test block is validated once."""
    n_days = (last_day - first_day).days + 1
    fold_of_block = _deal(seed, -(-n_days // BLOCK_DAYS))
    role_of_block = ["test" if f < 0 else "val" if f == fold else "train" for f in fold_of_block]
    return _roles_by_day(role_of_block, n_days, last_day)


def _deal(seed, n_blocks):
    """Fold of every block (index = age of the block, 0 is the one that ends at the event), -1 for a test block."""
    rng = np.random.default_rng(seed)
    fold_of_block = np.full(n_blocks, -1)
    recent = -(-RECENT_DAYS // BLOCK_DAYS)
    for stratum in (np.arange(min(recent, n_blocks)), np.arange(min(recent, n_blocks), n_blocks)):
        order = rng.permutation(stratum)
        n_test = round(TEST_SHARE * len(order))
        fold_of_block[order[n_test:]] = np.arange(len(order) - n_test) % FOLDS
    return fold_of_block


def _roles_by_day(role_of_block, n_days, last_day):
    by_age = [role_of_block[age // BLOCK_DAYS] for age in range(n_days)]      # age = days before the event
    roles = {}
    for age in range(n_days):
        near = by_age[max(age - GAP_DAYS, 0):age + GAP_DAYS + 1]
        if all(r == by_age[age] for r in near):
            roles[last_day - dt.timedelta(days=age)] = by_age[age]
    return roles


def event_roles(event_day, history, seed, fold, first_day=FIRST_DAY, real_day=EVENT_TIME.date()):
    """Roles of the `history` days up to `event_day`, which is the real event day or a pseudo-event a whole
    number of blocks before it, so the blocks are the same calendar blocks as in block_roles.

    The test blocks are the test blocks of block_roles(seed) and stay unread. The other blocks are dealt
    into the folds in the order of their age (the first one that is not a test block to fold 0, the next
    to fold 1, ...), so every event has the same geometry of training and validation blocks relative to
    itself, and the blocks closest to it are in different folds."""
    shift = (real_day - event_day).days
    if shift < 0 or shift % BLOCK_DAYS or history % BLOCK_DAYS:
        raise ValueError(f"the event day must lie a whole number of {BLOCK_DAYS}-day blocks before {real_day}, "
                         f"and the history must be a whole number of blocks")
    n_blocks = -(-((real_day - first_day).days + 1) // BLOCK_DAYS)
    is_test = _deal(seed, n_blocks)[shift // BLOCK_DAYS:][:history // BLOCK_DAYS] < 0
    if len(is_test) < history // BLOCK_DAYS:
        raise ValueError("the history reaches back beyond the first day of the split")
    role_of_block, i = [], 0
    for t in is_test:
        role_of_block.append("test" if t else "val" if i % FOLDS == fold else "train")
        i += not t
    return _roles_by_day(role_of_block, history, event_day)


def held_out_days(seed, first_day=FIRST_DAY, real_day=EVENT_TIME.date()):
    """The calendar days of the test blocks of block_roles(seed) and the days within GAP_DAYS of them."""
    n_days = (real_day - first_day).days + 1
    is_test = _deal(seed, -(-n_days // BLOCK_DAYS)) < 0
    out = set()
    for age in range(n_days):
        if is_test[age // BLOCK_DAYS]:
            out.update(real_day - dt.timedelta(days=age + g) for g in range(-GAP_DAYS, GAP_DAYS + 1))
    return out


def transfer_days(days, train_events, val_events, history, seed, first_day=FIRST_DAY, real_day=EVENT_TIME.date()):
    """Split by period instead of by block: the `history` days up to each day of train_events are training
    days and those up to each day of val_events are validation days, every day counted to its own event day
    (column 'event'). Used to train on one winter and predict another. The days of the test blocks and the
    days within GAP_DAYS of them are left out, so the held-out days stay unread and keep their gap."""
    out, held = [], held_out_days(seed, first_day, real_day)
    for role, events in (("train", train_events), ("val", val_events)):
        for ev in events:
            to_event = np.array([(ev - d).days for d in days["date"]])
            keep = (to_event >= 0) & (to_event < history) & ~days["date"].isin(held).to_numpy()
            out.append(days[keep].assign(days_to_event=to_event[keep], role=role, event=ev.isoformat()))
    out = pd.concat(out, ignore_index=True)
    if out.duplicated(["station", "date"]).any():
        raise ValueError("the periods of the events overlap")
    return out


def list_days(kind, data_dir, stations, min_coverage, event_day=None, history=None):
    """Table of the usable station-days: station, date, days_to_event, coverage (and row in the feature file).
    With event_day the days are counted to that day instead of the real event, and only the `history` days
    up to it are kept."""
    frames = []
    for s in stations:
        f = Path(data_dir) / ("raw5hz" if kind == "raw" else "features") / (f"{s}/index.csv" if kind == "raw" else f"{s}_index.csv")
        df = pd.read_csv(f, parse_dates=["date"])
        df["row"] = np.arange(len(df))
        df["station"] = s
        frames.append(df[["station", "date", "days_to_event", "coverage", "row"]])
    days = pd.concat(frames, ignore_index=True)
    days["date"] = days["date"].dt.date
    if event_day is not None:
        days["days_to_event"] = [(event_day - d).days for d in days["date"]]
        days = days[days["days_to_event"] < history]
    return days[(days["coverage"] >= min_coverage) & (days["days_to_event"] >= 0)].reset_index(drop=True)


def split(days, seed, fold=0, with_test=False, event_day=None, history=None, first_day=FIRST_DAY,
          real_day=EVENT_TIME.date()):
    """Adds the 'role' column; days in a gap are dropped, and so are the test days unless with_test.
    With event_day the roles are those of event_roles."""
    roles = (block_roles(seed, fold, first_day, real_day) if event_day is None
             else event_roles(event_day, history, seed, fold, first_day, real_day))
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
