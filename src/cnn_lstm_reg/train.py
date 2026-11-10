"""Trains one model (model.py) on one kind of input and scores it on the validation blocks.

    uv run python -u -m cnn_lstm_reg.train --kind raw|features|pca [--arch lstm|transformer]
                                           [--tokenizer cnn|patch] --out runs/NAME [options]

Writes to --out: run.json (command line, settings, commit, versions, the days and their roles), train.log lines
on stdout (one per LOG_EVERY steps: loss, learning rate, speed, ETA), metrics.jsonl (one line per epoch),
best.pt and val_predictions.csv of the best epoch, and an empty file DONE at the end.

The test blocks of data.split are never read here. Two baselines are printed first, on the same validation
days: the mean of the training targets, and a linear fit on the day's level alone (log of its standard
deviation, per station), which is what a model can get from loudness without any other property of the day.
"""
import argparse
import datetime as dt
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from torch import nn

from . import data
from .build_cache import commit
from .model import DayModel

LOG_EVERY = 10
STATIONS = ["FRT", "KOVA", "MAYA", "NARI", "SVRC"]


def scores(y, p):
    """Errors of predictions p of log1p(days to event) y."""
    return dict(mae_log=float(np.mean(np.abs(y - p))),
                median_abs_days=float(np.median(np.abs(np.expm1(y) - np.expm1(p)))),
                spearman=float(spearmanr(y, p).statistic) if np.ptp(p) > 1e-6 else 0.0)


def day_level(kind, days, data_dir):
    """log of the day's standard deviation (raw cache) or of its median window raw_std (feature tables)."""
    out = np.empty(len(days))
    feats = {}
    for i, r in enumerate(days.itertuples()):
        if kind == "raw":
            out[i] = np.log(np.nanstd(np.load(Path(data_dir) / "raw5hz" / r.station / f"{r.date.isoformat()}.npy")))
        else:
            if r.station not in feats:
                feats[r.station] = np.load(Path(data_dir) / "features" / f"{r.station}.npz")["x"][:, :, 1]
            out[i] = np.log(np.nanmedian(feats[r.station][r.row]))
    return out


def baselines(kind, days, data_dir):
    y = data.target(days["days_to_event"])
    tr, va = (days["role"] == "train").to_numpy(), (days["role"] == "val").to_numpy()
    level = day_level(kind, days, data_dir)
    stations = sorted(days["station"].unique())
    onehot = np.stack([(days["station"] == s).to_numpy(float) for s in stations], axis=1)
    X = np.column_stack([onehot, onehot * level[:, None]])          # level and offset per station
    fit = LinearRegression(fit_intercept=False).fit(X[tr], y[tr])
    level_pred = fit.predict(X[va])
    return dict(train_mean=scores(y[va], np.full(va.sum(), y[tr].mean())), level_only=scores(y[va], level_pred)), level_pred


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        ps.append(model(x.to(device)).float().cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--kind", required=True, choices=["raw", "features", "pca"])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--stations", default=",".join(STATIONS))
    ap.add_argument("--norm", choices=["station", "day"], default="station", help="raw input only, see data.RawDays")
    ap.add_argument("--pca", type=float, default=7, help="pca input: number of components (or a share of variance)")
    ap.add_argument("--same-days-as-features", action="store_true",
                    help="raw input: use only the station-days that also have features")
    ap.add_argument("--min-coverage", type=float, default=0.5,
                    help="least share of a day with data (raw) or with complete windows (features, pca)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--arch", choices=["lstm", "transformer"], default="lstm")
    ap.add_argument("--tokenizer", choices=["cnn", "patch"], default="cnn", help="patch = no CNN, see model.py")
    ap.add_argument("--lr", type=float, help="peak learning rate (default 1e-3 lstm, 5e-4 transformer)")
    ap.add_argument("--weight-decay", type=float, help="default 0.01 lstm, 0.05 transformer")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0, help="of the model and the batches")
    ap.add_argument("--split-seed", type=int, default=0, help="of the blocks; keep it fixed across runs")
    ap.add_argument("--fold", type=int, default=0, choices=range(data.FOLDS), help="which fold validates")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.lr is None:
        args.lr = 1e-3 if args.arch == "lstm" else 5e-4
    if args.weight_decay is None:
        args.weight_decay = 1e-2 if args.arch == "lstm" else 5e-2
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    stations = args.stations.split(",")
    days = data.list_days(args.kind, args.data, stations, args.min_coverage)
    if args.kind == "raw" and args.same_days_as_features:
        keep = data.list_days("features", args.data, stations, 0.0)[["station", "date"]]
        days = days.merge(keep, on=["station", "date"])
    days = data.split(days, args.split_seed, args.fold)
    if (days["role"] == "train").sum() == 0 or (days["role"] == "val").sum() == 0:
        sys.exit("no training or no validation days")

    if args.kind == "raw":
        full = data.RawDays(days, args.data, args.norm)
        channels = 2
    else:
        pca = None if args.kind == "features" else (int(args.pca) if args.pca >= 1 else args.pca)
        full = data.FeatureDays(days, args.data, pca)
        channels = full.channels
    idx = {r: np.where(days["role"] == r)[0].tolist() for r in ("train", "val")}
    workers = args.workers if args.kind == "raw" else 0
    train_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(full, idx["train"]), batch_size=args.batch,
                                               shuffle=True, drop_last=True, num_workers=workers)
    val_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(full, idx["val"]), batch_size=args.batch,
                                             num_workers=workers)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "DONE").unlink(missing_ok=True)
    base, level_pred = baselines(args.kind, days, args.data)
    model = DayModel(args.kind, channels, args.arch, args.tokenizer, hidden=args.hidden).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    (args.out / "run.json").write_text(json.dumps(dict(
        argv=sys.argv, started=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        commit=commit(), versions=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__),
        device=torch.cuda.get_device_name(0) if args.device.startswith("cuda") else args.device,
        parameters=n_params, baselines=base,
        days={r: int((days["role"] == r).sum()) for r in ("train", "val")},
        days_per_station=days.groupby(["station", "role"]).size().unstack(fill_value=0).to_dict("index"),
        split=[dict(station=r.station, date=r.date.isoformat(), role=r.role) for r in days.itertuples()]), indent=1))
    print(f"{args.kind}, {args.tokenizer} + {args.arch}, fold {args.fold}, seed {args.seed}: {len(idx['train'])} training and {len(idx['val'])} validation station-days, "
          f"{channels} input channels, {n_params:,} parameters, device {args.device}", flush=True)
    for name, s in base.items():
        print(f"baseline {name}: val MAE(log) {s['mae_log']:.3f} | median |error| {s['median_abs_days']:.0f} days | "
              f"Spearman {s['spearman']:.2f}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total = args.epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.1)
    loss_fn = nn.HuberLoss(delta=1.0)
    step, best, t0 = 0, math.inf, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = []
        for x, y in train_loader:
            x, y = x.to(args.device, non_blocking=True), y.to(args.device)
            loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            running.append(loss.item())
            if step % LOG_EVERY == 0 or step == total:
                speed = step * args.batch / (time.time() - t0)
                print(f"step {step}/{total} | epoch {epoch} | loss {np.mean(running[-LOG_EVERY:]):.4f} | "
                      f"lr {sched.get_last_lr()[0]:.2e} | {speed:.1f} days/s | "
                      f"ETA {(total - step) * args.batch / speed / 60:.1f} min", flush=True)
        y, p = predict(model, val_loader, args.device)
        s = scores(y, p)
        with open(args.out / "metrics.jsonl", "a") as fh:
            fh.write(json.dumps(dict(epoch=epoch, step=step, train_loss=float(np.mean(running)), **s)) + "\n")
        print(f"epoch {epoch} | train loss {np.mean(running):.4f} | val MAE(log) {s['mae_log']:.3f} | "
              f"median |error| {s['median_abs_days']:.0f} days | Spearman {s['spearman']:.2f}"
              f"{' | best' if s['mae_log'] < best else ''}", flush=True)
        if s["mae_log"] < best:
            best = s["mae_log"]
            torch.save(dict(model=model.state_dict(), epoch=epoch, scores=s, settings=vars(args)), args.out / "best.tmp")
            os.replace(args.out / "best.tmp", args.out / "best.pt")
            val_days = days.iloc[idx["val"]]
            val_days.assign(predicted_log=p, predicted_days=np.expm1(p), level_only_log=level_pred,
                            train_mean_log=float(data.target(days.loc[days["role"] == "train", "days_to_event"]).mean()))[
                ["station", "date", "days_to_event", "predicted_log", "predicted_days", "level_only_log", "train_mean_log"]
            ].to_csv(args.out / "val_predictions.csv", index=False)
    (args.out / "DONE").touch()
    print(f"DONE | best val MAE(log) {best:.3f} | baselines: train mean {base['train_mean']['mae_log']:.3f}, "
          f"level only {base['level_only']['mae_log']:.3f}", flush=True)


if __name__ == "__main__":
    main()
