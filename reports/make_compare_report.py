"""PDF report of the LSTM and transformer runs: four inputs x two sequence models x seeds x folds.

usage: make_compare_report.py RUNS_DIR DATA_DIR OUT.pdf

Needs, under RUNS_DIR, the runs of scripts/train_compare.sh (<input>_<arch>_s<seed>_f<fold>/ with run.json,
metrics.jsonl and val_predictions.csv; best.pt is not needed) and, under DATA_DIR, features/<STATION>_index.csv
(for the table each feature day came from). Only validation days are read; the test blocks stay held out.
Under DATA_DIR it also needs raw5hz/ and features/ of one station for the figure of an example day (a training
day; no test day is read). If RUNS_DIR/placebo holds the runs of scripts/train_placebo.sh, the pseudo-event
placebo is added as section 9; RUNS_DIR/winters (scripts/train_winters.sh) as section 10 and
RUNS_DIR/placebo_balikesir (scripts/train_placebo_balikesir.sh) as section 11.

    uv run --with reportlab --with matplotlib python reports/make_compare_report.py runs data \
        reports/elazig_2020_lstm_transformer_report.pdf

The page layout (fonts, colours, table style) is copied from reports/make_report.py and reports/cm_grid.py of
seismic-features-classification (commit 6c64090), so that the reports of the two repositories look the same.
"""
import datetime as dt
import io
import json
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_LEFT  # noqa: E402
from reportlab.lib.pagesizes import A4, landscape  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,  # noqa: E402
                                TableStyle)

from cnn_lstm_reg import placebo as plc  # noqa: E402
from cnn_lstm_reg import winters as wnt  # noqa: E402
from cnn_lstm_reg.build_cache import EVENT_TIME  # noqa: E402
from cnn_lstm_reg.data import (BLOCK_DAYS, CLIP, EVENTS, FOLDS, GAP_DAYS, RECENT_DAYS, TEST_SHARE,  # noqa: E402
                               block_roles, list_days)
from cnn_lstm_reg.model import SPAN, TOKENS, DayModel  # noqa: E402
from cnn_lstm_reg.summarize import rho  # noqa: E402

# --------------------------------------------------------------------------- layout (as in the sfc reports)

FONT_DIR = "/usr/share/fonts/noto"
pdfmetrics.registerFont(TTFont("Noto", f"{FONT_DIR}/NotoSans-Regular.ttf"))
pdfmetrics.registerFont(TTFont("Noto-Bold", f"{FONT_DIR}/NotoSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("Noto-Italic", f"{FONT_DIR}/NotoSans-Italic.ttf"))
pdfmetrics.registerFontFamily("Noto", normal="Noto", bold="Noto-Bold", italic="Noto-Italic", boldItalic="Noto-Bold")

ACCENT = colors.HexColor("#1f4e79")
LIGHT = colors.HexColor("#e8eef5")
GRID = colors.HexColor("#b0b8c4")
INK, MUTED, GRIDC = "#2b2b2b", "#6b6b6b", "#d9d9d6"

ss = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("title", parent=ss["Title"], fontName="Noto-Bold", fontSize=20, leading=24,
                            textColor=ACCENT, alignment=TA_LEFT),
    "h1": ParagraphStyle("h1", parent=ss["Heading1"], fontName="Noto-Bold", fontSize=15, leading=19,
                         textColor=ACCENT, spaceBefore=4, spaceAfter=6),
    "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontName="Noto-Bold", fontSize=12, leading=15,
                         textColor=ACCENT, spaceBefore=8, spaceAfter=4),
    "body": ParagraphStyle("body", parent=ss["BodyText"], fontName="Noto", fontSize=9, leading=12.5),
    "small": ParagraphStyle("small", parent=ss["BodyText"], fontName="Noto", fontSize=7.5, leading=10,
                            textColor=colors.HexColor("#444444")),
    "cell": ParagraphStyle("cell", fontName="Noto", fontSize=7.5, leading=9.5),
    "head": ParagraphStyle("head", fontName="Noto-Bold", fontSize=7.5, leading=9.5, textColor=colors.white),
}


def P(text, style="cell"):
    return Paragraph(str(text), S[style])


def table(rows, widths, head_rows=1, zebra=True):
    data = [[c if isinstance(c, Paragraph) else P(c, "head" if i < head_rows else "cell") for c in r]
            for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=head_rows)
    st = [("BACKGROUND", (0, 0), (-1, head_rows - 1), ACCENT),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("GRID", (0, 0), (-1, -1), 0.4, GRID),
          ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
          ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    if zebra:
        for i in range(head_rows, len(rows)):
            if (i - head_rows) % 2:
                st.append(("BACKGROUND", (0, i), (-1, i), LIGHT))
    t.setStyle(TableStyle(st))
    return t


def note(title, lines, W):
    """Shaded box of short lines: a cheat note for presenting, not part of the description of the runs."""
    box = Table([[[P(f"<b>Cheat note: {title}</b>", "body")] + [P("• " + x, "body") for x in lines]]], colWidths=[W])
    box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff6d6")),
                             ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#d9b84a")),
                             ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                             ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7)]))
    return [Spacer(1, 7), box]


SCALING_NOTE = [
    "Station scale: every day is divided by one number per station (the median standard deviation of its training "
    "days). A loud day stays loud, so the model sees how the day compares with the station's usual level.",
    "Day scale: every day is divided by its own standard deviation. The loudness of the day as a whole is removed; "
    "what is left is its shape (the course within the day, how spiky it is, its frequency content).",
    "A result that holds with day scale comes from the shape; one that holds only with station scale comes mainly "
    "from the loudness of the day. Day scale does not remove everything that goes along with a loud day."]


def fig_to_image(fig, width_cm):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    w, h = fig.get_size_inches()
    return Image(buf, width=width_cm * cm, height=width_cm * cm * h / w)


# --------------------------------------------------------------------------- data

STATIONS = ["FRT", "KOVA", "MAYA", "NARI", "SVRC"]
INPUTS = [("features", "features", "12 window features + mask, 13 × 432"),
          ("pca", "pca", "7 principal components + mask, 8 × 432"),
          ("raw_station", "raw, station scale", "5 Hz signal + mask, 2 × 432 000; the day's level kept"),
          ("raw_day", "raw, day scale", "5 Hz signal + mask, 2 × 432 000; the day's level removed")]
ARCHS = [("lstm", "LSTM"), ("transformer", "Transformer")]
RANGES = [(0, 14), (15, 30), (31, 90), (91, 180), (181, 365), (366, 10**6)]
GROUPS = [("final", "Final month, every day (0–30 days)"), ("summer", "Summer placebo block, every day"),
          ("spring", "Spring placebo block, every day"), ("winter", "Winter placebo block, every day"),
          ("sampled", "Sampled days, 31–365 days (every 2nd or 9th day)"),
          ("earlier", "Sampled days, more than a year before")]
SHADE = {"final": "#c0392b", "summer": "#e69f00", "spring": "#009e73", "winter": "#0072b2"}


def minus(text):
    return text.replace("-", "−")


def pm(a, b, digits=2):
    return minus(f"{a:.{digits}f} ± {b:.{digits}f}")


def load(runs):
    """{(input, arch): dict(per_seed={seed: frame}, mean=frame, info=[run.json ...], curves=[...])}; the frames
    hold the out-of-fold predictions of the validation days, with y = log1p(days to the event) and the fold."""
    out = {}
    for key, _, _ in INPUTS:
        for arch, _ in ARCHS:
            per_seed, info = {}, []
            for d in sorted(Path(runs).glob(f"{key}_{arch}_s*_f*")):
                m = re.fullmatch(rf"{key}_{arch}_s(\d+)_f(\d)", d.name)
                if not m or not (d / "DONE").exists():
                    continue
                r = json.loads((d / "run.json").read_text())
                train = [s["date"] for s in r.pop("split") if s["role"] == "train"]
                r["closest_train"] = min((EVENT_TIME.date() - dt.date.fromisoformat(x)).days for x in train)
                epochs = [json.loads(line) for line in (d / "metrics.jsonl").read_text().splitlines() if line.strip()]
                r["best_epoch"] = min(epochs, key=lambda e: e["mae_log"])["epoch"]
                r["seed"], r["fold"] = int(m.group(1)), int(m.group(2))
                info.append(r)
                p = pd.read_csv(d / "val_predictions.csv").assign(fold=int(m.group(2)))
                per_seed.setdefault(int(m.group(1)), []).append(p)
            if not per_seed:
                sys.exit(f"no finished runs of {key}_{arch} in {runs}")
            per_seed = {s: pd.concat(v, ignore_index=True).assign(y=lambda t: np.log1p(t["days_to_event"]))
                        for s, v in per_seed.items()}
            mean = (pd.concat(per_seed.values())
                    .groupby(["station", "date", "days_to_event", "fold"], as_index=False).mean(numeric_only=True))
            mean["predicted_days"] = np.expm1(mean["predicted_log"])
            out[(key, arch)] = dict(per_seed=per_seed, mean=mean, info=info)
    return out


def scores(d, col="predicted_log"):
    """MAE of log1p(days) and Spearman, over all validation days and over the last year."""
    year = d[d["days_to_event"] <= 365]
    return [np.abs(d.y - d[col]).mean(), rho(d.y, d[col]), np.abs(year.y - year[col]).mean(), rho(year.y, year[col])]


def seed_scores(r, col="predicted_log"):
    m = np.array([scores(x, col) for x in r["per_seed"].values()])
    return m.mean(axis=0), m.std(axis=0)


def feature_groups(data):
    """station, date -> group of the feature day, from the sfc table it was imported from."""
    frames = []
    for s in STATIONS:
        d = pd.read_csv(Path(data) / "features" / f"{s}_index.csv")
        block = d["table"].str.extract(r"placebo/[A-Z]+_(spring|summer|winter)/")[0]
        g = np.where(block.notna(), block, np.where(d["days_to_event"] <= 30, "final",
                                                    np.where(d["days_to_event"] <= 365, "sampled", "earlier")))
        frames.append(pd.DataFrame(dict(station=s, date=d["date"], group=g, days_to_event=d["days_to_event"])))
    return pd.concat(frames, ignore_index=True)


def spread(x):
    """median (10 %–90 %) of predicted days"""
    if not len(x):
        return "–"
    lo, mid, hi = np.percentile(x, [10, 50, 90])
    return f"<b>{mid:.0f}</b> ({lo:.0f}–{hi:.0f})"


# --------------------------------------------------------------------------- task and models

EXAMPLE_STATION = "FRT"
ROLE_COLOURS = {"train": "#1f4e79", "val": "#e69f00", "test": "#9a9a9a", "gap": "#ffffff"}
BOX = {"input": "#e8eef5", "conv": "#d6e6d2", "tokens": "#fdf0cf", "seq": "#f6d9cf", "out": "#e6e0f0"}
RAW_CONVS = [(15, 5, 16), (9, 4, 32), (9, 5, 64), (7, 3, 64)]      # kernel, stride, channels: as in model.py
FEATURE_CONVS = [(5, 1, 64), (5, 2, 64)]


def spaced(n):
    return f"{n:,}".replace(",", " ")


def receptive(convs):
    """Input samples one output of the stack depends on."""
    field, step = 1, 1
    for kernel, stride, _ in convs:
        field += (kernel - 1) * step
        step *= stride
    return field


def example_days(data, seed):
    """A training day of EXAMPLE_STATION (fold 0) with nearly complete data, for the raw input and the features."""
    roles = block_roles(seed, 0)
    out = {}
    for kind in ("raw", "features"):
        d = list_days(kind, data, [EXAMPLE_STATION], 0.9)
        d = d[d["date"].map(roles).eq("train") & d["days_to_event"].between(20, 120)]
        if d.empty:
            sys.exit(f"no training day of {EXAMPLE_STATION} for the example figure")
        out[kind] = d.sort_values("coverage", ascending=False).iloc[0]
    return out


def fig_example(data, seed):
    """What the model is given for one day: the raw input (whole day and ten minutes) and the feature input."""
    ex = example_days(data, seed)
    raw = np.load(Path(data) / "raw5hz" / EXAMPLE_STATION / f"{ex['raw'].date.isoformat()}.npy")
    x = np.arcsinh(np.where(np.isfinite(raw), raw, 0.0) / np.nanstd(raw))
    feats = np.load(Path(data) / "features" / f"{EXAMPLE_STATION}.npz")["x"][ex["features"].row]          # (432, 12)
    names = json.loads((Path(data) / "features" / f"{EXAMPLE_STATION}_provenance.json").read_text())["parameters"]["features"]
    fig = plt.figure(figsize=(11.2, 5.6))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.5], hspace=0.75, wspace=0.12)
    ax = fig.add_subplot(gs[0, :])
    env = x.reshape(-1, 50)                                             # 10 s per column
    t = np.arange(len(env)) / 360
    ax.fill_between(t, env.min(axis=1), env.max(axis=1), color="#1f4e79", lw=0)
    ax.set_xlim(0, 24)
    ax.set_title(f"raw input, channel 1: {EXAMPLE_STATION} {ex['raw'].date}, {ex['raw'].days_to_event} days before the event "
                 f"(target {np.log1p(ex['raw'].days_to_event):.2f}) · 0.1–2 Hz at 5 Hz, 432 000 samples, day scale, asinh",
                 fontsize=7, color=INK, loc="left")
    ax.set_xlabel("hour of the day (UTC)", fontsize=6.5, color=INK)
    ax = fig.add_subplot(gs[1, 0])
    a = 12 * 3600 * 5
    ax.plot(np.arange(3000) / 5, x[a:a + 3000], color="#1f4e79", lw=0.4)
    for k in range(0, 601, 60):
        ax.axvline(k, color="#c0392b", lw=0.5, ls=":")
    ax.set_xlim(0, 600)
    ax.set_title("ten minutes of it (12:00–12:10); dotted: the stretches of the one-minute tokens", fontsize=7,
                 color=INK, loc="left")
    ax.set_xlabel("seconds", fontsize=6.5, color=INK)
    ax = fig.add_subplot(gs[1, 1])
    mask = np.isfinite(raw).reshape(-1, 300).mean(axis=1)
    ax.fill_between(np.arange(1440) / 60, 0, mask, color="#6b6b6b", lw=0, step="mid")
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"raw input, channel 2: the data mask (1 = data), shown per minute; coverage {ex['raw'].coverage:.2f}",
                 fontsize=7, color=INK, loc="left")
    ax.set_xlabel("hour of the day (UTC)", fontsize=6.5, color=INK)
    ax = fig.add_subplot(gs[2, :])
    med = np.nanmedian(feats, axis=0)
    iqr = np.nanpercentile(feats, 75, axis=0) - np.nanpercentile(feats, 25, axis=0)
    z = np.clip((feats - med) / np.where(iqr > 0, iqr, 1.0), -CLIP, CLIP).T
    im = ax.imshow(z, aspect="auto", cmap="RdBu_r", vmin=-CLIP, vmax=CLIP, extent=(0, 24, len(names) - 0.5, -0.5),
                   interpolation="nearest")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_title(f"feature input: {EXAMPLE_STATION} {ex['features'].date}, {ex['features'].days_to_event} days before the "
                 f"event · 12 features × 432 windows of 200 s; here each feature is centred and scaled by this "
                 f"day's own median and quartiles, clipped at ±{CLIP:g} (white: no value)", fontsize=7, color=INK, loc="left")
    ax.set_xlabel("hour of the day (UTC)", fontsize=6.5, color=INK)
    fig.colorbar(im, ax=ax, fraction=0.012, pad=0.01).ax.tick_params(labelsize=5.5)
    for a_ in fig.axes:
        style_axis(a_)
    return fig


def fig_split(seed, days=420):
    """Target and split over the last `days` days: the role of every calendar day in each fold."""
    code = {"gap": 0, "train": 1, "val": 2, "test": 3}
    grid = np.zeros((FOLDS, days), dtype=int)
    for f in range(FOLDS):
        roles = block_roles(seed, f)
        for age in range(days):
            grid[f, age] = code[roles.get(EVENT_TIME.date() - dt.timedelta(days=age), "gap")]
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(11.2, 3.3), sharex=True, gridspec_kw=dict(height_ratios=[1, 1.25], hspace=0.12))
    age = np.arange(days)
    a0.plot(age, np.log1p(age), color="#c0392b", lw=1.2)
    a0.set_ylabel("target\nlog(1 + days)", fontsize=6.5, color=INK)
    for d_, lab in ((0, "0 d → 0"), (7, "7 d → 2.08"), (30, "30 d → 3.43"), (90, "90 d → 4.51"), (365, "365 d → 5.90")):
        a0.plot([d_], [np.log1p(d_)], marker="o", ms=3, color="#c0392b")
        near = d_ <= 7
        a0.annotate(lab, (d_, np.log1p(d_)), textcoords="offset points", xytext=(5, -2) if near else (0, -10),
                    ha="left" if near else "center", fontsize=6, color=INK)
    from matplotlib.colors import ListedColormap
    a1.imshow(grid, aspect="auto", cmap=ListedColormap([ROLE_COLOURS[k] for k in ("gap", "train", "val", "test")]),
              vmin=0, vmax=3, extent=(-0.5, days - 0.5, FOLDS - 0.5, -0.5), interpolation="nearest")
    for b in range(0, days, BLOCK_DAYS):
        a1.axvline(b - 0.5, color="white", lw=0.3)
    a1.axvline(RECENT_DAYS - 0.5, color=INK, lw=0.7, ls="--")
    a1.set_yticks(range(FOLDS))
    a1.set_yticklabels([f"fold {f}" for f in range(FOLDS)])
    a1.set_xlabel(f"days before the event (blocks of {BLOCK_DAYS} days; dashed: the last {RECENT_DAYS} days are dealt out on their own)",
                  fontsize=6.5, color=INK)
    a1.set_xlim(days - 0.5, -32)
    from matplotlib.patches import Patch
    a1.legend(handles=[Patch(fc=ROLE_COLOURS["train"], label="training"), Patch(fc=ROLE_COLOURS["val"], label="validation"),
                       Patch(fc=ROLE_COLOURS["test"], label="test block (held out, not read)"),
                       Patch(fc=ROLE_COLOURS["gap"], ec=GRIDC, label=f"not used (within {GAP_DAYS} days of another role)")],
              fontsize=6, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.42))
    for a_ in (a0, a1):
        style_axis(a_)
    return fig


def chain(ax, y, items, height=1.0, gap=0.35):
    """A row of boxes joined by arrows; items = (text, width, colour key). Returns the right end."""
    from matplotlib.patches import FancyBboxPatch
    x = 0.0
    for i, (text, width, key) in enumerate(items):
        ax.add_patch(FancyBboxPatch((x, y - height / 2), width, height, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=BOX[key], ec="#7a8794", lw=0.7))
        ax.text(x + width / 2, y, text, ha="center", va="center", fontsize=6.2, color=INK, linespacing=1.35)
        if i < len(items) - 1:
            ax.annotate("", (x + width + gap, y), (x + width, y), arrowprops=dict(arrowstyle="-|>", color="#555555", lw=0.8))
        x += width + gap
    return x - gap


def conv_boxes(convs, length, first_period, unit):
    """Boxes of a stack of convolutions with the shape and the time step after each."""
    out, period = [], first_period
    for kernel, stride, ch in convs:
        length = (length + 2 * (kernel // 2) - kernel) // stride + 1
        period *= stride
        out.append((f"Conv1d, kernel {kernel}, stride {stride}\nGroupNorm(8) · GELU\n{ch} × {spaced(length)}\none value per {period:g} {unit}",
                    2.75, "conv"))
    return out, length


def fig_tokenizer():
    """The CNN in front of both sequence models, for the raw input and for the feature inputs."""
    fig, ax = plt.subplots(figsize=(11.2, 3.0))
    raw, n_raw = conv_boxes(RAW_CONVS, 432000, 0.2, "s")
    feat, n_feat = conv_boxes(FEATURE_CONVS, 432, 200, "s")
    assert n_raw == TOKENS["raw"] and n_feat == TOKENS["features"]
    ax.text(0, 2.95, "raw signal (station scale or day scale)", fontsize=7, color=INK, fontweight="bold")
    end = chain(ax, 2.2, [("one day\n2 × 432 000\n5 Hz signal + mask", 2.1, "input")] + raw
                + [(f"tokens\n{spaced(n_raw)} × 64\none per minute", 1.9, "tokens")])
    ax.text(0, 0.95, "features (13 channels) or pca (8 channels)", fontsize=7, color=INK, fontweight="bold")
    chain(ax, 0.2, [("one day\n13 × 432 or 8 × 432\n200 s windows + mask", 2.1, "input")] + feat
          + [(f"tokens\n{n_feat} × 64\none per 400 s", 1.9, "tokens")])
    ax.set_xlim(-0.1, end + 0.1)
    ax.set_ylim(-0.5, 3.2)
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    return fig


def fig_sequence_models():
    """The two models that read the tokens."""
    fig, ax = plt.subplots(figsize=(11.2, 3.3))
    ax.text(0, 3.25, "LSTM", fontsize=7, color=INK, fontweight="bold")
    chain(ax, 2.4, [("tokens\nT × 64", 1.5, "tokens"),
                    ("bidirectional LSTM, 1 layer\nforward: 64 units\nbackward: 64 units", 3.1, "seq"),
                    ("outputs\nT × 128", 1.5, "tokens"),
                    ("mean over\nall T tokens\n128", 1.9, "seq"),
                    ("dropout 0.1\nlinear 128 → 1", 2.0, "seq"),
                    ("prediction of\nlog(1 + days)", 1.9, "out")], height=1.1)
    ax.text(0, 1.05, "Transformer", fontsize=7, color=INK, fontweight="bold")
    end = chain(ax, 0.2, [("tokens\nT × 64", 1.5, "tokens"),
                          ("+ position\nlearned vector\nper time of day", 1.9, "seq"),
                          ("3 encoder layers (pre-norm):\nself-attention, 4 heads\nMLP 64 → 128 → 64, GELU\ntokens without data masked", 3.3, "seq"),
                          ("LayerNorm\nT × 64", 1.5, "tokens"),
                          ("mean over\ntokens with data\n64", 1.9, "seq"),
                          ("dropout 0.1\nlinear 64 → 1", 2.0, "seq"),
                          ("prediction of\nlog(1 + days)", 1.9, "out")], height=1.25)
    ax.set_xlim(-0.1, end + 0.1)
    ax.set_ylim(-0.6, 3.5)
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    return fig


def parameter_rows():
    rows = [["Input", "Model", "CNN tokenizer", "Position table", "Sequence model", "Output layer", "All parameters"]]
    for key, kt, channels in (("raw", "raw (both scalings)", 2), ("features", "features", 13), ("pca", "pca", 8)):
        for arch, at in ARCHS:
            m = DayModel(key, channels, arch, "cnn")
            n = {k: sum(p.numel() for p in c.parameters()) for k, c in m.named_children()}
            pos = m.position.numel() if arch == "transformer" else 0
            rows.append([P(f"<b>{kt}</b>"), at, spaced(n["tokenizer"]), spaced(pos) if pos else "–",
                         spaced(n["lstm" if arch == "lstm" else "encoder"]), spaced(n["head"]),
                         P(f"<b>{spaced(sum(p.numel() for p in m.parameters()))}</b>")])
    return rows


def intro_story(data, info0, seeds, folds, n_runs, W):
    """Sections 1-3: the task, the models, training and scoring."""
    seed = info0["settings"]["split_seed"]
    st = info0["settings"]
    story = [
        Paragraph("1. The task", S["h1"]),
        Paragraph(
            "<b>Question.</b> Can the time that is left until the mainshock be read from one day of a station's "
            "recording? If the ground motion changed in some systematic way as the earthquake came closer, a model "
            "given a day it has not seen should place it at roughly the right distance from the event.", S["body"]),
        Spacer(1, 4),
        Paragraph(
            f"<b>Sample and target.</b> One sample is one station-day: the N component of one station from 00:00 to "
            f"24:00 UTC. The model gives one number for it, and the target is log(1 + d), where d is the number of "
            f"calendar days from that day to the day of the mainshock ({EVENT_TIME:%d %B %Y}; d = 0 on that day, "
            f"whose recording is cut at {EVENT_TIME:%H}:00 UTC so that the mainshock itself is never in the input). "
            f"The logarithm makes an error of a few days count as much close to the event as an error of months "
            f"far from it: the days 0, 7, 30, 90 and 365 have the targets 0, 2.08, 3.43, 4.51 and 5.90. A model "
            f"sees a single day at a time and has no information on the date or on the neighbouring days.", S["body"]),
        Spacer(1, 4),
        Paragraph(
            "<b>Four inputs for the same day.</b> Each is a table of channels × time steps, and its last channel is "
            "a 0/1 mask that is 0 where data are missing.", S["body"]),
        Spacer(1, 3),
        table([["Input", "Shape per day", "What it holds", "Scaling"],
               [P("<b>raw, station scale</b>"), "2 × 432 000",
                "The signal band-passed to 0.1–2 Hz and resampled to 5 Hz (as in seismic-features-classification); "
                "gaps shorter than 2 s are interpolated, longer ones stay missing.",
                "Divided by the station's median daily standard deviation of the training days, then asinh. "
                "A loud day stays loud in the input."],
               [P("<b>raw, day scale</b>"), "2 × 432 000", "The same signal.",
                "Divided by the day's own standard deviation, then asinh. The level of the day is removed; only "
                "the shape of the signal is left."],
               [P("<b>features</b>"), "13 × 432",
                "The 12 features of seismic-features-classification for each of the day's 432 windows of 200 s "
                "(statistics of the window, mean frequency and period, two Lyapunov exponents, sample entropy, "
                "correlation dimension).",
                f"Per station: robust scaling fitted on the training windows, clipped at ±{CLIP:g}. A window with a "
                f"missing feature is zero with mask 0."],
               [P("<b>pca</b>"), "8 × 432", "The first 7 principal components of the scaled features.",
                "PCA fitted on the training windows of the station."]],
              [W * .14, W * .1, W * .4, W * .36]),
        Spacer(1, 5),
        Paragraph(
            f"A station-day is used when at least {100 * st['min_coverage']:.0f} % of it has data (raw) or complete "
            f"windows (features). The feature tables hold fewer days than the raw cache (section 6), so the feature "
            f"inputs and the raw inputs are trained and scored on different sets of days.", S["small"]),
        PageBreak(),
        P("<b>One day as the models get it</b> (a training day)", "cell"), Spacer(1, 3),
        fig_to_image(fig_example(data, seed), W / cm * .97),
        PageBreak(),
        Paragraph("2. The models", S["h1"]),
        Paragraph(
            "Both models work in two steps. A small convolutional network (the tokenizer) first shortens the day "
            "into a sequence of 64-dimensional vectors, the tokens: one per minute for the raw signal, one per 400 s "
            "for the feature inputs. A sequence model then reads all tokens of the day, and the average of its "
            "outputs is turned into the prediction by one linear layer. The two models share the tokenizer and "
            "the output layer and differ only in the sequence model.", S["body"]),
        Spacer(1, 5),
        P("<b>Step 1: the CNN tokenizer.</b> Each box is one layer with the shape of its output (channels × time "
          "steps).", "cell"), Spacer(1, 2),
        fig_to_image(fig_tokenizer(), W / cm * .97), Spacer(1, 2),
        Paragraph(
            f"The strides of the four raw layers multiply to {SPAN['raw']}, so 300 samples (60 s) become one token; "
            f"through the widths of the kernels a token depends on {receptive(RAW_CONVS)} samples "
            f"({receptive(RAW_CONVS) / 5:.0f} s) around its minute. A feature token covers {SPAN['features']} windows "
            f"and depends on {receptive(FEATURE_CONVS)} windows ({receptive(FEATURE_CONVS) * 200 / 60:.0f} min). The "
            f"weights are the same at every position of the day.", S["small"]),
        PageBreak(),
        P("<b>Step 2: the sequence model and the output</b> (T = 1 440 tokens for the raw signal, 216 for the "
          "feature inputs)", "cell"), Spacer(1, 2),
        fig_to_image(fig_sequence_models(), W / cm * .97), Spacer(1, 4),
        Paragraph(
            "<b>LSTM.</b> A recurrent network reads the tokens one after the other and carries a memory of 64 "
            "numbers along; here one reads the day forwards and a second one backwards, and their outputs are put "
            "side by side, so every position has 128 numbers that depend on the whole day. Tokens from stretches "
            "without data are read like the others; the model can recognise them by the mask channel.", S["body"]),
        Spacer(1, 4),
        Paragraph(
            "<b>Transformer.</b> Every token is compared with every other token of the day at once "
            "(self-attention), so a relation between two distant hours does not have to be carried through all "
            "the tokens between them. Since attention does not know the order of the tokens, a learned vector for "
            "the token's time of day is added first; it is the same on every day and so carries nothing about the "
            "date. Tokens whose stretch of the day has no data at all are left out of the attention and of the "
            "average.", S["body"]),
        Spacer(1, 6),
        P("<b>Number of trained parameters</b>", "cell"), Spacer(1, 3),
        table(parameter_rows(), [W * .2, W * .14, W * .14, W * .13, W * .14, W * .12, W * .13]),
        Spacer(1, 4),
        Paragraph(
            "The position table has one 64-dimensional vector per token, which is why the raw transformer is the "
            "largest model. Every model has far more parameters than a fold has training station-days (section 8).",
            S["small"]),
        PageBreak(),
        Paragraph("3. Training and scoring", S["h1"]),
        Paragraph(
            f"<b>Training.</b> Huber loss (δ = 1) on log(1 + d), AdamW, a one-cycle learning rate (rising over the "
            f"first 10 % of the steps, then falling) with peak 1e-3 for the LSTM and 5e-4 for the transformer, "
            f"weight decay 0.01 and 0.05, gradient norm clipped at 1, batches of {st['batch']} station-days, "
            f"{st['epochs']} epochs. The epoch with the lowest validation error is the one that is scored.", S["body"]),
        Spacer(1, 4),
        Paragraph(
            f"<b>Split.</b> Neighbouring days look alike, so days are not dealt out one by one. The calendar is cut "
            f"into {BLOCK_DAYS}-day blocks counted back from the event, the same at every station, and a block is "
            f"training, validation or test as a whole; a day within {GAP_DAYS} days of a block of another role is "
            f"not used. {100 * TEST_SHARE:.0f} % of the blocks are test blocks and were not read. The other blocks "
            f"form {FOLDS} folds (the last {RECENT_DAYS} days and the earlier days dealt out separately): in fold "
            f"k the blocks of fold k are validated and the rest are trained on, so over the {FOLDS} folds every "
            f"one of these days is predicted once by a model that did not train on its block (out-of-fold "
            f"predictions).", S["body"]),
        Spacer(1, 4),
        fig_to_image(fig_split(seed), W / cm * .97), Spacer(1, 4),
        Paragraph(
            f"<b>Runs.</b> 4 inputs × 2 models × {len(seeds)} seeds × {len(folds)} folds = {n_runs} runs; the seed "
            f"changes the initial weights and the order of the days, not the split.", S["body"]),
        Spacer(1, 4),
        Paragraph(
            "<b>Scores.</b> MAE of log(1 + d) (lower is better; 0.69 is a factor of 2 in 1 + d) and the Spearman "
            "rank correlation between the true and the predicted time (1 = the days are put in the right order, "
            "0 = no relation), over the out-of-fold predictions of a seed; given as mean ± standard deviation "
            "over the seeds, for all validation days and for the days of the last year.", S["body"]),
        Spacer(1, 4),
        Paragraph(
            "<b>Baselines</b>, on the same validation days: the mean of the training targets (a model that "
            "learned nothing), and a straight-line fit per station on the day's level alone, i.e. the logarithm "
            "of its standard deviation (what can be had from how loud the day is, without anything else).", S["body"]),
        Spacer(1, 4),
        Paragraph(
            "<b>What a score can and cannot show.</b> With one event, the time to the event is the calendar date "
            "counted backwards. A validation block lies between training blocks, so a model can score well by "
            "placing a day in the calendar from anything that changes slowly (season, noise level, the "
            "instrument), without any relation to the earthquake. A good score is therefore necessary for a "
            "precursor but does not show one; the pseudo-event placebo of section 9 asks how well the same "
            "models do for days on which nothing happened.", S["body"]),
        Spacer(1, 10),
    ]
    return story


# --------------------------------------------------------------------------- pseudo-event placebo

PLACEBO_SCORES = [("mae", "MAE(log)", False), ("gain", "Gain over level only", True),
                  ("rho_folds", "Spearman, mean of the folds", True), ("auc_month", "Final-month AUC", True)]
PLACEBO_COLOURS = {("raw_station", "lstm"): "#1f4e79", ("raw_station", "transformer"): "#56b4e9",
                   ("raw_day", "lstm"): "#d55e00", ("raw_day", "transformer"): "#e69f00"}


def load_placebo(runs, sub="placebo", real_day=EVENT_TIME.date()):
    """(table of placebo.table, settings of the runs) or None when there are no finished placebo runs."""
    d = Path(runs) / sub
    t = plc.table(d, real_day.isoformat()) if d.is_dir() else pd.DataFrame()
    if t.empty or not t["real"].any():
        return None
    info = [json.loads((f / "run.json").read_text()) for f in sorted(d.glob("e*_f?")) if (f / "DONE").exists()]
    for i in info:
        i.pop("split")
    return t, info


def rank_of_real(g, score, higher):
    """(value of the real event, values of the pseudo-events, rank of the real event, events with the score)"""
    v = g.loc[g[score].notna(), ["real", score]]
    real = float(v.loc[v["real"], score].iloc[0])
    pseudo = v.loc[~v["real"], score].to_numpy()
    return real, pseudo, 1 + int((pseudo >= real).sum() if higher else (pseudo <= real).sum()), len(v)


def placebo_cell(g, score, higher):
    real, pseudo, rank, n = rank_of_real(g, score, higher)
    return P(minus(f"<b>{real:.2f}</b> · {np.median(pseudo):.2f} ({pseudo.min():.2f} to {pseudo.max():.2f}) · "
                   f"<b>{rank} of {n}</b>"))


def fig_placebo(t, real_day=EVENT_TIME.date()):
    """Every score against the event day, one line per input and model; the real event is the last point."""
    fig, axes = plt.subplots(1, len(PLACEBO_SCORES), figsize=(11.2, 2.9), squeeze=False)
    real_day = pd.Timestamp(real_day)
    for ax, (score, title, _) in zip(axes[0], PLACEBO_SCORES):
        for (key, arch), g in t.groupby(["input", "arch"], sort=False):
            g = g.sort_values("event")
            x = pd.to_datetime(g["event"])
            c = PLACEBO_COLOURS[(key, arch)]
            ax.plot(x, g[score], color=c, lw=0.9, marker="o", ms=2.5,
                    label=f"{dict((k, n) for k, n, _ in INPUTS)[key]} · {dict(ARCHS)[arch]}")
            ax.plot(x[g["real"]], g.loc[g["real"], score], color=c, marker="o", ms=6, mec="#c0392b", mew=1.2, ls="")
        if score in ("gain", "rho_folds"):
            ax.axhline(0, color=MUTED, lw=0.6, ls="--")
        if score == "auc_month":
            ax.axhline(0.5, color=MUTED, lw=0.6, ls="--")
        ax.axvline(real_day, color="#c0392b", lw=0.6, ls=":")
        ax.set_title(title, fontsize=8, color=INK, pad=4, fontweight="bold")
        ax.set_xlabel("event day (red: the earthquake)", fontsize=7, color=INK)
        ax.tick_params(axis="x", labelrotation=45)
        style_axis(ax)
    axes[0, 0].legend(fontsize=5.5, frameon=False, loc="best")
    fig.tight_layout(w_pad=0.8)
    return fig


def placebo_story(pl, W, real_day=EVENT_TIME.date(), heading="9. Pseudo-event placebo", what=None, split=None,
                  cut_hour=EVENT_TIME.hour, reading=None, scaling=True):
    """Section of a pseudo-event placebo; `what` and `split` replace the two paragraphs written for Elazığ."""
    t, info = pl
    s0 = info[0]["settings"]
    events = sorted(t["event"].unique(), reverse=True)
    pseudo = [e for e in events if e != real_day.isoformat()]
    configs = [(k, a) for k, _, _ in INPUTS for a, _ in ARCHS if ((t["input"] == k) & (t["arch"] == a)).any()]
    names = dict((k, n) for k, n, _ in INPUTS)
    tr = [i["days"]["train"] for i in info]
    commits = sorted({i["commit"] for i in info})
    story = [PageBreak(), Paragraph(heading, S["h1"]),
             Paragraph(
                 what or f"<b>What was run.</b> The same training with the days counted to another day: the day of the "
                 f"earthquake and {len(pseudo)} pseudo-events, {BLOCK_DAYS}, {2 * BLOCK_DAYS}, … days before it "
                 f"({pseudo[-1]} to {pseudo[0]}), on which no earthquake happened. Every event gets only the "
                 f"{s0['history']} days up to it, so all events have the same length of history; the real event is "
                 f"run again under this limit and its scores here are not those of section 4. Raw signal only "
                 f"(the feature tables do not hold every day of 2019), stations {s0['stations'].replace(',', ', ')} "
                 f"(SVRC has no data from February to September 2019), both scalings, both models, {FOLDS} folds, "
                 f"one seed: {len(info)} runs with {min(tr)}–{max(tr)} training station-days each.", S["body"]),
             Spacer(1, 4),
             Paragraph(
                 split or f"<b>Split.</b> The blocks are the calendar blocks of the main runs, and their test blocks are not "
                 f"read here either. For every event the other blocks are dealt into the folds in the order of "
                 f"their age (the block that ends at the event to fold 0, the next to fold 1, …), so the blocks "
                 f"closest to an event are validated by different folds and each has training days on its far "
                 f"side. This differs from the split of sections 4–8.", S["body"]),
             Spacer(1, 4),
             Paragraph(
                 f"<b>Scores per event</b> (out-of-fold predictions of its validation days): MAE of log(1 + days); "
                 f"the gain, i.e. the MAE of the level-only baseline minus the model's MAE; the Spearman "
                 f"correlation per fold, averaged; and the final-month AUC, the share of pairs of a day 0–"
                 f"{plc.MONTH} days before the event and an earlier day in which the closer day gets the shorter "
                 f"prediction (0.5 = chance). Rank 1 is the best of the events. With {len(events)} events the "
                 f"smallest share a rank can give is 1/{len(events)} = {1 / len(events):.2f}; the events share most "
                 f"of their days and are not independent.", S["body"]),
             Spacer(1, 6)]
    rows = [["Input", "Model"] + [P(f"{title}<br/>real · pseudo median (range) · rank of real", "head")
                                  for _, title, _ in PLACEBO_SCORES]]
    for key in dict.fromkeys(k for k, _ in configs):
        for arch in [a for k, a in configs if k == key]:
            g = t[(t["input"] == key) & (t["arch"] == arch)]
            rows.append([P(f"<b>{names[key]}</b>"), dict(ARCHS)[arch]] + [placebo_cell(g, s, h) for s, _, h in PLACEBO_SCORES])
        g = t[(t["input"] == key) & (t["arch"] == [a for k, a in configs if k == key][0])]
        rows.append([P(f"<b>{names[key]}</b>"), "level only", placebo_cell(g, "mae_level", False), "–", "–",
                     placebo_cell(g, "auc_month_level", True)])
    story += [KeepTogether([P("<b>The real event among the pseudo-events</b>", "cell"), Spacer(1, 3),
                            table(rows, [W * .14, W * .1] + [W * .19] * 4, zebra=False)]),
              *placebo_notes(t, events, W, reading, scaling)[0],
              PageBreak(), fig_to_image(fig_placebo(t, real_day), W / cm * .97), Spacer(1, 8)]

    g0 = t[(t["input"] == configs[0][0]) & (t["arch"] == configs[0][1])].set_index("event")
    rows = [["Event day", "Days before the earthquake", "Validation station-days", f"of them 0–{plc.MONTH} days",
             "Level only<br/>MAE · AUC"] + [P(f"{names[k]} · {dict(ARCHS)[a]}<br/>MAE · AUC", "head") for k, a in configs]]
    rows[0] = [c if isinstance(c, Paragraph) else P(c, "head") for c in rows[0]]

    def two(a, b):
        return minus(f"{a:.2f} · " + (f"{b:.2f}" if np.isfinite(b) else "–"))

    for e in events:
        cells = []
        for k, a in configs:
            r = t[(t["input"] == k) & (t["arch"] == a) & (t["event"] == e)]
            cells.append(two(r["mae"].iloc[0], r["auc_month"].iloc[0]) if len(r) else "–")
        r0 = g0.loc[e]
        rows.append([P(f"<b>{e}</b>" + (" (earthquake)" if e == real_day.isoformat() else "")),
                     str((real_day - dt.date.fromisoformat(e)).days), str(int(r0["days"])), str(int(r0["month_days"])),
                     two(r0["mae_level"], r0["auc_month_level"])] + cells)
    short = [e for e in events if g0.loc[e, "month_days"] < g0["month_days"].max()]
    story += [P("<b>Every event</b>: MAE(log) · final-month AUC", "cell"), Spacer(1, 3),
              table(rows, [W * .15, W * .09, W * .09, W * .09, W * .12] + [W * .46 / len(configs)] * len(configs)),
              Spacer(1, 5),
              Paragraph(
                  "Notes. The station-days are those of the first row's input; both scalings use the same days. "
                  "The events do not have the same validation days relative to themselves (columns 3 and 4), so "
                  "the MAE of two events is not taken over the same range of times to the event. "
                  + (f"Events with fewer validation days in their last {plc.MONTH} days ({', '.join(short)}) have a "
                     f"test block there, which is not read. " if short else "")
                  + f"The recording of the day of the earthquake is cut at {cut_hour}:00 UTC, before the mainshock; the last "
                  "day of a pseudo-event is a whole day. The epoch that is scored is chosen on the validation days, "
                  "for every event in the same way. Commit recorded by the placebo runs: " + "; ".join(commits) + ".",
                  S["small"])]
    story += placebo_notes(t, events, W, reading, scaling)[1]
    return story


def placebo_notes(t, events, W, reading, scaling):
    real, pseudo = t[t["real"]], t[~t["real"]]
    g0 = t[(t["input"] == t["input"].iloc[0]) & (t["arch"] == t["arch"].iloc[0])]
    lv_real, lv_pseudo = g0[g0["real"]]["auc_month_level"].iloc[0], g0[~g0["real"]]["auc_month_level"].median()
    near = [f"{span_text(t[t['event'] == e]['rho_folds'])} for {e}" for e in events[1:3]]
    out = note("how to read the table above", [
        f"A cell is: real · pseudo median (range) · rank. Real is the score with the countdown aimed at the "
        f"earthquake, pseudo the same score for the {len(events) - 1} dates without one. Rank 1 is the best; 1 of "
        f"{len(events)} is the most a rank can say.",
        "MAE: lower is better. Gain above 0: better than loudness alone. Spearman of the folds: are the days put in "
        "the right order (1 perfect, 0 none); the score to trust most. Final-month AUC: 0.5 is chance.",
        "The level only rows are loudness alone: a straight line on the level of the day."], W)
    after = note("what the tables of this section show", [
        f"Spearman of the folds for the earthquake: {span_text(real['rho_folds'])} over the four setups; the highest of "
        f"any pseudo-event: {minus(f'{pseudo.rho_folds.max():.2f}')}. For the two dates closest to the earthquake: "
        + "; ".join(near) + ".",
        f"Loudness alone, final-month AUC: {minus(f'{lv_real:.2f}')} for the earthquake, median "
        f"{minus(f'{lv_pseudo:.2f}')} for the pseudo-events.",
        "Table of every event: one line per target date, cells are MAE · final-month AUC. Compare a "
        "cell with the level only cell of its line, not with other lines: the lines differ in their validation "
        f"days. A low MAE on a line with few days in the last {plc.MONTH} is a narrow range of targets, not a "
        "good prediction."] + ([reading] if reading else []), W)
    if scaling:
        after += note("station scale and day scale", SCALING_NOTE, W)
    return out, after


WINTER_ORDER = ["2015-16", "2018-19", "2019-20"]
QUAKE_WINTER = "2019-20"


def load_winters(runs):
    d = Path(runs) / "winters"
    if not d.is_dir():
        return None
    tr, di = wnt.transfer_table(d), wnt.direct_table(d)
    if tr.empty:
        return None
    info = [json.loads((f / "run.json").read_text()) for f in sorted(d.glob("transfer_*")) if (f / "DONE").exists()]
    for i in info:
        i.pop("split")
    return tr, di, info


def winter_name(w):
    return f"{w} (earthquake)" if w == QUAKE_WINTER else w


def winters_story(wn, W):
    tr, di, info = wn
    names = dict((k, n) for k, n, _ in INPUTS)
    configs = [(k, a) for k, _, _ in INPUTS for a, _ in ARCHS if ((tr["input"] == k) & (tr["arch"] == a)).any()]
    s0 = info[0]["settings"]
    pairs = sorted({(a, b) for a, b in zip(tr["trained_on"], tr["predicts"])},
                   key=lambda x: (x[0].count("+"), x[1] == QUAKE_WINTER, x[0], x[1]))
    days = tr.drop_duplicates("predicts").set_index("predicts")
    story = [PageBreak(), Paragraph("10. SVRC: winters with and without the earthquake", S["h1"]),
             Paragraph(
                 f"<b>Why.</b> In section 9 the day of the earthquake and mid-winter coincide. SVRC is the one "
                 f"station with continuous recordings in other winters: {WINTER_ORDER[0]} and {WINTER_ORDER[1]}, "
                 f"which did not end with an earthquake, and {QUAKE_WINTER}, which did.", S["body"]),
             Spacer(1, 4),
             Paragraph(
                 f"<b>Transfer between winters.</b> A model is trained on the {s0['history']} days up to 24 January "
                 f"of one winter (or of two) and predicts the days of another winter, every day counted to the 24 "
                 f"January of its own winter. Nothing of the predicted winter is used in training, and the last "
                 f"epoch is scored. If the time to the earthquake is read from the season, a model that has only "
                 f"seen a winter without an earthquake orders the days of the earthquake winter about as well as "
                 f"it orders those of another winter without one. Raw signal, both scalings, both models, "
                 f"{int(tr['seeds'].max())} seeds; scores are the mean ± standard deviation over the seeds. Days in "
                 f"test blocks or within {GAP_DAYS} days of one are not read. Usable days per winter: "
                 + "; ".join(f"{winter_name(w)} {int(days.loc[w, 'days'])} (of them {int(days.loc[w, 'month_days'])} in the "
                             f"last {plc.MONTH} days, closest {int(days.loc[w, 'nearest'])} days before 24 January)"
                             for w in WINTER_ORDER if w in days.index) + ".", S["body"]),
             Spacer(1, 6)]
    story += winters_note(tr, names)
    for score, title, level in (("rho", "Spearman correlation of predicted and true days to 24 January", "rho_level"),
                                ("auc_month", f"Final-month AUC (days 0–{plc.MONTH} against earlier days; 0.5 = chance)",
                                 "auc_month_level"),
                                ("mae", "MAE of log(1 + days)", "mae_level")):
        rows = [["Trained on", "Predicts", "Level only"] + [P(f"{names[k]} · {dict(ARCHS)[a]}", "head") for k, a in configs]]
        for a_, b_ in pairs:
            g = tr[(tr["trained_on"] == a_) & (tr["predicts"] == b_)]
            cells = []
            for k, a in configs:
                r = g[(g["input"] == k) & (g["arch"] == a)]
                cells.append(P(pm(r[score].iloc[0], r[f"{score}_sd"].iloc[0])) if len(r) and np.isfinite(r[score].iloc[0]) else "–")
            lv = g[level].iloc[0]
            rows.append([a_, P(f"<b>{winter_name(b_)}</b>"), minus(f"{lv:.2f}") if np.isfinite(lv) else "–"] + cells)
        story.append(KeepTogether([P(f"<b>{title}</b>", "cell"), Spacer(1, 3),
                                   table(rows, [W * .17, W * .17, W * .1] + [W * .14] * len(configs)), Spacer(1, 7)]))
    story.append(Paragraph(
        "Level only: a straight line on the day's level, fitted on the training winter and applied to the predicted "
        "one. A dash means that the predicted winter has fewer than five usable days in its last "
        f"{plc.MONTH} days. The winters differ in their days (above), in particular {WINTER_ORDER[1]} has almost no "
        "day close to 24 January, so its rows are scored on earlier days only; the recordings of the three winters "
        "may also differ for reasons that have nothing to do with season or earthquake (instrument, site).",
        S["small"]))
    thin = days.loc[WINTER_ORDER[1]]
    story += note("how to read the three tables", [
        "A line is one pair: trained on the winter(s) of column 1, predicting the winter of column 2. The first "
        "six lines train on one winter, the last three on two. Cells are mean ± standard deviation over the seeds.",
        "Spearman is the main table; the final-month AUC and the MAE tell the same story. Compare a cell with the "
        "level only cell of its line.",
        f"Lines predicting {WINTER_ORDER[1]}: no day closer than {int(thin['nearest'])} days to 24 January and "
        f"{int(thin['month_days'])} days in the last {plc.MONTH}. Their low MAE is a narrow range of targets, and "
        "their AUC is rough.",
        "Station scale and day scale: see the note in section 9."], W)
    if not di.empty:
        ev = sorted(di["event"].unique(), reverse=True)
        d0 = di.drop_duplicates("event").set_index("event")
        rows = [["Event day", "Validation days", f"of them 0–{plc.MONTH} days", "Level only<br/>MAE · AUC"]
                + [P(f"{names[k]} · {dict(ARCHS)[a]}<br/>MAE · Spearman (folds) · AUC", "head") for k, a in configs]]
        rows[0] = [c if isinstance(c, Paragraph) else P(c, "head") for c in rows[0]]
        for e in ev:
            cells = []
            for k, a in configs:
                r = di[(di["input"] == k) & (di["arch"] == a) & (di["event"] == e)]
                cells.append(minus(f"{r['mae'].iloc[0]:.2f} · {r['rho_folds'].iloc[0]:.2f} · "
                                   + (f"{r['auc_month'].iloc[0]:.2f}" if np.isfinite(r['auc_month'].iloc[0]) else "–"))
                             if len(r) else "–")
            lv = d0.loc[e]
            rows.append([P(f"<b>{e}</b>" + (" (earthquake)" if e == EVENT_TIME.date().isoformat() else "")),
                         str(int(lv["days"])), str(int(lv["month_days"])),
                         minus(f"{lv['mae_level']:.2f} · " + (f"{lv['auc_month_level']:.2f}" if np.isfinite(lv["auc_month_level"]) else "–"))]
                        + cells)
        story += [PageBreak(),
                  Paragraph(
                      f"<b>The placebo of section 9 at SVRC alone.</b> The countdown is trained towards the "
                      f"earthquake and towards a day in January of the two other winters ({', '.join(ev[1:])}; the "
                      f"closest days to 24 January that the recordings and the blocks allow), each with the 98 days "
                      f"before it, {FOLDS} folds by block age, {int(di['seeds'].max())} seeds. A fold has between about "
                      f"20 and 60 training days here, so these numbers are rough; the scores are the mean over the "
                      f"seeds.", S["body"]),
                  Spacer(1, 5),
                  table(rows, [W * .15, W * .08, W * .08, W * .11] + [W * .58 / len(configs)] * len(configs))]
        quake = EVENT_TIME.date().isoformat()
        best = [f"{names[k]} · {dict(ARCHS)[a]}" for k, a in configs
                if di[(di["input"] == k) & (di["arch"] == a)].set_index("event")["rho_folds"].idxmax() == quake]
        story += note("how to read this table", [
            "A line is one target date; a model cell is MAE · Spearman of the folds · final-month AUC.",
            f"The earthquake has the highest Spearman of the three dates in {len(best)} of {len(configs)} setups"
            + (f" ({', '.join(best)})" if best else "") + ".",
            f"Too little data to conclude from: about 20 to 60 training days per fold, and "
            f"{' and '.join(str(int(d0.loc[e, 'month_days'])) for e in ev[1:])} days in the last {plc.MONTH} for the "
            f"two other dates."], W)
    return story


def span_text(v):
    """'0.60' or '0.68 to 0.73' for the values v"""
    a, b = f"{np.min(v):.2f}", f"{np.max(v):.2f}"
    return minus(a if a == b else f"{a} to {b}")


def winter_groups(g):
    """Rows of single-winter training by direction: winters without the earthquake to each other, the earthquake
    winter to those, those to the earthquake winter."""
    g = g[~g["trained_on"].str.contains("\\+")]
    quake_in, quake_out = g["trained_on"] == QUAKE_WINTER, g["predicts"] == QUAKE_WINTER
    return g[~quake_in & ~quake_out], g[quake_in], g[quake_out]


def winters_note(tr, names):
    """Short reading of the transfer tables, for the configuration with the highest Spearman."""
    single = tr[~tr["trained_on"].str.contains("\\+")]
    k, a = single.groupby(["input", "arch"])["rho"].mean().idxmax()
    qq, eq, qe = winter_groups(tr[(tr["input"] == k) & (tr["arch"] == a)])
    quiet = pd.concat([qq, eq])
    day = single[single["input"].str.endswith("day")]
    lines = [
        f"From one winter without an earthquake to the other: Spearman {span_text(qq.rho)} (level only {span_text(qq.rho_level)}). A "
        f"countdown to 24 January is learned where no earthquake followed.",
        f"From the earthquake winter to the winters without: {span_text(eq.rho)}. What is learned from the earthquake winter works "
        f"equally where no earthquake follows.",
        f"From the winters without to the earthquake winter: {span_text(qe.rho)}, but level only already gives {span_text(qe.rho_level)}. "
        f"The model adds {span_text(qe.rho - qe.rho_level)} to level only here, {span_text(quiet.rho - quiet.rho_level)} in "
        f"the winters without an earthquake.",
        f"Order, not timing: for the last {plc.MONTH} days before the earthquake the median prediction is "
        f"{qe.month_pred.min():.0f} to {qe.month_pred.max():.0f} days."]
    if len(day):
        lines.append(f"Day scale (the day's level removed), both models: {span_text(day.rho)}. What transfers between "
                     f"winters is mainly the level of the day.")
    lines.append("Reading: the countdown at SVRC does not need an earthquake; season alone can produce it. Not ruled "
                 "out: a precursor underneath the seasonal trend. One station, three winters, "
                 f"{int(tr['days'].min())} to {int(tr['days'].max())} usable days each.")
    return ([Paragraph(f"<b>What this shows</b> ({names[k]} · {dict(ARCHS)[a]}, trained on one winter, mean of the "
                       f"seeds).", S["body"])]
            + [Paragraph("• " + x, S["body"]) for x in lines] + [Spacer(1, 6)])


def winters_bullet(wn):
    tr = wn[0]
    names = dict((k, n) for k, n, _ in INPUTS)
    day = tr["input"].str.endswith("day")
    parts = []
    for (k, a), g in tr[~day].groupby(["input", "arch"], sort=False):
        qq, eq, qe = winter_groups(g)
        parts.append(f"{dict(ARCHS)[a]} {span_text(qq.rho)} between the two winters without an earthquake, "
                     f"{span_text(eq.rho)} from the earthquake winter to those, {span_text(qe.rho)} from those to the "
                     f"earthquake winter")
    text = ("<b>SVRC winters (section 10).</b> With station scaling a countdown to 24 January transfers between "
            "winters whether or not they end with the earthquake. Spearman of models trained on one winter and "
            "predicting another: " + "; ".join(parts) + ".")
    if day.any():
        text += f" With day scaling: {span_text(pd.concat(winter_groups(tr[day])).rho)} over all pairs and both models."
    return text


def placebo_bullet(pl, label="Pseudo-event placebo (section 9)", what="Raw signal"):
    t, info = pl
    names = dict((k, n) for k, n, _ in INPUTS)
    parts = []
    for (key, arch), g in t.groupby(["input", "arch"], sort=False):
        r = {s: rank_of_real(g, s, h) for s, _, h in PLACEBO_SCORES}
        parts.append(f"{names[key]} {dict(ARCHS)[arch]}: MAE {r['mae'][2]}, gain {r['gain'][2]}, Spearman "
                     f"{r['rho_folds'][2]}, final-month AUC {r['auc_month'][2]} (of {r['auc_month'][3]})")
    n = t["event"].nunique()
    return (f"<b>{label}.</b> {what}, {info[0]['settings']['history']} days of history "
            f"per event, the earthquake and {n - 1} pseudo-events. Rank of the earthquake among the {n} events "
            f"(1 = best): " + "; ".join(parts) + ".")


# --------------------------------------------------------------------------- figures

def style_axis(ax):
    ax.tick_params(length=2, colors=MUTED, labelcolor=INK, labelsize=6)
    for sp in ax.spines.values():
        sp.set_color(GRIDC)


def fig_scatter(res):
    """Predicted against true days to the event (both logarithmic), one panel per input and model; predictions
    averaged over the seeds. Red: the final 30 days."""
    fig, axes = plt.subplots(len(ARCHS), len(INPUTS), figsize=(11.2, 5.4), squeeze=False, sharex=True, sharey=True)
    lim = (0.7, 2500)
    for i, (arch, at) in enumerate(ARCHS):
        for j, (key, kt, _) in enumerate(INPUTS):
            ax, d = axes[i, j], res[(key, arch)]["mean"]
            last = d["days_to_event"] <= 30
            ax.plot(lim, lim, color=MUTED, lw=0.6, ls="--")
            ax.scatter(d.loc[~last, "days_to_event"] + 1, d.loc[~last, "predicted_days"] + 1, s=3, color="#1f4e79",
                       alpha=0.35, lw=0)
            ax.scatter(d.loc[last, "days_to_event"] + 1, d.loc[last, "predicted_days"] + 1, s=4, color="#c0392b",
                       alpha=0.6, lw=0)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            style_axis(ax)
            if i == 0:
                ax.set_title(kt, fontsize=8, color=INK, pad=4, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"{at}\npredicted days + 1", fontsize=7, color=INK)
            if i == len(ARCHS) - 1:
                ax.set_xlabel("true days to the event + 1", fontsize=7, color=INK)
    fig.tight_layout(w_pad=0.6, h_pad=0.6)
    return fig


def fig_feature_blocks(res, groups):
    """Feature inputs, last 400 days: prediction against the true time, with the blocks of consecutive days shaded."""
    spans = {g: (v["days_to_event"].min(), v["days_to_event"].max()) for g, v in groups.groupby("group") if g in SHADE}
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 5.0), squeeze=False, sharex=True, sharey=True)
    for i, (arch, at) in enumerate(ARCHS):
        for j, (key, kt, _) in enumerate(INPUTS[:2]):
            ax, d = axes[i, j], res[(key, arch)]["mean"]
            d = d[d["days_to_event"] <= 400]
            for g, (a, b) in spans.items():
                ax.axvspan(a - 0.5, b + 0.5, color=SHADE[g], alpha=0.12, lw=0)
                if i == 0:
                    ax.text((a + b) / 2, 1500, {"final": "final month"}.get(g, g), ha="center", va="top", fontsize=6,
                            color=SHADE[g])
            ax.plot([0, 400], [1, 401], color=MUTED, lw=0.6, ls="--")
            ax.scatter(d["days_to_event"], d["predicted_days"] + 1, s=5, color="#1f4e79", alpha=0.55, lw=0)
            ax.set_yscale("log")
            ax.set_ylim(0.7, 2000)
            ax.set_xlim(405, -5)
            style_axis(ax)
            ax.set_title(f"{kt} · {at}", fontsize=8, color=INK, pad=4, fontweight="bold")
            if j == 0:
                ax.set_ylabel("predicted days + 1", fontsize=7, color=INK)
            if i == 1:
                ax.set_xlabel("true days to the event", fontsize=7, color=INK)
    fig.tight_layout(w_pad=0.6, h_pad=0.8)
    return fig


# --------------------------------------------------------------------------- report

def build(runs, data, out):
    res = load(runs)
    groups = feature_groups(data)
    seeds = sorted({s for r in res.values() for s in r["per_seed"]})
    folds = sorted({i["fold"] for r in res.values() for i in r["info"]})
    n_runs = sum(len(r["info"]) for r in res.values())
    sc = {k: seed_scores(r) for k, r in res.items()}
    base = {key: {c: scores(res[(key, "lstm")]["per_seed"][seeds[0]], c) for c in ("level_only_log", "train_mean_log")}
            for key, _, _ in INPUTS}
    info0 = res[("raw_station", "lstm")]["info"][0]
    commits = sorted({i["commit"] for r in res.values() for i in r["info"]})

    doc = SimpleDocTemplate(out, pagesize=landscape(A4), leftMargin=1.6 * cm, rightMargin=1.6 * cm,
                            topMargin=1.4 * cm, bottomMargin=1.4 * cm,
                            title="Days to the event from one full day – LSTM and transformer", author="cnn_lstm_reg")
    W = landscape(A4)[0] - 3.2 * cm

    story = [Paragraph("Days to the event from one full day – LSTM and transformer", S["title"]),
             Paragraph(f"Elazığ earthquake, 24 January 2020, 17:55 UTC (Mw 6.8) · stations {', '.join(STATIONS)} · "
                       f"N component · generated {dt.date.today():%d %B %Y}", S["body"]),
             Spacer(1, 8)]
    story += intro_story(data, info0, seeds, folds, n_runs, W)
    story += [
             PageBreak(),
             Paragraph("Results in short", S["h1"])]

    def cmp_text(key, i, digits=2):
        return " and ".join(f"{pm(sc[(key, a)][0][i], sc[(key, a)][1][i], digits)} ({t})" for a, t in ARCHS)

    gp = {}
    for key in ("features", "pca"):
        for arch, _ in ARCHS:
            d = res[(key, arch)]["mean"].merge(groups[["station", "date", "group"]], on=["station", "date"], how="left")
            gp[(key, arch)] = d

    def med(key, arch, lo, hi):
        d = res[(key, arch)]["mean"]
        return np.median(d.loc[d["days_to_event"].between(lo, hi), "predicted_days"])

    def gmed(key, arch, g):
        d = gp[(key, arch)]
        return np.median(d.loc[d["group"] == g, "predicted_days"])

    fold_rho = {k: r["mean"][r["mean"]["days_to_event"] <= 365].groupby("fold")
                .apply(lambda g: rho(g.y, g.predicted_log), include_groups=False) for k, r in res.items()}
    station_rho = {k: r["mean"][r["mean"]["days_to_event"] <= 365].groupby("station")
                   .apply(lambda g: rho(g.y, g.predicted_log), include_groups=False) for k, r in res.items()}
    closest = {f: min(i["closest_train"] for i in res[("features", "lstm")]["info"] if i["fold"] == f) for f in folds}
    gf = gp[("features", "lstm")]
    final_folds = sorted(gf.loc[gf["group"] == "final", "fold"].unique())
    final_fold = int(final_folds[0])
    if len(final_folds) != 1 or any(min(i["closest_train"] for i in r["info"] if i["fold"] == f) != closest[f]
                                    for r in res.values() for f in folds):
        sys.exit("the final month is validated by more than one fold, or the folds differ between the inputs: "
                 "the text of the report does not hold for these runs")
    n_final = int((gf["group"] == "final").sum())
    final_reach = closest[final_fold] - GAP_DAYS - 1
    n_summer = int((gf["group"] == "summer").sum())
    n_summer_same = int(((gf["group"] == "summer") & (gf["fold"] == final_fold)).sum())
    raw_keys = [(k, a) for k in ("raw_station", "raw_day") for a, _ in ARCHS]
    others = [station_rho[k][s] for k in raw_keys for s in STATIONS if s != "FRT"]
    bullets = [
        "<b>Feature inputs, last year.</b> MAE(log): features "
        + cmp_text("features", 2) + ", pca " + cmp_text("pca", 2)
        + f"; level-only baseline {base['features']['level_only_log'][2]:.2f}. Spearman: features "
        + cmp_text("features", 3) + ", pca " + cmp_text("pca", 3) + ".",
        "<b>Raw signal, last year.</b> MAE(log): station scale " + cmp_text("raw_station", 2) + ", day scale "
        + cmp_text("raw_day", 2) + f"; level-only baseline {base['raw_station']['level_only_log'][2]:.2f}. "
        "Spearman: station scale " + cmp_text("raw_station", 3) + ", day scale " + cmp_text("raw_day", 3) + ".",
        "<b>The final month.</b> Median prediction for the days 0–30 before the event, in days: "
        + "; ".join(f"{kt} {med(key, 'lstm', 0, 30):.0f} (LSTM) and {med(key, 'transformer', 0, 30):.0f} "
                    f"(transformer)" for key, kt, _ in INPUTS) + ".",
        "<b>Transformer on the feature inputs, by kind of day.</b> Median prediction in days, features and pca: "
        + "; ".join(f"{name} {gmed('features', 'transformer', g):.0f} and {gmed('pca', 'transformer', g):.0f}"
                    for g, name in [("final", "final month"), ("summer", "summer placebo block (178–207 days "
                                                                         "before)"),
                                    ("spring", "spring placebo block"), ("winter", "winter placebo block")])
        + f". Within the final month the days 0–14 and 15–30 get {med('features', 'transformer', 0, 14):.0f} and "
          f"{med('features', 'transformer', 15, 30):.0f} days (features), "
          f"{med('pca', 'transformer', 0, 14):.0f} and {med('pca', 'transformer', 15, 30):.0f} days (pca).",
        f"<b>Where the final month is validated.</b> All {n_final} validation station-days of the final month are in "
        f"fold {final_fold}, together with the other days up to {final_reach} days before the event; the closest "
        f"training day of that fold is {closest[final_fold]} days before the event (folds "
        + ", ".join(f"{f}: {c}" for f, c in closest.items() if f != final_fold)
        + f" days). The models that predict the final month therefore had no training day from it. "
          f"{n_summer_same} of the {n_summer} validation station-days of the summer placebo block are in the same fold.",
        "<b>Folds.</b> Last-year Spearman of the single folds (predictions averaged over the seeds): "
        + "; ".join(f"{kt} {at} " + minus(f"{fold_rho[(key, a)].min():.2f} to {fold_rho[(key, a)].max():.2f}")
                    for key, kt, _ in INPUTS[:2] for a, at in ARCHS) + ".",
        "<b>Stations, raw signal.</b> Last-year Spearman at FRT: "
        + minus(f"{min(station_rho[k]['FRT'] for k in raw_keys):.2f}–{max(station_rho[k]['FRT'] for k in raw_keys):.2f}")
        + " in the four raw runs; at the other stations "
        + minus(f"{min(others):.2f}–{max(others):.2f}") + ".",
    ]
    pl = load_placebo(runs)
    if pl is not None:
        bullets.append(placebo_bullet(pl))
    wn = load_winters(runs)
    if wn is not None:
        bullets.append(winters_bullet(wn))
    bk_time, _, bk_stations = EVENTS["balikesir_2025"]
    bk = load_placebo(runs, "placebo_balikesir", bk_time.date())
    if bk is not None:
        bullets.append(placebo_bullet(bk, "Pseudo-event placebo at Balıkesir (section 11)",
                                      f"Mainshock of {bk_time:%d %B %Y}, raw signal"))
    for b in bullets:
        story += [Paragraph(b, S["body"]), Spacer(1, 4)]
    story.append(Paragraph(
        "With one event the days to the event are the calendar date counted backwards, and a validation block "
        "lies between training blocks, so a score here measures how well a day can be placed in the calendar "
        "from its recording; it does not by itself show a relation to the earthquake. The feature inputs and the "
        "raw inputs are scored on different days (the feature tables hold fewer days), so their rows are not "
        "comparable with each other.", S["small"]))

    # ---- 1 scores
    story += [PageBreak(), Paragraph("4. Scores", S["h1"]),
              Paragraph(f"Out-of-fold predictions of the validation days; mean ± standard deviation over "
                        f"{len(seeds)} seeds. The baselines do not depend on the seed.", S["small"]), Spacer(1, 4)]
    for title, i_mae, i_rho, count in [("All validation days", 0, 1, lambda d: len(d)),
                                       ("Validation days of the last year (0–365 days before the event)", 2, 3,
                                        lambda d: int((d["days_to_event"] <= 365).sum()))]:
        rows = [["Input", "Station-days", "MAE(log) train mean", "MAE(log) level only", "MAE(log) LSTM",
                 "MAE(log) transformer", "Spearman level only", "Spearman LSTM", "Spearman transformer"]]
        for key, kt, _ in INPUTS:
            b = base[key]
            rows.append([P(f"<b>{kt}</b>"), str(count(res[(key, "lstm")]["mean"])), f"{b['train_mean_log'][i_mae]:.2f}",
                         f"{b['level_only_log'][i_mae]:.2f}"]
                        + [P(f"<b>{pm(sc[(key, a)][0][i_mae], sc[(key, a)][1][i_mae])}</b>") for a, _ in ARCHS]
                        + [minus(f"{b['level_only_log'][i_rho]:.2f}")]
                        + [P(f"<b>{pm(sc[(key, a)][0][i_rho], sc[(key, a)][1][i_rho])}</b>") for a, _ in ARCHS])
        story.append(KeepTogether([P(f"<b>{title}</b>", "cell"), Spacer(1, 3),
                                   table(rows, [W * .16, W * .09] + [W * .107] * 7), Spacer(1, 9)]))

    rows = [["Input", "Model"] + STATIONS]
    for key, kt, _ in INPUTS:
        n = res[(key, "lstm")]["mean"]
        n = n[n["days_to_event"] <= 365].groupby("station").size()
        for a, at in ARCHS:
            rows.append([P(f"<b>{kt}</b>"), at] + [minus(f"{station_rho[(key, a)].get(s, np.nan):.2f}") for s in STATIONS])
        lv = res[(key, "lstm")]["mean"]
        lv = lv[lv["days_to_event"] <= 365].groupby("station").apply(lambda g: rho(g.y, g.level_only_log),
                                                                     include_groups=False)
        rows.append([P(f"<b>{kt}</b>"), P("level only · <font color='#555555'>station-days</font>")]
                    + [P(minus(f"{lv.get(s, np.nan):.2f}") + f" · <font color='#555555'>{n.get(s, 0)}</font>")
                       for s in STATIONS])
    story.append(KeepTogether([P("<b>Spearman per station, last year</b> (predictions averaged over the seeds)", "cell"),
                               Spacer(1, 3), table(rows, [W * .18, W * .2] + [W * .11] * 5, zebra=False)]))

    # ---- 2 predictions
    story += [PageBreak(), Paragraph("5. Predicted against true time", S["h1"]),
              Paragraph("Every point is one validation station-day, predicted by the models that did not train on "
                        "its block and averaged over the seeds. Both axes are logarithmic (days + 1); the dashed "
                        "line is a correct prediction. Red: the final 30 days.", S["small"]),
              Spacer(1, 4), fig_to_image(fig_scatter(res), W / cm * .93), PageBreak()]
    rows = [["Input", "Model"] + [f"{a}–{b} days" if b < 10**6 else f"more than {a - 1} days" for a, b in RANGES]]
    for key, kt, _ in INPUTS:
        d0 = res[(key, "lstm")]["mean"]
        for a, at in ARCHS:
            d = res[(key, a)]["mean"]
            rows.append([P(f"<b>{kt}</b>"), at] + [P(spread(d.loc[d["days_to_event"].between(lo, hi), "predicted_days"]))
                                                   for lo, hi in RANGES])
        rows.append([P(f"<b>{kt}</b>"), P("<font color='#555555'>station-days</font>")]
                    + [P(f"<font color='#555555'>{int(d0['days_to_event'].between(lo, hi).sum())}</font>")
                       for lo, hi in RANGES])
    story += [P("<b>Predicted days per range of the true time</b>: median (10th–90th percentile)", "cell"),
              Spacer(1, 3), table(rows, [W * .16, W * .12] + [W * .12] * len(RANGES), zebra=False)]

    # ---- 3 feature days by kind
    counts = groups.groupby(["group", "station"]).size().unstack(fill_value=0)
    span = groups[groups["group"].isin(SHADE)].groupby("group")["days_to_event"].agg(["min", "max"])
    story += [PageBreak(), Paragraph("6. Feature inputs: predictions by kind of day", S["h1"]),
              Paragraph(
                  "The feature tables do not hold every day. They hold every day of the final month, every 2nd day "
                  "of days 31–90 and every 9th day of days 91–365 (the days the sfc run sampled), single days of "
                  "the earlier years, and every day of the three 30-day placebo blocks ("
                  + ", ".join(f"{g} {int(span.loc[g, 'min'])}–{int(span.loc[g, 'max'])} days before the event"
                              for g in ("summer", "spring", "winter")) + "; no placebo blocks at SVRC). "
                  "Below: the validation days of each kind and what the models predict for them. Days in test "
                  "blocks or in the gaps between blocks are not included.", S["small"]),
              Spacer(1, 4)]
    rows = [["Kind of day", "Validation station-days", "True days to the event", "Fold: station-days"]
            + [P(f"{kt} · {at}<br/>predicted days", "head") for key, kt, _ in INPUTS[:2] for a, at in ARCHS]]
    for g, name in GROUPS:
        d0 = gp[("features", "lstm")]
        d0 = d0[d0["group"] == g]
        rows.append([P(f"<b>{name}</b>"), str(len(d0)),
                     f"{int(d0['days_to_event'].min())}–{int(d0['days_to_event'].max())}" if len(d0) else "–",
                     ", ".join(f"{f}: {n}" for f, n in d0.groupby("fold").size().items())]
                    + [P(spread(gp[(key, a)].loc[gp[(key, a)]["group"] == g, "predicted_days"]))
                       for key, _, _ in INPUTS[:2] for a, _ in ARCHS])
    story += [P("<b>Predicted days per kind of day</b>: median (10th–90th percentile), predictions averaged over "
                "the seeds", "cell"), Spacer(1, 3),
              table(rows, [W * .27, W * .08, W * .08, W * .13] + [W * .11] * 4), Spacer(1, 4),
              Paragraph(f"The closest training day of fold {final_fold} is {closest[final_fold]} days before the "
                        f"event, so log(1 + days) of its training days is never below "
                        f"{np.log1p(closest[final_fold]):.2f}; the transformer's predictions for the final month "
                        f"and for the summer block lie below that floor.", S["small"]), Spacer(1, 4),
              fig_to_image(fig_feature_blocks(res, groups), W / cm * .76)]

    # ---- 4 folds
    story += [PageBreak(), Paragraph("7. The single folds", S["h1"]),
              Paragraph("Last year only. Per fold: Spearman of the predictions averaged over the seeds, and below it "
                        "the lowest and highest value of the single seeds. A fold's validation days are the days "
                        "of its blocks, so the folds differ in how many days they hold and where in the year "
                        "these lie. Closest training day, in days before the event: "
                        + ", ".join(f"fold {f}: {c}" for f, c in closest.items()) + ".", S["small"]), Spacer(1, 4)]
    rows = [["Input", "Model"] + [f"Fold {f}" for f in folds]]
    for key, kt, _ in INPUTS:
        for a, at in ARCHS:
            r = res[(key, a)]
            cells = []
            for f in folds:
                per = [rho(x[(x.fold == f) & (x.days_to_event <= 365)].y,
                           x[(x.fold == f) & (x.days_to_event <= 365)].predicted_log) for x in r["per_seed"].values()]
                cells.append(P(minus(f"<b>{fold_rho[(key, a)].get(f, np.nan):.2f}</b><br/>"
                                     f"<font color='#555555'>{np.nanmin(per):.2f} to {np.nanmax(per):.2f}</font>")))
            rows.append([P(f"<b>{kt}</b>"), at] + cells)
        d = res[(key, "lstm")]["mean"]
        d = d[d["days_to_event"] <= 365]
        rows.append([P(f"<b>{kt}</b>"), P("<font color='#555555'>station-days · of them in the final month</font>")]
                    + [P(f"<font color='#555555'>{int((d.fold == f).sum())} · "
                         f"{int(((d.fold == f) & (d.days_to_event <= 30)).sum())}</font>") for f in folds])
    story.append(table(rows, [W * .18, W * .26] + [W * .14] * len(folds), zebra=False))

    # ---- 5 runs
    story += [PageBreak(), Paragraph("8. The runs", S["h1"])]
    rows = [["Input", "Model", "Parameters", "Learning rate (peak)", "Weight decay", "Training station-days per fold",
             "Epoch with the lowest validation error"]]
    for key, kt, _ in INPUTS:
        for a, at in ARCHS:
            info = res[(key, a)]["info"]
            tr = [i["days"]["train"] for i in info]
            be = [i["best_epoch"] for i in info]
            rows.append([P(f"<b>{kt}</b>"), at, f"{info[0]['parameters']:,}".replace(",", " "),
                         f"{info[0]['settings']['lr']:g}", f"{info[0]['settings']['weight_decay']:g}",
                         f"{min(tr)}–{max(tr)}", f"median {int(np.median(be))} ({min(be)}–{max(be)})"])
    story += [table(rows, [W * .16, W * .12, W * .11, W * .13, W * .11, W * .17, W * .2]), Spacer(1, 6),
              Paragraph(
                  f"Run on {info0['device']}, torch {info0['versions']['torch']}, python "
                  f"{info0['versions']['python']}. Commit recorded by the runs: {'; '.join(commits)}. The code was "
                  "copied to the server before it was committed; its files were compared afterwards and are "
                  "those of commit 25620fb. The epoch that is scored is chosen on the validation days themselves, "
                  "which favours every model a little and both models in the same way.", S["small"])]
    if pl is not None:
        story += placebo_story(pl, W, reading=(
            "Reading: with station scale the earthquake scores best, but the dates two and four weeks before it "
            "score almost as well, and loudness alone already separates its final month. The models follow "
            "something that changes slowly through December and January, not the day of the earthquake."))
    if wn is not None:
        story += winters_story(wn, W)
    if bk is not None:
        b0 = bk[1][0]["settings"]
        n_seeds = len({i["settings"]["seed"] for i in bk[1]})
        n_pseudo = bk[0]["event"].nunique() - 1
        tr_days = [i["days"]["train"] for i in bk[1]]
        story += placebo_story(
            bk, W, bk_time.date(), "11. Pseudo-event placebo at a second earthquake: Balıkesir 2025",
            what=(f"<b>What was run.</b> The placebo of section 9 at the Sındırgı (Balıkesir) mainshock of "
                  f"{bk_time:%d %B %Y}, {bk_time:%H:%M} UTC (Mw 6.1), which did not happen in winter: the day of the "
                  f"earthquake and {n_pseudo} pseudo-events {BLOCK_DAYS}, {2 * BLOCK_DAYS}, … days before it. The "
                  f"recordings of the stations {', '.join(bk_stations)} start in February 2025, so every event gets "
                  f"only the {b0['history']} days up to it (182 in section 9) and no earlier pseudo-event is "
                  f"possible. Raw signal, both scalings, both models, {FOLDS} folds, {n_seeds} seeds: {len(bk[1])} runs "
                  f"with {min(tr_days)}–{max(tr_days)} training station-days each; the predictions of the seeds are "
                  f"averaged day by day before scoring. The M6.1 of 27 October 2025 is not used as a second "
                  f"event: its history contains the August mainshock and its aftershocks."),
            split=(f"<b>Split.</b> {BLOCK_DAYS}-day blocks counted back from the mainshock; "
                   f"{100 * TEST_SHARE:.0f} % of the 13 blocks are test blocks and were not read. For every event "
                   f"the other blocks are dealt into the folds in the order of their age, as in section 9."),
            cut_hour=bk_time.hour, scaling=False,
            reading=("Reading: no countdown to this earthquake. Where it ranks first on Spearman the value lies "
                     "inside what the pseudo-events reach, and on MAE it is among the worst events."
                     + (f" At Elazığ the Spearman of the earthquake was {span_text(pl[0][pl[0]['real']]['rho_folds'])}."
                        if pl is not None else "")
                     + f" Low power: {min(tr_days)} to {max(tr_days)} training station-days per run. Station scale "
                       "and day scale: see the note in section 9."))

    def footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Noto", 7)
        canvas.setFillColor(colors.HexColor("#777777"))
        canvas.drawString(1.6 * cm, 0.8 * cm, "Days to the event from one full day · Elazığ 2020 · LSTM and transformer")
        canvas.drawRightString(landscape(A4)[0] - 1.6 * cm, 0.8 * cm, f"page {doc_.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    build(*sys.argv[1:])
    print(sys.argv[3])
