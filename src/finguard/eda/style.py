"""One visual system for every Phase 1 figure.

Grey first: data is drawn in a quiet grey and ONE accent (blue) marks what the title is
about. A second hue (orange) is used only when two series differ in kind. Colours come
from a validated palette (categorical slots 1-2 pass CVD and contrast checks on the light
surface). Titles state the finding; a source line sits under every figure.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from finguard import config

ACCENT = "#2a78d6"      # categorical slot 1 (blue)
SECOND = "#eb6834"      # categorical slot 2 (orange)
GREY = "#b4b2aa"        # quiet data grey
GREY_DARK = "#898781"   # muted ink
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]  # blue ramp

SOURCE = "Source: IEEE-CIS train_transaction + train_identity (590,540 transactions), FinGuard Phase 1 EDA"


def apply() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 9.5, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.titlecolor": INK, "axes.labelcolor": INK2, "axes.labelsize": 9.5,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": GREY_DARK, "ytick.color": GREY_DARK, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.7, "lines.linewidth": 2.0,
        "legend.frameon": False, "legend.fontsize": 8.5, "figure.dpi": 110,
    })


def new(width: float = 10, height: float = 5.2, ncols: int = 1, nrows: int = 1, **kw):
    apply()
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), **kw)
    return fig, axes


def headline(fig, title: str, subtitle: str = "", source: str = SOURCE, top: float = 0.86) -> None:
    """Finding as the title, method as the subtitle, provenance at the bottom."""
    budget = fig.get_figwidth() * 8.4          # characters that fit at 13 pt bold
    size = 13 if len(title) <= budget else max(10.5, 13 * budget / len(title))
    fig.text(0.012, 0.975, title, ha="left", va="top", fontsize=size, fontweight="bold", color=INK)
    if subtitle:
        fig.text(0.012, 0.925, subtitle, ha="left", va="top", fontsize=9.5, color=INK2)
    fig.text(0.012, 0.012, source, ha="left", va="bottom", fontsize=7.5, color=GREY_DARK)
    fig.subplots_adjust(top=top, bottom=0.12, right=0.92)


def grid(ax, axis: str = "x") -> None:
    ax.grid(True, axis=axis, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def ref_line(ax, value: float, label: str, orient: str = "v", **kw) -> None:
    style = dict(color=GREY_DARK, linestyle=(0, (4, 3)), linewidth=1.0, zorder=1)
    style.update(kw)
    if orient == "v":
        ax.axvline(value, **style)
        ax.text(value, 1.0, f" {label}", transform=ax.get_xaxis_transform(), ha="left", va="bottom",
                fontsize=8, color=INK2)
    else:
        ax.axhline(value, **style)
        ax.text(1.008, value, label, transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=8, color=INK2, clip_on=False)


def pct(v: float, digits: int = 1) -> str:
    return f"{v * 100:.{digits}f}%"


def save(fig, name: str, fig_dir: Path | None = None) -> Path:
    fig_dir = Path(fig_dir or config.FIG_DIR)
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / f"{name}.png"
    fig.savefig(path, dpi=160)
    return path


def wilson(k: np.ndarray, n: np.ndarray, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = np.where(n > 0, k / np.maximum(n, 1), 0)
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return np.clip(centre - half, 0, 1), np.clip(centre + half, 0, 1)


def rate_table(key: pd.Series, y: pd.Series, min_n: int = 0, missing_label: str = "absent") -> pd.DataFrame:
    """Fraud rate per category with a Wilson 95% interval."""
    k = key.astype("string").fillna(missing_label)
    t = pd.DataFrame({"k": k, "y": y.to_numpy()}).groupby("k").y.agg(n="size", frauds="sum")
    t = t[t.n >= min_n].copy()
    t["rate"] = t.frauds / t.n
    t["lo"], t["hi"] = wilson(t.frauds.to_numpy(), t.n.to_numpy())
    return t
