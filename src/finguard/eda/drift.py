"""Train-vs-later-period drift measures."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from finguard import config
from finguard.data.schema import feature_group


def adversarial_auc_per_feature(train: pd.DataFrame, later: pd.DataFrame, cols: list[str],
                                sample: int = 300_000, seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """How well each single feature separates training rows from later rows.

    0.5 = no drift; 1.0 = the feature alone tells the periods apart. Text columns are
    frequency-encoded on the pooled sample. NaN is ranked lowest, so a change in the
    missing rate also shows up as drift.
    """
    rng = np.random.default_rng(seed)
    half = sample // 2
    a = train[cols].iloc[rng.choice(len(train), min(half, len(train)), replace=False)]
    b = later[cols].iloc[rng.choice(len(later), min(half, len(later)), replace=False)]
    both = pd.concat([a, b], ignore_index=True)
    lab = np.r_[np.zeros(len(a)), np.ones(len(b))]
    rows = []
    for c in cols:
        col = both[c]
        if pd.api.types.is_numeric_dtype(col):
            f = col.to_numpy(dtype="float64")
        else:
            f = col.map(col.value_counts(dropna=False)).to_numpy(dtype="float64")
        f = np.where(np.isnan(f), -1e12, f)
        auc = roc_auc_score(lab, f)
        rows.append({"column": c, "group": feature_group(c), "adversarial_auc": max(auc, 1 - auc)})
    return pd.DataFrame(rows).sort_values("adversarial_auc", ascending=False).reset_index(drop=True)


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Population stability index with quantile bins fitted on `expected`; NaN is its own bin."""
    e, a = expected.astype("float64"), actual.astype("float64")
    edges = np.unique(np.nanquantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return float("nan")
    def dist(x):
        cut = pd.cut(x, edges, include_lowest=True)
        p = cut.value_counts(normalize=False, sort=False).to_numpy(dtype="float64")
        p = np.append(p, x.isna().sum())
        return (p + 1) / (p.sum() + len(p))
    pe, pa = dist(e), dist(a)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def new_category_share(train: pd.Series, later: pd.Series) -> float:
    """Share of non-missing later values never seen in training."""
    seen = set(train.dropna().unique())
    later = later.dropna()
    return float((~later.isin(seen)).mean()) if len(later) else float("nan")
