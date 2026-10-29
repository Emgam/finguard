"""Honest single-feature signal strength.

* Numeric columns: rank AUC of the raw value with NaN ranked lowest (what a tree can see
  with one split family), plus the same AUC on the first and second half of the period to
  expose instability.
* Categorical columns and numeric codes: target encoding fitted on the EARLIER 60% of the
  period and scored on the LATER 40%, so the number is not inflated by in-sample fitting.

Strength = max(AUC, 1 - AUC): 0.5 is no signal, 1.0 is perfect separation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from finguard import config
from finguard.data.schema import NUMERIC_CODE_COLS, feature_group


def _rank_auc(x: np.ndarray, y: np.ndarray) -> float:
    x = np.where(np.isnan(x), -1e12, x)
    return roc_auc_score(y, x)


def univariate_strength(df: pd.DataFrame, cols: list[str], early_share: float = 0.6, prior_weight: float = 20.0) -> pd.DataFrame:
    y = df[config.TARGET].to_numpy()
    t = df[config.TIME].to_numpy()
    early = t <= np.quantile(t, early_share)
    first_half = t <= np.median(t)
    base = y[early].mean()
    rows = []
    for c in cols:
        if not pd.api.types.is_numeric_dtype(df[c]) or c in NUMERIC_CODE_COLS:
            s = df[c].astype("string").fillna("<missing>")
            stats = pd.DataFrame({"k": s[early].to_numpy(), "y": y[early]}).groupby("k").y.agg(["sum", "count"])
            enc = (stats["sum"] + base * prior_weight) / (stats["count"] + prior_weight)
            late = s[~early].map(enc).astype("float64").fillna(base).to_numpy()
            auc = roc_auc_score(y[~early], late)
            rows.append({"column": c, "group": feature_group(c), "kind": "categorical (fit early, score late)",
                         "auc": auc, "auc_first_half": np.nan, "auc_second_half": np.nan, "n_unique": int(df[c].nunique())})
        else:
            x = df[c].to_numpy(dtype="float64")
            rows.append({"column": c, "group": feature_group(c), "kind": "numeric (rank, NaN lowest)",
                         "auc": _rank_auc(x, y), "auc_first_half": _rank_auc(x[first_half], y[first_half]),
                         "auc_second_half": _rank_auc(x[~first_half], y[~first_half]), "n_unique": int(df[c].nunique())})
    out = pd.DataFrame(rows)
    out["strength"] = np.maximum(out.auc, 1 - out.auc)
    out["half_shift"] = (out.auc_first_half - out.auc_second_half).abs()
    return out.sort_values("strength", ascending=False).reset_index(drop=True)


def redundant_within_blocks(df: pd.DataFrame, blocks: pd.DataFrame, threshold: float = 0.95,
                            sample: int = 150_000, seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """Greedy de-duplication inside each missing-value block: a column is redundant if
    |Pearson r| >= threshold with an earlier kept column of the same block."""
    smp = df.sample(min(sample, len(df)), random_state=seed)
    rows = []
    for _, b in blocks.iterrows():
        cols = [c for c in b["columns"] if pd.api.types.is_numeric_dtype(df[c])]
        sub = smp[cols].dropna()
        if len(cols) < 2 or len(sub) < 1000:
            continue
        r = np.nan_to_num(np.abs(np.corrcoef(sub.to_numpy(dtype="float64").T)))
        kept: list[int] = []
        for i, c in enumerate(cols):
            match = [j for j in kept if r[i, j] >= threshold]
            rows.append({"column": c, "block": b.block, "redundant": bool(match),
                         "duplicate_of": cols[match[0]] if match else None})
            if not match:
                kept.append(i)
    return pd.DataFrame(rows)
