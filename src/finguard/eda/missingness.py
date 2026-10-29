"""Missing-value analysis: per column, per block of columns that go missing together, per row."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from finguard import config
from finguard.data.schema import feature_group

BUCKETS = [(-1e-9, 0.0, "0%"), (0.0, 0.10, "(0-10%]"), (0.10, 0.50, "(10-50%]"), (0.50, 0.75, "(50-75%]"),
           (0.75, 0.90, "(75-90%]"), (0.90, 0.99, "(90-99%]"), (0.99, 1.0, ">99%")]


def feature_columns(df: pd.DataFrame) -> list[str]:
    drop = {config.KEY, config.TARGET, "has_identity", "day", "rel_hour", "rel_weekday", "proxy_entity"}
    return [c for c in df.columns if c not in drop]


def missing_by_column(train: pd.DataFrame, test: pd.DataFrame | None = None) -> pd.DataFrame:
    cols = feature_columns(train)
    out = pd.DataFrame({"group": [feature_group(c) for c in cols], "missing_train": train[cols].isna().mean().values}, index=cols)
    if test is not None:
        out["missing_test"] = test[cols].isna().mean().values
        out["missing_shift"] = out.missing_test - out.missing_train
    return out


def bucket_counts(missing: pd.Series) -> pd.Series:
    labels = []
    for v in missing:
        for lo, hi, lab in BUCKETS:
            if lo < v <= hi or (v == 0 and lab == "0%"):
                labels.append(lab)
                break
    return pd.Series(labels).value_counts().reindex([b[2] for b in BUCKETS], fill_value=0)


def nan_blocks(df: pd.DataFrame, cols: list[str] | None = None, min_size: int = 2) -> pd.DataFrame:
    """Group columns whose missing-value masks are identical.

    Returns one row per block: block id, member columns, size, missing share and the fraud
    rate when the block is present vs missing.
    """
    cols = cols or feature_columns(df)
    y = df[config.TARGET].to_numpy() if config.TARGET in df else None
    masks: dict[int, list[str]] = {}
    for c in cols:
        m = df[c].isna().to_numpy()
        if 0 < m.mean() < 1:
            masks.setdefault(hash(np.packbits(m).tobytes()), []).append(c)
    rows = []
    for members in masks.values():
        if len(members) < min_size:
            continue
        m = df[members[0]].isna().to_numpy()
        row = {"columns": members, "n_cols": len(members), "first": members[0], "last": members[-1],
               "groups": sorted({feature_group(c) for c in members}), "missing_share": float(m.mean())}
        if y is not None:
            row["fraud_rate_present"] = float(y[~m].mean())
            row["fraud_rate_missing"] = float(y[m].mean()) if m.any() else np.nan
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("n_cols", ascending=False).reset_index(drop=True)
    out.insert(0, "block", [f"B{i + 1:02d}" for i in range(len(out))])
    out["label"] = out.apply(lambda r: f"{r.block} {r['first']}..{r['last']} ({r.n_cols})", axis=1)
    return out


def informative_missingness(df: pd.DataFrame, lo: float = 0.005, hi: float = 0.995) -> pd.DataFrame:
    """Fraud rate when a column is missing vs present, and the AUC of the missing indicator alone."""
    y = df[config.TARGET].to_numpy()
    rows = []
    for c in feature_columns(df):
        m = df[c].isna().to_numpy()
        if lo < m.mean() < hi:
            rows.append({"column": c, "group": feature_group(c), "missing_share": m.mean(),
                         "fraud_if_missing": y[m].mean(), "fraud_if_present": y[~m].mean(),
                         "auc_present_indicator": roc_auc_score(y, ~m)})
    return pd.DataFrame(rows).sort_values("auc_present_indicator", ascending=False).reset_index(drop=True)


def block_missing_by(df: pd.DataFrame, blocks: pd.DataFrame, by: str) -> pd.DataFrame:
    """Missing share of each block (rows) per category of `by` (columns)."""
    out = {}
    for _, b in blocks.iterrows():
        out[b.label] = df[b["first"]].isna().groupby(df[by]).mean()
    return pd.DataFrame(out).T
