"""Proxy entity for IEEE-CIS.

IMPORTANT: IEEE-CIS HAS NO CUSTOMER, ACCOUNT OR CARDHOLDER IDENTIFIER.

The proxy entity groups transactions that share card1, addr1 and the same "first-seen"
day (transaction day minus D1). The construction comes from top solutions of the 2019
Kaggle competition. It is a heuristic:

* it is NOT a customer: one person may map to several entities and, rarely, one entity
  may mix people;
* its meaning rests on the host's group-level description of D-columns ("timedelta");
  the exact meaning of D1 is undisclosed;
* every analysis built on it must be called "proxy-entity analysis", never
  "customer profiling", in code, figures, reports and the UI.

Phase 1 only defines the entity and measures its properties. Point-in-time entity
features (velocity, amount ratios) are specified in the feature-engineering plan and
built in Phase 2, from earlier transactions only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from finguard import config

ENTITY_COL = "proxy_entity"


def build_proxy_entity(df: pd.DataFrame) -> pd.Series:
    """Return the proxy-entity key, or <NA> when card1, addr1 or D1 is missing."""
    day_abs = np.floor(df[config.TIME].to_numpy(dtype="float64") / config.SECONDS_PER_DAY)
    first_seen = day_abs - df["D1"].to_numpy(dtype="float64")
    parts = pd.DataFrame({
        "c": df["card1"].astype("Int64"),
        "a": df["addr1"].astype("Int64"),
        "f": pd.array(np.where(np.isnan(first_seen), np.nan, first_seen), dtype="Float64").astype("Int64"),
    }, index=df.index)
    key = parts["c"].astype("string") + "_" + parts["a"].astype("string") + "_" + parts["f"].astype("string")
    return key.rename(ENTITY_COL)


def coverage(entity: pd.Series) -> dict:
    """Coverage with explicit denominators (all rows vs rows that have an entity)."""
    vc = entity.value_counts()            # NA keys are excluded
    multi_rows = int(vc[vc >= 2].sum())
    n_with = int(entity.notna().sum())
    return {
        "rows_with_entity": n_with / len(entity),
        "n_entities": int(len(vc)),
        "n_single_transaction_entities": int((vc == 1).sum()),
        "n_multi_transaction_entities": int((vc >= 2).sum()),
        "share_of_all_rows_in_multi_entities": multi_rows / len(entity),
        "share_of_entity_rows_in_multi_entities": multi_rows / max(n_with, 1),
        "median_size_of_multi_entities": float(vc[vc >= 2].median()),
    }


def label_purity(entity: pd.Series, y: pd.Series) -> pd.DataFrame:
    """Per entity with >= 2 transactions: size, frauds, fraud share."""
    d = pd.DataFrame({"e": entity, "y": y}).dropna(subset=["e"])
    g = d.groupby("e").y.agg(size="size", frauds="sum")
    g = g[g["size"] >= 2]
    g["fraud_share"] = g["frauds"] / g["size"]
    return g


def split_overlap(entity: pd.Series, y: pd.Series, train_mask: np.ndarray, test_mask: np.ndarray) -> dict:
    """How many test frauds belong to an entity already labelled fraud in the training part.

    A high share means a model can 'recognise' entities instead of learning fraud patterns,
    which inflates offline metrics (the label is propagated within linked transactions).
    """
    e_tr, e_te = entity[train_mask], entity[test_mask]
    y_tr, y_te = y[train_mask], y[test_mask]
    known = set(e_tr.dropna())
    known_fraud = set(e_tr[y_tr.astype(bool)].dropna())
    seen = e_te.isin(known).to_numpy() & e_te.notna().to_numpy()
    seen_fraud = e_te.isin(known_fraud).to_numpy()
    yt = y_te.to_numpy().astype(bool)
    return {
        "test_rows_entity_seen_in_train": float(seen.mean()),
        "test_frauds_entity_had_fraud_in_train": float(seen_fraud[yt].mean()),
        "fraud_rate_if_entity_had_fraud": float(yt[seen_fraud].mean()) if seen_fraud.any() else float("nan"),
        "fraud_rate_otherwise": float(yt[~seen_fraud].mean()),
    }
