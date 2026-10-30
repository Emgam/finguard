"""Entity store (synthetic) and scorer (integration, skipped when the Phase 2 artifacts are absent)."""
from pathlib import Path
from unittest import SkipTest

import numpy as np
import pandas as pd

import _synthetic
from finguard import config
from finguard.serving.entity_store import EntityStore

DAY = 86_400


def _table():
    t0 = 10 * DAY
    return pd.DataFrame({
        "TransactionID": [1, 2, 3, 4, 5],
        "TransactionDT": [t0, t0 + 60, t0 + 60, t0 + 3 * DAY, t0 + 100],
        "TransactionAmt": [10.0, 20.0, 30.0, 40.0, 5.0],
        "card1": [1000.0] * 5, "addr1": [300.0, 300.0, 300.0, 300.0, np.nan],
        "D1": [0.0, 0.0, 0.0, 3.0, 0.0],
        "DeviceInfo": ["iOS Device", "SM-G892A", None, None, None],
        "P_emaildomain": ["gmail.com", "gmail.com", "yahoo.com", None, None],
    })


def test_history_uses_earlier_rows_and_file_order_for_ties():
    store = EntityStore(_table())
    t = store.table
    assert store.history(t.iloc[[0]]).TransactionID.tolist() == []
    assert store.history(t.iloc[[2]]).TransactionID.tolist() == [1, 2]          # same time, earlier in file
    assert store.history(t.iloc[[1]]).TransactionID.tolist() == [1]
    assert store.history(t.iloc[[3]]).TransactionID.tolist() == [1, 2, 3]
    assert len(store.history(t.iloc[[4]])) == 0                                  # no key: never a guessed entity


def test_summary_counts_and_labels_the_proxy_entity():
    store = EntityStore(_table())
    s = store.summary(store.table.iloc[[3]])
    assert s["available"] and s["earlier_transactions"] == 3 and s["earlier_in_24h"] == 0
    assert s["device_brands_seen"] == ["apple ios", "samsung"] and "not a customer" in s["entity"]
    assert abs(s["amount_vs_earlier_mean"] - 2.0) < 1e-9
    assert store.summary(store.table.iloc[[4]])["available"] is False


def test_history_on_synthetic_data_is_strictly_earlier():
    df = _synthetic.frame(3000, seed=21).reset_index(drop=True)
    store = EntityStore(df)
    for i in np.random.RandomState(0).choice(len(df), 40, replace=False):
        h = store.history(df.iloc[[i]])
        assert (h[config.TIME] <= df.loc[i, config.TIME]).all() and i not in h.index


def test_scorer_on_cached_features():
    art = Path(config.ROOT / "artifacts" / "phase2" / "model_bundle.pkl")
    cache = Path(config.ROOT / "data" / "processed" / "main_numeric" / "X.npy")
    if not (art.exists() and cache.exists()):
        raise SkipTest("Phase 2 artifacts or feature cache not present")
    from finguard.serving.scorer import DISCLAIMER, Scorer
    sc = Scorer()
    ids = [3543502, 3550979, 3548165]
    out = sc.score_cached(ids)
    assert [o["transaction_id"] for o in out] == ids
    for o in out:
        assert 1 <= o["score"] <= 99 and o["band"] in ("High", "Medium", "Low") and DISCLAIMER in o["notes"]
        assert 0 < len(o["reasons"]) <= 5 and all("customer" not in r["text"].lower() for r in o["reasons"])
        assert all(r["signal_group"] for r in o["reasons"] if r["anonymized"])
        assert o["probability_displayed"] not in ("100%", "0%")
