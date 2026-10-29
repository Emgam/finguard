import numpy as np
import pandas as pd

from finguard.features import proxy_entity as P


def _df():
    day = 86400.0
    return pd.DataFrame({
        "TransactionDT": [10 * day, 12 * day, 12 * day + 5, 15 * day, 15 * day],
        "card1": [1000, 1000, 1000, 2000, 2000],
        "addr1": [300.0, 300.0, 300.0, np.nan, 315.0],
        "D1": [0.0, 2.0, 2.0, 0.0, np.nan],
        "isFraud": [1, 1, 1, 0, 0],
    })


def test_proxy_entity_key_and_missing_parts():
    e = P.build_proxy_entity(_df())
    # same card1, addr1 and first-seen day (day - D1 = 10) -> same entity
    assert e.iloc[0] == e.iloc[1] == e.iloc[2] == "1000_300_10"
    # missing addr1 or D1 -> no entity (never guessed)
    assert pd.isna(e.iloc[3]) and pd.isna(e.iloc[4])


def test_label_purity_counts_only_multi_transaction_entities():
    df = _df()
    e = P.build_proxy_entity(df)
    pur = P.label_purity(e, df.isFraud)
    assert len(pur) == 1
    assert pur.iloc[0]["size"] == 3 and pur.iloc[0]["fraud_share"] == 1.0


def test_split_overlap_detects_known_fraud_entities():
    e = pd.Series(["a", "a", "b", "c", "a", "c"])
    y = pd.Series([1, 1, 0, 0, 1, 0])
    tr = np.array([True, True, True, True, False, False])
    out = P.split_overlap(e, y, tr, ~tr)
    assert out["test_frauds_entity_had_fraud_in_train"] == 1.0
    assert out["test_rows_entity_seen_in_train"] == 1.0
