import numpy as np
import pandas as pd

from finguard.data import ingest, schema


def _txn(ids, dt=None):
    return pd.DataFrame({"TransactionID": ids, "TransactionDT": dt if dt is not None else np.arange(len(ids)) * 3600.0 + 86400,
                         "TransactionAmt": np.ones(len(ids))})


def test_schema_column_counts():
    assert len(schema.TRAIN_TXN_COLS) == 394
    assert len(schema.TEST_TXN_COLS) == 393
    assert len(schema.IDENTITY_COLS) == 41


def test_normalize_identity_columns_renames_only_hyphenated_ids():
    df = pd.DataFrame(columns=["TransactionID", "id-01", "id-38", "DeviceType"])
    out, renamed = ingest.normalize_identity_columns(df)
    assert list(out.columns) == ["TransactionID", "id_01", "id_38", "DeviceType"]
    assert renamed == ["id-01", "id-38"]


def test_join_preserves_rows_and_flags_identity():
    t = _txn([1, 2, 3])
    i = pd.DataFrame({"TransactionID": [2], "id_01": [0.5]})
    out = ingest.join_transaction_identity(t, i)
    assert len(out) == 3
    assert out.has_identity.tolist() == [0, 1, 0]
    assert out.id_01.isna().sum() == 2


def test_join_rejects_duplicate_keys():
    t = _txn([1, 1, 2])
    i = pd.DataFrame({"TransactionID": [2]})
    try:
        ingest.join_transaction_identity(t, i)
    except ValueError:
        return
    raise AssertionError("duplicate TransactionID was not rejected")


def test_join_rejects_orphan_identity_rows():
    t = _txn([1, 2])
    i = pd.DataFrame({"TransactionID": [3]})
    try:
        ingest.join_transaction_identity(t, i)
    except ValueError:
        return
    raise AssertionError("orphan identity row was not rejected")


def test_add_time_index_uses_training_origin():
    df = pd.DataFrame({"TransactionDT": [86400.0, 86400.0 + 3600 * 25, 86400.0 + 86400 * 3]})
    out = ingest.add_time_index(df.copy(), origin=86400.0)
    assert out.day.tolist() == [0, 1, 3]
    assert out.rel_hour.tolist() == [0, 1, 0]
