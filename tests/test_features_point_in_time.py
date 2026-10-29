"""Leakage tests for the Phase 2 feature builder."""
import numpy as np
import pandas as pd

import _synthetic
from finguard.features.build import ENTITY_FEATURES, FORBIDDEN, FeatureBuilder, entity_features

DAY = 86_400


def _tiny():
    # one proxy entity (card1=1000, addr1=300, first-seen day 10) with four transactions,
    # plus a row whose D1 is missing (no entity)
    return pd.DataFrame({
        "TransactionDT": [10 * DAY, 10 * DAY + 30, 10 * DAY + 3600, 18 * DAY, 11 * DAY],
        "TransactionAmt": [10.0, 20.0, 60.0, 30.0, 5.0],
        "card1": [1000, 1000, 1000, 1000, 1000],
        "addr1": [300.0, 300.0, 300.0, 300.0, 300.0],
        "D1": [0.0, 0.0, 0.0, 8.0, np.nan],
    })


def test_entity_features_hand_computed():
    df = _tiny()
    f = entity_features(df, email=pd.Series(["a.com", "a.com", "b.com", None, "a.com"]))
    assert f.entity_prev_txn.tolist()[:4] == [0, 1, 2, 3]
    assert np.isnan(f.entity_secs_since_prev[0]) and f.entity_secs_since_prev[1] == 30
    assert f.entity_txn_24h.tolist()[:4] == [0, 1, 2, 0]            # last one is 8 days later
    assert f.entity_txn_7d[3] == 0
    assert abs(f.entity_amt_ratio[2] - 60 / 15) < 1e-5              # 60 / mean(10, 20)
    assert f.entity_n_emails.tolist()[:4] == [1, 1, 2, 2]           # distinct so far, NaN not counted
    assert abs(f.entity_age_days[3] - 8.0) < 1e-6
    assert f.iloc[4].isna().all()                                   # missing D1 -> no entity, never guessed


def test_future_rows_never_change_earlier_entity_features():
    df = _synthetic.frame(4000, seed=1)
    cut = df["TransactionDT"].quantile(0.6)
    past = df[df.TransactionDT <= cut]
    full = entity_features(df).loc[past.index]
    only_past = entity_features(past)
    pd.testing.assert_frame_equal(full, only_past, check_exact=True)


def test_file_order_does_not_matter_when_times_are_distinct():
    df = _synthetic.frame(2000, seed=2).drop_duplicates("TransactionDT")
    a = entity_features(df)
    shuffled = df.sample(frac=1.0, random_state=0)
    b = entity_features(shuffled).loc[a.index]
    pd.testing.assert_frame_equal(a, b)


def test_builder_statistics_come_from_the_training_window_only():
    df = _synthetic.frame(3000, days=60, seed=3)
    fb1 = FeatureBuilder(fit_days=(0, 39)).fit(df)
    later = df["day"] > 39
    df2 = df.copy()
    df2.loc[later, "card1"] = 9999.0                                 # change everything after the window
    df2.loc[later, "P_emaildomain"] = "protonmail.com"
    df2.loc[later, "TransactionAmt"] = 1e6
    df2.loc[later, "V1"] = 1.0
    fb2 = FeatureBuilder(fit_days=(0, 39)).fit(df2)
    assert fb1.vocab_ == fb2.vocab_
    assert all(fb1.freq_[k].equals(fb2.freq_[k]) for k in fb1.freq_)
    assert np.array_equal(fb1.amt_edges_, fb2.amt_edges_)
    assert fb1.blocks_ == fb2.blocks_ and fb1.patterns_ == fb2.patterns_


def test_training_rows_unchanged_by_later_rows_end_to_end():
    df = _synthetic.frame(3000, days=60, seed=4)
    fb = FeatureBuilder(fit_days=(0, 39)).fit(df)
    full = fb.transform(df)
    early = df[df["day"] <= 39]
    pd.testing.assert_frame_equal(full.loc[early.index], fb.transform(early), check_exact=True)


def test_labels_are_never_read():
    df = _synthetic.frame(2000, seed=5)
    a = FeatureBuilder(fit_days=(0, 39)).fit_transform(df)
    df2 = df.copy()
    df2["isFraud"] = 1 - df2["isFraud"]
    b = FeatureBuilder(fit_days=(0, 39)).fit_transform(df2)
    pd.testing.assert_frame_equal(a, b)


def test_forbidden_columns_and_names():
    df = _synthetic.frame(1500, seed=6)
    fb = FeatureBuilder(fit_days=(0, 39))
    X = fb.fit_transform(df)
    assert not (FORBIDDEN & set(X.columns))
    assert not any("customer" in c for c in X.columns)               # proxy entity, never "customer"
    assert set(ENTITY_FEATURES) <= set(X.columns)
    assert "amt_decile" not in fb.tree_columns and "amt_decile" in fb.columns_
    for c in ("D1", "D15", "id_31", "DeviceInfo", "id_07"):          # replaced or dropped raw columns
        assert c not in X.columns
    assert {"D1n", "D15n", "browser_family", "device_brand", "screen_w"} <= set(X.columns)


def test_unseen_categories_go_to_other_and_nan_stays_nan():
    df = _synthetic.frame(2000, seed=7)
    fb = FeatureBuilder(fit_days=(0, 39)).fit(df)
    later = df[df["day"] > 39].copy()
    later.loc[later.index[:5], "P_emaildomain"] = "never-seen.example"
    later.loc[later.index[5:10], "P_emaildomain"] = np.nan
    X = fb.transform(later)
    other = len(fb.vocab_["P_emaildomain"])
    assert (X["P_emaildomain"].iloc[:5] == other).all()
    assert X["P_emaildomain"].iloc[5:10].isna().all()
    assert (X["freq_email"].iloc[:5] == 0).all()


def test_scoring_one_transaction_against_its_history_matches_the_batch_value():
    # "Same values online and offline": an entity store holds every earlier transaction, so
    # scoring transaction i with only that history must reproduce the batch feature row.
    df = _synthetic.frame(3000, seed=10).drop_duplicates("TransactionDT")   # "earlier" is unambiguous
    email = df["P_emaildomain"]
    batch = entity_features(df, email=email)
    rs = np.random.RandomState(0)
    for i in rs.choice(df.index, 25, replace=False):
        history = df[df.TransactionDT < df.at[i, "TransactionDT"]]
        one = pd.concat([history, df.loc[[i]]])
        online = entity_features(one, email=email.loc[one.index]).loc[i]
        pd.testing.assert_series_equal(online, batch.loc[i], check_names=False)
