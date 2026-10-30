"""Novelty flags: hand-computed values, causality, and no guessed entities."""
import numpy as np
import pandas as pd

import _synthetic
from finguard.features.novelty import novelty_flags

DAY = 86_400


def _tiny():
    t0 = 10 * DAY
    df = pd.DataFrame({
        "TransactionDT": [t0, t0 + 100, t0 + 200, t0 + 300, t0 + 400, 15 * DAY, 12 * DAY],
        "TransactionAmt": [10.0, 20.0, 15.0, 100.0, 12.0, 50.0, 999.0],
        "card1": [1000.0] * 7,
        "addr1": [300.0] * 6 + [np.nan],                              # last row: no entity key
        "D1": [0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 2.0],
        "DeviceInfo": [None] * 7,
    })
    device = pd.Series([None, "apple", "apple", "samsung", None, "huawei", "apple"], dtype=object)
    return df, device


def test_flags_hand_computed():
    df, device = _tiny()
    f = novelty_flags(df, device=device)
    assert f.flag_burst_first_day.tolist()[:6] == [0, 0, 0, 1, 1, 0]        # 4th and 5th txn within 24 h
    assert f.flag_amount_above_max.tolist()[:6] == [0, 0, 0, 1, 0, 0]       # 100 > 3 x 20
    assert f.flag_new_device.tolist()[:6] == [0, 0, 0, 1, 0, 1]             # samsung, then huawei
    assert np.isnan(f.entity_past_max_amt[0]) and f.entity_past_max_amt[4] == 100
    assert f.iloc[6].isna().all()                                           # no key -> NaN, never a guess


def test_future_rows_never_change_earlier_flags():
    df = _synthetic.frame(4000, seed=11)
    cut = df["TransactionDT"].quantile(0.5)
    past = df[df.TransactionDT <= cut]
    pd.testing.assert_frame_equal(novelty_flags(df).loc[past.index], novelty_flags(past), check_exact=True)


def test_labels_are_never_read():
    df = _synthetic.frame(2000, seed=12)
    flipped = df.assign(isFraud=1 - df["isFraud"])
    pd.testing.assert_frame_equal(novelty_flags(df), novelty_flags(flipped))
