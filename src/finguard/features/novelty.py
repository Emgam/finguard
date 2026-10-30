"""Explainable novelty flags on the proxy-entity history (Project Report v3.0, section 6.3).

Three rules an analyst can check by eye, computed from EARLIER transactions of the same
proxy entity only (same ordering as the entity features: time, then file order). No label
is read. Rows without an entity key (card1, addr1 or D1 missing) get NaN, never a guess.

* ``flag_new_device``       the device brand is new for this proxy entity, which had already
                            used at least one other known device brand;
* ``flag_burst_first_day``  this is at least the 4th transaction within 24 h of the proxy
                            entity's first transaction in the data (``BURST_K`` earlier ones);
* ``flag_amount_above_max`` the amount is more than ``AMOUNT_MULTIPLE`` times the largest
                            earlier amount of the proxy entity (at least 2 earlier transactions).

The thresholds are fixed in advance, not tuned on outcomes. The flags are shown to the
analyst; they enter the risk score only if a combiner fitted on the validation month
improves the test month (measured in ``finguard.phase2b``).

IEEE-CIS has no customer identifier: "entity" always means the heuristic proxy entity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from finguard import config
from finguard.features import normalize as N
from finguard.features.proxy_entity import build_proxy_entity

BURST_K = 3
AMOUNT_MULTIPLE = 3.0
MIN_HISTORY_FOR_AMOUNT = 2
FLAGS = ["flag_new_device", "flag_burst_first_day", "flag_amount_above_max"]
FLAG_TEXT = {
    "flag_new_device": "New device brand for this proxy entity (another brand was used before)",
    "flag_burst_first_day": f"Burst: at least {BURST_K + 1} transactions within 24 h of the proxy entity's first transaction",
    "flag_amount_above_max": f"Amount more than {AMOUNT_MULTIPLE:g}x the proxy entity's largest earlier amount",
}


def novelty_flags(df: pd.DataFrame, device: pd.Series | None = None) -> pd.DataFrame:
    """Flags plus the context an analyst needs to read them, aligned to ``df.index``."""
    n = len(df)
    key = build_proxy_entity(df)
    codes = pd.factorize(key, use_na_sentinel=True)[0].astype("int64")
    t = df[config.TIME].to_numpy(dtype="float64")
    amt = df["TransactionAmt"].to_numpy(dtype="float64")
    if device is None:
        device = N.device_brand(df["DeviceInfo"])
    order = np.lexsort((np.arange(n), t, codes))
    c, ts, a = codes[order], t[order], amt[order]
    pos = np.arange(n)
    start = np.r_[True, c[1:] != c[:-1]]
    gstart = np.maximum.accumulate(np.where(start, pos, 0))
    prev = pos - gstart
    age_days = (ts - ts[gstart]) / config.SECONDS_PER_DAY

    cummax = pd.Series(a).groupby(c, sort=False).cummax().to_numpy()
    past_max = np.r_[np.nan, cummax[:-1]]
    past_max[start] = np.nan

    v = pd.Series(np.asarray(device, dtype=object), dtype="object").iloc[order].reset_index(drop=True)
    has = v.notna().to_numpy()
    first_occ = (~pd.DataFrame({"c": c, "v": v}).duplicated(["c", "v"])).to_numpy() & has
    hc = np.cumsum(has)
    earlier_known = (hc - has) - (hc[gstart] - has[gstart])          # known devices strictly before, same entity

    out = {
        "flag_new_device": (has & first_occ & (earlier_known > 0)).astype("float32"),
        "flag_burst_first_day": ((age_days < 1.0) & (prev >= BURST_K)).astype("float32"),
        "flag_amount_above_max": ((prev >= MIN_HISTORY_FOR_AMOUNT)
                                  & (a > AMOUNT_MULTIPLE * np.nan_to_num(past_max, nan=np.inf))).astype("float32"),
        "entity_past_max_amt": past_max.astype("float32"),
    }
    no_key = c < 0
    res = np.empty((n, len(out)), dtype="float32")
    for j, col in enumerate(out.values()):
        col = col.copy()
        col[no_key] = np.nan
        res[order, j] = col
    return pd.DataFrame(res, index=df.index, columns=list(out))
