"""Time-based splits for Phase 2.

Random splits are not used anywhere in FinGuard. Phase 1 showed that with a random split
47.5% of test frauds belong to a proxy entity already labelled fraud in training (18.5%
with the time split), so random-split metrics measure entity recognition, not detection.

Schemes use the relative day index of the labelled file (`day`, 0-based):

* ``main``             train 0-120, validation 121-151, test 152-181
* ``label_delay_30d``  train 0-90 (the 30 days before validation are left out, as if their
                       chargeback labels had not arrived yet), same validation and test.

Validation is used for early stopping, model selection, calibration and band thresholds.
The test month is touched once, for the final numbers.
"""
from __future__ import annotations

import numpy as np

from finguard import config

SCHEMES: dict[str, dict[str, tuple[int, int]]] = {
    "main": dict(config.SPLIT_DAYS),
    "label_delay_30d": {"train": (0, 90), "valid": config.SPLIT_DAYS["valid"], "test": config.SPLIT_DAYS["test"]},
}
PARTS = ("train", "valid", "test")


def masks(day: np.ndarray, scheme: str = "main") -> dict[str, np.ndarray]:
    """Boolean masks for train / valid / test. Rows outside every window are in none."""
    spec = SCHEMES[scheme]
    day = np.asarray(day)
    return {p: (day >= spec[p][0]) & (day <= spec[p][1]) for p in PARTS}


def assign(day: np.ndarray, scheme: str = "main") -> np.ndarray:
    """Split label per row: 'train', 'valid', 'test' or 'unused' (e.g. the label-delay gap)."""
    out = np.full(len(day), "unused", dtype=object)
    for p, m in masks(day, scheme).items():
        out[m] = p
    return out


def check(day: np.ndarray, scheme: str = "main") -> dict:
    """Verify the split is time-ordered and disjoint; return a summary. Raises on violation."""
    m = masks(day, scheme)
    day = np.asarray(day)
    overlap = (m["train"] & m["valid"]) | (m["train"] & m["test"]) | (m["valid"] & m["test"])
    if overlap.any():
        raise ValueError("split windows overlap")
    for p in PARTS:
        if not m[p].any():
            raise ValueError(f"split part '{p}' is empty")
    if not (day[m["train"]].max() < day[m["valid"]].min() <= day[m["valid"]].max() < day[m["test"]].min()):
        raise ValueError("split is not time-ordered: train < valid < test must hold")
    return {p: {"rows": int(m[p].sum()), "first_day": int(day[m[p]].min()), "last_day": int(day[m[p]].max())}
            for p in PARTS} | {"unused_rows": int((~(m["train"] | m["valid"] | m["test"])).sum())}
