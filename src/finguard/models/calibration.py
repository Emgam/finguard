"""Isotonic calibration and risk bands, both fitted on the validation month only.

* Calibration maps the raw score to a probability that matches observed fraud rates.
  Isotonic regression is monotone, so it never changes the ranking (it can create ties).
* Bands translate an alert budget into thresholds: High = top 1% of validation scores,
  Medium = the next 4%, Low = the rest. Thresholds are on the raw score (rank-equivalent
  to the calibrated probability, without isotonic plateaus), then frozen and applied to
  the test month. If the score distribution drifts, the share of test traffic in each
  band moves away from 1% / 4%; that share is reported as an alert-volume check.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from finguard.eda.style import wilson

BAND_ORDER = ["High", "Medium", "Low"]


def fit_isotonic(raw_valid: np.ndarray, y_valid: np.ndarray) -> IsotonicRegression:
    return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw_valid, y_valid)


@dataclass(frozen=True)
class Bands:
    high: float                   # raw-score threshold for High
    medium: float                 # raw-score threshold for Medium
    high_share: float = 0.01
    medium_share: float = 0.04

    def assign(self, raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw)
        return np.where(raw >= self.high, "High", np.where(raw >= self.medium, "Medium", "Low"))


def fit_bands(raw_valid: np.ndarray, high_share: float = 0.01, medium_share: float = 0.04) -> Bands:
    q_high = np.quantile(raw_valid, 1 - high_share)
    q_med = np.quantile(raw_valid, 1 - high_share - medium_share)
    return Bands(high=float(q_high), medium=float(q_med), high_share=high_share, medium_share=medium_share)


def band_table(y: np.ndarray, raw: np.ndarray, bands: Bands, p_cal: np.ndarray | None = None) -> pd.DataFrame:
    b = bands.assign(raw)
    d = pd.DataFrame({"band": b, "y": np.asarray(y).astype(int)})
    if p_cal is not None:
        d["p"] = p_cal
    agg = {"n": ("y", "size"), "frauds": ("y", "sum")} | ({"mean_calibrated_p": ("p", "mean")} if p_cal is not None else {})
    t = d.groupby("band").agg(**agg).reindex(BAND_ORDER).fillna(0)
    t["share_of_rows"] = t["n"] / t["n"].sum()
    t["fraud_rate"] = t["frauds"] / t["n"].clip(lower=1)
    t["lo"], t["hi"] = wilson(t["frauds"].to_numpy(), t["n"].clip(lower=1).to_numpy())
    t["share_of_frauds"] = t["frauds"] / max(t["frauds"].sum(), 1)
    t["planned_share"] = [bands.high_share, bands.medium_share, 1 - bands.high_share - bands.medium_share]
    return t.reset_index()


def display_probability(p: float) -> str:
    """How a calibrated probability is shown to an analyst.

    Isotonic regression ends in plateaus at exactly 0 and 1 (all validation rows in the top
    plateau were fraud), but on the test month 2 of the 163 transactions given 1.0 were
    legitimate. Certainty is never displayed: values are capped at '<1%' and '>99%'.
    """
    if p < 0.01:
        return "<1%"
    if p > 0.99:
        return ">99%"
    return f"{p:.0%}"
