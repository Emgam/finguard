"""Isolation Forest, as a measured experiment (Project Report v3.0, sections 6.3 and 11).

Two input sets:

* ``curated``: amount, proxy-entity velocity and history, three counting columns, the missing
  count and two rarity counts, the features an analyst would call "unusual behaviour";
* ``all``: every tree input.

Inputs get a signed log (Isolation Forest draws split values uniformly between a feature's
minimum and maximum, so heavy tails would waste splits) and training-window medians for NaN.

Two fits, for two purposes:

* ``fit_on="legit"``: training-period legitimate transactions only, as specified in the
  report, to measure the detector on its own;
* ``fit_on="all"``: all training-period transactions, labels unused, to build a feature.
  A detector fitted on legitimate rows only would score training frauds as unusual partly
  because they were left out of its fit, which leaks the label into a training feature.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import IsolationForest

from finguard import config

CURATED = ["TransactionAmt", "entity_prev_txn", "entity_secs_since_prev", "entity_txn_24h", "entity_txn_7d",
           "entity_amt_ratio", "entity_n_devices", "entity_n_emails", "entity_age_days", "C1", "C13", "C14",
           "n_missing", "freq_card1", "freq_addr1"]


def _signed_log(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


@dataclass
class AnomalyDetector:
    columns: list[str]
    n_estimators: int = 300
    max_samples: int = 256
    sample_rows: int = 200_000
    seed: int = config.RANDOM_SEED
    median_: np.ndarray | None = field(default=None, repr=False)
    model_: IsolationForest | None = field(default=None, repr=False)

    def _prep(self, X: np.ndarray) -> np.ndarray:
        Z = _signed_log(np.asarray(X, dtype="float64"))
        return np.where(np.isnan(Z), self.median_, Z)

    def fit(self, X: np.ndarray) -> "AnomalyDetector":
        rs = np.random.RandomState(self.seed)
        rows = rs.choice(len(X), min(self.sample_rows, len(X)), replace=False) if len(X) > self.sample_rows else np.arange(len(X))
        Z = _signed_log(np.asarray(X[np.sort(rows)], dtype="float64"))
        med = np.nanmedian(Z, axis=0)
        self.median_ = np.where(np.isnan(med), 0.0, med)
        self.model_ = IsolationForest(n_estimators=self.n_estimators, max_samples=self.max_samples,
                                      random_state=self.seed, n_jobs=2).fit(np.where(np.isnan(Z), self.median_, Z))
        return self

    def score(self, X: np.ndarray, chunk: int = 100_000) -> np.ndarray:
        """Anomaly score: higher = more unusual (negated sklearn score_samples)."""
        out = np.empty(len(X), dtype="float64")
        for i in range(0, len(X), chunk):
            out[i:i + chunk] = -self.model_.score_samples(self._prep(X[i:i + chunk]))
        return out
