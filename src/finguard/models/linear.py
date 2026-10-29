"""Logistic-regression baseline: the reference every other model must beat.

Preprocessing follows the Phase 1 plan (section 5) and is fitted on the training window only:

* numeric codes (card1/2/3/5, addr1/2) are not used as numbers; their training-window
  frequencies (`freq_*`) represent them;
* columns that duplicate another column of the same NaN block (|r| >= 0.95) are dropped
  ("trees only" in the Phase 1 inventory, recomputed here on the training window);
* heavy-tailed columns (|skew| > 2) get a signed log, then median imputation and
  standardisation; one missing indicator per NaN block (block_present_*) plus one per
  column that is missing on its own;
* categories: one-hot for the 30 most frequent training-window levels, the rest in "other".
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from finguard import config
from finguard.features.build import CODE_COLS, FeatureBuilder


def _signed_log(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


@dataclass
class LinearPreprocessor:
    builder: FeatureBuilder
    max_onehot: int = 30
    skew_threshold: float = 2.0
    corr_threshold: float = 0.95
    sample: int = 150_000
    seed: int = config.RANDOM_SEED
    numeric_: list = field(default_factory=list)
    log_: set = field(default_factory=set)
    median_: dict = field(default_factory=dict)
    mean_: dict = field(default_factory=dict)
    std_: dict = field(default_factory=dict)
    indicators_: list = field(default_factory=list)
    onehot_: dict = field(default_factory=dict)
    dropped_redundant_: list = field(default_factory=list)
    feature_names_: list = field(default_factory=list)

    def fit(self, frame: pd.DataFrame) -> "LinearPreprocessor":
        cats = set(self.builder.categorical_)
        num = [c for c in frame.columns if c not in cats and c not in CODE_COLS]
        num = [c for c in num if frame[c].nunique(dropna=True) > 1]
        smp = frame.sample(min(self.sample, len(frame)), random_state=self.seed)
        full = [c for c in num if frame[c].notna().all()]
        groups = [[c for c in b if c in num] for b in self.builder.blocks_] + [full]
        drop: list[str] = []
        for cols in groups:
            if len(cols) < 2:
                continue
            sub = smp[cols].dropna()
            if len(sub) < 1000:
                continue
            r = np.nan_to_num(np.abs(np.corrcoef(sub.to_numpy(dtype="float64").T)))
            kept: list[int] = []
            for i, c in enumerate(cols):
                if any(r[i, j] >= self.corr_threshold for j in kept):
                    drop.append(c)
                else:
                    kept.append(i)
        self.dropped_redundant_ = drop
        self.numeric_ = [c for c in num if c not in set(drop)]
        block_cols = {c for b in self.builder.blocks_ for c in b}
        self.indicators_ = []
        for c in self.numeric_:
            x = frame[c].to_numpy(dtype="float64")
            miss = np.isnan(x).mean()
            if c not in block_cols and 0.01 < miss < 0.99:
                self.indicators_.append(c)
            v = x[~np.isnan(x)]
            sk = pd.Series(v[:: max(len(v) // 200_000, 1)]).skew()
            if abs(sk) > self.skew_threshold:
                self.log_.add(c)
                v = _signed_log(v)
            self.median_[c] = float(np.median(v))
            self.mean_[c] = float(v.mean())
            self.std_[c] = float(v.std()) or 1.0
        self.onehot_ = {}
        for c in sorted(cats & set(frame.columns)):
            n_levels = len(self.builder.vocab_.get(c, [])) or int(np.nanmax(frame[c].to_numpy()) + 1)
            self.onehot_[c] = min(self.max_onehot, n_levels)
        self.feature_names_ = (list(self.numeric_) + [f"{c}__missing" for c in self.indicators_]
                               + [f"{c}=={k}" for c, n in self.onehot_.items() for k in range(n)]
                               + [f"{c}==other" for c in self.onehot_])
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        out = np.zeros((len(frame), len(self.feature_names_)), dtype="float64")
        j = 0
        for c in self.numeric_:
            x = frame[c].to_numpy(dtype="float64")
            if c in self.log_:
                x = _signed_log(x)
            x = np.where(np.isnan(x), self.median_[c], x)
            out[:, j] = (x - self.mean_[c]) / self.std_[c]
            j += 1
        for c in self.indicators_:
            out[:, j] = frame[c].isna().to_numpy()
            j += 1
        for c, n in self.onehot_.items():
            code = frame[c].to_numpy(dtype="float64")
            for k in range(n):
                out[:, j] = code == k
                j += 1
        for c, n in self.onehot_.items():
            code = frame[c].to_numpy(dtype="float64")
            out[:, j] = ~np.isnan(code) & (code >= n)
            j += 1
        return out


@dataclass
class LinearModel:
    prep: LinearPreprocessor
    clf: LogisticRegression
    C: float
    seconds: float
    grid: dict

    def raw(self, frame: pd.DataFrame) -> np.ndarray:
        return self.clf.decision_function(self.prep.transform(frame))


def train(train_frame: pd.DataFrame, y_train: np.ndarray, valid_frame: pd.DataFrame, y_valid: np.ndarray,
          builder: FeatureBuilder, Cs: tuple[float, ...] = (0.001, 0.003, 0.01, 0.03, 0.1, 1.0),
          max_iter: int = 1000) -> LinearModel:
    """Fit the preprocessor on the training window, pick C on validation PR-AUC."""
    from sklearn.metrics import average_precision_score
    t0 = time.time()
    prep = LinearPreprocessor(builder).fit(train_frame)
    Xtr, Xva = prep.transform(train_frame), prep.transform(valid_frame)
    grid, best = {}, None
    for C in Cs:
        clf = LogisticRegression(C=C, max_iter=max_iter, tol=1e-4).fit(Xtr, y_train)
        ap = float(average_precision_score(y_valid, clf.decision_function(Xva)))
        grid[C] = {"valid_pr_auc": ap, "iterations": int(np.max(clf.n_iter_))}
        if best is None or ap > grid[best[0]]["valid_pr_auc"]:
            best = (C, clf)
    return LinearModel(prep=prep, clf=best[1], C=best[0], seconds=time.time() - t0, grid=grid)
