"""LightGBM main model.

Trained on the training window, early-stopped on the validation month (average precision),
never shown the test month. Code numbers card1/2/3/5 and addr1/2 enter either as raw
numbers or as training-window categories (FeatureBuilder.codes_as); the choice is made on
validation PR-AUC.

TreeSHAP values come from LightGBM itself (`pred_contrib=True`), so the `shap` package is
not required.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np

from finguard import config
from finguard.models.dataset import FeatureSet

BASE_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.5,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "max_bin": 255,
    "cat_smooth": 10.0,
    "cat_l2": 10.0,
    "min_data_per_group": 100,
    "max_cat_threshold": 32,
    "metric": "average_precision",
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 2,
    "verbose": -1,
}


@dataclass
class TrainedModel:
    booster: lgb.Booster
    columns: list[str]
    categorical: list[str]
    params: dict
    best_iteration: int
    seconds: float
    valid_curve: list[float] = field(default_factory=list)

    def raw(self, X: np.ndarray) -> np.ndarray:
        """Raw score (log-odds). Ranking metrics and bands use this."""
        return self.booster.predict(X, raw_score=True, num_iteration=self.best_iteration)

    def proba(self, X: np.ndarray) -> np.ndarray:
        """Uncalibrated probability; use finguard.models.calibration for calibrated values."""
        return 1.0 / (1.0 + np.exp(-self.raw(X)))

    def contrib(self, X: np.ndarray) -> np.ndarray:
        """TreeSHAP values in log-odds; last column is the expected value. Rows sum to raw()."""
        return self.booster.predict(X, pred_contrib=True, num_iteration=self.best_iteration)


def train(fs: FeatureSet, columns: list[str] | None = None, params: dict | None = None, seed: int = config.RANDOM_SEED,
          num_boost_round: int = 3000, early_stopping: int | None = 100, train_mask: np.ndarray | None = None,
          log_every: int = 100) -> TrainedModel:
    """Fit on the scheme's train window; early-stop on its validation month."""
    columns = columns or [c for c in fs.columns if c in fs.builder.tree_columns]
    cats = [c for c in fs.categorical if c in columns]
    p = dict(BASE_PARAMS) | (params or {}) | {"seed": seed, "bagging_seed": seed + 1, "feature_fraction_seed": seed + 2,
                                               "data_random_seed": seed + 3}
    m = fs.masks
    tr = m["train"] if train_mask is None else train_mask
    idx = fs.cols(columns)
    t0 = time.time()
    dtrain = lgb.Dataset(np.asarray(fs.X[tr][:, idx]), label=fs.y[tr], feature_name=columns,
                         categorical_feature=cats, free_raw_data=True)
    dvalid = lgb.Dataset(np.asarray(fs.X[m["valid"]][:, idx]), label=fs.y[m["valid"]], reference=dtrain)
    curve: dict = {}
    callbacks = [lgb.record_evaluation(curve)]
    if log_every:
        callbacks.append(lgb.log_evaluation(log_every))
    if early_stopping:
        callbacks.append(lgb.early_stopping(early_stopping, verbose=False))
    booster = lgb.train(p, dtrain, num_boost_round=num_boost_round, valid_sets=[dvalid], valid_names=["valid"],
                        callbacks=callbacks)
    best = booster.best_iteration or booster.current_iteration()
    return TrainedModel(booster=booster, columns=columns, categorical=cats, params=p, best_iteration=int(best),
                        seconds=time.time() - t0, valid_curve=list(curve.get("valid", {}).get("average_precision", [])))


def predict_raw(model: TrainedModel, fs: FeatureSet, rows: np.ndarray) -> np.ndarray:
    return model.raw(np.asarray(fs.X[rows][:, fs.cols(model.columns)]))
