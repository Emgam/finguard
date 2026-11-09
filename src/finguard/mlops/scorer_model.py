"""MLflow pyfunc model: FinGuard's calibrated fraud scorer ("models from code" file).

Logged by ``finguard.mlops.tracking.register_model`` and loaded with
``mlflow.pyfunc.load_model("models:/finguard-fraud-scorer@champion")``.

* Input: a DataFrame with the Phase 2 model columns (the feature matrix, as in
  ``artifacts/phase2/lgbm_numeric/columns.json``); NaN is allowed, as in training.
* Output: ``raw_score`` (LightGBM log-odds), ``probability`` (isotonic calibration fitted on the
  validation month), ``score`` (100 x probability, capped at 1-99) and ``band`` (High / Medium /
  Low from the frozen validation-month thresholds), exactly as ``finguard.serving.Scorer``.

This file imports nothing from ``finguard``: the logged model needs only lightgbm, numpy and
pandas, so it loads in any environment that has them.
"""
from __future__ import annotations

import json

import mlflow
import numpy as np
import pandas as pd


class FraudScorer(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import lightgbm as lgb

        self.booster = lgb.Booster(model_file=context.artifacts["booster"])
        with open(context.artifacts["calibration"], encoding="utf-8") as f:
            cal = json.load(f)
        self.columns = cal["columns"]
        self.iso_x = np.asarray(cal["isotonic_x"], dtype="float64")
        self.iso_y = np.asarray(cal["isotonic_y"], dtype="float64")
        self.high, self.medium = float(cal["bands"]["high"]), float(cal["bands"]["medium"])

    def predict(self, context, model_input, params=None):
        frame = pd.DataFrame(model_input)
        X = frame[self.columns].to_numpy(dtype="float32")           # the training precision
        raw = self.booster.predict(X, raw_score=True)
        p = np.interp(raw, self.iso_x, self.iso_y)                  # = IsotonicRegression(out_of_bounds="clip")
        band = np.where(raw >= self.high, "High", np.where(raw >= self.medium, "Medium", "Low"))
        return pd.DataFrame({"raw_score": raw, "probability": p, "score": np.round(100 * np.clip(p, 0.01, 0.99), 1),
                             "band": band.astype(object)}, index=frame.index)


mlflow.models.set_model(FraudScorer())
