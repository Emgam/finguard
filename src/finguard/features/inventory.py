"""Column-by-column feature inventory: what each column is, how good and stable it is, and
what Phase 2 should do with it. Recommendations follow explicit rules (below) so the table
can be regenerated whenever the evidence changes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from finguard import config
from finguard.data import schema

NORMALISE_TEXT = {"id_30": "os_family + major version", "id_31": "browser_family + major version",
                  "DeviceInfo": "device_brand", "id_33": "split into screen width and height",
                  "P_emaildomain": "email provider group (keep raw domain for trees)",
                  "R_emaildomain": "email provider group (keep raw domain for trees)"}


def _interpretability(col: str) -> str:
    g = schema.feature_group(col)
    if col in ("TransactionAmt", "card4", "card6", "P_emaildomain", "R_emaildomain", "DeviceType", "DeviceInfo",
               "id_30", "id_31", "id_33"):
        return "readable"
    if col in schema.NUMERIC_CODE_COLS or col == "ProductCD":
        return "code (values undisclosed)"
    if g in ("C1-14", "D1-15", "M1-9", "V1-339", "id_01-38", "dist1-2"):
        return "masked"
    return "readable"


def build_inventory(train: pd.DataFrame, missing: pd.DataFrame, signal: pd.DataFrame, drift: pd.DataFrame,
                    redundant: pd.DataFrame, d_trend: pd.Series) -> pd.DataFrame:
    cols = missing.index.tolist()
    inv = pd.DataFrame(index=cols)
    inv["group"] = [schema.feature_group(c) for c in cols]
    inv["dtype"] = ["text" if c in schema.TEXT_COLS else ("numeric code" if c in schema.NUMERIC_CODE_COLS else "numeric") for c in cols]
    inv["n_unique"] = [int(train[c].nunique()) for c in cols]
    inv["missing_train"] = missing.missing_train.round(4)
    inv["missing_test"] = missing.missing_test.round(4) if "missing_test" in missing else np.nan
    inv["signal_strength"] = signal.set_index("column").strength.reindex(cols).round(4)
    inv["half_period_shift"] = signal.set_index("column").half_shift.reindex(cols).round(4)
    inv["drift_adversarial_auc"] = drift.set_index("column").adversarial_auc.reindex(cols).round(4)
    red = redundant.set_index("column") if len(redundant) else pd.DataFrame(columns=["redundant", "duplicate_of"])
    inv["redundant_with"] = red.duplicate_of.reindex(cols)
    inv["d_spearman_with_day"] = d_trend.reindex(cols).round(3)
    inv["interpretability"] = [_interpretability(c) for c in cols]

    treat, use_model, use_expl, reason = [], [], [], []
    for c, r in inv.iterrows():
        if c in (config.TIME,):
            treat.append("exclude raw; derive rel_hour, rel_weekday, day for splits and velocity")
            use_model.append("no (derive only)"); reason.append("test values lie outside the training range (adversarial AUC 1.0)")
        elif r.missing_train > 0.99:
            treat.append("drop"); use_model.append("no"); reason.append(">99% missing")
        elif c in NORMALISE_TEXT:
            treat.append(f"normalise: {NORMALISE_TEXT[c]}; missing as its own category"); use_model.append("yes, transformed")
            reason.append("raw strings drift (new versions/devices)")
        elif c in schema.NUMERIC_CODE_COLS:
            treat.append("categorical code: native categorical / frequency (training period only)"); use_model.append("yes")
            reason.append("numeric value has no order meaning")
        elif c in schema.TEXT_COLS:
            treat.append("categorical; missing as its own category"); use_model.append("yes"); reason.append("")
        elif r.group == "D1-15" and abs(r.d_spearman_with_day) >= 0.05:
            treat.append("normalise: floor(DT/86400) - D; keep NaN"); use_model.append("yes, transformed")
            reason.append(f"trends with time (Spearman {r.d_spearman_with_day:+.2f})")
        elif isinstance(r.redundant_with, str):
            treat.append(f"optional for trees; drop for LR and explanations (|r| >= 0.95 with {r.redundant_with})")
            use_model.append("trees only"); reason.append("near-duplicate")
        elif c == "TransactionAmt":
            treat.append("keep raw for trees; log1p for linear models; decimals only if they add lift within product")
            use_model.append("yes"); reason.append("skew 14")
        else:
            treat.append("keep; NaN as NaN for trees, median + block indicator for LR"); use_model.append("yes"); reason.append("")
        use_expl.append({"readable": "yes", "code (values undisclosed)": "partial (category, no meaning)",
                         "masked": "family level only"}[r.interpretability])
    inv["recommended_treatment"] = treat
    inv["use_in_model"] = use_model
    inv["use_in_explanations"] = use_expl
    inv["note"] = reason
    inv["monitor_drift"] = inv.drift_adversarial_auc > 0.6
    return inv
