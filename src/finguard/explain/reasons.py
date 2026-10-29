"""Reason families from TreeSHAP values, worded by fixed templates.

Why families and not single features: about 96% of IEEE-CIS columns are anonymized
(V, C, D, M, most id_XX). A reason like "V258 = 1" means nothing to an analyst and its
SHAP value is split unstably across correlated V columns. Summing SHAP values inside a
family (e.g. all V columns) gives a reason that is both honest and stable.

Rules for the templates:
* they state only what the data contains (values of readable fields, counts of the proxy
  entity's earlier transactions) and the direction of the contribution;
* anonymized families say that their meaning is not disclosed; they never guess it;
* "proxy entity" is always written out: it is a heuristic grouping, not a customer.
"""
from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pandas as pd

FAMILY_LABEL = {
    "entity_velocity": "Proxy-entity velocity",
    "entity_history": "Proxy-entity history",
    "amount": "Amount",
    "product": "Product code",
    "card": "Card attributes",
    "address": "Address and distance codes",
    "email": "Email domains",
    "device": "Device and browser",
    "identity_masked": "Anonymized identity signals (id_XX)",
    "time": "Relative time of day",
    "counting_C": "Anonymized counting features (C1-C14)",
    "timedelta_D": "Anonymized time-delta features (D1-D15)",
    "match_M": "Match flags (M1-M9)",
    "vesta_V": "Vesta engineered features (V1-V339)",
    "completeness": "Data completeness",
}
ANONYMIZED = {"identity_masked", "counting_C", "timedelta_D", "match_M", "vesta_V"}


def family(col: str) -> str:
    """Map a model column to its reason family."""
    if col in ("entity_prev_txn", "entity_secs_since_prev", "entity_txn_24h", "entity_txn_7d"):
        return "entity_velocity"
    if col.startswith("entity_"):
        return "entity_history"
    if col in ("TransactionAmt", "amt_decimals", "amt_decile"):
        return "amount"
    if col == "ProductCD":
        return "product"
    if col.startswith("card") or col.startswith("freq_card"):
        return "card"
    if col.startswith("addr") or col.startswith("dist") or col.startswith("freq_addr"):
        return "address"
    if "email" in col:
        return "email"
    if col in ("DeviceType", "browser_family", "browser_major", "os_family", "os_major", "device_brand",
               "screen_w", "screen_h", "freq_device"):
        return "device"
    if col.startswith("id_"):
        return "identity_masked"
    if col in ("rel_hour", "rel_weekday"):
        return "time"
    if col[0] == "C" and col[1:].isdigit():
        return "counting_C"
    if col[0] == "D" and col[1:].rstrip("n").isdigit():
        return "timedelta_D"
    if col[0] == "M" and col[1:].isdigit():
        return "match_M"
    if col[0] == "V" and col[1:].isdigit():
        return "vesta_V"
    if col in ("n_missing", "missing_pattern", "has_identity") or col.startswith("block_present_"):
        return "completeness"
    raise KeyError(f"no reason family for column {col!r}")


def family_matrix(contrib: np.ndarray, columns: list[str]) -> pd.DataFrame:
    """Sum SHAP values (log-odds) per family. `contrib` may include the bias column last."""
    contrib = np.asarray(contrib)
    if contrib.shape[1] == len(columns) + 1:
        contrib = contrib[:, :-1]
    fam = np.array([family(c) for c in columns])
    names = [f for f in FAMILY_LABEL if (fam == f).any()]
    return pd.DataFrame({f: contrib[:, fam == f].sum(axis=1) for f in names})


def top_families(fam: pd.DataFrame, k: int = 5) -> list[list[str]]:
    """Per row, the k families with the largest positive contribution (fewer if fewer are positive)."""
    vals = fam.to_numpy()
    order = np.argsort(-vals, axis=1, kind="stable")[:, :k]
    names = np.array(fam.columns)
    return [[names[j] for j in row if vals[i, j] > 0] for i, row in enumerate(order)]


def overlap(a: list[list[str]], b: list[list[str]], k: int = 5) -> np.ndarray:
    """Per row |A ∩ B| / k for two lists of top-k families."""
    return np.array([len(set(x) & set(y)) / k for x, y in zip(a, b)])


def stability(tops: list[list[list[str]]], k: int = 5) -> pd.DataFrame:
    """Mean pairwise overlap per row across several retrains (one entry of `tops` per model)."""
    pairs = list(combinations(range(len(tops)), 2))
    per_pair = np.vstack([overlap(tops[i], tops[j], k) for i, j in pairs])
    return pd.DataFrame({"mean_pairwise_overlap": per_pair.mean(axis=0), "min_pairwise_overlap": per_pair.min(axis=0)})


def decode(frame: pd.DataFrame, builder) -> pd.DataFrame:
    """Replace category codes by their training-window labels ('other' for unseen/rare levels)."""
    out = frame.copy().astype(object)
    for c in builder.categorical_:
        if c not in out.columns or c == "missing_pattern":
            continue
        levels = builder.vocab_[c]
        out[c] = [None if pd.isna(v) else (levels[int(v)] if int(v) < len(levels) else "other") for v in frame[c]]
    return out


# --------------------------------------------------------------------------- templates
def _fmt_secs(s: float) -> str:
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return "no earlier transaction"
    if s < 60:
        return f"{s:.0f} s"
    if s < 3600:
        return f"{s / 60:.0f} min"
    if s < 86400:
        return f"{s / 3600:.1f} h"
    return f"{s / 86400:.1f} days"


def _val(row: pd.Series, col: str, default="not given"):
    v = row.get(col, None)
    if v is None or (isinstance(v, float) and math.isnan(v)) or (v is pd.NA):
        return default
    return v


def _num(row: pd.Series, col: str, fmt: str = "{:.0f}", default="n/a") -> str:
    v = row.get(col, None)
    try:
        return default if v is None or math.isnan(float(v)) else fmt.format(float(v))
    except (TypeError, ValueError):
        return default


def render(fam_name: str, row: pd.Series) -> str:
    """One sentence for one family. `row` holds readable values of the transaction
    (decoded categories, raw amount, entity features)."""
    label = FAMILY_LABEL[fam_name]
    if fam_name == "entity_velocity":
        n24 = _num(row, "entity_txn_24h")
        if n24 == "n/a":
            return f"{label}: no proxy entity could be formed (card1, addr1 or D1 missing)."
        return (f"{label}: {n24} earlier transaction(s) of the same proxy entity in the previous 24 h, "
                f"{_num(row, 'entity_txn_7d')} in 7 days; previous one {_fmt_secs(row.get('entity_secs_since_prev'))} earlier.")
    if fam_name == "entity_history":
        if _num(row, "entity_prev_txn") == "n/a":
            return f"{label}: no proxy entity could be formed (card1, addr1 or D1 missing)."
        ratio = _num(row, "entity_amt_ratio", "{:.1f}x")
        ratio_txt = "first transaction of this proxy entity in the data" if ratio == "n/a" else f"amount is {ratio} its earlier average"
        return (f"{label}: {ratio_txt}; {_num(row, 'entity_n_devices')} device brand(s) and "
                f"{_num(row, 'entity_n_emails')} purchaser email domain(s) seen so far; "
                f"first seen {_num(row, 'entity_age_days', '{:.0f}')} day(s) earlier in the data.")
    if fam_name == "amount":
        return f"{label}: {_num(row, 'TransactionAmt', '{:,.2f}')} ({_num(row, 'amt_decimals')} non-zero decimal place(s))."
    if fam_name == "product":
        return f"{label}: {_val(row, 'ProductCD')} (product meaning not disclosed by the data provider)."
    if fam_name == "card":
        return (f"{label}: network {_val(row, 'card4')}, type {_val(row, 'card6')}; "
                f"card1 code seen {_num(row, 'freq_card1')} time(s) in the training window (card codes are undisclosed).")
    if fam_name == "address":
        parts = [("billing-region code", "addr1"), ("country code", "addr2"), ("distance field", "dist1")]
        given = [f"{name} {_num(row, col)}" for name, col in parts if _num(row, col) != "n/a"]
        missing = [name for name, col in parts if _num(row, col) == "n/a"]
        text = "; ".join(x for x in (", ".join(given), (", ".join(missing) + " not given") if missing else "") if x)
        return f"{label}: {text} (undisclosed codes)."
    if fam_name == "email":
        return f"{label}: purchaser {_val(row, 'P_emaildomain')}, recipient {_val(row, 'R_emaildomain')}."
    if fam_name == "device":
        return (f"{label}: {_val(row, 'DeviceType', 'device type not given')}, brand {_val(row, 'device_brand')}, "
                f"OS {_val(row, 'os_family')}, browser {_val(row, 'browser_family')} {_num(row, 'browser_major', '{:.0f}', '')}".rstrip() + ".")
    if fam_name == "time":
        return f"{label}: relative hour {_num(row, 'rel_hour')} (the real clock time and time zone are not disclosed)."
    if fam_name == "completeness":
        ident = "present" if _num(row, "has_identity") == "1" else "absent"
        return f"{label}: {_num(row, 'n_missing')} empty fields; identity record {ident}."
    if fam_name == "match_M":
        flags = ", ".join(f"{c}={_val(row, c)}" if _val(row, c) != "not given" else f"{c} not given"
                          for c in ("M4", "M5", "M6") if c in row.index)
        return f"{label} (meaning not disclosed): {flags}."
    if fam_name in ANONYMIZED:
        return f"{label}: meaning not disclosed by the data provider."
    raise KeyError(fam_name)


def reason_lines(fam_row: pd.Series, readable_row: pd.Series, k: int = 5) -> list[str]:
    """Top-k positive families of one transaction, each rendered as a sentence, largest first."""
    pos = fam_row[fam_row > 0].sort_values(ascending=False).head(k)
    return [f"{render(f, readable_row)} [+{v:.2f} log-odds]" for f, v in pos.items()]
