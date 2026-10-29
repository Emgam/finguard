"""Small synthetic table with the interim schema, for fast tests (no Kaggle data needed)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from finguard.data import schema

DAY = 86_400


def frame(n: int = 3000, days: int = 60, seed: int = 0) -> pd.DataFrame:
    rs = np.random.RandomState(seed)
    t = np.sort(rs.randint(DAY, (days + 1) * DAY, n)).astype("float64")
    df = {"TransactionID": np.arange(3_000_000, 3_000_000 + n), "isFraud": (rs.rand(n) < 0.05).astype("int8"),
          "TransactionDT": t, "TransactionAmt": np.round(rs.lognormal(3.5, 1.0, n), 3)}
    df["ProductCD"] = rs.choice(list("WCRHS"), n)
    df["card1"] = rs.choice([1000, 1001, 1002, 2000, 3000], n).astype("float32")
    for c in ("card2", "card3", "card5"):
        df[c] = rs.choice([100.0, 150.0, 226.0, np.nan], n).astype("float32")
    df["card4"] = rs.choice(["visa", "mastercard", None], n)
    df["card6"] = rs.choice(["debit", "credit"], n)
    df["addr1"] = rs.choice([299.0, 325.0, np.nan], n, p=[0.45, 0.45, 0.1]).astype("float32")
    df["addr2"] = rs.choice([87.0, np.nan], n).astype("float32")
    df["dist1"] = np.where(rs.rand(n) < 0.5, np.nan, rs.rand(n) * 100).astype("float32")
    df["dist2"] = np.where(rs.rand(n) < 0.9, np.nan, rs.rand(n) * 100).astype("float32")
    df["P_emaildomain"] = rs.choice(["gmail.com", "yahoo.com", "hotmail.com", None], n)
    df["R_emaildomain"] = rs.choice(["gmail.com", None, None], n)
    for c in schema.C_COLS:
        df[c] = rs.poisson(2, n).astype("float32")
    for c in schema.D_COLS:
        df[c] = np.where(rs.rand(n) < 0.3, np.nan, rs.randint(0, 5, n)).astype("float32")
    for c in schema.M_COLS:
        df[c] = rs.choice(["T", "F", None], n) if c != "M4" else rs.choice(["M0", "M1", "M2", None], n)
    vmask = rs.rand(n) < 0.6                         # one NaN block shared by many V columns
    for c in schema.V_COLS:
        df[c] = np.where(vmask, np.nan, rs.rand(n)).astype("float32")
    has_id = rs.rand(n) < 0.3
    for c in schema.ID_COLS:
        df[c] = np.where(has_id, rs.rand(n), np.nan).astype("float32")
    for c in ("id_12", "id_15", "id_16", "id_23", "id_27", "id_28", "id_29", "id_34", "id_35", "id_36", "id_37", "id_38"):
        df[c] = np.where(has_id, rs.choice(["Found", "NotFound"], n), None)
    df["id_30"] = np.where(has_id, rs.choice(["Windows 10", "iOS 11.2.1", "Android 7.0"], n), None)
    df["id_31"] = np.where(has_id, rs.choice(["chrome 63.0", "mobile safari 11.0", "opera"], n), None)
    df["id_33"] = np.where(has_id, rs.choice(["1920x1080", "2208x1242"], n), None)
    df["DeviceType"] = np.where(has_id, rs.choice(["desktop", "mobile"], n), None)
    df["DeviceInfo"] = np.where(has_id, rs.choice(["Windows", "iOS Device", "SM-G892A Build/NRD90M"], n), None)
    out = pd.DataFrame(df)
    for c in schema.TEXT_COLS:
        out[c] = out[c].astype("str").where(out[c].notna())
    out["has_identity"] = has_id.astype("int8")
    out["day"] = ((out["TransactionDT"] - DAY) // DAY).astype("int32")
    out["rel_hour"] = ((out["TransactionDT"] // 3600) % 24).astype("int8")
    out["rel_weekday"] = ((out["TransactionDT"] // DAY) % 7).astype("int8")
    return out
