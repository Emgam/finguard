"""Normalisation of drifting text fields into stable, readable categories.

Version strings change every few weeks (60.6% of browser strings in the later test period
never occur in training), so models and explanations use families, not raw strings.
All functions are pure, row-wise and use no labels: safe at scoring time.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

_VERSION = re.compile(r"(\d+)(?:\.\d+)?")


def browser_family(s: pd.Series) -> pd.Series:
    """id_31 ('chrome 63.0', 'mobile safari 11.0', 'samsung browser 6.2', ...) -> family."""
    low = s.astype("string").str.lower()
    rules = [("samsung", "samsung"), ("edge", "edge"), ("chrome", "chrome"), ("firefox", "firefox"),
             ("opera", "opera"), ("safari", "safari"), ("android", "android webview"),
             ("ie ", "internet explorer"), ("ie1", "internet explorer")]
    out = pd.Series(pd.NA, index=s.index, dtype="string")
    for key, fam in rules:
        hit = low.str.contains(key, regex=False, na=False) & out.isna()
        out[hit] = fam
    out[low.notna() & out.isna()] = "other"
    return out


def browser_major(s: pd.Series) -> pd.Series:
    """Major version number from id_31, NaN when absent ('chrome generic' -> NaN)."""
    return s.astype("string").str.extract(_VERSION, expand=False).astype("float32")


def os_major(s: pd.Series) -> pd.Series:
    """Major version number from id_30 ('Windows 10' -> 10, 'iOS 11.2.1' -> 11), NaN when absent."""
    return s.astype("string").str.extract(_VERSION, expand=False).astype("float32")


def screen_size(s: pd.Series) -> tuple[pd.Series, pd.Series]:
    """id_33 ('2208x1242') -> (width, height) as floats; NaN when absent or malformed."""
    parts = s.astype("string").str.extract(r"^(\d+)x(\d+)$")
    return parts[0].astype("float32").rename("screen_w"), parts[1].astype("float32").rename("screen_h")


def os_family(s: pd.Series) -> pd.Series:
    """id_30 ('Windows 10', 'iOS 11.2.1', 'Mac OS X 10_13_2', 'Android 7.0') -> family."""
    low = s.astype("string").str.lower()
    out = pd.Series(pd.NA, index=s.index, dtype="string")
    for key, fam in [("windows", "windows"), ("ios", "ios"), ("mac", "mac"), ("android", "android"), ("linux", "linux")]:
        hit = low.str.startswith(key, na=False) & out.isna()
        out[hit] = fam
    out[low.notna() & out.isna()] = "other"
    return out


_BRANDS = [("sm-", "samsung"), ("samsung", "samsung"), ("gt-", "samsung"), ("ios device", "apple ios"),
           ("macos", "apple mac"), ("windows", "windows"), ("trident", "windows"), ("rv:", "windows"),
           ("moto", "motorola"), ("xt1", "motorola"), ("lg-", "lg"), ("lg", "lg"), ("huawei", "huawei"),
           ("hi6", "huawei"), ("redmi", "xiaomi"), ("mi ", "xiaomi"), ("pixel", "google"), ("nexus", "google"),
           ("htc", "htc"), ("zte", "zte"), ("blade", "zte"), ("lenovo", "lenovo"), ("asus", "asus"),
           ("sony", "sony"), ("linux", "linux")]


def device_brand(s: pd.Series) -> pd.Series:
    """DeviceInfo ('SAMSUNG SM-G892A Build/NRD90M', 'iOS Device', 'Windows') -> brand."""
    low = s.astype("string").str.lower()
    out = pd.Series(pd.NA, index=s.index, dtype="string")
    for key, brand in _BRANDS:
        hit = low.str.startswith(key, na=False) & out.isna()
        out[hit] = brand
    out[low.notna() & out.isna()] = "other"
    return out


_EMAIL_GROUPS = {
    "gmail": ["gmail.com", "gmail"],
    "yahoo": ["yahoo.com", "ymail.com", "rocketmail.com", "yahoo.com.mx", "yahoo.co.uk", "yahoo.co.jp",
              "yahoo.de", "yahoo.fr", "yahoo.es"],
    "microsoft": ["hotmail.com", "outlook.com", "live.com", "msn.com", "hotmail.es", "hotmail.co.uk",
                  "hotmail.de", "hotmail.fr", "outlook.es", "live.com.mx", "live.fr"],
    "apple": ["icloud.com", "me.com", "mac.com"],
    "aol": ["aol.com", "aim.com"],
    "anonymous": ["anonymous.com"],
    "protonmail": ["protonmail.com"],
}
_EMAIL_LOOKUP = {d: g for g, ds in _EMAIL_GROUPS.items() for d in ds}


def email_group(s: pd.Series) -> pd.Series:
    """Map an email domain to a provider group; unknown domains -> 'other', NaN stays NaN."""
    low = s.astype("string").str.lower()
    out = low.map(_EMAIL_LOOKUP).astype("string")
    out[low.notna() & out.isna()] = "other"
    return out


def amount_decimals(amount: pd.Series) -> pd.Series:
    """Number of non-zero decimals of TransactionAmt (0-3). Meaning undisclosed: see report."""
    cents = np.round(amount.to_numpy(dtype="float64") * 1000).astype("int64") % 1000
    dec = np.where(cents == 0, 0, np.where(cents % 100 == 0, 1, np.where(cents % 10 == 0, 2, 3)))
    return pd.Series(dec, index=amount.index, dtype="int8")
