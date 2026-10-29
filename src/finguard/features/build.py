"""Point-in-time feature builder (Phase 2).

Implements the Phase 1 feature-engineering plan (`reports/feature_engineering_plan.md`).
The five rules it enforces:

1. **Point in time.** Proxy-entity features for a transaction use only earlier transactions
   (stable time order: TransactionDT, then file order). No label is ever read.
2. **Training-window statistics only.** Category vocabularies, frequency counts, amount
   deciles, NaN blocks and missing-value patterns are learned in `fit` from rows whose `day`
   lies in `fit_days`, then frozen. `transform` never refits.
3. **Honest names.** Grouping features are `entity_*` (proxy entity), never `customer_*`;
   time features are relative (`rel_hour`, `rel_weekday`).
4. **Forbidden inputs** (TransactionID, raw TransactionDT, the label, the split day, the
   entity key itself) can never be returned as features: `transform` raises if they appear.
5. **Missingness stays visible.** NaN is kept (LightGBM routes it natively); block-presence
   flags, a row missing count and a hashed missing pattern make it explicit.

IEEE-CIS has no customer identifier. Every `entity_*` feature describes the heuristic proxy
entity (card1 + addr1 + first-seen day), which can merge or split real people.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from finguard import config
from finguard.data import schema
from finguard.eda.missingness import nan_blocks
from finguard.features import normalize as N
from finguard.features.proxy_entity import build_proxy_entity

FORBIDDEN = frozenset({config.KEY, config.TIME, config.TARGET, "day", "proxy_entity"})
TRENDING_D = ["D1", "D4", "D6", "D8", "D11", "D12", "D13", "D14", "D15"]       # replaced by Dn = day - D
SPARSE_DROP = ["id_07", "id_08"] + [f"id_{i}" for i in range(21, 28)]          # > 99% missing
REPLACED_TEXT = ["id_30", "id_31", "id_33", "DeviceInfo"]                       # replaced by normalised families
DERIVED_INPUTS = ["has_identity", "rel_hour", "rel_weekday"]                    # added by ingest, kept as features
CODE_COLS = list(schema.NUMERIC_CODE_COLS)
FREQ_SOURCES = {"freq_card1": "card1", "freq_card2": "card2", "freq_card3": "card3", "freq_card5": "card5",
                "freq_addr1": "addr1", "freq_addr2": "addr2", "freq_email": "P_emaildomain",
                "freq_device": "DeviceInfo"}
ENTITY_FEATURES = ["entity_prev_txn", "entity_secs_since_prev", "entity_txn_24h", "entity_txn_7d",
                   "entity_amt_ratio", "entity_n_devices", "entity_n_emails", "entity_age_days"]
LINEAR_ONLY = ["amt_decile"]          # trees are invariant to monotone binning of TransactionAmt


# --------------------------------------------------------------------------- proxy-entity features
def entity_features(df: pd.DataFrame, device: pd.Series | None = None, email: pd.Series | None = None) -> pd.DataFrame:
    """Causal proxy-entity features, aligned to `df.index`.

    For each transaction only *earlier* transactions of the same proxy entity are used
    (earlier = smaller TransactionDT; ties keep file order). Rows without an entity key
    (card1, addr1 or D1 missing) get NaN, never a guessed key. No label is read.
    """
    n = len(df)
    key = build_proxy_entity(df)
    codes = pd.factorize(key, use_na_sentinel=True)[0].astype("int64")      # -1 = no entity
    t = df[config.TIME].to_numpy(dtype="float64")
    amt = df["TransactionAmt"].to_numpy(dtype="float64")
    order = np.lexsort((np.arange(n), t, codes))                            # entity, time, file order
    c, ts, a = codes[order], t[order], amt[order]
    pos = np.arange(n)
    start = np.r_[True, c[1:] != c[:-1]]
    gstart = np.maximum.accumulate(np.where(start, pos, 0))
    prev = (pos - gstart).astype("float64")

    prev_t = np.r_[np.nan, ts[:-1]]
    prev_t[start] = np.nan
    secs = ts - prev_t

    # running sum inside each entity only, so other entities' rows cannot change the value (not even by rounding)
    sum_prev = pd.Series(a).groupby(c, sort=False).cumsum().to_numpy() - a
    ratio = np.where(prev > 0, a / np.maximum(sum_prev / np.maximum(prev, 1), 0.01), np.nan)

    composite = c.astype("float64") * 1e8 + ts                               # entity blocks never overlap: t < 1e8 - 7d
    def window(seconds: float) -> np.ndarray:
        lo = np.maximum(np.searchsorted(composite, composite - seconds, side="left"), gstart)
        return (pos - lo).astype("float64")

    def distinct_so_far(values: pd.Series | None) -> np.ndarray:
        if values is None:
            return np.full(n, np.nan)
        v = pd.Series(values.to_numpy(), dtype="object").iloc[order].reset_index(drop=True)
        frame = pd.DataFrame({"c": c, "v": v})
        first = (~frame.duplicated(["c", "v"])).to_numpy() & v.notna().to_numpy()
        cs = np.cumsum(first)
        return (cs - cs[gstart] + first[gstart]).astype("float64")

    age = (ts - ts[gstart]) / config.SECONDS_PER_DAY
    cols = {
        "entity_prev_txn": prev, "entity_secs_since_prev": secs, "entity_txn_24h": window(86_400),
        "entity_txn_7d": window(7 * 86_400), "entity_amt_ratio": ratio,
        "entity_n_devices": distinct_so_far(device), "entity_n_emails": distinct_so_far(email),
        "entity_age_days": age,
    }
    no_key = c < 0
    out = np.empty((n, len(cols)), dtype="float64")
    for j, v in enumerate(cols.values()):
        v = v.copy()
        v[no_key] = np.nan
        out[order, j] = v
    return pd.DataFrame(out.astype("float32"), index=df.index, columns=list(cols))


def entity_features_for(df: pd.DataFrame) -> pd.DataFrame:
    """Entity features exactly as the builder computes them (device brand and purchaser email domain)."""
    return entity_features(df, device=N.device_brand(df["DeviceInfo"]), email=df["P_emaildomain"])


# --------------------------------------------------------------------------- helpers
def _encode(s: pd.Series, levels: list) -> np.ndarray:
    """Category -> integer code (float32). Unseen or rare -> len(levels) ('other'); NaN stays NaN."""
    codes = pd.Categorical(s, categories=levels).codes.astype("float32")
    missing = s.isna().to_numpy()
    codes[(codes < 0) & ~missing] = len(levels)
    codes[missing] = np.nan
    return codes


def _freq(s: pd.Series, counts: pd.Series) -> np.ndarray:
    out = np.array(s.map(counts).astype("float64").to_numpy(), dtype="float64", copy=True)
    out[np.isnan(out) & s.notna().to_numpy()] = 0.0                         # unseen in training window
    return out.astype("float32")


def _day_abs(df: pd.DataFrame) -> np.ndarray:
    return np.floor(df[config.TIME].to_numpy(dtype="float64") / config.SECONDS_PER_DAY)


# --------------------------------------------------------------------------- builder
@dataclass
class FeatureBuilder:
    """Fit on the training window, then transform any period with frozen statistics.

    `codes_as`: how the six numeric codes (card1/2/3/5, addr1/2) reach the model:
    "numeric" (raw value, trees split on ranges) or "category" (training-window vocabulary,
    native categorical). The choice is made on the validation month in notebook 06.
    """
    fit_days: tuple[int, int] = config.SPLIT_DAYS["train"]
    codes_as: str = "numeric"
    min_level_count: int = 30
    max_levels: int = 250
    min_pattern_count: int = 200
    # learned in fit
    vocab_: dict = field(default_factory=dict, repr=False)
    freq_: dict = field(default_factory=dict, repr=False)
    blocks_: list = field(default_factory=list, repr=False)
    patterns_: list = field(default_factory=list, repr=False)
    amt_edges_: np.ndarray | None = field(default=None, repr=False)
    columns_: list = field(default_factory=list, repr=False)
    categorical_: list = field(default_factory=list, repr=False)
    fit_rows_: int = 0

    # ---- column groups
    def _raw_feature_cols(self, df: pd.DataFrame) -> list[str]:
        skip = set(FORBIDDEN) | set(SPARSE_DROP) | set(REPLACED_TEXT) | set(TRENDING_D) | set(DERIVED_INPUTS)
        return [c for c in df.columns if c not in skip]

    def _missing_source_cols(self, df: pd.DataFrame) -> list[str]:
        skip = set(FORBIDDEN) | set(DERIVED_INPUTS)
        return [c for c in df.columns if c not in skip]

    def _text_categoricals(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        out = {c: df[c] for c in schema.TEXT_COLS if c not in REPLACED_TEXT and c not in SPARSE_DROP}
        out["browser_family"] = N.browser_family(df["id_31"])
        out["os_family"] = N.os_family(df["id_30"])
        out["device_brand"] = N.device_brand(df["DeviceInfo"])
        out["email_group_P"] = N.email_group(df["P_emaildomain"])
        out["email_group_R"] = N.email_group(df["R_emaildomain"])
        return out

    def _pattern(self, df: pd.DataFrame) -> pd.Series:
        bits = np.zeros(len(df), dtype="int64")
        for k, cols in enumerate(self.blocks_):
            bits |= df[cols[0]].notna().to_numpy().astype("int64") << k
        return pd.Series(bits, index=df.index)

    # ---- fit
    def fit(self, df: pd.DataFrame) -> "FeatureBuilder":
        win = df["day"].between(*self.fit_days).to_numpy()
        if not win.any():
            raise ValueError("no rows in the fit window")
        tr = df.loc[win]
        self.fit_rows_ = int(win.sum())
        text = self._text_categoricals(tr)
        cat_sources = dict(text)
        if self.codes_as == "category":
            cat_sources |= {c: tr[c] for c in CODE_COLS}
        elif self.codes_as != "numeric":
            raise ValueError("codes_as must be 'numeric' or 'category'")
        self.vocab_ = {}
        for c, s in cat_sources.items():
            vc = s.value_counts()
            self.vocab_[c] = vc[vc >= self.min_level_count].index[: self.max_levels].tolist()
        self.freq_ = {name: tr[src].value_counts() for name, src in FREQ_SOURCES.items()}
        self.amt_edges_ = np.quantile(tr["TransactionAmt"].to_numpy(dtype="float64"), np.linspace(0, 1, 11)[1:-1])
        blocks = nan_blocks(tr[self._missing_source_cols(tr)], min_size=2)   # label-free input
        self.blocks_ = [list(cols) for cols in blocks["columns"]]
        vc = self._pattern(tr).value_counts()
        self.patterns_ = vc[vc >= self.min_pattern_count].index.tolist()
        self.columns_, self.categorical_ = [], []
        return self

    # ---- transform
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.amt_edges_ is None:
            raise RuntimeError("call fit() first")
        out: dict[str, np.ndarray] = {}
        cats: list[str] = []
        text = self._text_categoricals(df)
        for c in self._raw_feature_cols(df):
            if c in text:
                continue                                                     # encoded below
            if c in CODE_COLS and self.codes_as == "category":
                out[c] = _encode(df[c], self.vocab_[c])
                cats.append(c)
            else:
                out[c] = df[c].to_numpy(dtype="float32")
        for c, s in text.items():
            out[c] = _encode(s, self.vocab_[c])
            cats.append(c)
        day_abs = _day_abs(df)
        for d in TRENDING_D:                                                 # remove the calendar trend
            out[f"{d}n"] = (day_abs - df[d].to_numpy(dtype="float64")).astype("float32")
        for c in DERIVED_INPUTS:
            out[c] = df[c].to_numpy(dtype="float32")
        amt = df["TransactionAmt"]
        out["amt_decimals"] = N.amount_decimals(amt).to_numpy(dtype="float32")
        out["amt_decile"] = np.searchsorted(self.amt_edges_, amt.to_numpy(dtype="float64"), side="right").astype("float32")
        out["browser_major"] = N.browser_major(df["id_31"]).to_numpy(dtype="float32")
        out["os_major"] = N.os_major(df["id_30"]).to_numpy(dtype="float32")
        w, h = N.screen_size(df["id_33"])
        out["screen_w"], out["screen_h"] = w.to_numpy(dtype="float32"), h.to_numpy(dtype="float32")
        p, r = df["P_emaildomain"], df["R_emaildomain"]
        both = (p.notna() & r.notna()).to_numpy()
        out["email_match"] = np.where(both, (p.astype("string") == r.astype("string")).fillna(False).to_numpy(), np.nan).astype("float32")
        for name, src in FREQ_SOURCES.items():
            out[name] = _freq(df[src], self.freq_[name])
        ent = entity_features_for(df)
        for c in ENTITY_FEATURES:
            out[c] = ent[c].to_numpy()
        out["n_missing"] = df[self._missing_source_cols(df)].isna().sum(axis=1).to_numpy(dtype="float32")
        for k, cols in enumerate(self.blocks_):
            out[f"block_present_{k + 1:02d}"] = df[cols[0]].notna().to_numpy(dtype="float32")
        out["missing_pattern"] = _encode(self._pattern(df), self.patterns_)
        cats.append("missing_pattern")

        bad = FORBIDDEN & set(out)
        if bad:
            raise AssertionError(f"forbidden columns in features: {sorted(bad)}")
        frame = pd.DataFrame(out, index=df.index)
        if self.columns_ and list(frame.columns) != self.columns_:
            raise AssertionError("feature columns differ from the first transform")
        self.columns_, self.categorical_ = list(frame.columns), cats
        return frame

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    # ---- column sets per model family
    @property
    def tree_columns(self) -> list[str]:
        return [c for c in self.columns_ if c not in LINEAR_ONLY]

    @property
    def tree_categorical(self) -> list[str]:
        return [c for c in self.categorical_ if c in self.tree_columns]
