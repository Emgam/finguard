"""Entity store: each proxy entity's earlier transactions, for online scoring and the
``entity_history`` tool.

IEEE-CIS has no customer identifier. The store groups transactions by the proxy entity
(card1 + addr1 + first-seen day), a heuristic grouping, never a customer. "Earlier" means a
smaller TransactionDT, or the same TransactionDT and an earlier position in the file, exactly
as the batch feature builder orders transactions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from finguard import config
from finguard.features import normalize as N
from finguard.features.proxy_entity import build_proxy_entity


class EntityStore:
    def __init__(self, table: pd.DataFrame):
        self.table = table
        self.key = build_proxy_entity(table)
        self.time = table[config.TIME].to_numpy(dtype="float64")
        self.pos = np.arange(len(table))
        codes, uniques = pd.factorize(self.key, use_na_sentinel=True)
        order = np.lexsort((self.pos, self.time, codes))               # entity, time, file order
        c = codes[order]
        bounds = np.flatnonzero(np.r_[True, c[1:] != c[:-1]])
        starts = bounds[c[bounds] >= 0]                               # -1 (no key) sorts first and is skipped
        ends = np.r_[starts[1:], len(c)]
        self._groups = {uniques[c[s]]: order[s:e] for s, e in zip(starts, ends)}
        self._row_of = pd.Series(self.pos, index=table[config.KEY].to_numpy())

    @classmethod
    def from_interim(cls, parts: tuple[str, ...] = ("train",)) -> "EntityStore":
        from finguard.data.ingest import load_table
        frames = [load_table(p) for p in parts]
        return cls(pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0].reset_index(drop=True))

    def key_of(self, row: pd.Series | pd.DataFrame):
        frame = row.to_frame().T if isinstance(row, pd.Series) else row
        return build_proxy_entity(frame).iloc[0]

    def history(self, row: pd.Series | pd.DataFrame) -> pd.DataFrame:
        """Earlier transactions of the row's proxy entity (empty when it has no key)."""
        frame = row.to_frame().T if isinstance(row, pd.Series) else row
        key = build_proxy_entity(frame).iloc[0]
        if pd.isna(key) or key not in self._groups:
            return self.table.iloc[[]]
        idx = self._groups[key]
        t0 = float(frame[config.TIME].iloc[0])
        tid = frame[config.KEY].iloc[0] if config.KEY in frame else None
        p0 = int(self._row_of.get(tid, len(self.table))) if tid is not None else len(self.table)
        earlier = (self.time[idx] < t0) | ((self.time[idx] == t0) & (self.pos[idx] < p0))
        return self.table.iloc[idx[earlier]]

    def summary(self, row: pd.Series | pd.DataFrame, window_days: float = 7.0) -> dict:
        """What the entity_history tool shows: counts, gaps, amount against the past, devices and emails."""
        frame = row.to_frame().T if isinstance(row, pd.Series) else row
        key = build_proxy_entity(frame).iloc[0]
        base = {"entity": "proxy entity (card1 + addr1 + first-seen day); a heuristic grouping, not a customer"}
        if pd.isna(key):
            return base | {"entity_key": None, "available": False,
                           "reason": "card1, addr1 or D1 is missing, so no proxy entity can be formed"}
        h = self.history(frame)
        t0 = float(frame[config.TIME].iloc[0])
        amt0 = float(frame["TransactionAmt"].iloc[0])
        ht = h[config.TIME].to_numpy(dtype="float64")
        amounts = h["TransactionAmt"].to_numpy(dtype="float64")
        devices = N.device_brand(h["DeviceInfo"]).dropna() if len(h) else pd.Series(dtype="string")
        emails = h["P_emaildomain"].dropna() if len(h) else pd.Series(dtype="string")
        win = ht >= t0 - window_days * config.SECONDS_PER_DAY
        return base | {
            "entity_key": key, "available": True, "earlier_transactions": int(len(h)),
            "earlier_in_24h": int((ht >= t0 - config.SECONDS_PER_DAY).sum()),
            f"earlier_in_{window_days:g}d": int(win.sum()),
            "seconds_since_previous": float(t0 - ht.max()) if len(h) else None,
            "days_since_first": float((t0 - ht.min()) / config.SECONDS_PER_DAY) if len(h) else 0.0,
            "amount": amt0,
            "earlier_amount_mean": float(amounts.mean()) if len(h) else None,
            "earlier_amount_max": float(amounts.max()) if len(h) else None,
            "amount_vs_earlier_mean": float(amt0 / max(amounts.mean(), 0.01)) if len(h) else None,
            "device_brands_seen": sorted(set(devices.astype(str))),
            "purchaser_email_domains_seen": sorted(set(emails.astype(str))),
        }
