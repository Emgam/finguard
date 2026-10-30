"""Calibrated 0-100 risk scorer with reasons (Project Report v3.0, sections 6.5-6.6 and 8.1).

What the ``fraud_score`` and ``explain_transaction`` tools return for one transaction:

* ``score``: 100 x the isotonic-calibrated probability, rounded to one decimal and capped at
  1-99 (the calibrator's plateaus at exactly 0 and 1 would claim certainty), and the displayed
  probability, capped at "<1%" and ">99%";
* ``band``: High / Medium / Low from the frozen validation-month thresholds;
* ``model_version``: registry entry (or model-file hash) of the model that scored it;
* ``novelty_flags``: shown next to the score, not part of it;
* ``reasons``: the top 5 reason families (TreeSHAP, log-odds) worded by the fixed templates;
  an anonymized family also names the anonymized signal group (the NaN block of its columns)
  that contributed most, e.g. "V279-V321", with no invented meaning.

Two ways to get features:
* ``score_transaction(row, history)``: online, from the transaction and its proxy entity's
  earlier transactions (an ``EntityStore``); unit-tested to equal the batch values;
* ``score_cached(transaction_ids)``: from the Phase 2 feature cache (labelled period).
"""
from __future__ import annotations

import hashlib
import json
import pickle
import re
from functools import cached_property
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from finguard import config
from finguard.explain import reasons as R
from finguard.features.novelty import FLAG_TEXT, FLAGS, novelty_flags
from finguard.models.calibration import display_probability

ART = Path(config.ROOT / "artifacts" / "phase2")
DISCLAIMER = "Flagged for review, not a finding of fraud."


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _num(col: str) -> int:
    m = re.search(r"(\d+)", col)
    return int(m.group(1)) if m else 0


class Scorer:
    def __init__(self, art_dir: Path = ART):
        with open(Path(art_dir) / "model_bundle.pkl", "rb") as f:
            b = pickle.load(f)
        self.builder, self.iso, self.bands = b["builder"], b["isotonic"], b["bands"]
        self.columns = b["columns"]["columns"]
        self.model_path = Path(config.ROOT / b["model_file"])
        self.booster = lgb.Booster(model_file=str(self.model_path))
        self.model_sha = _sha(self.model_path)[:12]
        self.model_version = self._registry_version() or f"lgbm_{b['codes_as']}@{self.model_sha}"
        self._block_of = {c: k for k, cols in enumerate(self.builder.blocks_) for c in cols}

    def _registry_version(self) -> str | None:
        reg = Path(config.ROOT / "artifacts" / "registry.json")
        if not reg.exists():
            return None
        r = json.loads(reg.read_text())
        for m in r.get("models", []):
            if m.get("model_sha256", "")[:12] == self.model_sha:
                aliases = [a for a, mid in r.get("aliases", {}).items() if mid == m["model_id"]]
                return m["model_id"] + (f" ({', '.join(aliases)})" if aliases else "")
        return None

    # ------------------------------------------------------------------ core
    def _matrix(self, feats: pd.DataFrame) -> np.ndarray:
        return feats[self.columns].to_numpy(dtype="float32")

    def score_features(self, feats: pd.DataFrame) -> pd.DataFrame:
        raw = self.booster.predict(self._matrix(feats), raw_score=True)
        p = self.iso.predict(raw)
        # isotonic plateaus reach exactly 0 and 1; certainty is never shown, so the score is capped at 1-99
        return pd.DataFrame({"raw_score": raw, "probability": p, "score": np.round(100 * np.clip(p, 0.01, 0.99), 1),
                             "probability_displayed": [display_probability(v) for v in p],
                             "band": self.bands.assign(raw)}, index=feats.index)

    def signal_group(self, contrib: np.ndarray, family: str) -> str | None:
        """Name the anonymized signal group (NaN block) that contributed most within a family."""
        cols = [(c, v) for c, v in zip(self.columns, contrib) if R.family(c) == family]
        if not cols:
            return None
        top = max(cols, key=lambda cv: cv[1])[0]
        block = self._block_of.get(top.rstrip("n") if top.endswith("n") and top[:-1] in self._block_of else top)
        if block is None:
            return top
        members = sorted((c for c, _ in cols if self._block_of.get(c.rstrip("n") if c.endswith("n") else c) == block), key=_num)
        return members[0] if len(members) == 1 else f"{members[0]}-{members[-1]}"

    def explain_features(self, feats: pd.DataFrame, k: int = 5) -> list[list[dict]]:
        X = self._matrix(feats)
        contrib = self.booster.predict(X, pred_contrib=True)
        fam = R.family_matrix(contrib, self.columns)
        readable = R.decode(feats[self.columns].reset_index(drop=True), self.builder)
        out = []
        for i in range(len(feats)):
            row = fam.iloc[i]
            pos = row[row > 0].sort_values(ascending=False).head(k)
            items = []
            for f, v in pos.items():
                text = R.render(f, readable.iloc[i])
                group = self.signal_group(contrib[i, :-1], f) if f in R.ANONYMIZED else None
                if group:
                    text = f"{text[:-1]}; largest contribution from anonymized signal group {group}."
                items.append({"family": f, "label": R.FAMILY_LABEL[f], "anonymized": f in R.ANONYMIZED,
                              "checkable": f not in R.ANONYMIZED, "contribution_log_odds": round(float(v), 3),
                              "signal_group": group, "text": text})
            out.append(items)
        return out

    def _result(self, scored: pd.Series, reasons: list[dict], flags: pd.Series | None, tid) -> dict:
        nf = []
        if flags is not None:
            for f in FLAGS:
                v = flags.get(f)
                nf.append({"flag": f, "on": None if pd.isna(v) else bool(v), "rule": FLAG_TEXT[f]})
        return {"transaction_id": None if tid is None else int(tid), "score": float(scored["score"]),
                "probability_displayed": scored["probability_displayed"], "band": scored["band"],
                "model_version": self.model_version, "novelty_flags": nf, "reasons": reasons,
                "notes": [DISCLAIMER, "Reasons explain the model's score, not the cause of fraud.",
                          "Proxy entity = card1 + addr1 + first-seen day: a heuristic grouping, not a customer."]}

    # ------------------------------------------------------------------ online path
    def score_transaction(self, row: pd.DataFrame, history: pd.DataFrame, explain: bool = True) -> dict:
        """Score one transaction (a one-row frame in the interim schema) given its proxy entity's history."""
        frame = pd.concat([history, row]) if len(history) else row
        if frame.index.has_duplicates:
            frame = frame.reset_index(drop=True)
            target = frame.index[-len(row):]
        else:
            target = row.index
        feats = self.builder.transform(frame).loc[target]
        flags = novelty_flags(frame).loc[target]
        scored = self.score_features(feats)
        reasons = self.explain_features(feats)[0] if explain else []
        tid = row[config.KEY].iloc[0] if config.KEY in row else None
        return self._result(scored.iloc[0], reasons, flags.iloc[0], tid)

    # ------------------------------------------------------------------ cached path (labelled period)
    @cached_property
    def _cache(self):
        from finguard.models import dataset
        fs = dataset.load("main", "numeric")
        return fs, pd.Series(np.arange(len(fs.transaction_id)), index=fs.transaction_id)

    def score_cached(self, transaction_ids, explain: bool = True) -> list[dict]:
        fs, pos = self._cache
        rows = pos.reindex(list(transaction_ids))
        if rows.isna().any():
            raise KeyError(f"not in the labelled feature cache: {list(rows[rows.isna()].index)}")
        idx = rows.to_numpy(dtype=int)
        feats = pd.DataFrame(np.asarray(fs.X[idx]), columns=fs.columns)
        scored = self.score_features(feats)
        reasons = self.explain_features(feats) if explain else [[] for _ in idx]
        return [self._result(scored.iloc[i], reasons[i], None, fs.transaction_id[j]) for i, j in enumerate(idx)]
