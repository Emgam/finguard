"""Synthetic demo world: 48 invented transactions and the three FinGuard project documents.

It lets the API, the dashboard and the container run without the Kaggle data, the Phase 2 model
or the Phase 3 index: in CI (the deployment gate) and for a first look. Nothing here is IEEE-CIS
data and every API response says ``"world": "synthetic"``; the dashboard shows a banner.

What is real: the knowledge-base code (the project documents are parsed, chunked and indexed by
the Phase 3 pipeline, with the lexical re-ranker), the indicator map, the case store, the tools'
validation, the orchestrator, the evidence policy and the reports. What is invented: the
transactions, their scores, reasons and proxy-entity histories (TransactionIDs 9000001-9000048,
outside the IEEE-CIS range, so they can never be confused with real ones).
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from finguard import config
from finguard.agents.case_store import CaseStore
from finguard.agents.tools import FUNCTIONS, IndicatorMap, ToolError
from finguard.explain.reasons import ANONYMIZED, FAMILY_LABEL
from finguard.features.novelty import FLAG_TEXT, FLAGS

N_TRANSACTIONS = 48
FIRST_ID = 9_000_001
MODEL_VERSION = "synthetic-demo (no model)"
NOTE = "Synthetic demo transaction: invented for the demo world, not IEEE-CIS data."
SIGNAL_GROUP = {"vesta_V": "V279-V321", "counting_C": "C1-C14", "timedelta_D": "D1-D15", "match_M": "M1-M9",
                "identity_masked": "id_01-id_11"}


def _transactions(seed: int = 7) -> pd.DataFrame:
    rs = np.random.RandomState(seed)
    rows = []
    for i in range(N_TRANSACTIONS):
        if i < 6:
            band, score = "High", round(float(rs.uniform(90, 99)), 1)
        elif i < 18:
            band, score = "Medium", round(float(rs.uniform(55, 89)), 1)
        else:
            band, score = "Low", round(float(rs.uniform(1, 45)), 1)
        rows.append({"transaction_id": FIRST_ID + i, "relative_day": 152 + (i * 7) % 30, "relative_hour": (i * 5) % 24,
                     "amount": round(float(rs.lognormal(4.2, 0.9)), 2), "product_code": "WCRHS"[i % 5],
                     "card_network": ["visa", "mastercard", "discover", "american express"][i % 4],
                     "card_type": "credit" if i % 3 else "debit", "has_identity_record": bool(i % 2),
                     "score": score, "band": band, "probability_displayed": f"{max(1, min(99, round(score)))}%",
                     "purchaser_email": ["gmail.com", "anonymous.com", "yahoo.com", "outlook.com"][i % 4],
                     "recipient_email": ["gmail.com", None, "hotmail.com"][i % 3],
                     "entity_available": i % 11 != 3, "earlier": int(rs.poisson(6)) if i % 7 else 0,
                     "burst": int(rs.poisson(1)), "card1_seen": int(rs.randint(5, 900))})
    return pd.DataFrame(rows)


def _reasons(t: dict, seed: int) -> list[dict]:
    rs = np.random.RandomState(seed)
    decimals = len(f"{t['amount']:.2f}".split(".")[1].rstrip("0"))
    checkable = {
        "amount": f"Amount: {t['amount']:.2f} ({decimals} non-zero decimal place(s)).",
        "email": f"Email domains: purchaser {t['purchaser_email']}, recipient {t['recipient_email'] or 'not given'}.",
        "card": f"Card attributes: network {t['card_network']}, type {t['card_type']}; card1 code seen {t['card1_seen']} "
                "time(s) in the training window (card codes are undisclosed).",
        "product": f"Product code: {t['product_code']}.",
        "time": f"Relative time of day: hour {t['relative_hour']} (the real clock time is unknown).",
    }
    if t["entity_available"]:
        checkable["entity_velocity"] = (f"Proxy-entity velocity: {t['burst']} earlier transaction(s) in the previous 24 hours.")
    fams = list(checkable) + sorted(ANONYMIZED)
    chosen = [str(f) for f in rs.choice(fams, size=5, replace=False)]
    contrib = sorted(rs.uniform(0.2, 3.0, size=5), reverse=True)
    out = []
    for f, v in zip(chosen, contrib):
        anon = f in ANONYMIZED
        text = (f"{FAMILY_LABEL[f]}: meaning not disclosed by the data provider; largest contribution from anonymized "
                f"signal group {SIGNAL_GROUP[f]}." if anon else checkable[f])
        out.append({"family": f, "label": FAMILY_LABEL[f], "anonymized": anon, "checkable": not anon,
                    "contribution_log_odds": round(float(v), 3), "signal_group": SIGNAL_GROUP.get(f) if anon else None,
                    "text": text})
    return out


def _project_index(work_dir: Path):
    """Index the project documents (in Git) with the Phase 3 pipeline: lexical models only."""
    from finguard.rag import corpus
    from finguard.rag.index import LsaEmbedder
    from finguard.rag.retrieve import LexicalReranker, Retriever
    entries = [e for e in corpus.load_registry() if e.get("source") == "project"]
    work_dir.mkdir(parents=True, exist_ok=True)
    reg = work_dir / "registry.json"
    reg.write_text(json.dumps({"about": "synthetic world: project documents only", "documents": entries}), encoding="utf-8")
    index, _ = corpus.build(registry=reg, raw_dir=work_dir / "raw", project_dir=config.ROOT / "corpus" / "docs",
                            index_dir=work_dir / "index", embedder=LsaEmbedder(dims=64))
    return Retriever(index, reranker=LexicalReranker(), threshold=None, check_names=True), "hybrid_rerank"


class SyntheticContext:
    """Has what the tools and the orchestrator read from a ``DataContext``."""

    def __init__(self, case_dir: Path):
        self.frame = _transactions()
        self.by_id = {int(r["transaction_id"]): r for r in self.frame.to_dict("records")}   # plain Python values
        self.retriever = _project_index(Path(tempfile.mkdtemp(prefix="finguard-synthetic-")))
        self.case_store = CaseStore(Path(case_dir) / "case_store.sqlite")
        self.indicator = IndicatorMap.load()
        self._searched: dict = {}

    def get(self, transaction_id: int) -> dict:
        t = self.by_id.get(int(transaction_id))
        if t is None:
            raise ToolError(f"transaction {transaction_id} is not in the synthetic demo world "
                            f"({FIRST_ID}-{FIRST_ID + N_TRANSACTIONS - 1})", retryable=False)
        return t

    def versions(self) -> dict:
        retriever, variant = self.retriever
        return {"model_version": MODEL_VERSION, "index_version": retriever.index.version,
                "embedder": retriever.index.embedder.name, "reranker": retriever.reranker.name,
                "retrieval_variant": variant, "indicator_map": self.indicator.version}


# --------------------------------------------------------------------------- tool functions
def fraud_score(ctx: SyntheticContext, transaction_id: int) -> dict:
    t = ctx.get(transaction_id)
    i = t["transaction_id"] - FIRST_ID
    flags = [{"flag": f, "on": bool((i + k) % 5 == 0) if t["entity_available"] else None, "rule": FLAG_TEXT[f]}
             for k, f in enumerate(FLAGS)]
    return {"transaction_id": t["transaction_id"], "score": t["score"], "probability_displayed": t["probability_displayed"],
            "band": t["band"], "score_scale": "0-100 (synthetic)", "band_definition": "synthetic demo bands",
            "model_version": MODEL_VERSION, "novelty_flags": flags, "period": "test", "relative_day": t["relative_day"],
            "anomaly_score": None, "anomaly_note": "not used",
            "transaction": {"amount": t["amount"], "product_code": t["product_code"], "card_network": t["card_network"],
                            "card_type": t["card_type"], "has_identity_record": t["has_identity_record"],
                            "relative_hour": t["relative_hour"]},
            "notes": [NOTE, "Flagged for review, not a finding of fraud."]}


def explain_transaction(ctx: SyntheticContext, transaction_id: int) -> dict:
    t = ctx.get(transaction_id)
    return {"transaction_id": t["transaction_id"], "top": 5, "reasons": _reasons(t, seed=t["transaction_id"] % 10_000),
            "note": "Reasons explain the score, not the cause of fraud. " + NOTE}


def entity_history(ctx: SyntheticContext, transaction_id: int, window_days: float = 7) -> dict:
    t = ctx.get(transaction_id)
    base = {"window_days": float(window_days), "short_window_hours": 24,
            "entity": "proxy entity (card1 + addr1 + first-seen day); a heuristic grouping, not a customer"}
    if not t["entity_available"]:
        return base | {"entity_key": None, "available": False,
                       "reason": "card1, addr1 or D1 is missing, so no proxy entity can be formed"}
    n = t["earlier"]
    mean = round(t["amount"] / (1.0 + (t["transaction_id"] % 4)), 2) if n else None
    return base | {"entity_key": f"syn_{t['transaction_id']}", "available": True, "earlier_transactions": n,
                   "earlier_in_24h": min(n, t["burst"]), f"earlier_in_{window_days:g}d": min(n, t["burst"] + 2),
                   "seconds_since_previous": 3600.0 * (1 + t["transaction_id"] % 9) if n else None,
                   "days_since_first": float(3 + t["transaction_id"] % 40) if n else 0.0, "amount": t["amount"],
                   "earlier_amount_mean": mean, "earlier_amount_max": round(mean * 1.6, 2) if n else None,
                   "amount_vs_earlier_mean": round(t["amount"] / mean, 3) if n else None,
                   "device_brands_seen": ["samsung", "windows"][: 1 + t["transaction_id"] % 2] if n else [],
                   "purchaser_email_domains_seen": [t["purchaser_email"]] if n else []}


SYNTHETIC_FUNCTIONS = {**FUNCTIONS, "fraud_score": fraud_score, "explain_transaction": explain_transaction,
                       "entity_history": entity_history}   # indicator_map, policy_search and case_store are the real ones
