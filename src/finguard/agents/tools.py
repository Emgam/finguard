"""The six tools (Project Report v3.0, section 8.1) and the data they read.

All tools are deterministic and read-only. Each call is checked against its JSON input schema,
run, checked against its output schema, and recorded in the evidence with an ID. A tool never
returns the dataset's label (``isFraud``): the table the tools read has that column removed.
"""
from __future__ import annotations

import copy

import json
import re
import time
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from finguard import config
from finguard.agents import schemas
from finguard.agents.case_store import CaseStore
from finguard.agents.evidence import Evidence
from finguard.rag.corpus import CORPUS_DIR

INDICATOR_MAP = CORPUS_DIR / "indicator_map.json"
TOOLS = ["fraud_score", "explain_transaction", "entity_history", "indicator_map", "policy_search", "case_store"]


class ToolError(Exception):
    def __init__(self, message: str, retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable


def clean(o):
    """Plain JSON values: numpy and pandas scalars to Python, NaN and missing to null."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [clean(v) for v in o]
    if o is None or isinstance(o, (str, bool)):
        return o
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else float(o)
    try:
        if pd.isna(o):
            return None
    except (TypeError, ValueError):
        pass
    return str(o)


# --------------------------------------------------------------------------- indicator map
class IndicatorMap:
    def __init__(self, data: dict):
        self.data = data
        self.version = data["version"]
        self.triggers = data["triggers"]
        self.by_type = data.get("by_request_type", {})

    @classmethod
    def load(cls, path: Path = INDICATOR_MAP) -> "IndicatorMap":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def lookup(self, reason_families: list[str], novelty_flags: list[str], entity_available: bool = True,
               request_type: str | None = None) -> dict:
        order = list(dict.fromkeys(list(novelty_flags) + list(reason_families) + ([] if entity_available else ["no_entity_key"])))
        triggers, typologies, checks, queries, seen_q = [], [], [], [], set()
        for t in order:
            entry = self.triggers.get(t)
            if entry is None:
                continue
            triggers.append({"trigger": t, "label": entry["label"]})
            typologies += [x for x in entry["typologies"] if x not in typologies]
            checks += [{"text": c, "trigger": t} for c in entry["checks"] if c not in [x["text"] for x in checks]]
            for q in entry["queries"]:
                if q["query"] not in seen_q:
                    seen_q.add(q["query"])
                    queries.append({"query": q["query"], "tier": q.get("tier"), "trigger": t})
        for q in self.by_type.get(request_type or "", []):
            if q["query"] not in seen_q:
                seen_q.add(q["query"])
                queries.append({"query": q["query"], "tier": q.get("tier"), "trigger": f"request:{q.get('why', request_type)}"})
        return {"map_version": self.version, "triggers": triggers, "typologies": typologies, "checks": checks,
                "queries": queries,
                "note": "Typologies are what the indicators are consistent with in the playbook, never a finding."}


# --------------------------------------------------------------------------- data context
def period_of(day: int) -> str:
    for name, label in (("train", "training"), ("valid", "validation"), ("test", "test")):
        lo, hi = config.SPLIT_DAYS[name]
        if lo <= day <= hi:
            return label
    return "outside the labelled period"


class DataContext:
    """Loads the transaction table, the scorer, the entity store and the knowledge base once
    per process. Every attribute can be injected (tests use small synthetic stand-ins)."""

    def __init__(self, table: pd.DataFrame | None = None, scorer=None, store=None, retriever=None,
                 case_store: CaseStore | None = None, indicator: IndicatorMap | None = None):
        if table is not None:
            self.__dict__["table"] = table.drop(columns=[config.TARGET], errors="ignore").reset_index(drop=True)
        for name, value in (("scorer", scorer), ("store", store), ("retriever", retriever),
                            ("case_store", case_store), ("indicator", indicator)):
            if value is not None:
                self.__dict__[name] = value
        self._scored: dict[int, dict] = {}
        self._searched: dict[tuple, dict] = {}

    @cached_property
    def table(self) -> pd.DataFrame:
        from finguard.data.ingest import load_table
        return load_table("train").drop(columns=[config.TARGET], errors="ignore").reset_index(drop=True)

    @cached_property
    def position(self) -> pd.Series:
        return pd.Series(np.arange(len(self.table)), index=self.table[config.KEY].to_numpy())

    @cached_property
    def scorer(self):
        from finguard.serving.scorer import Scorer
        return Scorer()

    @cached_property
    def store(self):
        from finguard.serving.entity_store import EntityStore
        return EntityStore(self.table)

    @cached_property
    def retriever(self):
        from finguard.rag.tool import deployed
        return deployed()

    @cached_property
    def case_store(self) -> CaseStore:
        return CaseStore()

    @cached_property
    def indicator(self) -> IndicatorMap:
        return IndicatorMap.load()

    def warm_up(self) -> "DataContext":
        for name in ("table", "position", "scorer", "store", "retriever", "case_store", "indicator"):
            getattr(self, name)
        return self

    def row(self, transaction_id: int) -> pd.DataFrame:
        pos = self.position.get(int(transaction_id))
        if pos is None:
            raise ToolError(f"transaction {transaction_id} is not in the labelled IEEE-CIS data loaded in FinGuard",
                            retryable=False)
        return self.table.iloc[[int(pos)]]

    def scored(self, transaction_id: int) -> dict:
        tid = int(transaction_id)
        if tid not in self._scored:
            row = self.row(tid)
            self._scored[tid] = self.scorer.score_transaction(row, self.store.history(row), explain=True)
        return self._scored[tid]

    def versions(self) -> dict:
        retriever, variant = self.retriever
        return {"model_version": getattr(self.scorer, "model_version", None), "index_version": retriever.index.version,
                "embedder": retriever.index.embedder.name, "reranker": getattr(retriever.reranker, "name", None),
                "retrieval_variant": variant, "indicator_map": self.indicator.version}


# --------------------------------------------------------------------------- the tools
def fraud_score(ctx: DataContext, transaction_id: int) -> dict:
    row = ctx.row(transaction_id)
    r = ctx.scored(transaction_id)
    x = row.iloc[0]
    day = int(x["day"]) if "day" in row else None
    period = period_of(day) if day is not None else "unknown"
    notes = list(r["notes"])
    if period == "training":
        notes.append("This transaction is in the model's training period: its score is in-sample.")
    return {"transaction_id": int(transaction_id), "score": r["score"], "probability_displayed": r["probability_displayed"],
            "band": r["band"], "score_scale": "0-100 (100 x calibrated probability, capped at 1-99)",
            "band_definition": "High = top 1% of validation-month scores, Medium = the next 4%, Low = the rest",
            "model_version": r["model_version"],
            "novelty_flags": r["novelty_flags"], "period": period, "relative_day": day,
            "anomaly_score": None, "anomaly_note": "Isolation Forest not used: it added nothing in Phase 2.",
            "transaction": {"amount": x.get("TransactionAmt"), "product_code": x.get("ProductCD"),
                            "card_network": x.get("card4"), "card_type": x.get("card6"),
                            "has_identity_record": bool(x.get("has_identity", 0)) if "has_identity" in row else None,
                            "relative_hour": x.get("rel_hour")},
            "notes": notes}


def explain_transaction(ctx: DataContext, transaction_id: int) -> dict:
    r = ctx.scored(transaction_id)
    return {"transaction_id": int(transaction_id), "top": 5, "reasons": r["reasons"],
            "note": "Reasons explain the model's score, not the cause of fraud. Anonymized families have no disclosed meaning."}


def entity_history(ctx: DataContext, transaction_id: int, window_days: float = 7) -> dict:
    row = ctx.row(transaction_id)
    s = ctx.store.summary(row, window_days=float(window_days))
    return {"window_days": float(window_days), "short_window_hours": 24, **s}


def indicator_map(ctx: DataContext, reason_families: list[str], novelty_flags: list[str],
                  entity_available: bool = True, request_type: str | None = None) -> dict:
    return ctx.indicator.lookup(reason_families, novelty_flags, entity_available, request_type)


def policy_search(ctx: DataContext, query: str, tier: int | None = None, jurisdiction: str | None = None, k: int = 3) -> dict:
    """Searches are cached per process (the index does not change while it runs): a cross-encoder
    on a CPU takes seconds per query, and the fallback path repeats the orchestrated path's queries.
    Each call gets its own copy, because the evidence record numbers the passages in place."""
    key = (query.strip(), tier, jurisdiction, int(k))
    if key not in ctx._searched:
        retriever, variant = ctx.retriever
        ctx._searched[key] = retriever.search(key[0], k=key[3], variant=variant, tier=tier, jurisdiction=jurisdiction)
    return copy.deepcopy(ctx._searched[key])


def case_store(ctx: DataContext, action: str = "read", transaction_id: int | None = None, entity_key: str | None = None) -> dict:
    return ctx.case_store.read(transaction_id=transaction_id, entity_key=entity_key)


FUNCTIONS = {"fraud_score": fraud_score, "explain_transaction": explain_transaction, "entity_history": entity_history,
             "indicator_map": indicator_map, "policy_search": policy_search, "case_store": case_store}


class Tools:
    """The allow-list. ``call`` validates, runs and records one tool call in the evidence."""

    def __init__(self, ctx: DataContext, functions: dict | None = None):
        self.ctx = ctx
        self.functions = functions or FUNCTIONS

    def call(self, evidence: Evidence, tool: str, args: dict | None = None) -> dict:
        args = dict(args or {})
        t0 = time.time()
        if tool not in TOOLS or tool not in self.functions:
            return evidence.add(tool, args, error=f"'{tool}' is not an allowed tool", retryable=True)
        problems = schemas.errors(args, schemas.TOOL_INPUT[tool])
        if problems:
            return evidence.add(tool, args, error="invalid arguments: " + "; ".join(problems), retryable=True)
        try:
            out = clean(self.functions[tool](self.ctx, **args))
        except ToolError as e:
            return evidence.add(tool, args, error=str(e), retryable=e.retryable, seconds=time.time() - t0)
        except Exception as e:                                   # a bug or a missing file: recorded, never raised
            return evidence.add(tool, args, error=f"{type(e).__name__}: {e}", retryable=True, seconds=time.time() - t0)
        problems = schemas.errors(out, schemas.TOOL_OUTPUT[tool])
        if problems:
            return evidence.add(tool, args, error="invalid output: " + "; ".join(problems), retryable=True,
                                seconds=time.time() - t0)
        return evidence.add(tool, args, output=out, seconds=time.time() - t0)


TX_ID = re.compile(r"\b(?:TX[-_ ]?)?(\d{7})\b", re.I)


def transaction_ids(text: str) -> list[int]:
    """7-digit TransactionIDs in a request ('TX-3548165' or '3548165'); identifiers like '10.4' never match."""
    return [int(m) for m in TX_ID.findall(text or "")]
