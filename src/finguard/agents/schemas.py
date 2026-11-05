"""JSON schemas: tool inputs and outputs, and the three agents' outputs.

Tool inputs are strict (unknown arguments are refused). Agent outputs tolerate extra keys,
which code ignores: only the fields named here are ever read.

Every tool call is validated against its input schema before it runs and its output schema
after; every agent output is validated before code reads it. A failure is an error the graph
routes (re-plan, retry or fallback), never something the next step has to tolerate.
"""
from __future__ import annotations

from jsonschema import Draft202012Validator

REQUEST_TYPES = ["investigate", "why_high_risk", "what_to_check", "typology", "out_of_scope"]
TRANSACTION_TYPES = ("investigate", "why_high_risk", "what_to_check")

# Project Report v3.0, section 8.2: tools that must have succeeded before any agent writes.
REQUIRED_TOOLS = {
    "investigate": ["fraud_score", "explain_transaction", "entity_history", "indicator_map", "policy_search"],
    "what_to_check": ["fraud_score", "explain_transaction", "entity_history", "indicator_map", "policy_search"],
    "why_high_risk": ["fraud_score", "explain_transaction", "entity_history"],
    "typology": ["policy_search"],
    "out_of_scope": [],
}
ALLOWED_TOOLS = {
    "investigate": ["fraud_score", "explain_transaction", "entity_history", "indicator_map", "policy_search", "case_store"],
    "what_to_check": ["fraud_score", "explain_transaction", "entity_history", "indicator_map", "policy_search", "case_store"],
    "why_high_risk": ["fraud_score", "explain_transaction", "entity_history", "indicator_map", "policy_search", "case_store"],
    "typology": ["policy_search"],
    "out_of_scope": [],
}
MAX_TOOL_CALLS = 8
MAX_REPLANS = 2

_TX = {"type": "integer", "minimum": 1}
CITE = {"type": "string", "pattern": "^[TS][0-9]{1,3}$"}
CLAIM = {"type": "object", "required": ["text", "cites"],
         "properties": {"text": {"type": "string", "minLength": 3, "maxLength": 800},
                        "cites": {"type": "array", "items": CITE, "maxItems": 6}}}

TOOL_INPUT = {
    "fraud_score": {"type": "object", "required": ["transaction_id"], "additionalProperties": False,
                    "properties": {"transaction_id": _TX}},
    "explain_transaction": {"type": "object", "required": ["transaction_id"], "additionalProperties": False,
                            "properties": {"transaction_id": _TX}},
    "entity_history": {"type": "object", "required": ["transaction_id"], "additionalProperties": False,
                       "properties": {"transaction_id": _TX, "window_days": {"type": "number", "minimum": 1, "maximum": 90}}},
    "indicator_map": {"type": "object", "required": ["reason_families", "novelty_flags"], "additionalProperties": False,
                      "properties": {"reason_families": {"type": "array", "items": {"type": "string"}, "maxItems": 10},
                                     "novelty_flags": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
                                     "entity_available": {"type": "boolean"},
                                     "request_type": {"enum": REQUEST_TYPES}}},
    "policy_search": {"type": "object", "required": ["query"], "additionalProperties": False,
                      "properties": {"query": {"type": "string", "minLength": 3, "maxLength": 400},
                                     "tier": {"type": ["integer", "null"], "minimum": 1, "maximum": 3},
                                     "jurisdiction": {"type": ["string", "null"]},
                                     "k": {"type": "integer", "minimum": 1, "maximum": 5}}},
    "case_store": {"type": "object", "additionalProperties": False,
                   "properties": {"action": {"enum": ["read"]}, "transaction_id": _TX,
                                  "entity_key": {"type": ["string", "null"]}}},
}

TOOL_OUTPUT = {
    "fraud_score": {"type": "object", "required": ["transaction_id", "score", "probability_displayed", "band",
                                                   "model_version", "novelty_flags", "period"]},
    "explain_transaction": {"type": "object", "required": ["transaction_id", "reasons"],
                            "properties": {"reasons": {"type": "array", "maxItems": 5}}},
    "entity_history": {"type": "object", "required": ["entity", "available"]},
    "indicator_map": {"type": "object", "required": ["map_version", "typologies", "checks", "queries"]},
    "policy_search": {"type": "object", "required": ["status", "passages"]},
    "case_store": {"type": "object", "required": ["cases"]},
}

TOOL_DESCRIPTION = {
    "fraud_score": "Calibrated 0-100 risk score, band (High/Medium/Low), displayed probability, model version, "
                   "novelty flags and the period the transaction falls in. Input: transaction_id.",
    "explain_transaction": "Top 5 reason families that pushed the score up (TreeSHAP), each worded by a fixed "
                           "template and marked checkable or anonymized. Input: transaction_id.",
    "entity_history": "History of the transaction's proxy entity (card1 + addr1 + first-seen day; a heuristic, "
                      "never a customer): earlier transactions, bursts, amount against the past, devices and "
                      "emails seen. Input: transaction_id, window_days (default 7).",
    "indicator_map": "Fixed table: typologies, suggested checks and knowledge-base queries for the alert's "
                     "reason families and novelty flags. Arguments are filled by code from earlier tools.",
    "policy_search": "Search the knowledge base (card-not-present fraud guidance, Visa dispute rules, EBA/ECB "
                     "fraud reporting, FATF, Federal Reserve classifiers, BCT circular, FinGuard glossaries and a "
                     "fictional playbook). Returns cited passages or 'not covered by the sources'. Input: query, "
                     "optional tier (1 CNP fraud and system, 2 AML/KYC, 3 jurisdiction and internal), jurisdiction, "
                     "k (max 5). Use {\"from_indicator_map\": true} as args to search the indicator map's queries.",
    "case_store": "Earlier FinGuard alerts and analyst decisions for the same transaction or proxy entity "
                  "(FinGuard's own database). Input: action 'read', transaction_id.",
}

PLAN = {
    "type": "object", "required": ["request_type", "tool_calls"],
    "properties": {
        "request_type": {"enum": REQUEST_TYPES},
        "transaction_id": {"type": ["integer", "null"]},
        "tool_calls": {"type": "array", "maxItems": MAX_TOOL_CALLS, "items": {
            "type": "object", "required": ["tool"],
            "properties": {"tool": {"type": "string"}, "args": {"type": "object"}}}},
        "rationale": {"type": "string", "maxLength": 600},
    },
}

INVESTIGATION = {
    "type": "object", "required": ["interpretations", "suggested_checks"],
    "properties": {
        "interpretations": {"type": "array", "items": CLAIM, "maxItems": 8},
        "suggested_checks": {"type": "array", "items": CLAIM, "maxItems": 8},
        "open_questions": {"type": "array", "items": {"type": "string", "maxLength": 300}, "maxItems": 5},
    },
}

GUIDANCE = {"type": "object", "required": ["quote", "cites"],
            "properties": {"quote": {"type": "string", "minLength": 20, "maxLength": 600},
                           "cites": {"type": "array", "items": CITE, "minItems": 1, "maxItems": 2},
                           "note": {"type": "string", "maxLength": 400}}}

REPORT = {
    "type": "object", "required": ["summary", "interpretation", "guidance", "suggested_checks"],
    "properties": {
        "summary": CLAIM,
        "interpretation": {"type": "array", "items": CLAIM, "maxItems": 8},
        "guidance": {"type": "array", "items": GUIDANCE, "maxItems": 6},
        "suggested_checks": {"type": "array", "items": CLAIM, "maxItems": 8},
    },
}


def errors(instance, schema) -> list[str]:
    """Readable validation errors ('' path = top level); an empty list means valid."""
    out = []
    for e in sorted(Draft202012Validator(schema).iter_errors(instance), key=lambda e: list(e.path)):
        where = "/".join(str(p) for p in e.path) or "(top level)"
        out.append(f"{where}: {e.message}")
    return out
