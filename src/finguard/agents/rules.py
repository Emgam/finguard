"""Rule-based request handling shared by the code-level guard, the fallback path and the
scripted test model.

* ``out_of_scope``: requests FinGuard must refuse whatever the LLM says (blocking a card,
  judging a person, filing a report, contacting a customer). Code checks this before any LLM
  call, so a persuasive model cannot talk its way past it.
* ``classify`` and ``default_plan``: the fixed request types and tool sequences the fallback
  path uses (Project Report v3.0, section 8.3).
"""
from __future__ import annotations

import re

from finguard.agents.schemas import REQUIRED_TOOLS

_ASK = r"(^\s*|\b(please|can you|could you|would you|will you|should (i|we|you|finguard)|go ahead and|you must|i want you to|now)\s+)"
OUT_OF_SCOPE = [
    (re.compile(_ASK + r"(block|freeze|cancel|suspend|close|stop|decline|reject)\b.{0,40}\b(card|account|payment|transaction|customer)s?\b",
                re.I | re.M),
     "act on a card, an account or a payment"),
    (re.compile(r"\b(is|are|was)\s+(this|the|that)\s+(customer|cardholder|person|user|client|account holder|entity)\b.{0,60}"
                r"\b(criminal|fraudster|launderer|terrorist|guilty|thief|scammer|liar)", re.I),
     "judge whether a person is a criminal"),
    (re.compile(_ASK + r"(file|submit|send|lodge|report)\b.{0,40}\b(SAR|STR|suspicious (activity|transaction) reports?)\b", re.I | re.M),
     "file a report with the authorities"),
    (re.compile(_ASK + r"(contact|call|phone|email|e-mail|text)\s+(the\s+)?(customer|cardholder|client)\b", re.I | re.M),
     "contact a customer"),
    (re.compile(r"\b(who is|identify|name of|real name|home address|phone number of)\b.{0,40}\b(customer|cardholder|person behind)\b", re.I),
     "identify a person"),
]

REFUSAL = ("FinGuard cannot {what}. It scores and explains card-not-present transactions, shows the proxy entity's "
           "history and retrieves cited guidance; decisions about people, cards, payments and reports belong to the "
           "analyst and the compliance officer under the institution's procedures.")


def out_of_scope(request: str) -> str | None:
    """What the request asks FinGuard to do that it must not, or None."""
    for pattern, what in OUT_OF_SCOPE:
        if pattern.search(request or ""):
            return what
    return None


def classify(request: str, has_transaction: bool) -> str:
    if out_of_scope(request):
        return "out_of_scope"
    if not has_transaction:
        return "typology"
    text = request.lower()
    if re.search(r"\b(check|next|verify|look at|look into|follow[- ]up|to do)\b", text):
        return "what_to_check"
    if re.search(r"\bwhy\b|\breasons?\b|\bexplain|\bhigh[- ]risk|\brisky\b|\bscored?\b|\bpourquoi\b", text):
        return "why_high_risk"
    return "investigate"


def default_plan(request_type: str, transaction_id: int | None, request: str) -> list[dict]:
    """The fixed tool sequence for a request type (the fallback path, and the scripted model's plan)."""
    if request_type == "out_of_scope":
        return []
    if request_type == "typology":
        return [{"tool": "policy_search", "args": {"query": request.strip()[:400], "k": 5}}]
    tx = {"transaction_id": int(transaction_id)}
    calls = [{"tool": "fraud_score", "args": tx}, {"tool": "explain_transaction", "args": tx},
             {"tool": "entity_history", "args": {**tx, "window_days": 7}}]
    if "indicator_map" in REQUIRED_TOOLS[request_type]:
        calls += [{"tool": "indicator_map", "args": {}}, {"tool": "policy_search", "args": {"from_indicator_map": True}}]
    if request_type == "investigate":
        calls.append({"tool": "case_store", "args": {"action": "read", **tx}})
    return calls
