"""The evidence-policy validator (Project Report v3.0, sections 8.2-8.3). Code, not a prompt:
a persuasive model output cannot talk its way past it.

Checks on every agent-written item (summary, interpretation, guidance, suggested checks):

* ``uncited`` / ``unknown_citation``: each item cites at least one evidence ID that exists;
* ``untraceable_number``: every number appears in a tool output or a cited-able passage,
  exactly or rounded (a percentage may be a rounded fraction);
* ``quote_not_found`` / ``guidance_not_passage``: a guidance quote is copied verbatim from a
  passage it cites;
* ``forbidden_wording``: no "proves", no statement that a transaction is fraud or a person a
  criminal, no instruction to block a card, contact a customer or file a report;
* ``customer_word``: the proxy entity is never called a customer (regulatory terms such as
  "strong customer authentication" are allowed);
* ``anonymized_meaning``: no meaning given to an anonymized column (V, C, D, M, id_XX);
* ``transaction_in_general_answer``: a typology answer makes no claim about a transaction.

Plus, before any agent writes: ``missing_evidence`` (the request type's required tools).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from finguard.agents.evidence import NUMBER, Evidence
from finguard.agents.schemas import REQUIRED_TOOLS
from finguard.rag.text import normalize_for_match

ALLOWED_CUSTOMER = re.compile(r"\b(strong customer authentication|customer due diligence|know your customer|"
                              r"customer authentication|customer identification|customer-initiated)\b", re.I)
NEGATION = re.compile(r"\b(not|never|no|without|avoid|instead of|rather than)\b|n't\b", re.I)

FORBIDDEN = [
    (re.compile(r"\bprov(e|es|ed|en|ing)\b", re.I), "says 'proves': use 'consistent with'", True),
    (re.compile(r"\b(this|the|that)\s+(transaction|payment|alert|card|proxy entity|cardholder)\s+(is|was)\s+"
                r"(definitely\s+|clearly\s+|certainly\s+|surely\s+|obviously\s+)?(a\s+)?(fraud|fraudulent|criminal)\b", re.I),
     "states that a transaction is fraud", False),
    (re.compile(r"\b(is|was)\s+(a\s+)?(confirmed|proven|definite|certain)\s+fraud\b", re.I), "states confirmed fraud", False),
    (re.compile(r"\b(cardholder|proxy entity|account holder|user|person|entity)\s+(is|was)\s+(a\s+)?"
                r"(criminal|fraudster|thief|scammer|money launderer)\b", re.I), "judges a person", False),
    (re.compile(r"\b(block|freeze|decline|cancel|suspend)\s+(the\s+|this\s+)?(card|account|payment|transaction)s?\b", re.I),
     "tells the analyst to act on a card or payment", True),
    (re.compile(r"\bcontact\s+(the\s+)?(customer|cardholder)\b", re.I), "tells the analyst to contact a customer", True),
    (re.compile(r"\bfile\s+(a|the)\s+(SAR|STR|suspicious (transaction|activity) report)\b", re.I),
     "tells the analyst to file a report", True),
]
ANONYMIZED_MEANING = re.compile(r"\b(V\d{1,3}|C\d{1,2}|D\d{1,2}|M\d|id_\d{2})\b[^.;]{0,60}\b(means|measures|represents|"
                                r"refers to|stands for|captures|tracks|encodes|is the number|is the count)\b")
TRANSACTION_ID = re.compile(r"\b(?:TX[-_ ]?)?\d{7}\b", re.I)


# --------------------------------------------------------------------------- numbers
def _decimals(s: str) -> int:
    return len(s.split(".", 1)[1]) if "." in s else 0


class NumberIndex:
    """Numbers found in the evidence, as text and as values."""

    def __init__(self, tokens: list[str]):
        self.strings = {self._plain(t) for t in tokens}
        self.values = sorted({float(self._plain(t)) for t in tokens if self._plain(t) not in ("", "-", "+")})

    @staticmethod
    def _plain(t: str) -> str:
        return t.rstrip("%").replace(",", "").lstrip("+")

    def traceable(self, token: str) -> bool:
        s = self._plain(token)
        if s in self.strings:
            return True
        try:
            v = float(s)
        except ValueError:
            return False
        d = _decimals(s)
        tol = 0.5 * 10 ** -(d + 2)
        for e in self.values:
            if abs(e - v) <= 1e-9 or abs(round(e, d) - v) <= tol:
                return True
            if token.endswith("%") and 0 <= e <= 1 and abs(round(100 * e, d) - v) <= tol:
                return True
        return False


# --------------------------------------------------------------------------- result
@dataclass
class PolicyResult:
    violations: list[dict] = field(default_factory=list)
    claims: int = 0
    cited: int = 0
    numbers: int = 0
    traceable: int = 0

    @property
    def ok(self) -> bool:
        return not self.violations

    def metrics(self) -> dict:
        return {"claims": self.claims, "cited": self.cited, "numbers": self.numbers, "traceable": self.traceable,
                "violations": len(self.violations), "codes": sorted({v["code"] for v in self.violations})}

    def feedback(self) -> str:
        """The problems, worded for the revising agent."""
        return "\n".join(f"- {v['section']}[{v['index']}] {v['code']}: {v['detail']} | text: {v['text'][:160]!r}"
                         for v in self.violations)


def missing_evidence(evidence: Evidence, request_type: str) -> list[str]:
    return [t for t in REQUIRED_TOOLS.get(request_type, []) if not evidence.succeeded(t)]


# --------------------------------------------------------------------------- the check
def _wording(text: str) -> list[tuple[str, str]]:
    out = []
    for pattern, detail, negatable in FORBIDDEN:
        for m in pattern.finditer(text):
            before = text[max(0, m.start() - 40):m.start()]
            sentence = text[max(0, text.rfind(".", 0, m.start()) + 1):m.end() + 80]
            if negatable and (NEGATION.search(before) or (detail.endswith("file a report")
                                                          and re.search(r"compliance officer|whether", sentence, re.I))):
                continue
            out.append(("forbidden_wording", detail))
    if re.search(r"\bcustomers?\b", ALLOWED_CUSTOMER.sub("", text), re.I):
        out.append(("customer_word", "calls the proxy entity or the cardholder a customer: write 'proxy entity'"))
    if ANONYMIZED_MEANING.search(text):
        out.append(("anonymized_meaning", "gives a meaning to an anonymized column; their meaning is not disclosed"))
    return out


def check(report: dict, evidence: Evidence, request_type: str) -> PolicyResult:
    """Validate an agent-written report (summary, interpretation, guidance, suggested_checks)."""
    res = PolicyResult()
    ids, passages = evidence.ids(), evidence.passages()
    numbers = NumberIndex(evidence.numbers())
    search_ids = {i["id"] for i in evidence.items if i["tool"] == "policy_search" and i["ok"]}

    def bad(section, index, code, detail, text):
        res.violations.append({"section": section, "index": index, "code": code, "detail": detail, "text": text})

    def claim(section, index, text, cites, is_guidance=False):
        res.claims += 1
        unknown = [c for c in cites if c not in ids]
        if not cites:
            bad(section, index, "uncited", "cites no evidence", text)
        elif unknown:
            bad(section, index, "unknown_citation", f"cites {unknown}, which are not in the evidence", text)
        else:
            res.cited += 1
        if not is_guidance:
            for tok in NUMBER.findall(text):
                res.numbers += 1
                if numbers.traceable(tok):
                    res.traceable += 1
                else:
                    bad(section, index, "untraceable_number", f"{tok} appears in no tool output or passage", text)
            for code, detail in _wording(text):
                bad(section, index, code, detail, text)
        if request_type == "typology":
            if any(c.startswith("T") and c not in search_ids for c in cites) or TRANSACTION_ID.search(text):
                bad(section, index, "transaction_in_general_answer", "a general answer makes no transaction claim", text)

    s = report.get("summary") or {}
    if s:
        claim("summary", 0, s.get("text", ""), s.get("cites", []))
    for name in ("interpretation", "suggested_checks"):
        for n, c in enumerate(report.get(name, [])):
            claim(name, n, c.get("text", ""), c.get("cites", []))
    for n, g in enumerate(report.get("guidance", [])):
        quote, cites = g.get("quote", ""), g.get("cites", [])
        claim("guidance", n, quote, cites, is_guidance=True)
        cited_passages = [passages[c] for c in cites if c in passages]
        if cites and len(cited_passages) < len(cites):
            bad("guidance", n, "guidance_not_passage", "guidance must cite retrieved passages (S IDs)", quote)
        q = normalize_for_match(quote.strip().strip('"“”').rstrip("."))
        if cited_passages and not any(q and q in normalize_for_match(p.get("text", "")) for p in cited_passages):
            bad("guidance", n, "quote_not_found", "the quote is not copied verbatim from the cited passage", quote)
        note = g.get("note") or ""
        if note:
            for tok in NUMBER.findall(note):
                res.numbers += 1
                if numbers.traceable(tok):
                    res.traceable += 1
                else:
                    bad("guidance", n, "untraceable_number", f"{tok} in the note appears in no evidence", note)
            for code, detail in _wording(note):
                bad("guidance", n, code, detail, note)
    return res
