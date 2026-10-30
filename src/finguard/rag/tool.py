"""The ``policy_search`` tool (Project Report v3.0, section 8.1), as the agents will call it.

Input: a query and optional filters (tier, jurisdiction). Output (JSON): ranked passages
labelled S1-S5 with document, publisher, date, jurisdiction, section, pages and text, or
``{"status": "not_covered"}`` when a name in the query (a scheme, country, body or identifier)
appears in none of the passages, or when retrieval confidence is below the abstention threshold
tuned during evaluation. Read-only. The tool never interprets: it returns sources.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from finguard.rag.corpus import INDEX_DIR, load_index
from finguard.rag.retrieve import Retriever

SCHEMA = {
    "name": "policy_search",
    "description": ("Search the FinGuard knowledge base (card-not-present fraud guidance, dispute rules, AML/KYC obligations, "
                    "project glossaries). Returns cited passages or 'not covered by the sources'."),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to look up, in plain words or with identifiers (e.g. 'Visa 10.4')."},
            "tier": {"type": "integer", "enum": [1, 2, 3], "description": "1 = CNP fraud and system documents, 2 = AML/KYC, 3 = jurisdiction and internal."},
            "jurisdiction": {"type": "string", "description": "e.g. EU, International, US, TN, Dataset, Project."},
            "k": {"type": "integer", "minimum": 1, "maximum": 5, "default": 5},
        },
        "required": ["query"],
    },
}


@lru_cache(maxsize=1)
def _retriever(index_dir: str = str(INDEX_DIR)) -> tuple[Retriever, str]:
    """The retriever as evaluated: the variant chosen on the development half, its abstention
    threshold and the grounding check, all read from ``abstention.json`` written by `evaluate`."""
    index = load_index(Path(index_dir))
    abst = Path(index_dir) / "abstention.json"
    cfg = json.loads(abst.read_text()) if abst.exists() else {}
    return (Retriever(index, threshold=cfg.get("threshold"), check_names=cfg.get("grounding_check", True)),
            cfg.get("variant", "hybrid_rerank"))


def policy_search(query: str, tier: int | None = None, jurisdiction: str | None = None, k: int = 5) -> dict:
    if not query or not query.strip():
        return {"status": "error", "message": "empty query", "passages": []}
    retriever, variant = _retriever()
    return retriever.search(query.strip(), k=max(1, min(int(k), 5)), variant=variant, tier=tier, jurisdiction=jurisdiction)
