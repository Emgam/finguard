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
from finguard.rag.retrieve import Retriever, make_reranker

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


def deployed(index_dir: Path = INDEX_DIR) -> tuple[Retriever, str]:
    """The retriever exactly as evaluated: the index, the variant chosen on the development half,
    the re-ranker whose scores the abstention threshold was tuned on, the threshold and the
    grounding check, all read from ``abstention.json`` written by `evaluate`.

    A threshold is only meaningful for the re-ranker and index it was tuned on (a cross-encoder
    scores in logits, the lexical re-ranker between 0 and 1), so a rebuilt index refuses to load
    until it is evaluated again."""
    index = load_index(Path(index_dir))
    abst = Path(index_dir) / "abstention.json"
    if not abst.exists():
        raise RuntimeError("no abstention.json: run `python -m finguard.rag evaluate` after building the index")
    cfg = json.loads(abst.read_text(encoding="utf-8"))
    if cfg.get("index_version") and cfg["index_version"] != index.version:
        raise RuntimeError(f"the index ({index.version}) was rebuilt after the evaluation ({cfg['index_version']}): "
                           "run `python -m finguard.rag evaluate` again")
    reranker = make_reranker(cfg.get("reranker"))                # None (older files): FINGUARD_RERANKER or lexical
    return (Retriever(index, reranker=reranker, threshold=cfg.get("threshold"), check_names=cfg.get("grounding_check", True)),
            cfg.get("variant", "hybrid_rerank"))


@lru_cache(maxsize=1)
def _retriever(index_dir: str = str(INDEX_DIR)) -> tuple[Retriever, str]:
    return deployed(Path(index_dir))


def policy_search(query: str, tier: int | None = None, jurisdiction: str | None = None, k: int = 5) -> dict:
    if not query or not query.strip():
        return {"status": "error", "message": "empty query", "passages": []}
    retriever, variant = _retriever()
    return retriever.search(query.strip(), k=max(1, min(int(k), 5)), variant=variant, tier=tier, jurisdiction=jurisdiction)
