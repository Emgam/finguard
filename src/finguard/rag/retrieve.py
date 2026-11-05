"""Hybrid retrieval with re-ranking and abstention (Project Report v3.0, section 7.2).

Variants (compared in the evaluation, the simplest one meeting Hit@5 >= 0.80 is kept):

* ``bm25``           keyword search only;
* ``dense``          vector search only;
* ``hybrid``         BM25 top-30 and dense top-30 merged by reciprocal rank fusion (k = 60);
* ``hybrid_rerank``  hybrid, then the top 30 re-ranked.

Re-ranking is pluggable. With ``FINGUARD_RERANKER`` set to a cross-encoder model name (for
example ``cross-encoder/mmarco-mMiniLMv2-L12-H384-v1``, multilingual) and the
``sentence-transformers`` package installed, a cross-encoder scores each (query, passage)
pair. Without it, ``LexicalReranker`` is used: a fixed blend of IDF-weighted query coverage,
dense similarity, BM25 and exact identifier matches ("10.4", "R.16", "2018/05"). Its weights
were set before evaluation and are not tuned on the questions.

Abstention, two checks:

* grounding: every name in the query (a capitalised word after the first, an acronym, or an
  identifier mixing letters and digits, e.g. "Mastercard", "CTAF", "R.16") must appear in the
  returned passages or in their document's title, publisher or jurisdiction; a passage set that
  never mentions the named scheme, country or body cannot answer a question about it;
* confidence: the best re-rank score among the top results must reach a threshold tuned on
  the development half of the question set.

If either fails, the answer is "not covered by the sources" and no passage is returned; the
ungrounded names are reported so the agent can say what the sources do not cover.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np

from finguard.rag.index import KnowledgeIndex
from finguard.rag.text import fold, tokens

RRF_K = 60
CANDIDATES = 30
VARIANTS = ("bm25", "dense", "hybrid", "hybrid_rerank")
NOT_COVERED = "Not covered by the sources."
_IDENT = re.compile(r"[a-z]*\d+(?:[./-][a-z0-9]+)+|\b[a-z]\.\d+\b")


def identifiers(text: str) -> set[str]:
    return set(_IDENT.findall(fold(text)))


_WORD = re.compile(r"[A-Za-zÀ-ÿ0-9][\w.\-/]*")


def query_names(query: str) -> list[str]:
    """Names the passages must mention: capitalised words that do not start a sentence,
    acronyms, and identifiers that mix letters and digits."""
    out: list[str] = []
    start = True
    for m in _WORD.finditer(query):
        w = m.group(0).rstrip(".-/")
        prev = query[:m.start()].rstrip()
        start = not prev or prev[-1] in ".?!:;"
        is_name = ((not start and w[:1].isupper()) or (len(w) >= 2 and w.isupper() and w.isalpha())
                   or (re.search(r"\d", w) and re.search(r"[A-Za-z]", w)))
        if is_name and tokens(w) and w not in out:
            out.append(w)
    return out


class LexicalReranker:
    name = "lexical-v1"
    weights = {"coverage": 0.45, "dense": 0.35, "bm25": 0.20, "identifier": 0.10}

    def score(self, index: KnowledgeIndex, query: str, cand: np.ndarray, dense: np.ndarray, bm25: np.ndarray) -> np.ndarray:
        cov = np.array([index.bm25.coverage(query, i) for i in cand])
        b = bm25[cand] / max(float(bm25[cand].max()), 1e-9)
        d = np.clip(dense[cand], 0, 1)
        ids = identifiers(query)
        bonus = np.array([min(2, sum(x in fold(index.chunks[i]["indexed_text"]) for x in ids)) for i in cand]) if ids else 0
        w = self.weights
        return w["coverage"] * cov + w["dense"] * d + w["bm25"] * b + w["identifier"] * bonus


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder           # optional dependency
        self.name = model_name
        self.model = CrossEncoder(model_name)

    def score(self, index: KnowledgeIndex, query: str, cand: np.ndarray, dense: np.ndarray, bm25: np.ndarray) -> np.ndarray:
        pairs = [(query, index.chunks[i]["indexed_text"]) for i in cand]
        return np.asarray(self.model.predict(pairs), dtype="float64")


def make_reranker(name: str | None = None):
    name = name or os.environ.get("FINGUARD_RERANKER") or "lexical"           # empty = unset (Docker Compose)
    return LexicalReranker() if name in ("lexical", LexicalReranker.name) else CrossEncoderReranker(name)


@dataclass
class Retriever:
    index: KnowledgeIndex
    reranker: object = None
    threshold: float | None = None                                 # abstention threshold (None = never abstain)
    check_names: bool = True                                       # grounding check on names in the query

    def __post_init__(self):
        self.reranker = self.reranker or make_reranker()
        self._evidence: dict[int, set[str]] = {}

    # ---- grounding
    def _evidence_of(self, i: int) -> set[str]:
        if i not in self._evidence:
            c = self.index.chunks[i]
            d = self.index.docs.get(c["doc_id"], {})
            self._evidence[i] = set(tokens(" ".join([c["indexed_text"], str(d.get("title", "")), str(d.get("publisher", "")),
                                                    str(d.get("jurisdiction", ""))])))
        return self._evidence[i]

    def ungrounded(self, query: str, top) -> list[str]:
        """Names in the query that none of the passages (or their documents' metadata) mention.
        A compound ('IEEE-CIS', 'High-band') counts as mentioned when all of its parts are."""
        if not self.check_names:
            return []
        ev = set().union(*(self._evidence_of(int(i)) for i in top)) if len(top) else set()
        missing = []
        for name in query_names(query):
            toks = tokens(name)
            whole, parts = toks[0], toks[1:] or toks[:1]
            if whole not in ev and not all(t in ev for t in parts):
                missing.append(name)
        return missing

    # ---- filters
    def _mask(self, tier=None, jurisdiction=None, doc_ids=None) -> np.ndarray:
        def as_set(x):
            return None if x is None else {x} if isinstance(x, (str, int)) else set(x)
        tiers, jur, ids = as_set(tier), as_set(jurisdiction), as_set(doc_ids)
        keep = np.ones(len(self.index.chunks), dtype=bool)
        for i, c in enumerate(self.index.chunks):
            meta = self.index.docs.get(c["doc_id"], {})
            if tiers is not None and meta.get("tier") not in tiers:
                keep[i] = False
            elif jur is not None and meta.get("jurisdiction") not in jur:
                keep[i] = False
            elif ids is not None and c["doc_id"] not in ids:
                keep[i] = False
        return keep

    # ---- ranking
    def rank(self, query: str, variant: str = "hybrid_rerank", k: int = 5, conf_k: int | None = None,
             **filters) -> tuple[np.ndarray, np.ndarray, float]:
        """Return (top-k chunk indices, their scores, confidence over the top ``conf_k``, default k)."""
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        keep = self._mask(**filters)
        bm = np.where(keep, self.index.bm25.scores(query), -np.inf)
        qv = self.index.embedder.embed_queries([query])[0]
        dn = np.where(keep, self.index.vectors @ qv, -np.inf)
        top_b = np.argsort(-bm, kind="stable")[:CANDIDATES]
        top_d = np.argsort(-dn, kind="stable")[:CANDIDATES]
        top_b, top_d = top_b[np.isfinite(bm[top_b])], top_d[np.isfinite(dn[top_d])]
        if variant == "bm25":
            order, scores = top_b, bm[top_b]
        elif variant == "dense":
            order, scores = top_d, dn[top_d]
        else:
            fused: dict[int, float] = {}
            for lst in (top_b, top_d):
                for r, i in enumerate(lst):
                    fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (RRF_K + r + 1)
            order = np.array(sorted(fused, key=lambda i: -fused[i]), dtype=int)
            scores = np.array([fused[i] for i in order])
            if variant == "hybrid_rerank" and len(order):
                rr = self.reranker.score(self.index, query, order, np.where(np.isfinite(dn), dn, 0), np.where(np.isfinite(bm), bm, 0))
                o = np.argsort(-rr, kind="stable")
                order, scores = order[o], rr[o]
        conf = self.confidence(query, order[:conf_k or k], dn, bm)
        return order[:k], scores[:k], conf

    def confidence(self, query: str, top: np.ndarray, dn: np.ndarray, bm: np.ndarray) -> float:
        if not len(top):
            return 0.0
        dn, bm = np.where(np.isfinite(dn), dn, 0), np.where(np.isfinite(bm), bm, 0)
        if isinstance(self.reranker, LexicalReranker):
            return float(LexicalReranker().score(self.index, query, top, dn, bm).max())
        return float(self.reranker.score(self.index, query, top, dn, bm).max())

    # ---- the tool-facing call
    def abstains(self, conf: float, ungrounded: list[str]) -> bool:
        return bool(ungrounded) or (self.threshold is not None and conf < self.threshold)

    def search(self, query: str, k: int = 5, variant: str = "hybrid_rerank", **filters) -> dict:
        order, scores, conf = self.rank(query, variant, k, **filters)
        missing = self.ungrounded(query, order)
        active = {f: v for f, v in filters.items() if v is not None}
        base = {"query": query, "filters": active, "variant": variant, "confidence": round(conf, 4),
                "threshold": self.threshold, "ungrounded_names": missing, "index_version": self.index.version,
                "embedder": self.index.embedder.name, "reranker": getattr(self.reranker, "name", "none")}
        if not len(order) or self.abstains(conf, missing):
            msg = NOT_COVERED + (f" The sources found do not mention: {', '.join(missing)}." if missing else "")
            return base | {"status": "not_covered", "message": msg, "passages": []}
        passages = []
        for n, (i, s) in enumerate(zip(order, scores), 1):
            c = self.index.chunks[int(i)]
            d = self.index.docs.get(c["doc_id"], {})
            pages = None if c["page_start"] is None else (str(c["page_start"]) if c["page_start"] == c["page_end"]
                                                         else f"{c['page_start']}-{c['page_end']}")
            passages.append({"label": f"S{n}", "chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "title": d.get("title"),
                             "publisher": d.get("publisher"), "published": d.get("published"),
                             "jurisdiction": d.get("jurisdiction"), "tier": d.get("tier"), "language": d.get("language"),
                             "authority": d.get("authority"), "url": d.get("url"), "section": c["section"],
                             "subsections": list(c.get("subsections") or []), "pages": pages,
                             "kind": c["kind"], "text": c["text"], "score": round(float(s), 4),
                             "fictional": bool(d.get("fictional", False))})
        return base | {"status": "ok", "passages": passages}


def query_terms(query: str) -> list[str]:
    return tokens(query)
