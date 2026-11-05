"""Keyword and vector indexes over the chunks.

* ``BM25``: Okapi BM25 (k1 = 1.5, b = 0.75), the same formula as the rank_bm25 package,
  implemented here so the knowledge base has no network dependency.
* Dense embeddings are pluggable:
    - ``SentenceTransformerEmbedder`` (e.g. ``intfloat/multilingual-e5-small``, English and
      French) when the ``sentence-transformers`` package and the model are available;
    - ``LsaEmbedder`` otherwise: TF-IDF over word unigrams and bigrams, reduced with truncated
      SVD (latent semantic analysis). It is the dense fallback used where no embedding model
      can be downloaded, and it is weaker on paraphrases.
  Select with the environment variable ``FINGUARD_EMBEDDER`` (a model name, or "lsa").
* ``DenseIndex``: exact cosine search in NumPy with metadata filters. A corpus of 10-14
  documents gives a few thousand chunks, where exact search takes milliseconds; Qdrant can
  replace it behind the same interface.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from finguard.rag.text import fold, tokens


# --------------------------------------------------------------------------- BM25
@dataclass
class BM25:
    k1: float = 1.5
    b: float = 0.75
    idf_: dict = field(default_factory=dict, repr=False)
    tf_: list = field(default_factory=list, repr=False)
    len_: np.ndarray | None = field(default=None, repr=False)
    avgdl_: float = 0.0

    def fit(self, docs: list[str]) -> "BM25":
        toks = [tokens(d) for d in docs]
        self.tf_ = []
        df: dict[str, int] = {}
        for t in toks:
            counts: dict[str, int] = {}
            for w in t:
                counts[w] = counts.get(w, 0) + 1
            self.tf_.append(counts)
            for w in counts:
                df[w] = df.get(w, 0) + 1
        n = len(docs)
        self.idf_ = {w: float(np.log((n - c + 0.5) / (c + 0.5) + 1.0)) for w, c in df.items()}
        self.len_ = np.array([len(t) for t in toks], dtype="float64")
        self.avgdl_ = float(self.len_.mean()) if n else 0.0
        return self

    def scores(self, query: str) -> np.ndarray:
        q = tokens(query)
        out = np.zeros(len(self.tf_))
        norm = self.k1 * (1 - self.b + self.b * self.len_ / max(self.avgdl_, 1e-9))
        for w in set(q):
            idf = self.idf_.get(w)
            if idf is None:
                continue
            tf = np.array([d.get(w, 0) for d in self.tf_], dtype="float64")
            out += idf * tf * (self.k1 + 1) / (tf + norm)
        return out

    def coverage(self, query: str, doc_index: int) -> float:
        """IDF-weighted share of the query's content terms present in one chunk (0-1)."""
        q = set(tokens(query))
        total = sum(self.idf_.get(w, max(self.idf_.values(), default=1.0)) for w in q)
        hit = sum(self.idf_.get(w, 0.0) for w in q if w in self.tf_[doc_index])
        return hit / total if total else 0.0


# --------------------------------------------------------------------------- embedders
class LsaEmbedder:
    name = "lsa-tfidf-svd256"

    def __init__(self, dims: int = 256, seed: int = 42):
        self.dims, self.seed = dims, seed
        self.vec_ = self.svd_ = None

    def fit(self, texts: list[str]) -> "LsaEmbedder":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec_ = TfidfVectorizer(preprocessor=fold, tokenizer=tokens, token_pattern=None, ngram_range=(1, 2),
                                    min_df=1, sublinear_tf=True)
        X = self.vec_.fit_transform(texts)
        k = max(2, min(self.dims, X.shape[1] - 1, X.shape[0] - 1))
        self.svd_ = TruncatedSVD(n_components=k, random_state=self.seed).fit(X)
        self.name = f"lsa-tfidf-svd{k}"
        return self

    def _embed(self, texts: list[str]) -> np.ndarray:
        Z = self.svd_.transform(self.vec_.transform(texts))
        return Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts)


class SentenceTransformerEmbedder:
    """Multilingual sentence embeddings (E5 models expect 'query: ' / 'passage: ' prefixes)."""

    def __init__(self, model_name: str = "intfloat/multilingual-e5-small"):
        from sentence_transformers import SentenceTransformer   # optional dependency
        self.name = model_name
        self.model = SentenceTransformer(model_name)
        self.e5 = "e5" in model_name.lower()

    def fit(self, texts: list[str]) -> "SentenceTransformerEmbedder":
        return self

    def _embed(self, texts: list[str], prefix: str) -> np.ndarray:
        texts = [f"{prefix}{t}" for t in texts] if self.e5 else texts
        return np.asarray(self.model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False))

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, "passage: ")

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, "query: ")


def make_embedder(name: str | None = None):
    name = name or os.environ.get("FINGUARD_EMBEDDER") or "auto"            # empty = unset (Docker Compose)
    if name == "lsa":
        return LsaEmbedder()
    try:
        return SentenceTransformerEmbedder("intfloat/multilingual-e5-small" if name == "auto" else name)
    except Exception:
        if name not in ("auto",):
            raise
        return LsaEmbedder()


# --------------------------------------------------------------------------- persisted index
@dataclass
class KnowledgeIndex:
    chunks: list[dict]
    docs: dict                       # doc_id -> registry entry
    bm25: BM25
    embedder: object
    vectors: np.ndarray
    version: str

    @staticmethod
    def build(chunks: list[dict], docs: dict, embedder=None) -> "KnowledgeIndex":
        texts = [c["indexed_text"] for c in chunks]
        bm25 = BM25().fit(texts)
        emb = (embedder or make_embedder()).fit(texts)
        vectors = emb.embed_passages(texts).astype("float32")
        h = hashlib.sha256()
        for c in chunks:
            h.update(c["chunk_id"].encode())
            h.update(c["indexed_text"].encode())
        h.update(emb.name.encode())
        return KnowledgeIndex(chunks, docs, bm25, emb, vectors, h.hexdigest()[:16])

    def save(self, directory: str | Path) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        np.save(d / "vectors.npy", self.vectors)
        lsa = self.embedder if isinstance(self.embedder, LsaEmbedder) else None
        with open(d / "index.pkl", "wb") as f:
            pickle.dump({"bm25": self.bm25, "lsa": lsa, "embedder": self.embedder.name}, f, protocol=pickle.HIGHEST_PROTOCOL)
        (d / "manifest.json").write_text(json.dumps({
            "index_version": self.version, "embedder": self.embedder.name, "n_chunks": len(self.chunks),
            "documents": {k: {"sha256": v.get("sha256"), "title": v.get("title")} for k, v in self.docs.items()}}, indent=1), encoding="utf-8")
        return d

    @staticmethod
    def load(directory: str | Path, docs: dict) -> "KnowledgeIndex":
        d = Path(directory)
        chunks = [json.loads(line) for line in Path(d / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
        with open(d / "index.pkl", "rb") as f:
            obj = pickle.load(f)
        emb = obj["lsa"] if obj["lsa"] is not None else SentenceTransformerEmbedder(obj["embedder"])
        manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        return KnowledgeIndex(chunks, docs, obj["bm25"], emb, np.load(d / "vectors.npy"), manifest["index_version"])
