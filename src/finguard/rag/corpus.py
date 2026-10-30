"""Document registry and corpus build.

The registry (``corpus/registry.json``, versioned in Git) holds one entry per document:
doc_id, title, publisher, url, published date, retrieval date, jurisdiction, tier (1 = card-not-
present fraud and system documents, 2 = AML/KYC obligations, 3 = jurisdiction or internal),
type, authority level, language, file name, SHA-256 and whether the document is fictional.

Files:
* published documents: ``data/corpus/raw/<file>`` (downloaded by hand from ``url``; not in
  Git, tracked by checksum in the registry);
* project documents (glossaries, playbook): ``corpus/docs/<file>`` (in Git).

``build()`` verifies every checksum, parses, chunks and indexes, and writes
``data/corpus/index/`` with a manifest naming the index version, the embedder and the
document hashes, so a retrieval result can always be traced to the exact files it came from.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from finguard import config
from finguard.rag.chunk import chunk_blocks
from finguard.rag.index import KnowledgeIndex
from finguard.rag.parse import parse

CORPUS_DIR = Path(config.ROOT / "corpus")
REGISTRY = CORPUS_DIR / "registry.json"
PROJECT_DOCS = CORPUS_DIR / "docs"
RAW_DIR = Path(config.ROOT / "data" / "corpus" / "raw")
INDEX_DIR = Path(config.ROOT / "data" / "corpus" / "index")
REQUIRED = ["doc_id", "title", "publisher", "url", "published", "jurisdiction", "tier", "type", "authority", "language", "file"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_registry(path: Path = REGISTRY) -> list[dict]:
    entries = json.loads(Path(path).read_text(encoding="utf-8"))["documents"]
    for e in entries:
        missing = [k for k in REQUIRED if k not in e]
        if missing:
            raise ValueError(f"registry entry {e.get('doc_id')} lacks {missing}")
    ids = [e["doc_id"] for e in entries]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate doc_id in the registry")
    return entries


def locate(entry: dict, raw_dir: Path = RAW_DIR, project_dir: Path = PROJECT_DOCS) -> Path:
    base = project_dir if entry.get("source") == "project" else raw_dir
    return base / entry["file"]


def record_hash(doc_id: str, registry: Path = REGISTRY) -> str:
    """Store the current SHA-256 of a document in the registry (after a project document is edited
    or regenerated, or a published one is replaced by a newer version)."""
    reg = json.loads(Path(registry).read_text(encoding="utf-8"))
    entry = next(e for e in reg["documents"] if e["doc_id"] == doc_id)
    entry["sha256"] = sha256(locate(entry))
    Path(registry).write_text(json.dumps(reg, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return entry["sha256"]


def register_dir(src_dir: Path, registry: Path = REGISTRY, raw_dir: Path = RAW_DIR) -> dict:
    """Copy every file in ``src_dir`` whose SHA-256 matches a registered published document into
    ``raw_dir`` under its registered name. Files keep whatever name they were downloaded with;
    the hash identifies them. Returns what was placed and which registered documents are still missing."""
    import shutil
    entries = [e for e in load_registry(registry) if e.get("source") != "project" and e.get("sha256")]
    by_hash = {e["sha256"]: e for e in entries}
    placed = {}
    for f in sorted(Path(src_dir).expanduser().iterdir()):
        if f.is_file():
            e = by_hash.get(sha256(f))
            if e:
                dest = Path(raw_dir) / e["file"]
                dest.parent.mkdir(parents=True, exist_ok=True)
                if f.resolve() != dest.resolve():
                    shutil.copyfile(f, dest)
                placed[e["doc_id"]] = f.name
    return {"placed": placed, "missing": [e["doc_id"] for e in entries if e["doc_id"] not in placed
                                           and not (Path(raw_dir) / e["file"]).exists()]}


def build(registry: Path = REGISTRY, raw_dir: Path = RAW_DIR, project_dir: Path = PROJECT_DOCS,
          index_dir: Path = INDEX_DIR, embedder=None, strict_hash: bool = True) -> tuple[KnowledgeIndex, dict]:
    t0 = time.time()
    entries = load_registry(registry)
    report = {"documents": [], "missing": [], "hash_mismatch": []}
    chunks: list[dict] = []
    docs: dict[str, dict] = {}
    for e in entries:
        path = locate(e, raw_dir, project_dir)
        if not path.exists():
            report["missing"].append({"doc_id": e["doc_id"], "file": str(path), "url": e["url"]})
            continue
        digest = sha256(path)
        if e.get("sha256") and e["sha256"] != digest:
            report["hash_mismatch"].append({"doc_id": e["doc_id"], "expected": e["sha256"], "found": digest})
            if strict_hash:
                continue
        blocks = parse(path)
        cs = chunk_blocks(blocks, e["doc_id"])
        for c in cs:
            d = c.to_dict()
            d["indexed_text"] = c.indexed_text(e["title"])
            chunks.append(d)
        docs[e["doc_id"]] = e | {"sha256_found": digest}
        pages = max((b.page or 0 for b in blocks), default=0)
        report["documents"].append({"doc_id": e["doc_id"], "pages": pages or None, "blocks": len(blocks),
                                    "headings": sum(b.kind == "heading" for b in blocks),
                                    "tables": sum(b.kind == "table" for b in blocks), "chunks": len(cs)})
    if not chunks:
        raise RuntimeError("no document could be loaded; see the 'missing' list")
    index = KnowledgeIndex.build(chunks, docs, embedder)
    index.save(index_dir)
    report |= {"index_version": index.version, "embedder": index.embedder.name, "n_chunks": len(chunks),
               "seconds": round(time.time() - t0, 1)}
    (Path(index_dir) / "build_report.json").write_text(json.dumps(report, indent=1))
    return index, report


def load_index(index_dir: Path = INDEX_DIR, registry: Path = REGISTRY) -> KnowledgeIndex:
    docs = {e["doc_id"]: e for e in load_registry(registry)}
    return KnowledgeIndex.load(index_dir, docs)
