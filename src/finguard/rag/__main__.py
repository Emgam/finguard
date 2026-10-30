"""Command line for the knowledge base.

    python -m finguard.rag register FILE --doc-id ID     # copy a downloaded document in, record its SHA-256
    python -m finguard.rag register-dir FOLDER            # copy in every file whose SHA-256 matches the registry
    python -m finguard.rag rehash DOC_ID [DOC_ID ...]     # record a new SHA-256 after editing a project document
    python -m finguard.rag build [--embedder lsa|MODEL]   # parse, chunk, index -> data/corpus/index/
    python -m finguard.rag freeze [--note TEXT]           # freeze the question set before evaluating
    python -m finguard.rag evaluate [--note TEXT]         # four variants, abstention, reports (+ run log)
    python -m finguard.rag search "QUERY" [--tier 1] [--jurisdiction EU] [--variant V]
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from finguard.rag import corpus, evaluate as ev
from finguard.rag.retrieve import VARIANTS, Retriever


def cmd_register(a) -> None:
    reg = json.loads(corpus.REGISTRY.read_text(encoding="utf-8"))
    entry = next((e for e in reg["documents"] if e["doc_id"] == a.doc_id), None)
    if entry is None:
        raise SystemExit(f"unknown doc_id {a.doc_id}; add it to corpus/registry.json first")
    src = Path(a.file)
    dest = corpus.locate(entry)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dest.resolve():
        shutil.copyfile(src, dest)
    entry["sha256"] = corpus.sha256(dest)
    entry["retrieved"] = entry.get("retrieved") or time.strftime("%Y-%m-%d")
    corpus.REGISTRY.write_text(json.dumps(reg, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"registered {a.doc_id}: {dest} sha256={entry['sha256'][:16]}...")


def cmd_register_dir(a) -> None:
    out = corpus.register_dir(Path(a.folder))
    for doc_id, name in out["placed"].items():
        print(f"  {doc_id:<32} <- {name}")
    print(f"placed {len(out['placed'])} document(s); still missing: {out['missing'] or 'none'}")


def cmd_rehash(a) -> None:
    for doc_id in a.doc_ids:
        print(f"{doc_id}: sha256 {corpus.record_hash(doc_id)[:16]}... recorded")


def cmd_build(a) -> None:
    from finguard.rag.index import make_embedder
    index, report = corpus.build(embedder=make_embedder(a.embedder) if a.embedder else None)
    print(json.dumps({k: v for k, v in report.items() if k != "documents"}, indent=1))
    for d in report["documents"]:
        print(f"  {d['doc_id']:<36} pages={d['pages']} headings={d['headings']} tables={d['tables']} chunks={d['chunks']}")


def cmd_freeze(a) -> None:
    print(json.dumps(ev.freeze(note=a.note), indent=1))


def cmd_evaluate(a) -> None:
    frozen = ev.freeze(note="re-frozen at evaluation") if a.refreeze else ev.check_frozen()
    index = corpus.load_index()
    res = ev.evaluate(index, ev.load_questions(), variants=a.variants)
    payload = ev.write_report(res, index, frozen)
    if a.note:
        ev.log_run(payload, a.note)
    (corpus.INDEX_DIR / "abstention.json").write_text(json.dumps(res["abstention"] | {"chosen_variant": res["chosen_variant"]}, indent=1))
    print(res["summary"].round(3).to_string(index=False))
    print(res["by_split"].round(3).to_string(index=False))
    print("chosen variant (development half):", res["chosen_variant"])
    print(json.dumps(payload["abstention"], indent=1))
    if res["unmapped_gold"]:
        print("WARNING: gold quotes not found in any chunk:", res["unmapped_gold"])


def cmd_search(a) -> None:
    index = corpus.load_index()
    abst = corpus.INDEX_DIR / "abstention.json"
    cfg = json.loads(abst.read_text()) if abst.exists() else {}
    variant = a.variant or cfg.get("variant", "hybrid_rerank")
    out = Retriever(index, threshold=cfg.get("threshold"), check_names=cfg.get("grounding_check", True)).search(
        a.query, k=a.k, variant=variant, tier=a.tier, jurisdiction=a.jurisdiction)
    for p in out.get("passages", []):
        p["text"] = p["text"][:400] + ("..." if len(p["text"]) > 400 else "")
    print(json.dumps(out, indent=1, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("register"); r.add_argument("file"); r.add_argument("--doc-id", required=True); r.set_defaults(fn=cmd_register)
    rd = sub.add_parser("register-dir"); rd.add_argument("folder"); rd.set_defaults(fn=cmd_register_dir)
    h = sub.add_parser("rehash"); h.add_argument("doc_ids", nargs="+"); h.set_defaults(fn=cmd_rehash)
    b = sub.add_parser("build"); b.add_argument("--embedder", default=None); b.set_defaults(fn=cmd_build)
    f = sub.add_parser("freeze"); f.add_argument("--note", default=""); f.set_defaults(fn=cmd_freeze)
    e = sub.add_parser("evaluate"); e.add_argument("--variants", nargs="+", default=list(VARIANTS)); e.add_argument("--refreeze", action="store_true")
    e.add_argument("--note", default="", help="record this run in reports/tables/phase3_retrieval_runs.json")
    e.set_defaults(fn=cmd_evaluate)
    s = sub.add_parser("search"); s.add_argument("query"); s.add_argument("--tier", type=int); s.add_argument("--jurisdiction")
    s.add_argument("--variant", default=None, choices=VARIANTS); s.add_argument("--k", type=int, default=5); s.set_defaults(fn=cmd_search)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
