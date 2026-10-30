"""Retrieval evaluation (Project Report v3.0, section 7.3).

Question set: ``corpus/questions.jsonl``, one JSON object per line:

    {"id": "q01", "question": "...", "answerable": true, "split": "dev" | "test",
     "gold": [{"doc_id": "...", "quote": "a short exact span of the gold passage"}]}

The set is frozen before the index is evaluated: ``corpus/questions_frozen.json`` stores its
SHA-256 and date, and evaluation stops if the file changed afterwards (``--refreeze`` records
a new version deliberately). A chunk is a gold hit when it belongs to the gold document and
contains the quote (compared after normalisation), so the gold survives re-chunking.

Metrics, on answerable questions: Hit@1, Hit@5, MRR@10 for each retrieval variant, and
Doc@5 (the gold document among the top 5, a looser check that credits an equivalent passage).
Abstention: the grounding check (names in the question must appear in the returned passages)
plus a confidence threshold tuned on the ``dev`` split (balanced accuracy of answering
answerable questions and abstaining on unanswerable ones), reported on ``test`` and on all
questions; the threshold-only rule is reported too, as an ablation. Exit criteria: Hit@5 >= 0.80; at least 8 of 10 unanswerable questions
answered "not covered by the sources".
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from finguard import config
from finguard.rag.corpus import CORPUS_DIR
from finguard.rag.index import KnowledgeIndex
from finguard.rag.retrieve import VARIANTS, Retriever
from finguard.rag.text import normalize_for_match

QUESTIONS = CORPUS_DIR / "questions.jsonl"
FROZEN = CORPUS_DIR / "questions_frozen.json"


def load_questions(path: Path = QUESTIONS) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def file_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(path: Path = QUESTIONS, frozen: Path = FROZEN, note: str = "") -> dict:
    rec = {"sha256": file_hash(path), "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"), "n_questions": len(load_questions(path)),
           "note": note}
    Path(frozen).write_text(json.dumps(rec, indent=1))
    return rec


def check_frozen(path: Path = QUESTIONS, frozen: Path = FROZEN) -> dict:
    if not Path(frozen).exists():
        raise RuntimeError("the question set is not frozen: run `python -m finguard.rag freeze` before evaluating")
    rec = json.loads(Path(frozen).read_text())
    if rec["sha256"] != file_hash(path):
        raise RuntimeError("questions.jsonl changed after it was frozen; re-freeze deliberately with --refreeze")
    return rec


def gold_chunks(index: KnowledgeIndex, q: dict) -> set[int]:
    hits = set()
    norm = [normalize_for_match(c["text"]) for c in index.chunks] if not hasattr(index, "_norm") else index._norm
    index._norm = norm
    for g in q.get("gold", []):
        quote = normalize_for_match(g["quote"])
        for i, c in enumerate(index.chunks):
            if c["doc_id"] == g["doc_id"] and quote in norm[i]:
                hits.add(i)
    return hits


def _tune_threshold(conf: np.ndarray, answerable: np.ndarray, flagged: np.ndarray | None = None) -> float:
    """Threshold with the best balanced accuracy; ``flagged`` questions abstain whatever the threshold.
    Ties go to the lowest threshold (fewest false abstentions)."""
    flagged = np.zeros(len(conf), dtype=bool) if flagged is None else np.asarray(flagged, dtype=bool)
    cands = np.unique(np.r_[0.0, conf, conf.max() + 1e-6])
    best, best_t = -1.0, 0.0
    for t in cands:
        answer = (conf >= t) & ~flagged
        ans_ok = answer[answerable].mean() if answerable.any() else 1.0
        abs_ok = (~answer[~answerable]).mean() if (~answerable).any() else 1.0
        score = (ans_ok + abs_ok) / 2
        if score > best + 1e-12:
            best, best_t = score, float(t)
    return best_t


TARGET_HIT5 = 0.80


def choose_variant(by_split: pd.DataFrame, variants=VARIANTS, target: float = TARGET_HIT5) -> str:
    """The simplest variant (in the order bm25, dense, hybrid, hybrid_rerank) whose Hit@5 on the
    development half reaches the target; the best development Hit@5 if none does."""
    dev = by_split[by_split.split == "dev"].set_index("variant").hit_at_5
    ok = [v for v in variants if v in dev.index and dev[v] >= target - 1e-9]
    return ok[0] if ok else str(dev.idxmax())


def evaluate(index: KnowledgeIndex, questions: list[dict], variants=VARIANTS, k: int = 5, abstain_variant: str = "auto"):
    """``abstain_variant="auto"``: abstention is tuned for the variant chosen on the development half."""
    retr = Retriever(index)
    rows = []
    unmapped = []
    for q in questions:
        gold = gold_chunks(index, q) if q["answerable"] else set()
        if q["answerable"] and not gold:
            unmapped.append(q["id"])
        gold_docs = {g["doc_id"] for g in q.get("gold", [])}
        for v in variants:
            # ranks are read down to 10 (MRR@10); confidence and grounding use the top k the tool returns
            order, _, conf = retr.rank(q["question"], v, k=10, conf_k=k, **q.get("filters", {}))
            ranks = [r for r, i in enumerate(order, 1) if int(i) in gold]
            first = ranks[0] if ranks else None
            docs5 = [index.chunks[int(i)]["doc_id"] for i in order[:k]]
            rows.append({"id": q["id"], "split": q.get("split", "test"), "answerable": q["answerable"], "variant": v,
                         "first_gold_rank": first, "hit@1": first == 1, "hit@5": first is not None and first <= k,
                         "doc@5": bool(gold_docs & set(docs5)), "rr@10": 1.0 / first if first else 0.0, "confidence": conf,
                         "ungrounded": "|".join(retr.ungrounded(q["question"], order[:k])),
                         "top_doc": docs5[0] if docs5 else None})
    det = pd.DataFrame(rows)
    ans = det[det.answerable]
    summary = ans.groupby("variant").agg(questions=("id", "size"), hit_at_1=("hit@1", "mean"), hit_at_5=("hit@5", "mean"),
                                        doc_at_5=("doc@5", "mean"), mrr_at_10=("rr@10", "mean")).reindex(list(variants))
    summary_by_split = ans.groupby(["variant", "split"]).agg(questions=("id", "size"), hit_at_1=("hit@1", "mean"),
                                                             hit_at_5=("hit@5", "mean"), doc_at_5=("doc@5", "mean"),
                                                             mrr_at_10=("rr@10", "mean")).reset_index()

    chosen_variant = choose_variant(summary_by_split, variants)
    if abstain_variant == "auto":
        abstain_variant = chosen_variant
    a = det[det.variant == abstain_variant].set_index("id")
    a["flagged"] = a.ungrounded.fillna("").str.len() > 0
    dev = a[a.split == "dev"]

    def rule(t: float, use_names: bool) -> dict:
        def part_stats(part: pd.DataFrame) -> dict:
            abstain = (part.confidence < t) | (part.flagged if use_names else False)
            un, an = ~part.answerable, part.answerable
            return {"unanswerable": int(un.sum()), "correct_abstentions": int((abstain & un).sum()),
                    "answerable": int(an.sum()), "false_abstentions": int((abstain & an).sum()),
                    "hit@5_after_abstention": float((~abstain & part["hit@5"])[an].mean()) if an.any() else None}
        return {"threshold": t, "grounding_check": use_names, "dev": part_stats(dev),
                "test": part_stats(a[a.split == "test"]), "all": part_stats(a)}

    # Each check is calibrated on its own: the threshold is tuned on dev without the grounding
    # check, then the two are combined (abstain if either fires). Tuning the threshold jointly
    # with the check lets five dev questions decide both rules; it is kept as an ablation
    # because it was tried (run 3) and over-fitted: dev 5/5, test 2/5.
    t_only = _tune_threshold(dev.confidence.to_numpy(), dev.answerable.to_numpy()) if len(dev) else 0.0
    t_joint = _tune_threshold(dev.confidence.to_numpy(), dev.answerable.to_numpy(), dev.flagged.to_numpy()) if len(dev) else 0.0
    chosen = rule(t_only, True)
    abstention = {"variant": abstain_variant, "tuned_on": "dev",
                  "rule": "abstain if a name in the question is in no returned passage, or if confidence < threshold "
                          "(threshold tuned on dev alone)", **chosen,
                  "ablations": {"threshold_only": rule(t_only, False), "grounding_only": rule(0.0, True),
                                "jointly_tuned": rule(t_joint, True)}}
    return {"summary": summary.reset_index(), "by_split": summary_by_split, "details": det, "abstention": abstention,
            "chosen_variant": chosen_variant, "unmapped_gold": unmapped}


def write_report(res: dict, index: KnowledgeIndex, frozen: dict, out_dir: Path = config.TABLE_DIR) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res["summary"].to_csv(out_dir / "phase3_retrieval_summary.csv", index=False)
    res["details"].to_csv(out_dir / "phase3_retrieval_details.csv", index=False)
    payload = {"index_version": index.version, "embedder": index.embedder.name, "questions": frozen,
               "summary": res["summary"].to_dict("records"), "by_split": res["by_split"].to_dict("records"),
               "chosen_variant": res["chosen_variant"],
               "selection_rule": f"simplest variant with development-half Hit@5 >= {TARGET_HIT5:.2f}",
               "abstention": res["abstention"], "unmapped_gold": res["unmapped_gold"]}
    (out_dir / "phase3_retrieval_results.json").write_text(json.dumps(payload, indent=1, default=float))
    return payload


RUNS = config.TABLE_DIR / "phase3_retrieval_runs.json"


def log_run(payload: dict, note: str, path: Path = RUNS) -> list[dict]:
    """Append one evaluation run (what changed, overall and per-split results) to the run log, so the
    report can show every evaluation of the frozen question set, not only the last one."""
    runs = json.loads(Path(path).read_text()) if Path(path).exists() else []
    runs.append({"run": len(runs) + 1, "at": time.strftime("%Y-%m-%d %H:%M:%S"), "note": note,
                 "index_version": payload["index_version"], "embedder": payload["embedder"],
                 "chosen_variant": payload.get("chosen_variant"), "abstention_variant": payload["abstention"]["variant"],
                 "summary": payload["summary"], "by_split": payload["by_split"],
                 "abstention": {k: payload["abstention"][k] for k in ("threshold", "grounding_check", "dev", "test", "all")},
                 "ablations": {k: {kk: v[kk] for kk in ("threshold", "dev", "test", "all")}
                               for k, v in payload["abstention"].get("ablations", {}).items()}})
    Path(path).write_text(json.dumps(runs, indent=1, default=float))
    return runs
