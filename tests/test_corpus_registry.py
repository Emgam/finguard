"""The versioned corpus and the frozen question set (run in CI: needs only files kept in Git).
The retrieval regression check runs when the index has been built locally (skipped otherwise)."""
import json
from unittest import SkipTest

from finguard.rag import corpus
from finguard.rag import evaluate as ev


def test_registry_is_complete_and_project_documents_match_their_hashes():
    entries = corpus.load_registry()
    assert 10 <= len(entries) <= 14
    for e in entries:
        assert e["tier"] in (1, 2, 3) and e["language"] in ("en", "fr")
        if e.get("source") == "project":
            path = corpus.locate(e)
            assert path.exists(), path
            assert e["sha256"] == corpus.sha256(path), f"{e['doc_id']} changed: run `python -m finguard.rag rehash {e['doc_id']}`"
        else:
            assert e["sha256"] and len(e["sha256"]) == 64 and e["url"].startswith("http")


def test_question_set_is_frozen_balanced_and_points_at_registered_documents():
    rec = ev.check_frozen()                                   # raises if questions.jsonl changed after freezing
    qs = ev.load_questions()
    assert rec["n_questions"] == len(qs)
    answerable = [q for q in qs if q["answerable"]]
    assert 40 <= len(answerable) <= 50 and len(qs) - len(answerable) == 10
    ids = {e["doc_id"] for e in corpus.load_registry()}
    assert all(g["doc_id"] in ids and len(g["quote"]) >= 15 for q in answerable for g in q["gold"])
    assert {q["split"] for q in qs} == {"dev", "test"} and len({q["id"] for q in qs}) == len(qs)


def test_retrieval_regression_on_the_built_index():
    if not (corpus.INDEX_DIR / "chunks.jsonl").exists():
        raise SkipTest("knowledge-base index not built (run `make kb`)")
    index = corpus.load_index()
    res = ev.evaluate(index, ev.load_questions())
    assert res["unmapped_gold"] == []
    s = res["summary"].set_index("variant").loc[res["chosen_variant"]]
    assert s["hit_at_5"] >= 0.78 and s["doc_at_5"] >= 0.95           # regression floor, just under the reported 0.80
    assert res["abstention"]["all"]["correct_abstentions"] >= 8
