"""Knowledge-base pipeline on a small synthetic corpus (no downloads, no network)."""
import json
import tempfile
from pathlib import Path

import numpy as np

from finguard.rag import corpus
from finguard.rag import evaluate as ev
from finguard.rag.chunk import MAX_TOKENS, chunk_blocks
from finguard.rag.index import BM25, LsaEmbedder
from finguard.rag.parse import Block, parse_markdown
from finguard.rag.retrieve import Retriever, identifiers, query_names
from finguard.rag.text import approx_tokens, clean, light_stem, normalize_for_match, tokens

DOC_A = """# Card disputes guide

## Reason code 10.4

Condition 10.4 covers other fraud in a card-absent environment. The cardholder states that the transaction was not authorised.
The merchant may respond with compelling evidence such as proof of delivery to the cardholder's address.

## Time limits

The acquirer must respond within 30 days of the dispute. Late responses are not accepted.

| Item | Days |
| --- | --- |
| Response | 30 |
"""

DOC_B = """# Reporting obligations

## Recommendation 20

If a financial institution suspects that funds are the proceeds of a criminal activity, it should report its suspicions promptly
to the financial intelligence unit.

## Recommendation 21

Financial institutions and their staff should be prohibited from disclosing the fact that a suspicious transaction report is
being filed. This is the tipping-off rule.
"""


def _corpus(tmp: Path) -> Path:
    docs = tmp / "docs"
    docs.mkdir()
    (docs / "a.md").write_text(DOC_A, encoding="utf-8")
    (docs / "b.md").write_text(DOC_B, encoding="utf-8")
    reg = {"documents": [
        {"doc_id": "a", "title": "Card disputes guide", "publisher": "Test", "url": "-", "published": "2024", "jurisdiction": "Card network",
         "tier": 1, "type": "guide", "authority": "test", "language": "en", "file": "a.md", "source": "project", "sha256": None},
        {"doc_id": "b", "title": "Reporting obligations", "publisher": "Test", "url": "-", "published": "2023", "jurisdiction": "International",
         "tier": 2, "type": "standard", "authority": "test", "language": "en", "file": "b.md", "source": "project", "sha256": None}]}
    (tmp / "registry.json").write_text(json.dumps(reg), encoding="utf-8")
    return tmp


def _index(tmp: Path):
    _corpus(tmp)
    return corpus.build(registry=tmp / "registry.json", project_dir=tmp / "docs", raw_dir=tmp / "raw",
                        index_dir=tmp / "index", embedder=LsaEmbedder(dims=16))[0]


def test_tokens_keep_identifiers_whole_and_split():
    t = tokens("Visa condition 10.4 and FATF R.16 (EBA/GL/2018/05); la déclaration")
    assert {"10.4", "10", "4", "r.16", "16", "eba/gl/2018/05", "2018", "declaration"} <= set(t)
    assert "and" not in t and "la" not in t
    assert identifiers("What does 10.4 mean under R.16?") == {"10.4", "r.16"}


def test_markdown_parse_and_chunks_respect_sections():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "a.md"
        p.write_text(DOC_A, encoding="utf-8")
        blocks = parse_markdown(p)
    kinds = [b.kind for b in blocks]
    assert kinds.count("heading") == 3 and kinds.count("table") == 1
    chunks = chunk_blocks(blocks, "a")
    # both sub-sections are short (< 300 tokens): they share one chunk, the second heading inline,
    # and the two-row table (< 120 tokens) stays in the text
    assert len(chunks) == 1 and chunks[0].section == "Card disputes guide > Reason code 10.4"
    assert "Time limits." in chunks[0].text and "Response | 30" in chunks[0].text
    assert chunks[0].subsections == ("Time limits",)                                       # cited with the section
    long_text = " ".join(f"Sentence {i} explains the time limit rules for disputes in detail." for i in range(60))
    blocks = [Block("heading", "Guide", None, 1), Block("heading", "Long part", None, 2), Block("text", long_text, None),
              Block("heading", "Short part", None, 2), Block("text", "A short closing note.", None),
              Block("heading", "Annex", None, 1), Block("text", "Annex text.", None)]
    chunks = chunk_blocks(blocks, "b")
    assert chunks[0].section == "Guide > Long part" and all(approx_tokens(c.text) <= MAX_TOKENS * 1.3 for c in chunks)
    assert chunks[-1].section == "Annex" and "Annex text." == chunks[-1].text            # never crosses a top-level section


def test_long_section_gets_overlapping_chunks_within_budget():
    sents = " ".join(f"Sentence {i} talks about chargebacks and evidence in some detail here." for i in range(200))
    chunks = chunk_blocks([Block("heading", "Long", 1, 1), Block("text", sents, 1)], "x")
    sizes = [approx_tokens(c.text) for c in chunks]
    assert len(chunks) >= 4 and max(sizes) <= MAX_TOKENS
    assert all(c1.text.split(". ")[-1] in c2.text for c1, c2 in zip(chunks, chunks[1:]))       # overlap carried over


def test_bm25_matches_hand_computation():
    bm = BM25().fit(["fraud chargeback", "chargeback", "report suspicion"])
    s = bm.scores("fraud")
    n, df = 3, 1
    idf = np.log((n - df + 0.5) / (df + 0.5) + 1)
    tf, dl, avg = 1, 2, (2 + 1 + 2) / 3
    expected = idf * tf * 2.5 / (tf + 1.5 * (1 - 0.75 + 0.75 * dl / avg))
    assert abs(s[0] - expected) < 1e-12 and s[1] == 0 and s[2] == 0


def test_build_search_filters_citations_and_abstention():
    with tempfile.TemporaryDirectory() as d:
        index = _index(Path(d))
        r = Retriever(index)
        out = r.search("what does reason code 10.4 cover", k=3)
        assert out["status"] == "ok" and out["passages"][0]["doc_id"] == "a"
        p = out["passages"][0]
        assert p["label"] == "S1" and p["section"] and p["publisher"] == "Test" and p["tier"] == 1
        only_b = r.search("reason code 10.4", tier=2)
        assert all(x["doc_id"] == "b" for x in only_b["passages"])
        strict = Retriever(index, threshold=0.99).search("zebra migration patterns in winter")
        assert strict["status"] == "not_covered" and strict["passages"] == []


def test_registry_rejects_changed_files():
    with tempfile.TemporaryDirectory() as d:
        tmp = _corpus(Path(d))
        reg = json.loads((tmp / "registry.json").read_text(encoding="utf-8"))
        reg["documents"][0]["sha256"] = "0" * 64
        (tmp / "registry.json").write_text(json.dumps(reg), encoding="utf-8")
        _, report = corpus.build(registry=tmp / "registry.json", project_dir=tmp / "docs", raw_dir=tmp / "raw",
                                 index_dir=tmp / "index", embedder=LsaEmbedder(dims=8))
        assert [m["doc_id"] for m in report["hash_mismatch"]] == ["a"]
        assert [x["doc_id"] for x in report["documents"]] == ["b"]


def test_evaluation_maps_gold_quotes_and_refuses_unfrozen_changes():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        index = _index(tmp)
        qs = [{"id": "q1", "question": "Who must not be told that a report is filed?", "answerable": True, "split": "dev",
               "gold": [{"doc_id": "b", "quote": "prohibited from disclosing the fact that a suspicious transaction report"}]},
              {"id": "q2", "question": "How long does the acquirer have to answer a dispute?", "answerable": True, "split": "test",
               "gold": [{"doc_id": "a", "quote": "respond within 30 days"}]},
              {"id": "u1", "question": "What is the boiling point of ethanol?", "answerable": False, "split": "dev", "gold": []}]
        qpath, fpath = tmp / "q.jsonl", tmp / "frozen.json"
        qpath.write_text("\n".join(json.dumps(q) for q in qs), encoding="utf-8")
        ev.freeze(qpath, fpath)
        ev.check_frozen(qpath, fpath)
        res = ev.evaluate(index, ev.load_questions(qpath), variants=("bm25", "hybrid_rerank"))
        assert res["unmapped_gold"] == []
        assert res["summary"].set_index("variant").loc["bm25", "hit_at_5"] == 1.0
        qpath.write_text(qpath.read_text(encoding="utf-8") + "\n")
        try:
            ev.check_frozen(qpath, fpath)
            raise AssertionError("a changed question file must be refused")
        except RuntimeError:
            pass


def test_normalize_for_match_ignores_hyphenation_and_case():
    assert normalize_for_match("Regula-\ntion  of “Payments”") == normalize_for_match("regulation of payments")


def test_plural_folding_and_marks_leave_identifiers_alone():
    assert [light_stem(w) for w in ("units", "policies", "psps", "process", "status", "analysis", "principaux", "fees")] == \
        ["unit", "policy", "psp", "process", "status", "analysis", "principal", "fee"]
    assert tokens("financial intelligence units") == tokens("financial intelligence unit")
    assert tokens("conditions 10.4 R.16")[:3] == ["condition", "10.4", "10"] and "r.16" in tokens("R.16")
    assert clean("FraudClassifier\u2120 model") == "FraudClassifier model" and clean("ScamClassifierSM model") == "ScamClassifier model"
    assert "\uf0b7" not in clean("\uf0b7 payment service providers")


def test_names_in_the_query_must_be_grounded_in_the_passages():
    assert query_names("What is the Mastercard reason code for CNP fraud under R.16?") == ["Mastercard", "CNP", "R.16"]
    assert query_names("How long does the acquirer have?") == []                 # a sentence-initial capital is not a name
    with tempfile.TemporaryDirectory() as d:
        index = _index(Path(d))
        r = Retriever(index)
        ok = r.search("What does reason code 10.4 cover?")
        assert ok["status"] == "ok" and ok["ungrounded_names"] == []
        out = r.search("What does Mastercard reason code 4837 cover?")
        assert out["status"] == "not_covered" and out["ungrounded_names"] == ["Mastercard"] and "Mastercard" in out["message"]
        assert Retriever(index, check_names=False).search("What does Mastercard reason code 4837 cover?")["status"] == "ok"
        # a name found only in the document's metadata (publisher, jurisdiction) counts as grounded
        assert r.search("What does the International standard say about tipping-off?")["status"] == "ok"


def test_threshold_tuning_counts_flagged_questions_as_abstentions():
    conf = np.array([0.9, 0.8, 0.7, 0.75, 0.6])
    answerable = np.array([True, True, True, False, False])
    assert ev._tune_threshold(conf, answerable) == 0.8                          # 0.75 and 0.6 below, 0.7 a false abstention
    assert ev._tune_threshold(conf, answerable, np.array([False, False, False, True, False])) == 0.7


def test_evaluation_confidence_matches_what_the_tool_sees():
    with tempfile.TemporaryDirectory() as d:
        index = _index(Path(d))
        r = Retriever(index)
        q = "How long does the acquirer have to answer a dispute?"
        _, _, conf10 = r.rank(q, "bm25", k=10, conf_k=5)
        assert abs(conf10 - r.search(q, k=5, variant="bm25")["confidence"]) < 1e-4


def test_register_dir_places_files_by_hash_whatever_their_name():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        downloads = tmp / "downloads"
        downloads.mkdir()
        (downloads / "Final_Report_v2 (1).pdf").write_bytes(b"%PDF-1.4 published document A")
        (downloads / "unrelated.pdf").write_bytes(b"%PDF-1.4 something else")
        reg = {"documents": [
            {"doc_id": "pub_a", "title": "A", "publisher": "P", "url": "https://example.org/a.pdf", "published": "2024",
             "jurisdiction": "EU", "tier": 1, "type": "guidelines", "authority": "x", "language": "en", "file": "pub_a.pdf",
             "sha256": corpus.sha256(downloads / "Final_Report_v2 (1).pdf")},
            {"doc_id": "pub_b", "title": "B", "publisher": "P", "url": "https://example.org/b.pdf", "published": "2024",
             "jurisdiction": "EU", "tier": 1, "type": "guidelines", "authority": "x", "language": "en", "file": "pub_b.pdf",
             "sha256": "0" * 64}]}
        (tmp / "registry.json").write_text(json.dumps(reg), encoding="utf-8")
        out = corpus.register_dir(downloads, registry=tmp / "registry.json", raw_dir=tmp / "raw")
        assert out == {"placed": {"pub_a": "Final_Report_v2 (1).pdf"}, "missing": ["pub_b"]}
        assert (tmp / "raw" / "pub_a.pdf").read_bytes() == b"%PDF-1.4 published document A"


def test_tool_uses_the_reranker_and_index_the_threshold_was_tuned_on():
    from finguard.rag.tool import deployed
    with tempfile.TemporaryDirectory() as d:
        index = _index(Path(d))
        cfg = {"variant": "bm25", "threshold": 0.5, "grounding_check": True, "reranker": "lexical-v1",
               "index_version": index.version}
        (Path(d) / "index" / "abstention.json").write_text(json.dumps(cfg), encoding="utf-8")
        retriever, variant = deployed(Path(d) / "index")
        assert variant == "bm25" and retriever.reranker.name == "lexical-v1" and retriever.threshold == 0.5
        cfg["index_version"] = "rebuilt-since"
        (Path(d) / "index" / "abstention.json").write_text(json.dumps(cfg), encoding="utf-8")
        try:
            deployed(Path(d) / "index")
            raise AssertionError("a threshold tuned on another index must be refused")
        except RuntimeError:
            pass
