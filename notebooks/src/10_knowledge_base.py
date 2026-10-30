# %% [markdown]
# # 10 · Knowledge base: corpus, hybrid retrieval, abstention and evaluation
#
# **FinGuard AI, Phase 3.** Run `python -m finguard.rag build` and `python -m finguard.rag evaluate`
# first; this notebook reads their outputs and shows the `policy_search` tool.
#
# The knowledge base answers three questions about an alert: what the pattern means, what to
# check next, and what the institution's obligations are if fraud is confirmed. It does not
# "verify a transaction against regulation". Exit criteria (Project Report v3.0): Hit@5 >= 0.80
# and at least 8 of 10 unanswerable questions answered "not covered by the sources".

# %%
import sys, pathlib, json
ROOT = pathlib.Path.cwd().resolve().parent if pathlib.Path.cwd().name == "notebooks" else pathlib.Path.cwd().resolve()
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from finguard import config
from finguard.evaluation import figures as F
from finguard.rag import corpus, evaluate as ev
from finguard.rag.text import approx_tokens

pd.set_option("display.max_columns", 30, "display.width", 200, "display.max_colwidth", 110)
T = config.TABLE_DIR
reg = pd.DataFrame(corpus.load_registry())
build = json.loads((corpus.INDEX_DIR / "build_report.json").read_text())
index = corpus.load_index()
print(f"index {index.version} | embedder {index.embedder.name} | {len(index.chunks):,} chunks")

# %% [markdown]
# ## 1. The corpus
# Every document has a registry entry (publisher, date, jurisdiction, tier, language, URL and
# SHA-256). Published documents are checked against their recorded hash on every build.

# %%
docs = pd.DataFrame(build["documents"]).merge(reg[["doc_id", "title", "publisher", "published", "jurisdiction", "tier", "language", "sha256"]],
                                              on="doc_id", how="right")
docs["sha256"] = docs.sha256.fillna("").str[:12]
print("missing documents:", [m["doc_id"] for m in build["missing"]] or "none", "| hash mismatches:", build["hash_mismatch"] or "none")
docs[["doc_id", "publisher", "published", "jurisdiction", "tier", "language", "pages", "headings", "tables", "chunks", "sha256"]]

# %%
sizes = pd.Series([approx_tokens(c["text"]) for c in index.chunks])
kinds = pd.Series([c["kind"] for c in index.chunks]).value_counts()
print("chunk size (approximate tokens):", sizes.describe(percentiles=[0.1, 0.5, 0.9]).round(0).to_dict())
print("chunk kinds:", kinds.to_dict())
print("share of text chunks within 300-500 tokens:",
      f"{((sizes >= 300) & (sizes <= 500))[pd.Series([c['kind'] == 'text' for c in index.chunks])].mean():.1%}")

# %% [markdown]
# ## 2. The question set
# Written before the index was evaluated and frozen by hash. Each answerable question names
# its gold passage by document and a quoted span; unanswerable questions are in the domain but
# not covered by the corpus.

# %%
frozen = ev.check_frozen()
qs = pd.DataFrame(ev.load_questions())
qs["gold_doc"] = qs.gold.map(lambda g: g[0]["doc_id"] if g else "(none)")
print("frozen:", frozen)
pd.crosstab(qs.gold_doc, [qs.answerable, qs.split], margins=True)

# %% [markdown]
# ## 3. Every evaluation of the frozen set
# The question set was frozen before the first evaluation. Fixes after it were chosen by
# reading misses on the development half only; the held-out half is reported each time but
# never used to choose. Run 3 is kept although it was replaced: it shows an abstention rule
# that fitted the five development unanswerables and failed on the held-out ones.

# %%
runs = json.loads((T / "phase3_retrieval_runs.json").read_text())
log = []
for r in runs:
    v = r.get("chosen_variant") or "hybrid_rerank"                 # runs 1-4 used hybrid + re-ranking throughout
    s = {x["variant"]: x for x in r["summary"]}[v]
    sp = {x["split"]: x["hit_at_5"] for x in (r["by_split"] or []) if x["variant"] == v}
    ab = r["abstention"]
    log.append({"run": r["run"], "variant": v, "hit@5 all": s["hit_at_5"], "hit@5 dev": sp.get("dev"), "hit@5 test": sp.get("test"),
                "abstained (of 10)": ab["all"]["correct_abstentions"], "false abstentions (of 50)": ab["all"].get("false_abstentions"),
                "what changed": r["note"]})
pd.DataFrame(log).set_index("run")

# %% [markdown]
# ## 4. Retrieval variants
# The keyword, vector, fused and re-ranked variants on the answerable questions. In this build
# environment the dense model is LSA (TF-IDF + SVD) and the re-ranker is the lexical blend,
# because no embedding or cross-encoder model could be downloaded; with `FINGUARD_EMBEDDER`
# and `FINGUARD_RERANKER` set, the same evaluation compares the neural ones. The variant is
# chosen on the development half: the simplest one at or above 0.80 there (the rule written in
# `rag/retrieve.py` before the first run), and `policy_search` uses it.

# %%
res = json.loads((T / "phase3_retrieval_results.json").read_text())
summary = pd.DataFrame(res["summary"])
by_split = pd.DataFrame(res["by_split"])
chosen = res["chosen_variant"]
fig = F.fig27_retrieval(summary, by_split, chosen)
summary.round(3)

# %%
def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h

det = pd.read_csv(T / "phase3_retrieval_details.csv")
rows = []
for part, sel in [("dev", det.split == "dev"), ("test", det.split == "test"), ("all", det.split.notna())]:
    a = det[sel & det.answerable & (det.variant == chosen)]
    k, n = int(a["hit@5"].sum()), len(a)
    lo, hi = wilson(k, n)
    rows.append({"half": part, "questions": n, "gold in top 5": k, "hit@5": round(k / n, 3), "95% interval": f"{lo:.2f}-{hi:.2f}",
                 "doc@5": round(a["doc@5"].mean(), 3)})
print("chosen on the development half:", chosen)
pd.DataFrame(rows).set_index("half")

# %% [markdown]
# Hit@5 is exactly at the target on all 50 questions and the interval is wide: with 23
# held-out questions one question moves Hit@5 by 0.04. All four variants reach 0.80 overall;
# they differ in how high the gold passage ranks (MRR). Doc@5 (the gold document among the top
# five) is close to 1, so most misses retrieve the right document but a neighbouring passage.

# %% [markdown]
# ## 5. Abstention
# Two checks, either of which abstains: a name in the question (a scheme, country, body or
# identifier) appears in none of the returned passages or their document's metadata; or the
# confidence is below a threshold tuned on the development half alone.

# %%
ab = res["abstention"]
a = det[det.variant == ab["variant"]].reset_index(drop=True)
flagged = a.ungrounded.fillna("").str.len().gt(0).to_numpy()
fig = F.fig28_abstention(a.confidence.to_numpy(), a.answerable.to_numpy(), flagged, ab["threshold"],
                         ab["all"]["correct_abstentions"], ab["all"]["unanswerable"], ab["all"]["false_abstentions"],
                         ab["all"]["answerable"])
rules = {"chosen: names + threshold": ab, **{f"ablation: {k.replace('_', ' ')}": v for k, v in ab["ablations"].items()}}
pd.DataFrame([{"rule": name, "threshold": round(r["threshold"], 3),
               **{f"{h} abstained": f"{r[h]['correct_abstentions']}/{r[h]['unanswerable']}" for h in ("dev", "test", "all")},
               "false abstentions (all)": f"{r['all']['false_abstentions']}/{r['all']['answerable']}",
               "hit@5 after abstention": round(r["all"]["hit@5_after_abstention"], 3)} for name, r in rules.items()]).set_index("rule")

# %%
qtext = pd.DataFrame(ev.load_questions()).set_index("id").question
u = a[~a.answerable].assign(question=lambda d: d.id.map(qtext), abstained=lambda d: (d.confidence < ab["threshold"]) | d.ungrounded.notna())
u[["id", "split", "question", "confidence", "ungrounded", "abstained"]]

# %% [markdown]
# The threshold is a trade-off between wrong answers and declined questions. The operating
# points below (grounding check always on) let the team choose; the evaluation's rule picks the
# best balanced accuracy on the development half.

# %%
ops = []
for t in sorted({0.0, 0.55, 0.60, 0.65, 0.70, round(ab["threshold"], 3)}):
    row = {"threshold": t}
    for h, sel in [("dev", a.split == "dev"), ("test", a.split == "test"), ("all", a.split.notna())]:
        x = a[sel]
        abstain = (x.confidence < t) | x.ungrounded.notna()
        row[f"{h}: abstained"] = f"{int((abstain & ~x.answerable).sum())}/{int((~x.answerable).sum())}"
        row[f"{h}: declined answerable"] = f"{int((abstain & x.answerable).sum())}/{int(x.answerable.sum())}"
    ops.append(row)
pd.DataFrame(ops).set_index("threshold")

# %% [markdown]
# ## 6. Where retrieval misses

# %%
miss = det[(det.variant == chosen) & det.answerable & ~det["hit@5"]].merge(qs[["id", "question", "gold_doc"]], on="id")
miss[["id", "split", "question", "gold_doc", "top_doc", "first_gold_rank", "doc@5"]]

# %% [markdown]
# ## 7. The `policy_search` tool
# What the agents will receive: cited passages labelled S1-S5, or "not covered by the sources"
# with the names the sources do not mention.

# %%
from finguard.rag.tool import SCHEMA, policy_search
print(json.dumps(SCHEMA, indent=1))
examples = {}
for q, kw in [("What does Visa dispute condition 10.4 cover and what evidence can the merchant provide?", {}),
              ("Which FATF Recommendation requires reporting suspicious transactions?", {"tier": 2}),
              ("What is the Mastercard reason code for card-not-present fraud?", {}),
              ("What does the Mastercard Ethoca alert service do?", {})]:
    out = policy_search(q, **kw)
    examples[q] = {k: out[k] for k in ("status", "confidence", "ungrounded_names")} | {
        "passages": [{k: p[k] for k in ("label", "doc_id", "section", "pages")} for p in out["passages"][:3]]}
    print("\nQ:", q, "| status:", out["status"], "| confidence:", out["confidence"], "| ungrounded:", out["ungrounded_names"])
    if out["status"] != "ok":
        print("  ", out["message"])
    for p in out["passages"][:3]:
        print(f"  [{p['label']}] {p['title']} ({p['publisher']}, {p['published']}) | {p['section'][:70]} | pages {p['pages']}")
        if p["subsections"]:
            print("       also covers:", "; ".join(p["subsections"])[:150])
        print("      ", p["text"][:220].replace("\n", " "), "...")

# %%
phase3 = {"index_version": res["index_version"], "embedder": res["embedder"], "n_chunks": len(index.chunks),
          "documents": len(build["documents"]), "chosen_variant": chosen, "hit_at_5": rows, "summary": res["summary"],
          "by_split": res["by_split"], "abstention": {k: ab[k] for k in ("rule", "threshold", "dev", "test", "all")},
          "ablations": ab["ablations"], "misses": miss[["id", "split", "gold_doc", "top_doc", "first_gold_rank"]].to_dict("records"),
          "runs": log, "operating_points": ops, "tool_examples": examples}
(T / "phase3_summary.json").write_text(json.dumps(phase3, indent=1, default=float))
print("saved reports/tables/phase3_summary.json")
