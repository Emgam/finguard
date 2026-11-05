# FinGuard AI

Decision-support prototype for card-not-present payment fraud review, built on the IEEE-CIS
Fraud Detection dataset. **This repository is in Phase 4: an LLM orchestrator and two writing agents
that investigate an alert with six tools and produce a cited case report, inside an evidence policy
checked in code. Phases 1-3 built the data foundation, the calibrated model with reasons and the
cited knowledge base.**

> IEEE-CIS has **no customer identifier**. All grouping of transactions in this project is a
> heuristic **proxy entity** (card1 + addr1 + first-seen day), never a customer.

## Phase 4: orchestrator and agents

```bash
python -m finguard.agents investigate "Investigate alert TX-3554656" --llm anthropic
python -m finguard.agents investigate "What does Visa dispute condition 10.4 cover?" --llm anthropic
python -m finguard.agents evaluate --llm anthropic        # 20 scripted cases, orchestrated vs fallback
```

| Piece | Where |
| --- | --- |
| Six tools with JSON schemas (score, explain, entity history, indicator map, policy search, case store) | `src/finguard/agents/tools.py`, `schemas.py` |
| LLM orchestrator, investigation agent, report agent (versioned prompts) | `agents.py`, `prompts/*.md`, `prompt_store.py` |
| Evidence-policy validator (code, not a prompt) | `policy.py` |
| Graph: LangGraph when installed, built-in engine otherwise; budgets; fallback path | `graph.py` |
| Reports (Facts by template), traces, case database | `report.py`, `trace.py`, `case_store.py` |
| 20 scripted cases and agent metrics | `cases/scripted_cases.jsonl`, `evaluate.py` |

LLM: `--llm anthropic` (`ANTHROPIC_API_KEY`; model `FINGUARD_LLM_MODEL`, default `claude-sonnet-5-5`),
`--llm openai` for any OpenAI-compatible endpoint (OpenAI, Groq, a local Ollama with
`FINGUARD_LLM_BASE_URL=http://localhost:11434/v1`), `--llm scripted` (deterministic test model, no key)
or `--llm none` (template report, no LLM). Reports, traces and the case database go to `data/cases/`.
Budgets: 8 tool calls, 2 re-plans, 2 report drafts and `FINGUARD_AGENT_TIME_BUDGET` seconds
(default 90); past any of them the fixed fallback path writes the report and says so.
Exit criteria (Project Report v3.0): 100% evidence-policy compliance, 100% of numbers traceable, at
least 90% of claims cited. With the scripted test model all three hold on the 20 cases
(`reports/tables/phase4_agent_summary_scripted.json`); that checks the pipeline, not an LLM; the LLM
evaluation is `python -m finguard.agents evaluate --llm anthropic`. Example reports:
`reports/cases_examples/`. Knowledge-base questions are scored against what the deployed retriever
returned (`outcome_follows_evidence`); where the retriever disagrees with a case label, the case is
listed under `retrieval_disagrees_with_label`: a knowledge-base (Phase 3) result that depends on the
embedder, re-ranker and threshold, not an agent error. Quick test run without the two slow tests on
the real data: `python -m pytest -q -k "not twenty_scripted and not retrieval_regression"`.

## Phase 3 results: knowledge base (frozen question set)

| Exit criterion | Target | Result |
| --- | --- | --- |
| Documents with registry and SHA-256 | 10-14 | 11 (8 published, 3 project) |
| Hit@5, 50 answerable questions | ≥ 0.80 | 0.80 (95% interval 0.67-0.89); development half 0.81, held-out half 0.78 |
| Unanswerable questions answered "not covered by the sources" | ≥ 8 of 10 | 9 of 10, while 16 of 50 answerable questions are also declined |

Retrieval here uses lexical models only (BM25; LSA as the dense fallback), because no embedding or
cross-encoder model could be downloaded in the build environment. Set `FINGUARD_EMBEDDER` and
`FINGUARD_RERANKER` to use neural ones. Full report: `reports/knowledge_base_report.md`.

## Phase 2 results (test month, days 152-181, scored once)

| | LightGBM (main) | Logistic regression (strengthened baseline) |
| --- | --- | --- |
| PR-AUC (95% interval) | 0.580 (0.552-0.609, proxy-entity blocks) | 0.459 (0.440-0.477) |
| ROC-AUC | 0.907 | 0.868 |
| Precision / recall, top 1% of traffic | 89.2% / 25.5% | 84.2% / 24.0% |
| Precision / recall, top 5% of traffic | 43.7% / 62.4% | 35.2% / 50.3% |

Isotonic calibration brings test ECE from 0.0178 to 0.0041. Risk bands: High 90.1% fraud, Medium 30.7%,
Low 1.3%; High + Medium = 5.6% of traffic, 64.9% of fraud. Novelty flags are shown next to the score but add
nothing to it; Isolation Forest is not used (PR-AUC 0.117 alone). The model gate passed 11 of 11 checks and
the model is registered as `lgbm_numeric-v1`; promotion to champion is a human step. Full report:
`reports/model_report.md` (section 11 for the completion items).

## Deliverables

| Phase | Deliverable | Where |
| --- | --- | --- |
| 3 | Knowledge-base report | `reports/knowledge_base_report.md` |
| 3 | Corpus registry, project documents, frozen question set | `corpus/` |
| 3 | Retrieval service: parsing, chunking, BM25 / dense / hybrid / re-ranking, abstention, `policy_search` tool | `src/finguard/rag/` |
| 3 | Executed notebook, figures 27-28, tables `phase3_*` | `notebooks/10_knowledge_base.ipynb`, `reports/` |
| 2 | Model report (section 11: novelty flags, Isolation Forest, strengthened baseline, scorer, gate) | `reports/model_report.md` |
| 2 | Pipelines | `python -m finguard.phase2`, `python -m finguard.phase2b` |
| 2 | Scorer and entity store (what `fraud_score` / `explain_transaction` return) | `src/finguard/serving/` |
| 2 | Model gate and registry | `python -m finguard.gate`, `python -m finguard.registry`, `artifacts/registry.json` |
| 2 | Executed notebooks, figures 17-26, tables `phase2_*` | `notebooks/05…09_*.ipynb`, `reports/` |
| 2 | Model bundle (builder, model, calibrator, band thresholds) | `artifacts/phase2/model_bundle.pkl` + `artifacts/phase2/lgbm_numeric/model.txt` |
| 1 | Data-quality report, feature-engineering plan | `reports/data_quality_report.md`, `reports/feature_engineering_plan.md` |
| 1 | Executed notebooks, figures 01-16 | `notebooks/01…04_*.ipynb`, `reports/figures/` |
| 4 | Agents: tools, orchestrator, investigation and report agents, evidence policy, fallback, traces | `src/finguard/agents/`, `python -m finguard.agents` |
| 4 | 20 scripted cases, agent metrics, example reports | `cases/`, `reports/tables/phase4_*`, `reports/cases_examples/` |
| 1-4 | Unit tests | `tests/` (84 tests; those needing the data, the artifacts or the index skip without them) |

| Notebook | Covers |
| --- | --- |
| 01_data_validation | checksums, schema, `id-01`/`id_01` fix, joins, duplicates, types, time, class imbalance, no customer ID |
| 02_eda_missingness_and_distributions | missingness blocks, fraud by segment, amount, identity, email, device, card |
| 03_eda_temporal_drift_leakage | time, drift, category novelty, single-feature signal, redundancy, proxy entity, leakage register |
| 04_feature_inventory_and_plan | per-column inventory, engineered-feature specification, Phase 2 checklist |
| 05_split_features_baseline | time split, leakage guards re-checked on the real data, feature set, logistic-regression baseline |
| 06_lightgbm_calibration_bands | model choice on validation, test results with bootstrap intervals, isotonic calibration, risk bands, model bundle |
| 07_explanations | TreeSHAP reason families, reason cards, stability across 5 retrains |
| 08_robustness_and_scorecard | new vs known proxy entities, 30-day label delay, ablation without entity features, weekly performance, scorecard |
| 09_phase2_additions | novelty flags, Isolation Forest, the baseline gap, the scorer (online = batch), gate and registry |
| 10_knowledge_base | corpus, chunks, frozen questions, evaluation runs, retrieval variants, abstention, `policy_search` |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
make install                      # pip install -e ".[dev]"
# put the five Kaggle CSVs in data/raw/  (see data/README.md)
make data validate                # join, fix, checksum -> data/interim/; data-quality checks
make eda                          # Phase 1 notebooks 01-04
make phase2                       # Phase 2 pipeline (about 1.5-2 h on 2 CPU cores; resumable)
make phase2-notebooks             # Phase 2 notebooks 05-08 (read the pipeline outputs)
make phase2b gate                 # Phase 2 completion items; model gate and registration
make phase2b-notebook             # notebook 09
make test                         # unit tests
```

The delivered `artifacts/phase2/` already holds the trained models' predictions, the main model and the
SHAP families, so `make phase2` only rebuilds the feature cache (about 1 minute) and skips every finished
stage. Delete `artifacts/phase2/` (or pass `--force`) to retrain from scratch. The five retrain models and
the categorical variant are not shipped (190 MB); their predictions are.

### Knowledge base

The published documents are copyrighted and not in the repository. Put the PDFs (downloaded from the URLs
in `corpus/registry.json`, any file names) in one folder, then:

```bash
python -m finguard.rag register-dir ~/Downloads/finguard-pdfs   # matched to the registry by SHA-256, copied in
make kb                           # parse, chunk, index -> data/corpus/index/ (about 2 minutes)
make kb-eval                      # evaluate on the frozen questions; writes abstention.json
make kb-notebook                  # notebook 10
python -m finguard.rag search "Visa 10.4 compelling evidence"
```

A file whose hash differs from the registry (a newer version, say) is not matched; record it deliberately
with `python -m finguard.rag register FILE --doc-id ID`, rebuild and re-evaluate. After editing a project document in `corpus/docs/`,
run `python -m finguard.rag rehash <doc_id>`. With `sentence-transformers` installed (`pip install -e ".[kb]"`):

```bash
FINGUARD_EMBEDDER=intfloat/multilingual-e5-small FINGUARD_RERANKER=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 make kb kb-eval
```

Without `make` (Windows PowerShell), run the same Python commands the Makefile lists, for example
`python -m finguard.rag build` then `python tools/run_notebooks.py notebooks\10_knowledge_base.ipynb`.
Paths can be overridden with `FINGUARD_RAW_DIR`, `FINGUARD_INTERIM_DIR` and `FINGUARD_REPORTS_DIR`; LightGBM
threads with `FINGUARD_THREADS` (default 2).

## Layout

```
src/finguard/
  config.py               paths, checksums, time split
  split.py                time-split schemes (main, 30-day label delay) and their checks
  phase2.py, phase2b.py   Phase 2 pipelines, stage by stage (resumable)
  gate.py, registry.py    model quality gate; file-based model registry (MLflow optional)
  data/                   schema, ingest (id-XX fix, join), validation
  eda/                    Phase 1 analysis helpers, visual style, figures 01-16
  features/normalize.py   browser / OS / device / email normalisers, amount decimals
  features/proxy_entity.py  proxy entity (NOT a customer), coverage, label purity, split overlap
  features/build.py       point-in-time feature builder (training-window statistics, causal entity features)
  features/novelty.py     explainable novelty flags (new device brand, first-day burst, amount above past max)
  features/inventory.py   rule-based per-column treatment table
  models/dataset.py       build / cache / load the feature matrix per split scheme
  models/linear.py        logistic-regression baseline and its preprocessing
  models/lgbm.py          LightGBM training, raw scores, TreeSHAP
  models/anomaly.py       Isolation Forest experiment
  models/calibration.py   isotonic calibration, risk bands, displayed probability
  serving/                entity store (online history) and the 0-100 scorer with reasons
  evaluation/metrics.py   ranking metrics, bootstrap (incl. entity blocks), paired difference, ECE
  evaluation/figures.py   figures 17-28
  explain/reasons.py      reason families, templates, stability
  rag/                    knowledge base: text, parse, chunk, index, retrieve, corpus, evaluate, tool, glossary
  agents/                 Phase 4: tools, LLM clients, agents, prompts, evidence policy, graph, reports, evaluation
corpus/                   registry.json, questions.jsonl (frozen), indicator_map.json, project documents in docs/
cases/                    the 20 scripted agent cases
tools/                    notebook build/run helpers, fallback test runner
artifacts/                Phase 2 outputs (models, predictions, SHAP families, model bundle), registry.json
```

## Notes on execution

The committed notebook outputs were produced with `tools/run_notebooks.py`, a small headless runner,
because the build environment had no Jupyter. The notebooks are standard `.ipynb` files and run unchanged
with `jupyter nbconvert --execute`.

LightGBM 4.6.0 was used with `deterministic=True` and 2 threads; other versions or thread counts can
shift results in the third decimal. The knowledge-base index was built with the LSA fallback
(`FINGUARD_EMBEDDER=lsa`); a neural embedder gives a different index version and different results.
