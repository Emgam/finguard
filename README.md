# FinGuard AI

Decision-support prototype for card-not-present payment fraud review, built on the IEEE-CIS
Fraud Detection dataset. **This repository is at the end of Phase 2: baseline and main models on a
time split, calibration, risk bands, explanations and robustness checks.**

> IEEE-CIS has **no customer identifier**. All grouping of transactions in this project is a
> heuristic **proxy entity** (card1 + addr1 + first-seen day), never a customer.

## Phase 2 results (test month, days 152-181, scored once)

| | LightGBM (main) | Logistic regression (baseline) |
| --- | --- | --- |
| PR-AUC (95% interval, proxy-entity blocks) | 0.580 (0.552-0.609) | 0.419 |
| ROC-AUC | 0.907 | 0.852 |
| Precision / recall, top 1% of traffic | 89.2% / 25.5% | 80.9% / 23.1% |
| Precision / recall, top 5% of traffic | 43.7% / 62.4% | 32.2% / 45.9% |

Isotonic calibration (fitted on validation) brings test ECE from 0.0178 to 0.0041. Risk bands with
validation thresholds: High 90.1% fraud, Medium 30.7%, Low 1.3%; High + Medium = 5.6% of traffic,
64.9% of fraud. Full results, robustness checks and limitations: `reports/model_report.md`.

## Deliverables

| Phase | Deliverable | Where |
| --- | --- | --- |
| 2 | Model report | `reports/model_report.md` |
| 2 | Pipeline (features, models, retrains, label-delay, ablation, SHAP) | `python -m finguard.phase2` |
| 2 | Executed notebooks | `notebooks/05…08_*.ipynb` |
| 2 | Figures 17-24, tables `phase2_*` | `reports/figures/`, `reports/tables/` |
| 2 | Model bundle for Phase 3 (builder, model, calibrator, band thresholds) | `artifacts/phase2/model_bundle.pkl` + `artifacts/phase2/lgbm_numeric/model.txt` |
| 1 | Data-quality report, feature-engineering plan | `reports/data_quality_report.md`, `reports/feature_engineering_plan.md` |
| 1 | Executed notebooks, figures 01-16 | `notebooks/01…04_*.ipynb`, `reports/figures/` |
| 1-2 | Unit tests | `tests/` (35 tests) |

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

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
make install                      # pip install -e ".[dev]"
# put the five Kaggle CSVs in data/raw/  (see data/README.md)
make data validate                # join, fix, checksum -> data/interim/; data-quality checks
make eda                          # Phase 1 notebooks 01-04
make phase2                       # Phase 2 pipeline (about 1.5-2 h on 2 CPU cores; resumable)
make phase2-notebooks             # Phase 2 notebooks 05-08 (read the pipeline outputs)
make test                         # unit tests
```

The delivered `artifacts/phase2/` already holds the trained models' predictions, the main model and the
SHAP families, so `make phase2` only rebuilds the feature cache (about 1 minute) and skips every finished
stage. Delete `artifacts/phase2/` (or pass `--force`) to retrain from scratch. The five retrain models and
the categorical variant are not shipped (190 MB); their predictions are.

Without `make` (Windows PowerShell), run the same Python commands the Makefile lists, for example
`python -m finguard.phase2` then `python tools/run_notebooks.py notebooks\05_split_features_baseline.ipynb ...`.
Paths can be overridden with `FINGUARD_RAW_DIR`, `FINGUARD_INTERIM_DIR` and `FINGUARD_REPORTS_DIR`.

## Layout

```
src/finguard/
  config.py               paths, checksums, time split
  split.py                time-split schemes (main, 30-day label delay) and their checks
  phase2.py               Phase 2 pipeline, stage by stage (resumable)
  data/                   schema, ingest (id-XX fix, join), validation
  eda/                    Phase 1 analysis helpers, visual style, figures 01-16
  features/normalize.py   browser / OS / device / email normalisers, amount decimals
  features/proxy_entity.py  proxy entity (NOT a customer), coverage, label purity, split overlap
  features/build.py       point-in-time feature builder (training-window statistics, causal entity features)
  features/inventory.py   rule-based per-column treatment table
  models/dataset.py       build / cache / load the feature matrix per split scheme
  models/linear.py        logistic-regression baseline and its preprocessing
  models/lgbm.py          LightGBM training, raw scores, TreeSHAP
  models/calibration.py   isotonic calibration, risk bands
  evaluation/metrics.py   ranking metrics, bootstrap (incl. entity blocks), paired difference, ECE
  evaluation/figures.py   figures 17-24
  explain/reasons.py      reason families, templates, stability
tools/                    notebook build/run helpers, fallback test runner
artifacts/phase2/         pipeline outputs (models, predictions, SHAP families, model bundle)
```

## Notes on execution

The committed notebook outputs were produced with `tools/run_notebooks.py`, a small headless runner,
because the build environment had no Jupyter. The notebooks are standard `.ipynb` files and run unchanged
with `jupyter nbconvert --execute` (`make eda`, `make phase2-notebooks`).

LightGBM 4.6.0 was used with `deterministic=True` and 2 threads; other versions or thread counts can
shift results in the third decimal.
