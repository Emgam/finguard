PY ?= python
NB = notebooks/01_data_validation.ipynb notebooks/02_eda_missingness_and_distributions.ipynb \
     notebooks/03_eda_temporal_drift_leakage.ipynb notebooks/04_feature_inventory_and_plan.ipynb
NB2 = notebooks/05_split_features_baseline.ipynb notebooks/06_lightgbm_calibration_bands.ipynb \
      notebooks/07_explanations.ipynb notebooks/08_robustness_and_scorecard.ipynb
NB2B = notebooks/09_phase2_additions.ipynb
NB3 = notebooks/10_knowledge_base.ipynb
RUN_NB = $(shell command -v jupyter >/dev/null 2>&1 && echo "jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1" || echo "$(PY) tools/run_notebooks.py")

.PHONY: install data validate notebooks eda eda-fallback test all all-phase2 phase2 phase2-notebooks phase2-fallback phase2b phase2b-notebook gate glossary kb kb-eval kb-notebook all-phase3 agents-eval agents-demo serve serve-demo demo release-check docker-up docker-demo

install:            ## editable install with dev tools
	$(PY) -m pip install -e ".[dev]"

data:               ## raw CSVs -> joined, fixed tables in data/interim (+ checksums)
	$(PY) -m finguard.data.ingest --check-md5

validate:           ## data-quality checks; exits 1 on any FAIL (use in CI)
	$(PY) -m finguard.data.validate

notebooks:          ## regenerate .ipynb files from notebooks/src/*.py
	$(PY) tools/build_notebooks.py

eda:                ## execute Phase 1 notebooks 01-04 with Jupyter
	$(PY) tools/build_notebooks.py 01 02 03 04
	jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 $(NB)

eda-fallback:       ## same, without Jupyter
	$(PY) tools/build_notebooks.py 01 02 03 04
	$(PY) tools/run_notebooks.py $(NB)

phase2:             ## Phase 2 pipeline: features, LR, LightGBM, retrains, label-delay, ablation, SHAP (resumable)
	$(PY) -m finguard.phase2

phase2-notebooks:   ## execute Phase 2 notebooks 05-08 (after `make phase2`)
	$(PY) tools/build_notebooks.py 05 06 07 08
	jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 $(NB2)

phase2-fallback:    ## same, without Jupyter
	$(PY) tools/build_notebooks.py 05 06 07 08
	$(PY) tools/run_notebooks.py $(NB2)

test:
	$(PY) -m pytest -q || $(PY) tools/run_tests.py

all: data validate eda test

all-phase2: phase2 phase2-notebooks test

phase2b:            ## Phase 2 additions: novelty flags, Isolation Forest, baseline check, strengthened baseline
	$(PY) -m finguard.phase2b

phase2b-notebook:   ## notebook 09 (after `make phase2b`); Jupyter if installed, else the headless runner
	$(PY) tools/build_notebooks.py 09
	$(RUN_NB) $(NB2B)

gate:               ## model quality gate (exit 1 blocks a release); registers the model with its report
	$(PY) -m finguard.gate --register

glossary:           ## regenerate the feature and reason-code glossary (a knowledge-base document)
	$(PY) -m finguard.rag.glossary

kb:                 ## Phase 3: parse, chunk and index the registered documents
	$(PY) -m finguard.rag build

kb-eval:            ## Phase 3: retrieval evaluation on the frozen question set
	$(PY) -m finguard.rag evaluate

kb-notebook:        ## notebook 10 (after `make kb kb-eval`)
	$(PY) tools/build_notebooks.py 10
	$(RUN_NB) $(NB3)

all-phase3: kb kb-eval kb-notebook test


agents-eval:        ## Phase 4: the 20 scripted cases, orchestrated vs fallback (LLM from FINGUARD_LLM)
	$(PY) -m finguard.agents evaluate

agents-demo:        ## Phase 4: one investigation printed as a report
	$(PY) -m finguard.agents investigate "Investigate alert TX-3554656"

serve:              ## Phase 5: API + dashboard on http://127.0.0.1:8000 (real data, model and index)
	$(PY) -m finguard.api serve

serve-demo:         ## Phase 5: same, on the synthetic demo world (no data needed)
	$(PY) -m finguard.api serve --world synthetic --llm scripted

demo:               ## Phase 5 exit criterion: timed alert-to-report run against the running API (< 60 s)
	$(PY) -m finguard.api demo

release-check:      ## every gate on the real data: data, model, prompts, tests, deployment demo
	$(PY) tools/release_check.py

docker-up:          ## Phase 5: build and start the container (real world; mounts data/ and artifacts/)
	docker compose up --build -d

docker-demo:        ## Phase 5: the container on the synthetic world, as in CI
	FINGUARD_WORLD=synthetic FINGUARD_LLM=scripted FINGUARD_INSTALL_KB=false docker compose up --build -d --wait
