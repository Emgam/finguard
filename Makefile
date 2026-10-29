PY ?= python
NB = notebooks/01_data_validation.ipynb notebooks/02_eda_missingness_and_distributions.ipynb \
     notebooks/03_eda_temporal_drift_leakage.ipynb notebooks/04_feature_inventory_and_plan.ipynb
NB2 = notebooks/05_split_features_baseline.ipynb notebooks/06_lightgbm_calibration_bands.ipynb \
      notebooks/07_explanations.ipynb notebooks/08_robustness_and_scorecard.ipynb

.PHONY: install data validate notebooks eda eda-fallback test all all-phase2 phase2 phase2-notebooks phase2-fallback

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
