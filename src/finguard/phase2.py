"""Phase 2 pipeline: every heavy computation, resumable stage by stage.

    python -m finguard.phase2                   # all stages (skips finished ones)
    python -m finguard.phase2 --stage lgbm      # one stage
    python -m finguard.phase2 --force           # recompute

Stages and outputs (artifacts/phase2/):

  features   data/processed/{scheme}_{codes}/      cached feature matrices
  lgbm       lgbm_{codes}/                         LightGBM, codes as numbers vs categories (seed 42)
  select     selection.json                        choice made on VALIDATION PR-AUC only
  lr         lr/                                   logistic-regression baseline
  retrain    retrain_seed{1..5}/                   same config, 5 other seeds (stability)
  gap        gap_{codes}/                          label-delay run: train days 0-90 only
  ablation   ablation_no_entity/                   selected config without entity_* features
  shap       shap/                                 TreeSHAP family sums for alerted test rows

The test month is only scored (predictions saved); nothing is fitted or chosen on it.
"""
from __future__ import annotations

import argparse
import gc
import json
import pickle
import time
from pathlib import Path

import numpy as np

from finguard import config
from finguard.evaluation.metrics import ranking
from finguard.explain import reasons
from finguard.features.build import ENTITY_FEATURES
from finguard.models import calibration, dataset, lgbm, linear

ART = Path(config.ROOT / "artifacts" / "phase2")
SEEDS_RETRAIN = (1, 2, 3, 4, 5)
SHAP_SAMPLE = 5000
T0 = time.time()


def log(*a) -> None:
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def _done(d: Path, force: bool) -> bool:
    return (d / "info.json").exists() and not force


def _write(d: Path, info: dict, preds: dict | None = None) -> None:
    d.mkdir(parents=True, exist_ok=True)
    if preds is not None:
        np.savez_compressed(d / "preds.npz", **preds)
    (d / "info.json").write_text(json.dumps(info, indent=1, default=float))


def load_preds(name: str) -> dict[str, np.ndarray]:
    with np.load(ART / name / "preds.npz") as z:
        return {k: z[k] for k in z.files}


def load_info(name: str) -> dict:
    return json.loads((ART / name / "info.json").read_text())


def selected_codes() -> str:
    return json.loads((ART / "selection.json").read_text())["codes_as"]


def _score_split(model: lgbm.TrainedModel, fs: dataset.FeatureSet) -> dict[str, np.ndarray]:
    m = fs.masks
    out = {}
    for part in ("valid", "test"):
        idx = np.flatnonzero(m[part])
        out[f"{part}_idx"], out[f"{part}_raw"] = idx, lgbm.predict_raw(model, fs, m[part])
    return out


def _fit_and_save(name: str, fs: dataset.FeatureSet, force: bool, save_model: bool = False, **kw) -> None:
    d = ART / name
    if _done(d, force):
        log(f"{name}: done already")
        return
    log(f"{name}: training")
    model = lgbm.train(fs, log_every=0, **kw)
    preds = _score_split(model, fs)
    va, te = fs.masks["valid"], fs.masks["test"]
    info = {"name": name, "scheme": fs.scheme, "best_iteration": model.best_iteration, "seconds": model.seconds,
            "n_features": len(model.columns), "n_categorical": len(model.categorical), "params": model.params,
            "valid": ranking(fs.y[va], preds["valid_raw"]), "test": ranking(fs.y[te], preds["test_raw"]),
            "valid_curve": model.valid_curve}
    if save_model:
        d.mkdir(parents=True, exist_ok=True)
        model.booster.save_model(str(d / "model.txt"), num_iteration=model.best_iteration)
        with open(d / "columns.json", "w") as f:
            json.dump({"columns": model.columns, "categorical": model.categorical}, f)
    _write(d, info, preds)
    log(f"{name}: best_iter={model.best_iteration} valid PR-AUC={info['valid']['pr_auc']:.4f} ({model.seconds:.0f}s)")


# --------------------------------------------------------------------------- stages
def stage_features(force: bool) -> None:
    for codes in ("numeric", "category"):
        if force or not (dataset.cache_dir("main", codes) / "X.npy").exists():
            log(f"features main/{codes}")
            dataset.save(dataset.build("main", codes), codes)
            gc.collect()


def stage_lgbm(force: bool) -> None:
    for codes in ("numeric", "category"):
        fs = dataset.load("main", codes)
        _fit_and_save(f"lgbm_{codes}", fs, force, save_model=True)
        del fs
        gc.collect()


def stage_select(force: bool) -> None:
    vals = {c: load_info(f"lgbm_{c}")["valid"]["pr_auc"] for c in ("numeric", "category")}
    choice = max(vals, key=vals.get)
    (ART / "selection.json").write_text(json.dumps({
        "codes_as": choice, "valid_pr_auc": vals, "rule": "higher validation PR-AUC; test month not used"}, indent=1))
    log(f"select: codes_as={choice} {vals}")


def stage_lr(force: bool) -> None:
    d = ART / "lr"
    if _done(d, force):
        log("lr: done already")
        return
    fs = dataset.load("main", "numeric")
    m = fs.masks
    cols = fs.builder.columns_
    log("lr: building frames")
    import pandas as pd
    tr = pd.DataFrame(np.asarray(fs.X[m["train"]]), columns=cols)
    va = pd.DataFrame(np.asarray(fs.X[m["valid"]]), columns=cols)
    model = linear.train(tr, fs.y[m["train"]], va, fs.y[m["valid"]], fs.builder)
    del tr
    gc.collect()
    te = pd.DataFrame(np.asarray(fs.X[m["test"]]), columns=cols)
    preds = {"valid_idx": np.flatnonzero(m["valid"]), "valid_raw": model.raw(va),
             "test_idx": np.flatnonzero(m["test"]), "test_raw": model.raw(te)}
    coef = pd.DataFrame({"feature": model.prep.feature_names_, "coef": model.clf.coef_[0]})
    d.mkdir(parents=True, exist_ok=True)
    coef.to_csv(d / "coefficients.csv", index=False)
    info = {"name": "lr", "C": model.C, "grid": model.grid, "seconds": model.seconds,
            "n_inputs": len(model.prep.feature_names_), "n_numeric": len(model.prep.numeric_),
            "n_dropped_redundant": len(model.prep.dropped_redundant_), "n_log": len(model.prep.log_),
            "n_indicators": len(model.prep.indicators_), "n_onehot_groups": len(model.prep.onehot_),
            "valid": ranking(fs.y[m["valid"]], preds["valid_raw"]), "test": ranking(fs.y[m["test"]], preds["test_raw"])}
    _write(d, info, preds)
    log(f"lr: C={model.C} valid PR-AUC={info['valid']['pr_auc']:.4f} ({model.seconds:.0f}s)")


def stage_retrain(force: bool) -> None:
    codes = selected_codes()
    main = load_info(f"lgbm_{codes}")
    fs = dataset.load("main", codes)
    for s in SEEDS_RETRAIN:
        _fit_and_save(f"retrain_seed{s}", fs, force, save_model=True, seed=s, num_boost_round=main["best_iteration"],
                      early_stopping=None)
        gc.collect()


def stage_gap(force: bool) -> None:
    codes = selected_codes()
    name = f"gap_{codes}"
    if _done(ART / name, force):
        log(f"{name}: done already")
        return
    fs = dataset.ensure("label_delay_30d", codes)
    _fit_and_save(name, fs, force)


def stage_ablation(force: bool) -> None:
    codes = selected_codes()
    fs = dataset.load("main", codes)
    cols = [c for c in fs.builder.tree_columns if c not in ENTITY_FEATURES]
    _fit_and_save("ablation_no_entity", fs, force, columns=cols)


def stage_shap(force: bool) -> None:
    d = ART / "shap"
    if (d / "families.pkl").exists() and not force:
        log("shap: done already")
        return
    import lightgbm as lgb
    codes = selected_codes()
    fs = dataset.load("main", codes)
    main = load_preds(f"lgbm_{codes}")
    bands = calibration.fit_bands(main["valid_raw"])
    test_idx, test_raw = main["test_idx"], main["test_raw"]
    alert = test_idx[bands.assign(test_raw) != "Low"]
    rs = np.random.RandomState(config.RANDOM_SEED)
    sample = np.sort(rs.choice(test_idx, SHAP_SAMPLE, replace=False))
    cols = json.loads((ART / f"lgbm_{codes}" / "columns.json").read_text())["columns"]
    booster = lgb.Booster(model_file=str(ART / f"lgbm_{codes}" / "model.txt"))
    X_alert = np.asarray(fs.X[alert][:, fs.cols(cols)])
    X_sample = np.asarray(fs.X[sample][:, fs.cols(cols)])
    out = {"columns": cols, "alert_idx": alert, "sample_idx": sample, "bands": bands}
    log(f"shap: main model on {len(alert)} alerted + {len(sample)} sampled test rows")
    c_alert = booster.predict(X_alert, pred_contrib=True)
    out["local_accuracy_max_abs_error"] = float(np.abs(c_alert.sum(1) - booster.predict(X_alert, raw_score=True)).max())
    out["main_alert"] = reasons.family_matrix(c_alert, cols)
    c_sample = booster.predict(X_sample, pred_contrib=True)
    out["main_sample"] = reasons.family_matrix(c_sample, cols)
    out["main_sample_abs_by_feature"] = np.abs(c_sample[:, :-1]).mean(axis=0)
    out["expected_value"] = float(c_alert[0, -1])
    del c_alert, c_sample
    for s in SEEDS_RETRAIN:                       # same transactions, models retrained with other seeds
        log(f"shap: retrain seed {s}")
        b = lgb.Booster(model_file=str(ART / f"retrain_seed{s}" / "model.txt"))
        out[f"seed{s}_alert"] = reasons.family_matrix(b.predict(X_alert, pred_contrib=True), cols)
        del b
        gc.collect()
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "families.pkl", "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    log("shap: done")


STAGES = {"features": stage_features, "lgbm": stage_lgbm, "select": stage_select, "lr": stage_lr,
          "retrain": stage_retrain, "gap": stage_gap, "ablation": stage_ablation, "shap": stage_shap}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all", choices=["all", *STAGES])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    ART.mkdir(parents=True, exist_ok=True)
    for name, fn in STAGES.items():
        if a.stage in ("all", name):
            fn(a.force)
    log("phase 2 pipeline finished")


if __name__ == "__main__":
    main()
