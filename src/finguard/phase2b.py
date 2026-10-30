"""Phase 2 additions: the Project Report v3.0 items the first Phase 2 run left out.

    python -m finguard.phase2b                      # all stages (skips finished ones)
    python -m finguard.phase2b --stage iforest      # one stage
    python -m finguard.phase2b --force

Stages and outputs (artifacts/phase2/):

  novelty          novelty/         novelty flags for every labelled row, fraud rate per flag,
                                    and the combiner test (flags + score, fitted on validation)
  iforest          iforest/         Isolation Forest alone: curated and all-feature inputs,
                                    fitted on training-period legitimate rows
  lr_variants      lr_variants/     why the baseline scores 0.419 here vs 0.445 in the design review
  iforest_feature  iforest_feature/ LightGBM main configuration + a label-free Isolation Forest score
  lr_v2            lr_v2/           the strengthened logistic-regression baseline (predictions saved)

Nothing is fitted or chosen on the test month; it is scored for the reported numbers.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from finguard import config, split
from finguard.evaluation.metrics import bootstrap, paired_difference, ranking
from finguard.features.novelty import FLAGS, novelty_flags
from finguard.models import calibration, dataset, lgbm
from finguard.models.anomaly import CURATED, AnomalyDetector

ART = Path(config.ROOT / "artifacts" / "phase2")
T0 = time.time()


def log(*a) -> None:
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def _write(d: Path, info: dict, arrays: dict | None = None) -> None:
    d.mkdir(parents=True, exist_ok=True)
    if arrays is not None:
        np.savez_compressed(d / "preds.npz", **arrays)
    (d / "info.json").write_text(json.dumps(info, indent=1, default=float))


def _done(d: Path, force: bool) -> bool:
    if (d / "info.json").exists() and not force:
        log(f"{d.name}: done already")
        return True
    return False


def _main_preds() -> dict:
    with np.load(ART / "lgbm_numeric" / "preds.npz") as z:
        return {k: z[k] for k in z.files}


def _overlap_top(a: np.ndarray, b: np.ndarray, k: float = 0.01) -> float:
    n = max(int(round(k * len(a))), 1)
    ta, tb = set(np.argsort(-a, kind="stable")[:n]), set(np.argsort(-b, kind="stable")[:n])
    return len(ta & tb) / n


# --------------------------------------------------------------------------- novelty flags
def _rate_rows(y: np.ndarray, flag: np.ndarray, name: str, part: str) -> list[dict]:
    from finguard.eda.style import wilson
    rows = []
    has_key = ~np.isnan(flag)
    for value, label in ((1.0, "flag on"), (0.0, "flag off")):
        k = has_key & (flag == value)
        n, f = int(k.sum()), int(y[k].sum())
        lo, hi = wilson(np.array([f]), np.array([max(n, 1)]))
        rows.append({"part": part, "flag": name, "group": label, "n": n, "share_of_keyed_rows": n / max(has_key.sum(), 1),
                     "frauds": f, "fraud_rate": f / max(n, 1), "lo": float(lo[0]), "hi": float(hi[0])})
    return rows


def _combiner_inputs(raw: np.ndarray, F: np.ndarray) -> np.ndarray:
    no_key = np.isnan(F[:, 0]).astype("float64")
    return np.column_stack([raw, np.nan_to_num(F, nan=0.0), no_key])


def stage_novelty(force: bool) -> None:
    from sklearn.linear_model import LogisticRegression
    from finguard.data.ingest import load_table
    d = ART / "novelty"
    if _done(d, force):
        return
    df = load_table("train")
    log("novelty: computing flags")
    flags = novelty_flags(df)
    y, day = df[config.TARGET].to_numpy(dtype="int8"), df["day"].to_numpy()
    del df
    gc.collect()
    m = split.masks(day)
    F = flags[FLAGS].to_numpy(dtype="float64")
    rows = []
    for part in ("train", "valid", "test"):
        for j, name in enumerate(FLAGS):
            rows += _rate_rows(y[m[part]], F[m[part], j], name, part)
    rates = pd.DataFrame(rows)

    main = _main_preds()
    bands = calibration.fit_bands(main["valid_raw"])
    te_band = bands.assign(main["test_raw"])
    Fte, yte = F[main["test_idx"]], y[main["test_idx"]].astype(int)
    any_flag = np.nan_to_num(Fte, nan=0).max(axis=1) > 0
    within = []
    for b in calibration.BAND_ORDER:
        for lab, k in (("any flag", any_flag), ("no flag", ~any_flag)):
            sel = (te_band == b) & k
            within.append({"band": b, "group": lab, "n": int(sel.sum()), "fraud_rate": float(yte[sel].mean()) if sel.any() else None})

    # combiner: decide on the validation month (two time-ordered halves), confirm on test
    va_idx, yva = main["valid_idx"], y[main["valid_idx"]].astype(int)
    Xva, Xte = _combiner_inputs(main["valid_raw"], F[va_idx]), _combiner_inputs(main["test_raw"], Fte)
    half = day[va_idx] <= np.median(day[va_idx])
    cv = []
    for fit_part, eval_part in ((half, ~half), (~half, half)):
        clf = LogisticRegression(C=1.0, max_iter=1000).fit(Xva[fit_part], yva[fit_part])
        cv.append(ranking(yva[eval_part], clf.decision_function(Xva[eval_part]))["pr_auc"]
                  - ranking(yva[eval_part], main["valid_raw"][eval_part])["pr_auc"])
    clf = LogisticRegression(C=1.0, max_iter=1000).fit(Xva, yva)
    comb_te = clf.decision_function(Xte)
    paired = paired_difference(yte, comb_te, main["test_raw"], B=1000)
    use = bool(np.mean(cv) > 0 and min(cv) > 0)
    info = {"thresholds": {"burst_k": 3, "amount_multiple": 3.0}, "rates": rates.to_dict("records"), "within_test_bands": within,
            "combiner": {"inputs": ["raw score"] + FLAGS + ["no entity key"], "coef": clf.coef_[0].tolist(),
                         "valid_time_cv_pr_auc_gain": cv, "test": ranking(yte, comb_te),
                         "paired_test_combined_minus_main": paired,
                         "decision": "blend into the score" if use else "keep flags as displayed indicators only",
                         "rule": "blend only if both time-ordered halves of the validation month gain PR-AUC"}}
    _write(d, info, {f: flags[f].to_numpy() for f in flags.columns} | {"combined_test_raw": comb_te})
    log(f"novelty: decision = {info['combiner']['decision']}; validation CV gains {np.round(cv, 4)}; "
        f"test gain {paired['pr_auc']['difference']:+.4f}")


# --------------------------------------------------------------------------- Isolation Forest
def stage_iforest(force: bool) -> None:
    d = ART / "iforest"
    if _done(d, force):
        return
    fs = dataset.load("main", "numeric")
    m, y = fs.masks, fs.y.astype(int)
    main = _main_preds()
    tree_cols = [c for c in fs.columns if c in fs.builder.tree_columns]
    variants = {"curated": CURATED, "all": tree_cols}
    info, arrays = {"variants": {}}, {}
    legit_tr = np.flatnonzero(m["train"] & (y == 0))
    for name, cols in variants.items():
        log(f"iforest: {name} ({len(cols)} inputs), fitted on {len(legit_tr):,} training-period legitimate rows (sampled)")
        idx = fs.cols(cols)
        det = AnomalyDetector(cols).fit(np.asarray(fs.X[legit_tr][:, idx]))
        res = {}
        for part in ("valid", "test"):
            s = det.score(np.asarray(fs.X[m[part]][:, idx]))
            arrays[f"{name}_{part}"] = s
            r = ranking(y[m[part]], s)
            r["overlap_top1pct_with_lightgbm"] = _overlap_top(s, main[f"{part}_raw"])
            res[part] = r
        res["test_ci95"] = bootstrap(y[m["test"]], arrays[f"{name}_test"], B=500)["pr_auc"]
        info["variants"][name] = res
        log(f"iforest: {name} valid PR-AUC {res['valid']['pr_auc']:.4f} test {res['test']['pr_auc']:.4f} "
            f"overlap {res['test']['overlap_top1pct_with_lightgbm']:.3f}")
    chosen = max(variants, key=lambda v: info["variants"][v]["valid"]["pr_auc"])
    info["chosen_for_feature"] = chosen
    info["rule"] = "variant with the higher validation PR-AUC becomes the candidate feature"
    log(f"iforest: label-free fit of '{chosen}' on all training-period rows, scoring every row")
    idx = fs.cols(variants[chosen])
    det = AnomalyDetector(variants[chosen], seed=config.RANDOM_SEED + 1).fit(np.asarray(fs.X[m["train"]][:, idx]))
    arrays["feature_all_rows"] = det.score(np.asarray(fs.X[:, idx]))
    _write(d, info, arrays)


def stage_iforest_feature(force: bool) -> None:
    d = ART / "iforest_feature"
    if _done(d, force):
        return
    with np.load(ART / "iforest" / "preds.npz") as z:
        feat = z["feature_all_rows"].astype("float32")
    fs = dataset.load("main", "numeric")
    X = np.hstack([np.asarray(fs.X), feat[:, None]])
    fs2 = dataset.FeatureSet(X=X, columns=fs.columns + ["iforest_score"], categorical=fs.categorical, y=fs.y, day=fs.day,
                             transaction_id=fs.transaction_id, entity=fs.entity, scheme=fs.scheme, builder=fs.builder)
    del fs
    gc.collect()
    cols = [c for c in fs2.columns if c in fs2.builder.tree_columns] + ["iforest_score"]
    log(f"iforest_feature: LightGBM with {len(cols)} inputs")
    model = lgbm.train(fs2, columns=cols, log_every=0)
    m = fs2.masks
    preds = {f"{p}_idx": np.flatnonzero(m[p]) for p in ("valid", "test")}
    preds |= {f"{p}_raw": lgbm.predict_raw(model, fs2, m[p]) for p in ("valid", "test")}
    y_te = fs2.y[m["test"]].astype(int)
    main = _main_preds()
    imp = dict(zip(model.booster.feature_name(), model.booster.feature_importance("gain")))
    rank = sorted(imp, key=imp.get, reverse=True)
    info = {"best_iteration": model.best_iteration, "seconds": model.seconds, "n_features": len(cols),
            "valid": ranking(fs2.y[m["valid"]], preds["valid_raw"]), "test": ranking(y_te, preds["test_raw"]),
            "paired_test_with_minus_without": paired_difference(y_te, preds["test_raw"], main["test_raw"], B=1000),
            "iforest_gain_rank": rank.index("iforest_score") + 1}
    _write(d, info, preds)
    log(f"iforest_feature: test PR-AUC {info['test']['pr_auc']:.4f} (main {ranking(y_te, main['test_raw'])['pr_auc']:.4f}); "
        f"gain rank {info['iforest_gain_rank']}")


# --------------------------------------------------------------------------- baseline gap
def _review_style(df: pd.DataFrame, tr: np.ndarray, variant: str) -> np.ndarray:
    """The design review's baseline preprocessing, fitted on the training window.

    Signed log on every numeric column, training-window median for NaN, one indicator per
    distinct NaN pattern (columns 1-99% missing), one-hot for text columns with <= 30 levels,
    standardisation. Built column by column in float32 to stay within memory.
    """
    from finguard.features.build import CODE_COLS, TRENDING_D
    text = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    drop = {config.KEY, config.TARGET, config.TIME, "day", "rel_hour", "rel_weekday"}
    num = [c for c in df.columns if c not in drop and c not in text]
    if variant in ("no_codes", "both"):
        num = [c for c in num if c not in CODE_COLS]
    day_abs = np.floor(df[config.TIME].to_numpy(dtype="float64") / config.SECONDS_PER_DAY)

    def column(c: str) -> np.ndarray:
        x = df[c].to_numpy(dtype="float64")
        return day_abs - x if variant in ("d_reexpressed", "both") and c in TRENDING_D else x

    blocks: dict = {}
    for c in num:
        miss_tr = np.isnan(column(c)[tr])
        if 0.01 < miss_tr.mean() < 0.99:
            blocks.setdefault(np.packbits(miss_tr).tobytes(), c)
    low = [c for c in text if df.loc[tr, c].nunique() <= 30]
    levels = [(c, v) for c in low for v in df.loc[tr, c].dropna().unique()]
    M = np.empty((len(df), len(num) + len(blocks) + len(levels)), dtype="float32")

    def put(j: int, x: np.ndarray) -> None:
        mu, sd = x[tr].mean(), x[tr].std() + 1e-6
        M[:, j] = (x - mu) / sd

    for j, c in enumerate(num):
        x = column(c)
        x = np.sign(x) * np.log1p(np.abs(x))
        med = np.nanmedian(x[tr]) if (~np.isnan(x[tr])).any() else 0.0
        put(j, np.where(np.isnan(x), med, x))
    j = len(num)
    for c in blocks.values():
        put(j, np.isnan(column(c)).astype("float64"))
        j += 1
    for c, v in levels:
        put(j, (df[c].astype("string") == v).fillna(False).to_numpy(dtype="float64"))
        j += 1
    return M


def stage_lr_variants(force: bool) -> None:
    from sklearn.linear_model import LogisticRegression
    from finguard.data.ingest import load_table
    d = ART / "lr_variants"
    if _done(d, force):
        return
    df = load_table("train")
    y = df[config.TARGET].to_numpy(dtype="int8").astype(int)
    m = split.masks(df["day"].to_numpy())
    rows = []
    for variant in ("review_style", "d_reexpressed", "no_codes", "both"):
        log(f"lr_variants: {variant}")
        M = _review_style(df, m["train"], variant)
        best = None
        for C in (0.01, 0.05):
            clf = LogisticRegression(C=C, max_iter=500).fit(M[m["train"]], y[m["train"]])
            ap = ranking(y[m["valid"]], clf.decision_function(M[m["valid"]]))["pr_auc"]
            if best is None or ap > best[1]:
                best = (C, ap, clf)
        s_te = best[2].decision_function(M[m["test"]])
        r = ranking(y[m["test"]], s_te)
        rows.append({"variant": variant, "inputs": M.shape[1], "C": best[0], "valid_pr_auc": best[1], "test_pr_auc": r["pr_auc"],
                     "test_ci95": bootstrap(y[m["test"]], s_te, B=500)["pr_auc"], "recall@5%": r["recall@5%"]})
        log(f"lr_variants: {variant} C={best[0]} valid {best[1]:.4f} test {r['pr_auc']:.4f}")
        del M
        gc.collect()
    _write(d, {"variants": rows,
               "meaning": {"review_style": "design-review preprocessing: raw D columns and card/address codes as numbers",
                           "d_reexpressed": "same, trending D columns as day - D",
                           "no_codes": "same, card/address codes removed",
                           "both": "trending D re-expressed and codes removed"}})


def stage_lr_baseline_v2(force: bool) -> None:
    """The strengthened baseline: design-review preprocessing with the Phase 1 rules applied
    (trending D columns re-expressed, card and address codes removed), C chosen on validation."""
    from sklearn.linear_model import LogisticRegression
    from finguard.data.ingest import load_table
    from finguard.evaluation.metrics import calibration as calib
    d = ART / "lr_v2"
    if _done(d, force):
        return
    df = load_table("train")
    y = df[config.TARGET].to_numpy(dtype="int8").astype(int)
    m = split.masks(df["day"].to_numpy())
    M = _review_style(df, m["train"], "both")
    del df
    gc.collect()
    grid, best = {}, None
    for C in (0.003, 0.01, 0.03):
        clf = LogisticRegression(C=C, max_iter=500).fit(M[m["train"]], y[m["train"]])
        ap = ranking(y[m["valid"]], clf.decision_function(M[m["valid"]]))["pr_auc"]
        grid[C] = ap
        if best is None or ap > grid[best[0]]:
            best = (C, clf)
        log(f"lr_v2: C={C} valid PR-AUC {ap:.4f}")
    clf = best[1]
    preds = {f"{p}_idx": np.flatnonzero(m[p]) for p in ("valid", "test")}
    preds |= {f"{p}_raw": clf.decision_function(M[m[p]]) for p in ("valid", "test")}
    main = _main_preds()
    assert np.array_equal(main["test_idx"], preds["test_idx"])
    y_te, y_va = y[m["test"]], y[m["valid"]]
    iso = calibration.fit_isotonic(preds["valid_raw"], y_va)
    info = {"name": "lr_v2", "preprocessing": "design-review style; trending D re-expressed as day - D; card/address codes removed",
            "inputs": int(M.shape[1]), "C": best[0], "grid_valid_pr_auc": grid,
            "valid": ranking(y_va, preds["valid_raw"]), "test": ranking(y_te, preds["test_raw"]),
            "test_ci95": bootstrap(y_te, preds["test_raw"], B=1000),
            "calibration_test_isotonic": calib(y_te, iso.predict(preds["test_raw"])),
            "paired_lightgbm_minus_lr_v2": paired_difference(y_te, main["test_raw"], preds["test_raw"], B=1000)}
    _write(d, info, preds)
    log(f"lr_v2: C={best[0]} test PR-AUC {info['test']['pr_auc']:.4f}; LightGBM - LR v2 = "
        f"{info['paired_lightgbm_minus_lr_v2']['pr_auc']['difference']:+.4f}")


STAGES = {"novelty": stage_novelty, "iforest": stage_iforest, "lr_variants": stage_lr_variants,
          "iforest_feature": stage_iforest_feature, "lr_v2": stage_lr_baseline_v2}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all", choices=["all", *STAGES])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for name, fn in STAGES.items():
        if a.stage in ("all", name):
            fn(a.force)
    log("phase 2 additions finished")


if __name__ == "__main__":
    main()
