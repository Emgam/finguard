"""Drift monitoring (Evidently): is the traffic the model scores still like the month its
calibration and alert thresholds were frozen on?

Reference = the validation month (days 121-151): isotonic calibration and the High / Medium
thresholds were fitted there, so it is what "normal" means for the deployed score. Windows:

| Window  | Current data                                   | Labels | Score |
| test    | the held-out test month (days 152-181), and    | yes    | yes   |
|         | each of its weeks (trend)                      |        |       |
| later   | Kaggle's unlabelled later file (days 212-394), | no     | no    |
|         | and each 30-day slice of it (trend)            |        |       |

Per window: Evidently's DataDriftPreset on readable inputs (amount, product, card network and
type, e-mail domains, device type, identity presence, distance, three counters, relative hour;
default methods: Wasserstein distance for numbers, Jensen-Shannon for categories, threshold
0.1), Evidently PSI on the calibrated score, and FinGuard's own PSI (``finguard.eda.drift``,
missing values as a bin) as a cross-check. With labels, the PR-AUC of both months is added.

Decision (thresholds in ``params.yaml``, section ``monitoring``):
    retrain review   score PSI >= 0.25, or >= 75% of the monitored inputs drifted
    review           score PSI >= 0.10, dataset drift (>= 50% of inputs drifted), the share of
                     traffic sent to analysts (High + Medium) moved more than 50% from plan (5%),
                     or, with labels, PR-AUC >= 15% below the reference month's (the validation
                     month was used for early stopping, so some drop is expected: the threshold
                     sits above that optimism)
    stable           otherwise
Nothing is retrained automatically: the decision is a signal for a person (as for promotion).

Outputs (reports/monitoring/): drift_<window>.html (Evidently report), drift_<window>.summary.json,
drift_trend.csv; reports/metrics/drift.json (flat, for ``dvc metrics``); one MLflow run per window.

    python -m finguard.mlops drift                       # both windows
    python -m finguard.mlops drift --window test --fail-on retrain   # exit 1 on a retrain signal
Without Evidently installed the PSI part still runs (no HTML report).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from finguard import config
from finguard.eda.drift import psi as psi_numeric
from finguard.mlops import optional, params, version

OUT_DIR = config.REPORTS_DIR / "monitoring"
METRICS = config.REPORTS_DIR / "metrics" / "drift.json"
ART = Path(config.ROOT / "artifacts" / "phase2")
NUMERICAL = ["TransactionAmt", "dist1", "C1", "C13", "C14", "rel_hour"]
CATEGORICAL = ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", "DeviceType", "has_identity", "M4", "M6"]
SCORE = "score"
POLICY = {"drift_share": 0.5, "drift_share_retrain": 0.75, "score_psi_review": 0.10, "score_psi_retrain": 0.25,
          "alert_share_planned": 0.05, "alert_share_tolerance": 0.5, "pr_auc_drop_review": 0.15}
LEVEL = {"stable": 0, "review": 1, "retrain review": 2}


def policy() -> dict:
    """Thresholds from params.yaml (section monitoring) over the defaults above."""
    return params("monitoring", POLICY)


# --------------------------------------------------------------------------- data
def psi_categorical(ref: pd.Series, cur: pd.Series, top: int = 20) -> float:
    """PSI over the reference's most frequent categories (+ 'other' + missing), add-one smoothed."""
    cats = list(ref.value_counts().index[:top])
    def dist(s):
        known = s.where(s.isin(cats))
        counts = [int((known == c).sum()) for c in cats] + [int((s.notna() & known.isna()).sum()), int(s.isna().sum())]
        p = np.asarray(counts, dtype="float64")
        return (p + 1) / (p.sum() + len(p))
    a, b = dist(ref), dist(cur)
    return float(np.sum((b - a) * np.log(b / a)))


def column_psi(ref: pd.DataFrame, cur: pd.DataFrame, col: str) -> float:
    if col in CATEGORICAL:
        return psi_categorical(ref[col].astype("object"), cur[col].astype("object"))
    return psi_numeric(ref[col].astype("float64"), cur[col].astype("float64"))


def load_frames(score: bool = True) -> dict[str, pd.DataFrame]:
    """Reference (validation month), the test month and the later file, with the score where it exists."""
    from finguard.data.ingest import load_table
    cols = [config.KEY, config.TARGET, "day"] + NUMERICAL + CATEGORICAL
    train = load_table("train", columns=cols)
    later = load_table("test", columns=[c for c in cols if c != config.TARGET])
    (v0, v1), (t0, t1) = config.SPLIT_DAYS["valid"], config.SPLIT_DAYS["test"]
    ref = train[train["day"].between(v0, v1)].copy()
    cur = train[train["day"].between(t0, t1)].copy()
    if score:
        scored = model_scores()
        if scored is not None:
            ref = ref.merge(scored, on=config.KEY, how="left")
            cur = cur.merge(scored, on=config.KEY, how="left")
    return {"reference": ref.reset_index(drop=True), "test": cur.reset_index(drop=True), "later": later.reset_index(drop=True)}


def model_scores() -> pd.DataFrame | None:
    """Calibrated probability and band for the validation and test months (saved Phase 2 predictions)."""
    import pickle
    try:
        sel = json.loads((ART / "selection.json").read_text(encoding="utf-8"))["codes_as"]
        with np.load(ART / f"lgbm_{sel}" / "preds.npz") as z:
            preds = {k: z[k] for k in z.files}
        from finguard.models import dataset
        tid = dataset.load("main", sel).transaction_id
    except (FileNotFoundError, KeyError) as exc:
        print(f"drift: no model scores ({exc}); monitoring the inputs only")
        return None
    iso = bands = None
    if (ART / "model_bundle.pkl").exists():
        with open(ART / "model_bundle.pkl", "rb") as f:
            b = pickle.load(f)
        iso, bands = b["isotonic"], b["bands"]
    parts = []
    for period in ("valid", "test"):
        raw = preds[f"{period}_raw"]
        p = iso.predict(raw) if iso is not None else raw
        band = bands.assign(raw) if bands is not None else np.full(len(raw), None)
        parts.append(pd.DataFrame({config.KEY: tid[preds[f"{period}_idx"]], "raw_score": raw, SCORE: p, "band": band}))
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------- Evidently
def evidently_report(ref: pd.DataFrame, cur: pd.DataFrame, drift_share: float, with_score: bool):
    """Run Evidently; returns the snapshot (save_html / dict) or None when Evidently is missing."""
    ev = optional("evidently")
    if ev is None:
        return None
    from evidently import DataDefinition, Dataset, Report
    from evidently.metrics import ValueDrift
    from evidently.presets import DataDriftPreset
    num = NUMERICAL + ([SCORE] if with_score else [])
    dd = DataDefinition(numerical_columns=num, categorical_columns=CATEGORICAL)
    keep = num + CATEGORICAL
    metrics = [DataDriftPreset(columns=NUMERICAL + CATEGORICAL, drift_share=drift_share)]
    if with_score:
        metrics.append(ValueDrift(column=SCORE, method="psi"))
    report = Report(metrics, include_tests=True)
    return report.run(Dataset.from_pandas(cur[keep], data_definition=dd), Dataset.from_pandas(ref[keep], data_definition=dd))


def parse_snapshot(snap: dict) -> dict:
    """Per-column drift (method, score, threshold, drifted) and the drifted share from a snapshot dict."""
    failed = {t.get("metric_config", {}).get("metric_id") for t in snap.get("tests", []) if t.get("status") == "FAIL"}
    cols, dataset = {}, {}
    for m in snap.get("metrics", []):
        cfg = m.get("config", {})
        kind = str(cfg.get("type", ""))
        if kind.endswith("DriftedColumnsCount"):
            dataset = {"drifted_columns": int(m["value"]["count"]), "drifted_share": float(m["value"]["share"])}
        elif kind.endswith("ValueDrift"):
            key = (cfg["column"], str(cfg.get("method")))
            cols[key] = {"column": cfg["column"], "method": cfg.get("method"), "threshold": cfg.get("threshold"),
                         "drift_score": float(m["value"]), "drifted": m.get("id") in failed}
    return {"dataset": dataset, "columns": list(cols.values())}


# --------------------------------------------------------------------------- one window
def analyse(ref: pd.DataFrame, cur: pd.DataFrame, pol: dict, html: Path | None = None) -> dict:
    with_score = SCORE in ref and SCORE in cur and ref[SCORE].notna().any() and cur[SCORE].notna().any()
    out = {"evidently": version("evidently"), "columns": [], "dataset": {}, "score": {}}
    snap = evidently_report(ref, cur, pol["drift_share"], with_score)
    if snap is not None:
        parsed = parse_snapshot(snap.dict())
        out["dataset"] = parsed["dataset"]
        by_col = {c["column"]: c for c in parsed["columns"] if c["column"] != SCORE}
        score_psi = next((c for c in parsed["columns"] if c["column"] == SCORE and "psi" in str(c["method"]).lower()), None)
        if html is not None:
            html.parent.mkdir(parents=True, exist_ok=True)
            snap.save_html(str(html))
            out["html"] = html.relative_to(config.ROOT).as_posix() if html.is_relative_to(config.ROOT) else str(html)
        if score_psi:
            out["score"]["evidently_psi"] = score_psi["drift_score"]
    else:
        by_col = {}
    for c in NUMERICAL + CATEGORICAL:
        p = column_psi(ref, cur, c)
        row = by_col.get(c, {"column": c, "method": "psi (finguard)", "threshold": pol["score_psi_review"],
                             "drift_score": p, "drifted": bool(p >= pol["score_psi_review"])})
        out["columns"].append(row | {"type": "categorical" if c in CATEGORICAL else "numerical", "psi": round(p, 6),
                                     "missing_ref": round(float(ref[c].isna().mean()), 4),
                                     "missing_cur": round(float(cur[c].isna().mean()), 4)})
    if not out["dataset"]:
        n = sum(c["drifted"] for c in out["columns"])
        out["dataset"] = {"drifted_columns": n, "drifted_share": n / len(out["columns"])}
    out["dataset"] |= {"monitored_columns": len(out["columns"]),
                       "dataset_drift": out["dataset"]["drifted_share"] >= pol["drift_share"]}
    if with_score:
        s = out["score"]
        s["psi"] = psi_numeric(ref[SCORE].astype("float64"), cur[SCORE].astype("float64"))
        if "band" in cur and cur["band"].notna().any():
            for name, d in (("reference", ref), ("current", cur)):
                s[f"alert_share_{name}"] = float(d["band"].isin(["High", "Medium"]).mean())
                s[f"high_share_{name}"] = float((d["band"] == "High").mean())
        if config.TARGET in cur and cur[config.TARGET].notna().all() and cur[config.TARGET].sum() > 0:
            from finguard.evaluation.metrics import ranking
            s["pr_auc_reference"] = ranking(ref[config.TARGET].to_numpy(), ref["raw_score"].to_numpy())["pr_auc"]
            s["pr_auc_current"] = ranking(cur[config.TARGET].to_numpy(), cur["raw_score"].to_numpy())["pr_auc"]
    out["decision"], out["reasons"] = decide(out, pol)
    return out


def decide(res: dict, pol: dict) -> tuple[str, list[str]]:
    """Map the window's measurements to stable / review / retrain review, with the reasons."""
    ds, sc = res["dataset"], res["score"]
    found: list[tuple[int, str]] = []
    share = ds.get("drifted_share", 0.0)
    if share >= pol["drift_share_retrain"]:
        found.append((2, f"{share:.0%} of the monitored inputs drifted (>= {pol['drift_share_retrain']:.0%})"))
    elif ds.get("dataset_drift"):
        found.append((1, f"dataset drift: {share:.0%} of the monitored inputs drifted (>= {pol['drift_share']:.0%})"))
    if "psi" in sc:
        if sc["psi"] >= pol["score_psi_retrain"]:
            found.append((2, f"score PSI {sc['psi']:.3f} >= {pol['score_psi_retrain']}"))
        elif sc["psi"] >= pol["score_psi_review"]:
            found.append((1, f"score PSI {sc['psi']:.3f} >= {pol['score_psi_review']}"))
    if "pr_auc_current" in sc and sc.get("pr_auc_reference"):
        drop = 1 - sc["pr_auc_current"] / sc["pr_auc_reference"]
        if drop >= pol["pr_auc_drop_review"]:
            found.append((1, f"PR-AUC {sc['pr_auc_current']:.3f} vs {sc['pr_auc_reference']:.3f} on the reference "
                             f"({drop:.0%} lower, >= {pol['pr_auc_drop_review']:.0%})"))
    if "alert_share_current" in sc:
        plan, tol = pol["alert_share_planned"], pol["alert_share_tolerance"]
        if abs(sc["alert_share_current"] - plan) > tol * plan:
            found.append((1, f"alert share {sc['alert_share_current']:.1%} vs plan {plan:.0%} (+/-{tol:.0%})"))
    level = max((lv for lv, _ in found), default=0)
    return next(k for k, v in LEVEL.items() if v == level), [r for _, r in found] or ["no threshold crossed"]


# --------------------------------------------------------------------------- windows and trend
def slices(df: pd.DataFrame, start: int, end: int, width: int) -> list[tuple[int, int, pd.DataFrame]]:
    out = []
    for a in range(start, end + 1, width):
        b = min(a + width - 1, end)
        if end - b < width // 2 and b < end:              # fold a short tail into the last slice
            b = end
        part = df[df["day"].between(a, b)]
        if len(part):
            out.append((a, b, part))
        if b == end:
            break
    return out


def run(windows: tuple[str, ...] = ("test", "later"), html: bool = True, log=print, frames: dict | None = None,
        out_dir: Path = OUT_DIR, metrics_path: Path = METRICS) -> dict:
    pol = policy()
    frames = frames or load_frames()
    ref = frames["reference"]
    (v0, v1) = config.SPLIT_DAYS["valid"]
    summaries, trend = {}, []
    names = {"test": "test month (held out, labelled)", "later": "later file (unlabelled)"}
    for w in windows:
        cur = frames[w]
        t0 = time.time()
        res = analyse(ref, cur, pol, out_dir / f"drift_{w}.html" if html else None)
        lo, hi = int(cur["day"].min()), int(cur["day"].max())
        summary = {"run_at": time.strftime("%Y-%m-%d %H:%M:%S"), "window": w, "policy": pol,
                   "reference": {"name": "validation month", "days": [v0, v1], "rows": int(len(ref))},
                   "current": {"name": names.get(w, w), "days": [lo, hi], "rows": int(len(cur)),
                               "labelled": config.TARGET in cur}} | res
        width = 7 if w == "test" else 30
        for a, b, part in slices(cur, lo, hi, width):
            r = analyse(ref, part, pol, None)
            trend.append({"window": w, "start_day": a, "end_day": b, "rows": int(len(part)),
                          "drifted_columns": r["dataset"]["drifted_columns"], "drifted_share": round(r["dataset"]["drifted_share"], 4),
                          "score_psi": round(r["score"]["psi"], 4) if "psi" in r["score"] else None,
                          "alert_share": round(r["score"]["alert_share_current"], 4) if "alert_share_current" in r["score"] else None,
                          "pr_auc": round(r["score"]["pr_auc_current"], 4) if "pr_auc_current" in r["score"] else None,
                          "decision": r["decision"]})
        summary["trend"] = [t for t in trend if t["window"] == w]
        summary["seconds"] = round(time.time() - t0, 1)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"drift_{w}.summary.json"
        path.write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
        summaries[w] = summary
        log(line(summary))
        from finguard.mlops import tracking
        tracking.log_drift(summary, [path] + ([out_dir / f"drift_{w}.html"] if html else []))
    if trend:
        pd.DataFrame(trend).to_csv(out_dir / "drift_trend.csv", index=False)
    write_metrics(summaries, metrics_path)
    worst = max((s["decision"] for s in summaries.values()), key=LEVEL.get, default="stable")
    return {"decision": worst, "windows": summaries}


def write_metrics(summaries: dict, path: Path = METRICS) -> None:
    flat = {}
    for w, s in summaries.items():
        flat[w] = {"drifted_share": round(s["dataset"]["drifted_share"], 4), "drifted_columns": s["dataset"]["drifted_columns"],
                   "decision_level": LEVEL[s["decision"]]} | {k: round(v, 4) for k, v in s["score"].items() if isinstance(v, float)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(flat, indent=1), encoding="utf-8")


def line(s: dict) -> str:
    ds, sc = s["dataset"], s["score"]
    score = f", score PSI {sc['psi']:.3f}" if "psi" in sc else ""
    alert = f", alert share {sc['alert_share_current']:.1%}" if "alert_share_current" in sc else ""
    drifted = [c["column"] for c in s["columns"] if c["drifted"]]
    return (f"  {s['window']:<6} {s['current']['rows']:>7,} rows, days {s['current']['days'][0]}-{s['current']['days'][1]}: "
            f"{ds['drifted_columns']}/{ds['monitored_columns']} inputs drifted{score}{alert} -> {s['decision'].upper()}"
            + (f"\n         drifted: {', '.join(drifted)}" if drifted else "") + f"\n         {'; '.join(s['reasons'])}")
