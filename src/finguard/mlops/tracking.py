"""Experiment tracking and model registry on MLflow.

Every evaluation the project already writes to JSON is also recorded as an MLflow run, so the
whole history can be compared in one UI (``mlflow ui``):

| Experiment                  | One run per                                   | Logged by                         |
| finguard-phase2-models      | trained model (LightGBM, retrains, label-delay | ``finguard.phase2``, ``phase2b``,  |
|                             | run, ablation, baselines) + the Phase 2 report | ``track sync``                    |
| finguard-phase3-retrieval   | retrieval evaluation of the frozen questions   | ``finguard.rag evaluate``          |
| finguard-phase4-agents      | agent evaluation (per LLM)                     | ``finguard.agents evaluate``       |
| finguard-monitoring         | drift report                                   | ``finguard.mlops drift``           |

Model registry: ``finguard.registry register`` (after the quality gate) logs the calibrated
scorer as an MLflow model (``scorer_model.py``: LightGBM + isotonic calibration + frozen bands)
and registers a version of ``finguard-fraud-scorer``; ``finguard.registry promote`` sets the
same alias (``champion``) there, with the approver as a version tag. The file registry
(``artifacts/registry.json``) stays the source of truth for the gate; MLflow mirrors it.

Configuration (environment):
    MLFLOW_TRACKING_URI     default: sqlite:///<project>/mlflow.db (artifacts in <project>/mlruns)
    FINGUARD_MLFLOW         auto (default: log when mlflow is installed) | off
    FINGUARD_MLFLOW_MODEL   registered model name (default: finguard-fraud-scorer)

Logging never breaks a pipeline: a tracking error is printed and the step goes on.
Backfill everything already computed, without retraining (idempotent):

    python -m finguard.mlops track sync
    mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

from finguard import config
from finguard.mlops import optional

ART = Path(config.ROOT / "artifacts" / "phase2")
T = config.TABLE_DIR
EXPERIMENTS = {"models": "finguard-phase2-models", "retrieval": "finguard-phase3-retrieval",
               "agents": "finguard-phase4-agents", "monitoring": "finguard-monitoring"}
MODEL_NAME = os.environ.get("FINGUARD_MLFLOW_MODEL", "finguard-fraud-scorer")
SCORER_FILE = Path(__file__).with_name("scorer_model.py")
PHASE2_FIGURES = ["fig17_model_comparison", "fig18_precision_recall_curves", "fig19_reliability", "fig20_risk_bands",
                  "fig21_weekly_performance", "fig22_new_vs_known_entities", "fig23_reason_family_importance",
                  "fig24_reason_stability", "fig25_novelty_flags", "fig26_isolation_forest"]


# --------------------------------------------------------------------------- setup
def enabled() -> bool:
    return os.environ.get("FINGUARD_MLFLOW", "auto").lower() not in ("off", "0", "false", "no") and \
        optional("mlflow") is not None


def tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{(config.ROOT / 'mlflow.db').as_posix()}"


def artifact_root() -> str:
    """Artifacts sit next to a SQLite store (<dir of mlflow.db>/mlruns), else in <project>/mlruns."""
    uri = tracking_uri()
    base = Path(uri[len("sqlite:///"):]).parent if uri.startswith("sqlite:///") else config.ROOT
    return (base / "mlruns").resolve().as_uri()


def client():
    import mlflow
    from mlflow import MlflowClient
    mlflow.set_tracking_uri(tracking_uri())
    return MlflowClient(tracking_uri=tracking_uri())


def experiment_id(key: str, c=None) -> str:
    c = c or client()
    name = EXPERIMENTS[key]
    exp = c.get_experiment_by_name(name)
    if exp is not None:
        return exp.experiment_id
    uri = tracking_uri()
    loc = f"{artifact_root()}/{name}" if uri.startswith("sqlite:") or uri.startswith("file:") else None
    return c.create_experiment(name, artifact_location=loc, tags={"project": "finguard"})


def safe(fn):
    """Decorator: tracking is a side channel; an error is printed, never raised into the pipeline."""
    def wrapper(*a, **k):
        if not enabled():
            return {"status": "skipped (mlflow not installed or FINGUARD_MLFLOW=off)"}
        try:
            return fn(*a, **k)
        except Exception as exc:                                  # pragma: no cover - depends on the store
            print(f"[mlflow] {fn.__name__} not logged: {type(exc).__name__}: {exc}")
            return {"status": f"error: {type(exc).__name__}: {exc}"}
    wrapper.__name__, wrapper.__doc__ = fn.__name__, fn.__doc__
    return wrapper


# --------------------------------------------------------------------------- helpers
def metric_name(*parts) -> str:
    """MLflow names allow letters, digits, _ - . /: precision@1% -> precision_at_1pct."""
    s = ".".join(str(p) for p in parts if p not in (None, ""))
    s = s.replace("@", "_at_").replace("%", "pct").replace("+", "plus")
    return re.sub(r"_+", "_", re.sub(r"[^\w.\-/]", "_", s))


ID_KEYS = ("variant", "label", "name", "mode", "part", "flag", "group", "band", "split", "half")


def flatten(d, prefix: str = "", depth: int = 4) -> dict[str, float]:
    """Numeric leaves of nested dicts (and of lists of records named by keys such as "variant" or
    "band") as dotted metric names; booleans become 0/1, strings and other lists are skipped."""
    out: dict[str, float] = {}
    if depth < 0 or d is None:
        return out
    if isinstance(d, dict):
        for k, v in d.items():
            out |= flatten(v, metric_name(prefix, k), depth - 1)
    elif isinstance(d, list) and d and all(isinstance(x, dict) for x in d):
        for x in d:
            ids = [k for k in ID_KEYS if isinstance(x.get(k), str)]
            if ids:
                out |= flatten({k: v for k, v in x.items() if k not in ids},
                               metric_name(prefix, "_".join(x[k] for k in ids)), depth - 1)
    elif isinstance(d, bool):
        out[prefix] = float(d)
    elif isinstance(d, (int, float)) and d == d and abs(d) != float("inf"):
        out[prefix] = float(d)
    return out


def _sha(path: Path, n: int = 16) -> str | None:
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest()[:n] if p.exists() else None


def _ms(when: str | float | None) -> int | None:
    if when is None:
        return None
    if isinstance(when, (int, float)):
        return int(when * 1000)
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return int(datetime.strptime(when, fmt).timestamp() * 1000)
        except ValueError:
            continue
    return None


def _existing(c, exp_id: str, source_hash: str):
    runs = c.search_runs([exp_id], filter_string=f"tags.finguard_source_hash = '{source_hash}'", max_results=1)
    return runs[0] if runs else None


def _code() -> str:
    from finguard.registry import code_fingerprint
    return code_fingerprint()


def log_run(key: str, run_name: str, source: str, source_hash: str, params: dict | None = None,
            metrics: dict | None = None, tags: dict | None = None, artifacts: list | None = None,
            steps: dict[str, list] | None = None, started: int | None = None, note: str = "", c=None) -> dict:
    """Create one finished run (or return the existing one for the same source content)."""
    from mlflow.entities import Metric, Param
    c = c or client()
    exp = experiment_id(key, c)
    old = _existing(c, exp, source_hash)
    if old is not None:
        return {"status": "exists", "run_id": old.info.run_id, "experiment": EXPERIMENTS[key]}
    now = int(time.time() * 1000)
    started = started or now
    run = c.create_run(exp, start_time=started, run_name=run_name,
                       tags={"finguard_source": source, "finguard_source_hash": source_hash, "project": "finguard",
                             **({"mlflow.note.content": note} if note else {}), **{k: str(v) for k, v in (tags or {}).items()}})
    rid = run.info.run_id
    ps = [Param(k[:250], str(v)[:6000]) for k, v in (params or {}).items() if v is not None]
    ms = [Metric(k[:250], float(v), started, 0) for k, v in (metrics or {}).items()]
    for name, values in (steps or {}).items():
        ms += [Metric(name, float(v), started, i) for i, v in enumerate(values) if v == v]
    for i in range(0, max(len(ps), 1), 100):                      # the store limits a batch to 100 params
        c.log_batch(rid, params=ps[i:i + 100])
    for i in range(0, len(ms), 1000):                             # ... and to 1000 metrics
        c.log_batch(rid, metrics=ms[i:i + 1000])
    for item in artifacts or []:
        path, sub = (item, None) if isinstance(item, (str, Path)) else item
        if Path(path).exists():
            c.log_artifact(rid, str(path), artifact_path=sub)
    c.set_terminated(rid, "FINISHED", end_time=max(now, started))
    return {"status": "logged", "run_id": rid, "experiment": EXPERIMENTS[key]}


# --------------------------------------------------------------------------- Phase 2
def _phase2_payload(name: str) -> dict:
    d = ART / name
    info = json.loads((d / "info.json").read_text(encoding="utf-8"))
    params = {k: v for k, v in info.items() if isinstance(v, (str, int, float, bool)) and k not in ("seconds", "name")}
    params |= {f"lgbm.{k}": v for k, v in (info.get("params") or {}).items()}
    metrics = flatten({k: v for k, v in info.items() if k not in ("params", "valid_curve", "best_iteration", "n_features",
                                                                  "n_categorical", "C", "n_inputs")})
    steps = {"valid.pr_auc_by_iteration": info["valid_curve"]} if isinstance(info.get("valid_curve"), list) else None
    model = d / "model.txt"
    tags = {"phase": "2", "model_sha256": _sha(model, 64), "code": _code(),
            "data_manifest_sha256": _sha(config.INTERIM_DIR / "manifest.json", 64)}
    arts = [d / "info.json", d / "columns.json", d / "coefficients.csv"]
    return {"params": params, "metrics": metrics, "steps": steps, "tags": {k: v for k, v in tags.items() if v},
            "artifacts": arts, "started": _ms((d / "info.json").stat().st_mtime),
            "hash": "p2:" + name + ":" + _sha(d / "info.json")}


@safe
def log_phase2_model(name: str) -> dict:
    """One trained Phase 2 model (``artifacts/phase2/<name>/info.json``)."""
    p = _phase2_payload(name)
    return log_run("models", name, f"phase2/{name}", p["hash"], p["params"], p["metrics"], p["tags"], p["artifacts"],
                   p["steps"], p["started"], note=f"Phase 2 model `{name}`: see artifacts/phase2/{name}/info.json")


@safe
def log_phase2_report() -> dict:
    """The Phase 2 report (notebooks 06-09): calibration, intervals, bands, gate, figures."""
    res_path = T / "phase2_results.json"
    res = json.loads(res_path.read_text(encoding="utf-8"))
    metrics = flatten({"test": res.get("test_metrics"), "ci95_block": {k: {"lo": v[0], "hi": v[1]} for k, v in
                                                                        (res.get("block_bootstrap_ci95_lightgbm") or {}).items()},
                       "calibration": res.get("calibration"), "bands": res.get("bands", {}).get("test"),
                       "paired_lightgbm_minus_lr": res.get("paired_lightgbm_minus_lr")})
    gate = ART / "gate_report.json"
    if gate.exists():
        g = json.loads(gate.read_text(encoding="utf-8"))
        metrics["gate.passed"] = float(g["passed"])
        metrics["gate.checks_failed"] = float(sum(not c["passed"] for c in g["checks"]))
    params = {"selected_codes": (res.get("selection") or {}).get("codes_as"),
              "band_threshold_high_raw": (res.get("bands") or {}).get("thresholds_raw", {}).get("high"),
              "band_threshold_medium_raw": (res.get("bands") or {}).get("thresholds_raw", {}).get("medium")}
    arts = [res_path, gate, T / "phase2b_summary.json", T / "phase2_scorecard.csv", T / "phase2_weekly.csv"]
    arts += [(config.FIG_DIR / f"{f}.png", "figures") for f in PHASE2_FIGURES]
    h = "p2report:" + "".join(_sha(p) or "-" for p in (res_path, gate))
    return log_run("models", "phase2-report", "phase2/report", h, params, metrics, {"phase": "2"}, arts,
                   started=_ms(res_path.stat().st_mtime), note="Phase 2 report: test metrics, intervals, calibration, bands, gate")


# --------------------------------------------------------------------------- Phase 3
def _retrieval_metrics(payload: dict) -> dict:
    m = flatten({"all": {r["variant"]: {k: v for k, v in r.items() if k != "variant"} for r in payload.get("summary") or []}})
    for r in payload.get("by_split") or []:
        m |= flatten({k: v for k, v in r.items() if k not in ("variant", "split")}, metric_name(r["split"], r["variant"]))
    ab = payload.get("abstention") or {}
    m |= flatten({k: ab.get(k) for k in ("threshold", "dev", "test", "all")}, "abstention")
    return m


@safe
def log_retrieval(payload: dict, note: str = "", started: str | None = None, source_hash: str | None = None,
                  run_name: str | None = None) -> dict:
    """One evaluation of the frozen question set (``finguard.rag evaluate``)."""
    ab = payload.get("abstention") or {}
    params = {"index_version": payload.get("index_version"), "embedder": payload.get("embedder"),
              "chosen_variant": payload.get("chosen_variant"),
              "abstention_variant": payload.get("abstention_variant") or ab.get("variant"),
              "grounding_check": ab.get("grounding_check"),
              "questions_sha256": (payload.get("questions") or {}).get("sha256") if isinstance(payload.get("questions"), dict) else None}
    h = source_hash or "p3:" + hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]
    arts = [T / "phase3_retrieval_results.json", T / "phase3_retrieval_summary.csv"] if source_hash is None else []
    name = run_name or f"retrieval {payload.get('chosen_variant') or ''} {payload.get('index_version') or ''}".strip()
    return log_run("retrieval", name,
                   "phase3/evaluate", h, params, _retrieval_metrics(payload), {"phase": "3"}, arts,
                   started=_ms(started), note=note or "retrieval evaluation (frozen question set)")


def sync_retrieval() -> list[dict]:
    out = []
    runs_path = T / "phase3_retrieval_runs.json"
    if runs_path.exists():
        for r in json.loads(runs_path.read_text(encoding="utf-8")):
            h = "p3run:" + hashlib.sha256(json.dumps(r, sort_keys=True, default=str).encode()).hexdigest()[:16]
            out.append(log_retrieval(r, note=f"run {r.get('run')}: {r.get('note', '')}", started=r.get("at"), source_hash=h,
                                     run_name=f"retrieval run {r.get('run')}"))
    res = T / "phase3_retrieval_results.json"
    if res.exists():
        out.append(log_retrieval(json.loads(res.read_text(encoding="utf-8")), note="current results (phase3_retrieval_results.json)",
                                 started=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(res.stat().st_mtime))))
    return out


# --------------------------------------------------------------------------- Phase 4
@safe
def log_agents(payload: dict, table: Path | None = None) -> dict:
    """One agent evaluation (``finguard.agents evaluate``), both paths."""
    params = {"llm": payload.get("llm"), "llm_model": payload.get("llm_model"), "graph": payload.get("graph"),
              "label": payload.get("label"), "prompts": ",".join(payload.get("prompts") or []),
              "cases_done": payload.get("cases_done"), "cases_total": payload.get("cases_total")}
    params |= {f"versions.{k}": v for k, v in (payload.get("versions") or {}).items()}
    metrics = flatten({"summary": payload.get("summary"), "exit_criteria": payload.get("exit_criteria")})
    h = "p4:" + hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]
    tag = f"_{payload['llm']}" if payload.get("llm") else ""
    arts = [T / f"phase4_agent_summary{tag}.json", Path(table) if table else T / f"phase4_agent_cases{tag}.csv"]
    return log_run("agents", f"agents {payload.get('llm')} {payload.get('llm_model') or ''}".strip(), "phase4/evaluate", h,
                   params, metrics, {"phase": "4", "llm": payload.get("llm")}, arts, started=_ms(payload.get("run_at")),
                   note=payload.get("note") or payload.get("label") or "agent evaluation (20 scripted cases)")


# --------------------------------------------------------------------------- monitoring
@safe
def log_drift(summary: dict, files: list[Path] = ()) -> dict:
    """One drift report (``finguard.mlops drift``)."""
    params = {"reference": summary.get("reference", {}).get("name"), "current": summary.get("current", {}).get("name"),
              "evidently": summary.get("evidently"), "decision": summary.get("decision"),
              "policy": json.dumps(summary.get("policy"), sort_keys=True)}
    metrics = flatten({"dataset": summary.get("dataset"), "score": summary.get("score"),
                       "columns": {c["column"]: {"drift_score": c.get("drift_score"), "psi": c.get("psi"),
                                                 "drifted": c.get("drifted")} for c in summary.get("columns", [])}})
    h = "drift:" + hashlib.sha256(json.dumps(summary, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return log_run("monitoring", f"drift {summary.get('window')}", "monitoring/drift", h, params, metrics,
                   {"decision": summary.get("decision"), "window": summary.get("window")}, list(files),
                   started=_ms(summary.get("run_at")), note=f"drift: {summary.get('window')} -> {summary.get('decision')}")


# --------------------------------------------------------------------------- registry
def calibration_payload(bundle: dict) -> dict:
    iso, b = bundle["isotonic"], bundle["bands"]
    return {"columns": bundle["columns"]["columns"], "categorical": bundle["columns"].get("categorical", []),
            "isotonic_x": [float(x) for x in iso.X_thresholds_], "isotonic_y": [float(y) for y in iso.y_thresholds_],
            "bands": {"high": float(b.high), "medium": float(b.medium), "high_share": float(b.high_share),
                      "medium_share": float(b.medium_share)}, "codes_as": bundle.get("codes_as")}


def _signature(columns: list[str]):
    from mlflow.models import ModelSignature
    from mlflow.types import ColSpec, Schema
    return ModelSignature(inputs=Schema([ColSpec("double", c) for c in columns]),
                          outputs=Schema([ColSpec("double", "raw_score"), ColSpec("double", "probability"),
                                          ColSpec("double", "score"), ColSpec("string", "band")]))


def _version_for_sha(c, sha: str):
    found = c.search_model_versions(f"name = '{MODEL_NAME}' and tags.model_sha256 = '{sha}'")
    return found[0] if found else None


@safe
def register_model(entry: dict, model_file: Path | None = None, bundle_file: Path | None = None) -> dict:
    """Log the calibrated scorer for a file-registry entry and register it as a model version."""
    import pickle
    import tempfile

    import mlflow
    c = client()
    model_file = Path(model_file or config.ROOT / entry["model_file"])
    bundle_file = Path(bundle_file or ART / "model_bundle.pkl")
    if not model_file.exists() or not bundle_file.exists():
        return {"status": f"skipped (missing {model_file.name if not model_file.exists() else bundle_file.name})"}
    sha = entry.get("model_sha256") or _sha(model_file, 64)
    old = _version_for_sha(c, sha)
    if old is not None:
        return {"status": "exists", "name": MODEL_NAME, "version": str(old.version), "run_id": old.run_id}
    with open(bundle_file, "rb") as f:
        bundle = pickle.load(f)
    if Path(bundle["model_file"]).name != model_file.name or bundle.get("codes_as") not in entry["name"]:
        return {"status": f"skipped (model_bundle.pkl belongs to {bundle['model_file']}, not {entry['model_file']})"}
    cal = calibration_payload(bundle)
    exp = experiment_id("models", c)
    source = c.search_runs([exp], filter_string=f"tags.finguard_source = 'phase2/{entry['name']}'",
                           order_by=["attributes.start_time DESC"], max_results=1)
    import lightgbm
    with tempfile.TemporaryDirectory() as tmp:
        cal_path = Path(tmp) / "calibration.json"
        cal_path.write_text(json.dumps(cal), encoding="utf-8")
        mlflow.set_tracking_uri(tracking_uri())
        run_kw = {"run_id": source[0].info.run_id} if source else {"experiment_id": exp, "run_name": f"register {entry['model_id']}"}
        with mlflow.start_run(**run_kw) as run:
            info = mlflow.pyfunc.log_model(
                name="fraud-scorer", python_model=str(SCORER_FILE), artifacts={"booster": str(model_file), "calibration": str(cal_path)},
                signature=_signature(cal["columns"]), registered_model_name=MODEL_NAME,
                pip_requirements=[f"lightgbm=={lightgbm.__version__}", "numpy", "pandas", f"mlflow=={mlflow.__version__}"],
                metadata={"finguard_model_id": entry["model_id"], "model_sha256": sha})
            mlflow.set_tags({"registered_model_id": entry["model_id"]})
    version = str(info.registered_model_version)
    for k, v in {"model_id": entry["model_id"], "model_sha256": sha, "code": entry.get("code"),
                 "data_manifest_sha256": entry.get("data_manifest_sha256"),
                 "gate_passed": (entry.get("gate") or {}).get("passed")}.items():
        if v is not None:
            c.set_model_version_tag(MODEL_NAME, version, k, str(v))
    test = (entry.get("metrics") or {}).get("test") or {}
    c.update_model_version(MODEL_NAME, version, description=(
        f"{entry['model_id']}: LightGBM + isotonic calibration + frozen bands. Test PR-AUC {test.get('pr_auc', float('nan')):.3f}, "
        f"gate {'passed' if (entry.get('gate') or {}).get('passed') else 'not passed'}."))
    return {"status": "registered", "name": MODEL_NAME, "version": version, "run_id": run.info.run_id, "model_uri": info.model_uri}


@safe
def set_alias(alias: str, version: str, approved_by: str | None = None, model_id: str | None = None) -> dict:
    c = client()
    c.set_registered_model_alias(MODEL_NAME, alias, str(version))
    if approved_by:
        c.set_model_version_tag(MODEL_NAME, str(version), f"{alias}_approved_by", approved_by)
    c.set_model_version_tag(MODEL_NAME, str(version), f"{alias}_since", time.strftime("%Y-%m-%d %H:%M:%S"))
    return {"status": "alias set", "name": MODEL_NAME, "alias": alias, "version": str(version), "model_id": model_id}


def sync_registry() -> list[dict]:
    """Mirror ``artifacts/registry.json`` into the MLflow registry: versions, then aliases."""
    from finguard import registry
    reg = registry.load()
    out, changed = [], False
    for m in reg["models"]:
        res = register_model(m)
        out.append({"model_id": m["model_id"]} | res)
        if res.get("version") and (m.get("mlflow") or {}).get("version") != res["version"]:
            m["mlflow"] = {"status": "registered", "name": res["name"], "version": res["version"], "run_id": res.get("run_id")}
            changed = True
    for alias, mid in reg.get("aliases", {}).items():
        m = next((x for x in reg["models"] if x["model_id"] == mid), None)
        v = ((m or {}).get("mlflow") or {}).get("version")
        if v:
            who = next((h.get("approved_by") for h in reversed(reg.get("history", []))
                        if h.get("event") == "promote" and h.get("alias") == alias and h.get("model_id") == mid), None)
            out.append(set_alias(alias, v, who, mid))
    if changed:
        registry.save(reg)
    return out


# --------------------------------------------------------------------------- backfill and status
def sync(what: str = "all", log=print) -> dict:
    """Record every result already on disk (idempotent: unchanged results are not logged twice)."""
    if not enabled():
        log("mlflow is not installed (pip install -e \".[mlops]\") or FINGUARD_MLFLOW=off: nothing to sync")
        return {"status": "skipped"}
    done: dict[str, list] = {}
    if what in ("all", "models"):
        names = sorted(p.parent.name for p in ART.glob("*/info.json"))
        done["models"] = [{"name": n} | log_phase2_model(n) for n in names]
        if (T / "phase2_results.json").exists():
            done["models"].append({"name": "phase2-report"} | log_phase2_report())
    if what in ("all", "retrieval"):
        done["retrieval"] = sync_retrieval()
    if what in ("all", "agents"):
        done["agents"] = [log_agents(json.loads(p.read_text(encoding="utf-8")), p.with_name(p.name.replace("summary", "cases")).with_suffix(".csv"))
                          for p in sorted(T.glob("phase4_agent_summary*.json"))]
    if what in ("all", "monitoring"):
        mon = config.REPORTS_DIR / "monitoring"
        done["monitoring"] = [log_drift(json.loads(p.read_text(encoding="utf-8")),
                                        [p, p.with_name(p.name.replace(".summary.json", ".html"))])
                              for p in sorted(mon.glob("drift_*.summary.json"))]
    if what in ("all", "registry"):
        done["registry"] = sync_registry()
    for k, v in done.items():
        counts: dict[str, int] = {}
        for r in v:
            s = str(r.get("status", "?")).split(" (")[0]
            counts[s] = counts.get(s, 0) + 1
        log(f"  {k:<11} " + ", ".join(f"{n} {s}" for s, n in sorted(counts.items())))
    log(f"tracking store: {tracking_uri()}\nopen the UI:    mlflow ui --backend-store-uri {_ui_uri()} --port 5000")
    return done


def _ui_uri() -> str:
    uri = tracking_uri()
    default = f"sqlite:///{(config.ROOT / 'mlflow.db').as_posix()}"
    return "sqlite:///mlflow.db" if uri == default else uri


def status() -> dict:
    if not enabled():
        return {"enabled": False, "reason": "mlflow not installed or FINGUARD_MLFLOW=off"}
    c = client()
    exps = {}
    for key, name in EXPERIMENTS.items():
        e = c.get_experiment_by_name(name)
        exps[name] = 0 if e is None else len(c.search_runs([e.experiment_id], max_results=1000))
    versions, aliases = [], {}
    try:
        rm = c.get_registered_model(MODEL_NAME)
        aliases = dict(rm.aliases or {})
        versions = sorted(int(v.version) for v in c.search_model_versions(f"name = '{MODEL_NAME}'"))
    except Exception:
        pass
    return {"enabled": True, "tracking_uri": tracking_uri(), "runs": exps, "registered_model": MODEL_NAME,
            "versions": versions, "aliases": aliases}
