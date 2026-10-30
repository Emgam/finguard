"""Model registry with champion / challenger aliases (Project Report v3.0, section 9).

File-based by default (``artifacts/registry.json``): each entry records the model file's
SHA-256, the data manifest's SHA-256, a code fingerprint (Git commit when available, else a
hash of ``src/``), parameters, metrics and the gate report. Promotion to an alias needs a
passing gate AND a named person's approval; nothing is promoted automatically.

When the ``mlflow`` package is installed and ``MLFLOW_TRACKING_URI`` is set, ``register``
also logs the run (parameters, metrics, the LightGBM model) and registers it in the MLflow
Model Registry under ``FINGUARD_MLFLOW_MODEL`` (default "finguard-lgbm"); ``promote`` then
sets the same alias there. This path was not exercised in the build environment (no MLflow).

    python -m finguard.registry register --name lgbm_numeric
    python -m finguard.registry promote MODEL_ID --alias champion --approved-by "Full Name"
    python -m finguard.registry show
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from finguard import config

REGISTRY = Path(config.ROOT / "artifacts" / "registry.json")
ART = Path(config.ROOT / "artifacts" / "phase2")


def _sha(path: Path) -> str | None:
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def code_fingerprint() -> str:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=config.ROOT, capture_output=True, text=True, timeout=5)
        if commit.returncode == 0 and commit.stdout.strip():
            dirty = subprocess.run(["git", "status", "--porcelain", "src"], cwd=config.ROOT, capture_output=True, text=True).stdout
            return "git:" + commit.stdout.strip()[:12] + ("+dirty" if dirty.strip() else "")
    except Exception:
        pass
    h = hashlib.sha256()
    for p in sorted(Path(config.ROOT / "src").rglob("*.py")):
        h.update(p.relative_to(config.ROOT).as_posix().encode())
        h.update(p.read_bytes())
    return "src:" + h.hexdigest()[:12]


def load() -> dict:
    return json.loads(REGISTRY.read_text()) if REGISTRY.exists() else {"models": [], "aliases": {}, "history": []}


def save(reg: dict) -> None:
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(json.dumps(reg, indent=1, default=float))


def register(name: str = "lgbm_numeric", gate_report: dict | None = None) -> dict:
    info = json.loads((ART / name / "info.json").read_text())
    model_file = ART / name / "model.txt"
    reg = load()
    sha = _sha(model_file)
    existing = next((m for m in reg["models"] if m["model_sha256"] == sha), None)
    if existing:
        if gate_report is not None:
            existing["gate"] = gate_report
            save(reg)
        return existing
    entry = {"model_id": f"{name}-v{len(reg['models']) + 1}", "name": name, "registered": time.strftime("%Y-%m-%d %H:%M:%S"),
             "model_file": str(model_file.relative_to(config.ROOT)), "model_sha256": sha,
             "data_manifest_sha256": _sha(config.INTERIM_DIR / "manifest.json"), "code": code_fingerprint(),
             "params": {k: v for k, v in info.get("params", {}).items() if k != "metric"},
             "best_iteration": info.get("best_iteration"), "metrics": {"valid": info.get("valid"), "test": info.get("test")},
             "gate": gate_report}
    reg["models"].append(entry)
    reg["history"].append({"event": "register", "model_id": entry["model_id"], "at": entry["registered"]})
    entry["mlflow"] = _mlflow_register(entry, model_file)
    save(reg)
    return entry


def promote(model_id: str, alias: str, approved_by: str) -> dict:
    if not approved_by or not approved_by.strip():
        raise ValueError("promotion needs the name of the person who approves it")
    reg = load()
    entry = next((m for m in reg["models"] if m["model_id"] == model_id), None)
    if entry is None:
        raise KeyError(model_id)
    if not (entry.get("gate") or {}).get("passed"):
        raise RuntimeError(f"{model_id} has no passing gate report; run `python -m finguard.gate` first")
    previous = reg["aliases"].get(alias)
    reg["aliases"][alias] = model_id
    reg["history"].append({"event": "promote", "alias": alias, "model_id": model_id, "previous": previous,
                           "approved_by": approved_by.strip(), "at": time.strftime("%Y-%m-%d %H:%M:%S")})
    if entry.get("mlflow", {}).get("version"):
        _mlflow_alias(alias, entry["mlflow"]["version"])
    save(reg)
    return reg


def _mlflow_register(entry: dict, model_file: Path) -> dict:
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        return {"status": "skipped (MLFLOW_TRACKING_URI not set)"}
    try:
        import lightgbm as lgb
        import mlflow
        import mlflow.lightgbm
    except ImportError:
        return {"status": "skipped (mlflow not installed)"}
    name = os.environ.get("FINGUARD_MLFLOW_MODEL", "finguard-lgbm")
    with mlflow.start_run(run_name=entry["model_id"]) as run:
        mlflow.log_params({k: v for k, v in entry["params"].items() if isinstance(v, (int, float, str, bool))})
        for part in ("valid", "test"):
            for k, v in (entry["metrics"].get(part) or {}).items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"{part}_{k.replace('@', '_at_').replace('%', 'pct')}", float(v))
        mlflow.set_tags({"model_sha256": entry["model_sha256"], "code": entry["code"], "data_manifest": entry["data_manifest_sha256"]})
        info = mlflow.lightgbm.log_model(lgb.Booster(model_file=str(model_file)), artifact_path="model", registered_model_name=name)
    return {"status": "registered", "run_id": run.info.run_id, "name": name,
            "version": getattr(info, "registered_model_version", None)}


def _mlflow_alias(alias: str, version) -> None:
    try:
        from mlflow import MlflowClient
        MlflowClient().set_registered_model_alias(os.environ.get("FINGUARD_MLFLOW_MODEL", "finguard-lgbm"), alias, str(version))
    except Exception as exc:                                     # pragma: no cover - optional path
        print(f"MLflow alias not set: {exc}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("register"); r.add_argument("--name", default="lgbm_numeric")
    p = sub.add_parser("promote"); p.add_argument("model_id"); p.add_argument("--alias", default="champion"); p.add_argument("--approved-by", required=True)
    sub.add_parser("show")
    a = ap.parse_args()
    if a.cmd == "register":
        print(json.dumps(register(a.name), indent=1, default=float)[:2000])
    elif a.cmd == "promote":
        print(json.dumps(promote(a.model_id, a.alias, a.approved_by)["aliases"], indent=1))
    else:
        reg = load()
        print(json.dumps({"aliases": reg["aliases"], "models": [{k: m[k] for k in ("model_id", "registered", "code")} | {
            "gate_passed": (m.get("gate") or {}).get("passed")} for m in reg["models"]], "history": reg["history"]}, indent=1))


if __name__ == "__main__":
    main()
