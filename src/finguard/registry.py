"""Model registry with champion / challenger aliases (Project Report v3.0, section 9).

File-based by default (``artifacts/registry.json``): each entry records the model file's
SHA-256, the data manifest's SHA-256, a code fingerprint (Git commit when available, else a
hash of ``src/``), parameters, metrics and the gate report. Promotion to an alias needs a
passing gate AND a named person's approval; nothing is promoted automatically.

When MLflow is installed (``pip install -e ".[mlops]"``), ``register`` also logs the calibrated
scorer (LightGBM + isotonic calibration + frozen bands, ``finguard/mlops/scorer_model.py``) and
registers it as a version of ``finguard-fraud-scorer`` (``FINGUARD_MLFLOW_MODEL``); ``promote``
sets the same alias there, with the approver as a version tag. The store defaults to
``sqlite:///mlflow.db`` in the project (``MLFLOW_TRACKING_URI`` overrides it); see
``finguard.mlops.tracking``. ``python -m finguard.mlops track sync`` mirrors entries made before.

    python -m finguard.registry register --name lgbm_numeric
    python -m finguard.registry promote MODEL_ID --alias champion --approved-by "Full Name"
    python -m finguard.registry show
"""
from __future__ import annotations

import argparse
import hashlib
import json
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
    return json.loads(REGISTRY.read_text(encoding="utf-8")) if REGISTRY.exists() else {"models": [], "aliases": {}, "history": []}


def save(reg: dict) -> None:
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(json.dumps(reg, indent=1, default=float), encoding="utf-8")


def register(name: str = "lgbm_numeric", gate_report: dict | None = None) -> dict:
    info = json.loads((ART / name / "info.json").read_text(encoding="utf-8"))
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
    if not (entry.get("mlflow") or {}).get("version"):          # registered before MLflow was installed
        res = _mlflow_register(entry, config.ROOT / entry["model_file"])
        if res.get("version"):
            entry["mlflow"] = {k: res.get(k) for k in ("status", "name", "version", "run_id")}
    if (entry.get("mlflow") or {}).get("version"):
        _mlflow_alias(alias, entry["mlflow"]["version"], approved_by.strip(), model_id)
    save(reg)
    return reg


def _mlflow_register(entry: dict, model_file: Path) -> dict:
    from finguard.mlops import tracking
    return tracking.register_model(entry, model_file)


def _mlflow_alias(alias: str, version, approved_by: str | None = None, model_id: str | None = None) -> None:
    from finguard.mlops import tracking
    res = tracking.set_alias(alias, str(version), approved_by, model_id)
    if not str(res.get("status", "")).startswith("alias"):
        print(f"MLflow alias not set: {res.get('status')}")


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
