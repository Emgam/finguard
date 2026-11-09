"""MLOps layer: data contracts (Pandera), experiment tracking and model registry (MLflow),
drift monitoring (Evidently) and the reproducible pipeline (DVC, see ``dvc.yaml``).

Every module here is optional at run time: the project's pipelines call into it, and each
call does nothing (with one printed line) when the library it needs is not installed, so the
model, the knowledge base, the agents and the API keep working without the ``mlops`` extra.

    python -m finguard.mlops check                 # which of the four tools are installed and wired
    python -m finguard.mlops contracts --stage data
    python -m finguard.mlops drift
    python -m finguard.mlops track sync            # backfill MLflow from the saved results
    python -m finguard.mlops metrics               # flat metric files for `dvc metrics show`
    python -m finguard.mlops pipeline              # which DVC stages already have their outputs
"""
from __future__ import annotations

import importlib
import importlib.metadata
from types import ModuleType

TOOLS = {"pandera": "pandera", "mlflow": "mlflow", "evidently": "evidently", "dvc": "dvc"}


def optional(module: str) -> ModuleType | None:
    """Import an optional dependency, or return None when it is not installed."""
    try:
        return importlib.import_module(module)
    except ImportError:
        return None


def version(dist: str) -> str | None:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def versions() -> dict[str, str | None]:
    return {name: version(dist) for name, dist in TOOLS.items()}


def params(section: str, defaults: dict | None = None) -> dict:
    """One section of ``params.yaml`` (the file DVC tracks) over the given defaults."""
    from finguard import config
    out = dict(defaults or {})
    path = config.ROOT / "params.yaml"
    yaml = optional("yaml")
    if path.exists() and yaml is not None:
        out |= (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get(section) or {}
    return out
