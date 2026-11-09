"""Flat metric files for DVC (``dvc metrics show``, ``dvc metrics diff main``).

The project's result files are nested (lists of variants, bands, intervals); DVC compares flat
key -> number maps best, so this step copies the headline numbers into reports/metrics/:

    model.json       selected LightGBM: validation / test PR-AUC, ROC-AUC, recall and precision
                     at 1%, the entity-block 95% interval, calibrated Brier and ECE, gate result
    retrieval.json   chosen retrieval variant: Hit@1/5 and MRR@10 (all, dev, test), abstention
    agents.json      per evaluated LLM and path: outcome accuracy, policy compliance, citation
                     and traceability rates, fallback rate, median seconds
    drift.json       written by ``finguard.mlops drift``

    python -m finguard.mlops metrics
"""
from __future__ import annotations

import json
from pathlib import Path

from finguard import config

ART = Path(config.ROOT / "artifacts" / "phase2")
T = config.TABLE_DIR
OUT = config.REPORTS_DIR / "metrics"
KEEP = ("pr_auc", "roc_auc", "precision@1%", "recall@1%", "precision@5%", "recall@5%")
AGENT_KEEP = ("outcome_accuracy", "policy_compliance", "claims_cited_final", "claims_cited_first_draft",
              "numbers_traceable_final", "fallback_rate", "mean_tool_calls", "mean_llm_calls", "median_seconds")


def _json(path: Path) -> dict | list | None:
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else None


def _key(k: str) -> str:
    return k.replace("@", "_at_").replace("%", "pct")


def model_metrics() -> dict:
    sel = (_json(ART / "selection.json") or {}).get("codes_as", "numeric")
    info = _json(ART / f"lgbm_{sel}" / "info.json") or {}
    out: dict = {"model": f"lgbm_{sel}"}
    for part in ("valid", "test"):
        out[part] = {_key(k): round(float(v), 4) for k, v in (info.get(part) or {}).items() if k in KEEP}
    res = _json(T / "phase2_results.json") or {}
    ci = (res.get("block_bootstrap_ci95_lightgbm") or {}).get("pr_auc")
    if ci:
        out["test"] |= {"pr_auc_ci95_lo": round(ci[0], 4), "pr_auc_ci95_hi": round(ci[1], 4)}
    cal = (res.get("calibration") or {}).get("LightGBM, isotonic") or {}
    out["calibration"] = {k: round(float(cal[k]), 5) for k in ("brier", "ece") if k in cal}
    gate = _json(ART / "gate_report.json")
    if gate:
        out["gate"] = {"passed": bool(gate["passed"]), "checks_failed": sum(not c["passed"] for c in gate["checks"])}
    return out


def retrieval_metrics() -> dict:
    res = _json(T / "phase3_retrieval_results.json") or {}
    chosen = res.get("chosen_variant")
    out: dict = {"variant": chosen, "index_version": res.get("index_version")}
    for r in res.get("summary") or []:
        if r["variant"] == chosen:
            out["all"] = {k: round(float(v), 4) for k, v in r.items() if k not in ("variant", "questions")}
    for r in res.get("by_split") or []:
        if r["variant"] == chosen:
            out[r["split"]] = {k: round(float(v), 4) for k, v in r.items() if k not in ("variant", "split", "questions")}
    ab = (res.get("abstention") or {}).get("all") or {}
    if ab:
        out["abstention"] = {"correct_abstentions": ab.get("correct_abstentions"), "unanswerable": ab.get("unanswerable"),
                             "false_abstentions": ab.get("false_abstentions"),
                             "hit_at_5_after_abstention": ab.get("hit@5_after_abstention")}
    return out


def agent_metrics() -> dict:
    out: dict = {}
    for p in sorted(T.glob("phase4_agent_summary*.json")):
        s = _json(p) or {}
        llm = s.get("llm") or p.stem.replace("phase4_agent_summary", "").lstrip("_") or "default"
        out[llm] = {mode: {k: (round(float(v), 4) if isinstance(v, (int, float)) else v) for k, v in vals.items()
                           if k in AGENT_KEEP and v is not None}
                    for mode, vals in (s.get("summary") or {}).items()}
        out[llm]["cases_done"] = s.get("cases_done") or (s.get("summary") or {}).get("orchestrated", {}).get("cases")
    return out


def write(out_dir: Path = OUT) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, fn in (("model", model_metrics), ("retrieval", retrieval_metrics), ("agents", agent_metrics)):
        files[name] = out_dir / f"{name}.json"
        files[name].write_text(json.dumps(fn(), indent=1), encoding="utf-8")
    return files
