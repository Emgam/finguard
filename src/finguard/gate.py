"""Model quality gate (Project Report v3.0, section 9.2). Exit code 1 blocks a release.

    python -m finguard.gate [--register]

Checks, all computed from the saved Phase 1-2 outputs:

| Gate          | Check                                                                              |
| Data          | no FAIL in the Phase 1 data-quality checks; the data contracts passed (Pandera:    |
|               | raw files, interim tables, feature matrix, model outputs; finguard.mlops.contracts)|
| Leakage       | no forbidden or ID/time-like input; statistics fitted on training rows only;       |
|               | time split ordered and disjoint                                                    |
| Model         | test PR-AUC 95% entity-block interval entirely above the strongest baseline;       |
|               | not worse than the current champion by more than 0.005                            |
| Calibration   | calibrated Brier below the raw model's; band fraud rates High > Medium > Low        |
| Explainability| TreeSHAP local accuracy; no ID/time-like feature among the top 20 by mean |SHAP|;   |
|               | top-5 reason-family overlap across retrains >= 0.6                                 |

With ``--register`` the model is added to the registry with this report. Promotion to
"champion" stays a separate, human step (``python -m finguard.registry promote``).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from finguard import config, split
from finguard.features.build import FORBIDDEN

ART = Path(config.ROOT / "artifacts" / "phase2")
T = config.TABLE_DIR
ID_LIKE = ("transactionid", "transactiondt", "day", "proxy_entity", "customer")
CHAMPION_TOLERANCE = 0.005


def _json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run(name: str = "lgbm_numeric") -> dict:
    checks = []

    def check(gate: str, what: str, passed: bool, detail: str) -> None:
        checks.append({"gate": gate, "check": what, "passed": bool(passed), "detail": detail})

    val = pd.read_csv(T / "validation_results.csv")
    n_fail = int((val["status"] == "FAIL").sum())
    check("Data", "Phase 1 data-quality checks", n_fail == 0, f"{n_fail} FAIL, {int((val['status'] == 'WARN').sum())} WARN")
    from finguard.mlops import contracts
    ok, detail = contracts.gate_check()
    check("Data", "data contracts (Pandera): raw, interim, features, scores", ok, detail)

    cols = json.loads((ART / name / "columns.json").read_text(encoding="utf-8"))["columns"]
    bad = sorted(set(cols) & FORBIDDEN) + [c for c in cols if c.lower() in ID_LIKE or "customer" in c.lower()]
    check("Leakage", "no forbidden or ID/time-like input", not bad, f"{len(cols)} inputs; offending: {bad or 'none'}")
    from finguard.models import dataset
    fs = dataset.load("main", "numeric")
    m = fs.masks
    check("Leakage", "statistics fitted on training rows only", fs.builder.fit_rows_ == int(m["train"].sum()),
          f"fit rows {fs.builder.fit_rows_:,} = training rows {int(m['train'].sum()):,}")
    try:
        s = split.check(fs.day, "main")
        check("Leakage", "time split ordered and disjoint", True,
              f"train {s['train']['first_day']}-{s['train']['last_day']}, valid {s['valid']['first_day']}-{s['valid']['last_day']}, "
              f"test {s['test']['first_day']}-{s['test']['last_day']}")
    except ValueError as exc:
        check("Leakage", "time split ordered and disjoint", False, str(exc))

    res = _json(T / "phase2_results.json")
    lo = res["block_bootstrap_ci95_lightgbm"]["pr_auc"][0]
    baselines = {"logistic regression (Phase 2)": res["test_metrics"]["Logistic regression"]["pr_auc"]}
    if (ART / "lr_v2" / "info.json").exists():
        baselines["logistic regression v2 (strengthened)"] = _json(ART / "lr_v2" / "info.json")["test"]["pr_auc"]
    best_name, best = max(baselines.items(), key=lambda kv: kv[1])
    check("Model", "PR-AUC interval entirely above the strongest baseline", lo > best,
          f"LightGBM entity-block 95% lower bound {lo:.3f} vs {best_name} {best:.3f}")
    from finguard import registry
    reg = registry.load()
    champ = reg["aliases"].get("champion")
    own = _json(ART / name / "info.json")["test"]["pr_auc"]
    if champ:
        c = next(x for x in reg["models"] if x["model_id"] == champ)
        cp = c["metrics"]["test"]["pr_auc"]
        check("Model", "not worse than the champion", own >= cp - CHAMPION_TOLERANCE, f"{own:.4f} vs champion {champ} {cp:.4f}")
    else:
        check("Model", "not worse than the champion", True, "no champion registered yet")

    cal = res["calibration"]
    check("Calibration", "calibrated Brier below raw", cal["LightGBM, isotonic"]["brier"] < cal["LightGBM, uncalibrated"]["brier"],
          f"{cal['LightGBM, isotonic']['brier']:.4f} vs raw {cal['LightGBM, uncalibrated']['brier']:.4f}")
    rates = {b["band"]: b["fraud_rate"] for b in res["bands"]["test"]}
    check("Calibration", "band fraud rates High > Medium > Low (test)", rates["High"] > rates["Medium"] > rates["Low"],
          f"High {rates['High']:.1%}, Medium {rates['Medium']:.1%}, Low {rates['Low']:.1%}")

    ex = _json(T / "phase2_explanations.json")
    check("Explainability", "TreeSHAP local accuracy", ex["local_accuracy_max_abs_error"] < 1e-4,
          f"max |sum SHAP - score| = {ex['local_accuracy_max_abs_error']:.1e}")
    top = pd.read_csv(T / "phase2_feature_shap.csv").head(20).feature.tolist()
    leaky = [f for f in top if f.lower() in ID_LIKE or f.lower().startswith(("transactionid", "transactiondt"))]
    check("Explainability", "no ID/time-like feature in the top 20", not leaky, f"top 3: {', '.join(top[:3])}; offending: {leaky or 'none'}")
    ov = ex["stability"]["mean_pairwise_overlap"]
    check("Explainability", "top-5 family overlap >= 0.6", ov >= 0.6, f"{ov:.2f}")

    report = {"model": name, "passed": all(c["passed"] for c in checks), "checks": checks}
    (ART / "gate_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="lgbm_numeric")
    ap.add_argument("--register", action="store_true")
    a = ap.parse_args()
    report = run(a.name)
    print(pd.DataFrame(report["checks"]).to_string(index=False))
    print("GATE", "PASSED" if report["passed"] else "FAILED")
    if a.register:
        from finguard import registry
        e = registry.register(a.name, gate_report=report)
        print(f"registered {e['model_id']} (code {e['code']}); promote with: "
              f"python -m finguard.registry promote {e['model_id']} --alias champion --approved-by \"<name>\"")
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
