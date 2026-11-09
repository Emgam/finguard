"""FinGuard MLOps commands.

    python -m finguard.mlops check [--strict]           # the four tools: installed, configured, working
    python -m finguard.mlops contracts [--stage data|model|all] [--sample N]
    python -m finguard.mlops drift [--window test|later|all] [--no-html] [--fail-on review|retrain]
    python -m finguard.mlops track sync [--what all|models|retrieval|agents|monitoring|registry]
    python -m finguard.mlops track status
    python -m finguard.mlops metrics                     # reports/metrics/*.json for `dvc metrics show`
    python -m finguard.mlops pipeline                    # dvc.yaml checks and which stages can be committed
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys

from finguard import config
from finguard.mlops import versions


def cmd_contracts(a) -> int:
    from finguard.mlops import contracts
    print(f"data contracts (Pandera), stage: {a.stage}")
    rep = contracts.run(a.stage, sample=a.sample)
    for st, r in rep["stages"].items():
        print(f"{st}: {'PASSED' if r['passed'] else 'FAILED'} -> {contracts.OUT[st].relative_to(config.ROOT)}")
    return 0 if rep["passed"] else 1


def cmd_drift(a) -> int:
    from finguard.mlops import drift
    windows = ("test", "later") if a.window == "all" else (a.window,)
    print(f"drift monitoring (Evidently {versions()['evidently'] or 'not installed: PSI only'}); reference = validation month")
    res = drift.run(windows, html=not a.no_html)
    print(f"decision: {res['decision'].upper()}  ->  {drift.OUT_DIR.relative_to(config.ROOT)}/")
    if a.fail_on != "never" and drift.LEVEL[res["decision"]] >= drift.LEVEL[{"review": "review", "retrain": "retrain review"}[a.fail_on]]:
        return 1
    return 0


def cmd_track(a) -> int:
    from finguard.mlops import tracking
    if a.action == "sync":
        tracking.sync(a.what)
    else:
        print(json.dumps(tracking.status(), indent=1))
    return 0


def cmd_metrics(a) -> int:
    from finguard.mlops import metrics
    for name, path in metrics.write().items():
        print(f"{name:<10} -> {path.relative_to(config.ROOT)}")
    return 0


def cmd_pipeline(a) -> int:
    from finguard.mlops import pipeline
    return pipeline.report()


# --------------------------------------------------------------------------- check
def _check_pandera() -> tuple[bool, str]:
    import numpy as np
    import pandas as pd

    from finguard.mlops import contracts
    if not contracts.available():
        return False, "not installed"
    df = pd.DataFrame({"a": np.array([1.0, np.nan, 3.0], dtype="float32")})
    ok = contracts.validate(contracts.feature_schema(["a"]), df)["passed"]
    bad = contracts.validate(contracts.feature_schema(["a"]), df.assign(a=np.float32(np.inf)))["passed"]
    runs = [p.name for p in contracts.OUT.values() if p.exists()]
    return ok and not bad, f"contracts load and catch an infinity; results present: {', '.join(runs) or 'none yet'}"


def _check_mlflow() -> tuple[bool, str]:
    from finguard.mlops import tracking
    if not tracking.enabled():
        return False, "not installed (or FINGUARD_MLFLOW=off)"
    s = tracking.status()
    runs = sum(s["runs"].values())
    alias = ", ".join(f"{k} -> v{v}" for k, v in s["aliases"].items()) or "no alias yet"
    return True, f"store {s['tracking_uri']}: {runs} runs, model versions {s['versions'] or 'none'} ({alias})"


def _check_evidently() -> tuple[bool, str]:
    import numpy as np
    import pandas as pd

    from finguard.mlops import drift, optional
    if optional("evidently") is None:
        return False, "not installed"
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({c: rng.normal(size=400) for c in drift.NUMERICAL} | {c: rng.choice(["a", "b"], 400) for c in drift.CATEGORICAL})
    cur = ref.assign(TransactionAmt=ref["TransactionAmt"] + 3)
    snap = drift.evidently_report(ref, cur, 0.5, with_score=False)
    parsed = drift.parse_snapshot(snap.dict())
    hit = [c["column"] for c in parsed["columns"] if c["drifted"]]
    summaries = sorted(p.name for p in drift.OUT_DIR.glob("drift_*.summary.json"))
    return hit == ["TransactionAmt"], f"report runs and flags a shifted column ({hit}); summaries: {', '.join(summaries) or 'none yet'}"


def _check_dvc() -> tuple[bool, str]:
    from finguard.mlops import pipeline
    problems = pipeline.validate()
    exe = shutil.which("dvc")
    if exe is None:
        return False, f"dvc command not found; dvc.yaml structure: {'OK' if not problems else problems[:2]}"
    v = subprocess.run([exe, "--version"], capture_output=True, text=True, cwd=config.ROOT).stdout.strip()
    init = (config.ROOT / ".dvc").exists()
    lock = (config.ROOT / "dvc.lock").exists()
    ok = not problems and init
    return ok, (f"dvc {v}; repo {'initialised' if init else 'NOT initialised (dvc init)'}; dvc.lock {'present' if lock else 'absent (dvc commit -f)'}; "
                f"dvc.yaml {'OK' if not problems else f'{len(problems)} problem(s): ' + '; '.join(problems[:2])}")


def cmd_check(a) -> int:
    v = versions()
    print("FinGuard MLOps check")
    results = {}
    for name, fn in (("pandera", _check_pandera), ("mlflow", _check_mlflow), ("evidently", _check_evidently), ("dvc", _check_dvc)):
        try:
            ok, detail = fn()
        except Exception as exc:                                    # a broken install is a finding, not a crash
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        results[name] = ok
        print(f"  {'OK  ' if ok else 'FAIL'} {name:<10} {v.get(name) or '-':<9} {detail}")
    if not all(results.values()):
        print('\ninstall the tools with:  pip install -e ".[mlops]"   (then: dvc init)')
    return 0 if all(results.values()) or not a.strict else 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="the four tools: installed, configured, working")
    c.add_argument("--strict", action="store_true", help="exit 1 unless all four pass")
    c.set_defaults(fn=cmd_check)
    k = sub.add_parser("contracts", help="Pandera data contracts")
    k.add_argument("--stage", default="all", choices=["data", "model", "all"])
    k.add_argument("--sample", type=int, default=None, help="feature-matrix rows to check (default: params.yaml)")
    k.set_defaults(fn=cmd_contracts)
    d = sub.add_parser("drift", help="Evidently drift reports and the monitoring decision")
    d.add_argument("--window", default="all", choices=["test", "later", "all"])
    d.add_argument("--no-html", action="store_true")
    d.add_argument("--fail-on", default="never", choices=["never", "review", "retrain"])
    d.set_defaults(fn=cmd_drift)
    t = sub.add_parser("track", help="MLflow tracking")
    t.add_argument("action", choices=["sync", "status"])
    t.add_argument("--what", default="all", choices=["all", "models", "retrieval", "agents", "monitoring", "registry"])
    t.set_defaults(fn=cmd_track)
    m = sub.add_parser("metrics", help="flat metric files for dvc metrics")
    m.set_defaults(fn=cmd_metrics)
    p = sub.add_parser("pipeline", help="dvc.yaml checks and stage status")
    p.set_defaults(fn=cmd_pipeline)
    a = ap.parse_args()
    if a.cmd == "contracts" and a.sample is None:
        from finguard.mlops.contracts import FEATURE_SAMPLE
        a.sample = FEATURE_SAMPLE
    sys.exit(a.fn(a))


if __name__ == "__main__":
    main()
