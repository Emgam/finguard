"""Every release gate on the real data, model and index (Project Report v3.0, section 9.2).

CI cannot run the gates that need the Kaggle data, the trained model files or the downloaded
documents; this script runs all of them on the machine that has them, in order, and writes one
record: reports/tables/phase5_release_check.json. Exit code 1 if any gate fails.

    python tools/release_check.py                       # all gates
    python tools/release_check.py --llm anthropic       # demo with a real LLM (default: FINGUARD_LLM)
    python tools/release_check.py --skip tests          # skip a gate (data, contracts, model, monitoring, prompts,
                                                        # tests, deployment)

| Gate        | Command                                                         | Blocks the release if   |
| data        | python -m finguard.data.validate                                | a data check FAILS      |
| contracts   | python -m finguard.mlops contracts (Pandera: raw, interim,      | a contract fails        |
|             | features, model outputs)                                        |                         |
| model       | python -m finguard.gate (contracts, leakage, model, calibration,| the model gate fails    |
|             | SHAP)                                                           |                         |
| monitoring  | python -m finguard.mlops drift --fail-on retrain (Evidently)    | a retrain signal        |
| prompts     | python -m finguard.agents prompts                               | a prompt changed without a version bump |
| tests       | python -m pytest -q (with the real-data retrieval and agent cases) | any test fails       |
| deployment  | serve the API, health check, alert-to-report demo               | not ready, or the demo fails or takes 60 s or more |
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "tables" / "phase5_release_check.json"
PY = sys.executable


def run(name: str, cmd: list[str], timeout: float) -> dict:
    print(f"\n=== {name}: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=ROOT, timeout=timeout, capture_output=True, text=True, encoding="utf-8", errors="replace")
        code, tail = p.returncode, (p.stdout + p.stderr).strip().splitlines()[-12:]
    except subprocess.TimeoutExpired:
        code, tail = -1, [f"timed out after {timeout:.0f} s"]
    for line in tail:
        print("   ", line)
    res = {"gate": name, "command": " ".join(cmd), "passed": code == 0, "exit_code": code,
           "seconds": round(time.time() - t0, 1), "tail": tail}
    print(f"--- {name}: {'PASSED' if res['passed'] else 'FAILED'} ({res['seconds']} s)", flush=True)
    return res


def deployment(llm: str | None, port: int, max_seconds: float) -> dict:
    """Start the API (real world), wait until ready, run the timed alert-to-report demo, stop it."""
    print(f"\n=== deployment: serve on port {port}, then the alert-to-report demo", flush=True)
    t0 = time.time()
    case_dir = Path(os.environ.get("FINGUARD_RELEASE_CASE_DIR", ROOT / "data" / "cases" / "release-check"))
    case_dir.mkdir(parents=True, exist_ok=True)              # demo cases stay out of the analysts' case store
    log = case_dir / "serve.log"
    env = {**os.environ, "FINGUARD_WORLD": "real", "PYTHONUTF8": "1", "FINGUARD_CASE_DIR": str(case_dir)}
    with open(log, "w", encoding="utf-8") as f:
        server = subprocess.Popen([PY, "-m", "finguard.api", "serve", "--port", str(port)], cwd=ROOT, env=env,
                                  stdout=f, stderr=subprocess.STDOUT)
    try:
        demo_out = ROOT / "reports" / "tables" / f"phase5_demo_{llm or 'default'}.json"
        cmd = [PY, "-m", "finguard.api", "demo", "--base-url", f"http://127.0.0.1:{port}", "--max-seconds", str(max_seconds),
               "--out", str(demo_out)] + (["--llm", llm] if llm else [])
        res = run("deployment demo", cmd, timeout=1800)
        if demo_out.exists():
            res["demo"] = json.loads(demo_out.read_text(encoding="utf-8"))
    finally:
        server.terminate()
        try:
            server.wait(20)
        except subprocess.TimeoutExpired:
            server.kill()
    res.update({"gate": "deployment", "seconds": round(time.time() - t0, 1), "server_log": str(log)})
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", default=os.environ.get("FINGUARD_LLM"), help="LLM for the deployment demo")
    ap.add_argument("--skip", nargs="*", default=[],
                    choices=["data", "contracts", "model", "monitoring", "prompts", "tests", "deployment"])
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--max-seconds", type=float, default=60.0)
    a = ap.parse_args()
    gates = [("data", [PY, "-m", "finguard.data.validate"], 3600), ("contracts", [PY, "-m", "finguard.mlops", "contracts"], 3600),
             ("model", [PY, "-m", "finguard.gate"], 3600),
             ("monitoring", [PY, "-m", "finguard.mlops", "drift", "--fail-on", "retrain"], 3600),
             ("prompts", [PY, "-m", "finguard.agents", "prompts"], 120), ("tests", [PY, "-m", "pytest", "-q"], 7200)]
    results = [run(n, c, t) for n, c, t in gates if n not in a.skip]
    if "deployment" not in a.skip:
        results.append(deployment(a.llm, a.port, a.max_seconds))
    passed = all(r["passed"] for r in results)
    record = {"run_at": time.strftime("%Y-%m-%d %H:%M:%S"), "passed": passed, "skipped": a.skip, "llm": a.llm,
              "gates": results}
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        record["commit"] = commit or None
    except OSError:
        record["commit"] = None
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
    print("\n" + "\n".join(f"{r['gate']:<11} {'PASSED' if r['passed'] else 'FAILED'}  {r['seconds']:>7} s" for r in results))
    print(f"RELEASE CHECK {'PASSED' if passed else 'FAILED'}  ->  {OUT.relative_to(ROOT)}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
