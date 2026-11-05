"""FinGuard API and dashboard (Phase 5).

    python -m finguard.api serve [--host 127.0.0.1] [--port 8000] [--world real|synthetic] [--llm NAME]
    python -m finguard.api demo  [--base-url URL] [--llm NAME] [--max-seconds 60]   # alert-to-report timing
    python -m finguard.api healthcheck [--base-url URL]                             # exit 0 when ready

The dashboard is at http://127.0.0.1:8000/ and the interactive API docs at /docs (FastAPI).
Settings come from the environment: FINGUARD_WORLD, FINGUARD_LLM, FINGUARD_API_TIME_BUDGET,
FINGUARD_API_MAX_CONCURRENT, FINGUARD_API_LLMS, FINGUARD_REQUIRE_CHAMPION, FINGUARD_CASE_DIR.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from finguard import config


def cmd_serve(a) -> None:
    try:
        import uvicorn
    except ImportError:
        sys.exit('uvicorn is not installed: pip install -e ".[api]"')
    if a.world:
        os.environ["FINGUARD_WORLD"] = a.world
    if a.llm:
        os.environ["FINGUARD_LLM"] = a.llm
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from finguard.api.app import create_app
    from finguard.api.service import Service
    app = create_app(Service.from_env())
    print(f"FinGuard dashboard: http://{'127.0.0.1' if a.host == '0.0.0.0' else a.host}:{a.port}/   API docs: /docs")
    uvicorn.run(app, host=a.host, port=a.port, log_level="info")


def cmd_demo(a) -> None:
    from finguard.api.client import demo
    out = a.out or config.TABLE_DIR / f"phase5_demo_{a.llm or 'default'}.json"
    try:
        res = demo(a.base_url, llm=a.llm, transaction_id=a.transaction_id, max_seconds=a.max_seconds, wait=a.wait, out=out)
    except Exception as e:
        sys.exit(f"demo failed: {e}")
    sys.exit(0 if res["passed"] else 1)


def cmd_healthcheck(a) -> None:
    from finguard.api.client import healthcheck
    sys.exit(healthcheck(a.base_url, ready=not a.live))


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m finguard.api", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="run the API and the dashboard")
    s.add_argument("--host", default=os.environ.get("FINGUARD_HOST", "127.0.0.1"))
    s.add_argument("--port", type=int, default=int(os.environ.get("FINGUARD_PORT", "8000")))
    s.add_argument("--world", choices=["real", "synthetic"], default=None)
    s.add_argument("--llm", default=None, help="default LLM: auto (Groq, then Ollama offline), groq, ollama, none, scripted ...")
    s.set_defaults(fn=cmd_serve)
    d = sub.add_parser("demo", help="time the alert-to-report path on a running API")
    d.add_argument("--base-url", default="http://127.0.0.1:8000")
    d.add_argument("--llm", default=None)
    d.add_argument("--transaction-id", type=int, default=None)
    d.add_argument("--max-seconds", type=float, default=60.0)
    d.add_argument("--wait", type=float, default=900.0, help="seconds to wait for the API to be ready")
    d.add_argument("--out", type=Path, default=None)
    d.set_defaults(fn=cmd_demo)
    h = sub.add_parser("healthcheck", help="exit 0 when the API is ready (used by Docker)")
    h.add_argument("--base-url", default=f"http://127.0.0.1:{os.environ.get('FINGUARD_PORT', '8000')}")
    h.add_argument("--live", action="store_true", help="only check that the server answers (ready or starting)")
    h.set_defaults(fn=cmd_healthcheck)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
