"""Run traces (Project Report v3.0, section 8.3): every plan, tool call, argument, output,
prompt version and model version, one JSON line per event, in ``data/cases/traces/<case>.jsonl``.
Langfuse or MLflow tracing can read the same events; the JSON lines need neither.
"""
from __future__ import annotations

import json
import time
from pathlib import Path


class Tracer:
    def __init__(self, path: Path | None):
        self.path = Path(path) if path else None
        self.t0 = time.time()
        self.events: list[dict] = []
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("", encoding="utf-8")

    def event(self, kind: str, **data) -> dict:
        e = {"t": round(time.time() - self.t0, 3), "event": kind, **data}
        self.events.append(e)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(e, ensure_ascii=False, default=str) + "\n")
        return e


class NullTracer(Tracer):
    def __init__(self):
        super().__init__(None)
