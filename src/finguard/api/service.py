"""The API's logic, independent of the web framework (Project Report v3.0, sections 4, 5 and 9).

``ROUTES`` is the contract: one entry per endpoint with its query, path and body JSON schemas.
``Service.handle`` validates a request against it and answers with a ``Reply``. The FastAPI app
(``app.py``) only translates HTTP to ``handle`` and back, so the same contract is unit-tested here
and served there; the OpenAPI page at ``/docs`` is generated from the same schemas.

Rules the service keeps:

* every reply carries the model, index and prompt versions (``X-FinGuard-*`` headers, and a
  ``versions`` field in JSON bodies);
* the API starts at once and loads the data, model and knowledge base in the background:
  ``/api/health`` answers 503 ``starting`` until it is ready, and data endpoints answer 503 too;
* an investigation runs the Phase 4 orchestrator with the API's time budget
  (``FINGUARD_API_TIME_BUDGET``, default 40 s; LLM calls are cut 10 s later), so alert-to-report
  stays under 60 s: past the budget the fixed fallback path writes the report and says so;
* at most ``FINGUARD_API_MAX_CONCURRENT`` investigations run at once (default 2); more get 429;
* decisions are recorded only with an analyst's name or role; nothing is decided by the system.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from jsonschema import Draft202012Validator

from finguard import __version__
from finguard.agents.case_store import DECISIONS
from finguard.agents.evidence import Evidence
from finguard.agents.graph import Orchestrator
from finguard.agents.llm import LLMError, make_llm
from finguard.agents.prompt_store import load_all, lock_problems
from finguard.agents.report import DISCLAIMER
from finguard.agents.tools import transaction_ids

log = logging.getLogger("finguard.api")

LLM_NAMES = ("auto", "groq", "ollama", "none", "scripted", "gemini", "openrouter", "openai", "anthropic")
TX = {"type": "integer", "minimum": 1_000_000, "maximum": 9_999_999}
CASE_ID = {"type": "string", "pattern": "^[A-Za-z0-9_-]{1,80}$"}
LLM = {"type": "string", "enum": list(LLM_NAMES)}


class ApiError(Exception):
    def __init__(self, status: int, error: str, message: str, details: Any = None):
        super().__init__(message)
        self.status, self.error, self.message, self.details = status, error, message, details


@dataclass
class Reply:
    status: int
    body: Any
    media_type: str = "application/json"
    headers: dict = field(default_factory=dict)

    def content(self) -> bytes:
        if self.media_type == "application/json":
            return json.dumps(self.body, ensure_ascii=False, default=_json_default).encode("utf-8")
        return self.body.encode("utf-8") if isinstance(self.body, str) else bytes(self.body)


def _json_default(v):
    if hasattr(v, "item"):                                       # numpy scalars
        return v.item()
    if v is pd.NA or v is pd.NaT:
        return None
    return str(v)


@dataclass
class Route:
    method: str
    path: str
    name: str
    summary: str
    query: dict = field(default_factory=dict)                  # name -> JSON schema (+ "default")
    path_params: dict = field(default_factory=dict)            # name -> JSON schema
    body: dict | None = None                                   # JSON schema of the JSON body
    media_type: str = "application/json"
    needs_ready: bool = True
    pattern: re.Pattern | None = None

    def __post_init__(self):
        rx = re.sub(r"\{(\w+)\}", lambda m: f"(?P<{m.group(1)}>[^/]+)", re.escape(self.path).replace(r"\{", "{").replace(r"\}", "}"))
        self.pattern = re.compile(f"^{rx}$")

    def openapi(self) -> dict:
        """The operation's parameters and body for FastAPI's ``openapi_extra``."""
        params = [{"name": n, "in": "path", "required": True, "schema": s} for n, s in self.path_params.items()]
        params += [{"name": n, "in": "query", "required": False,
                    "schema": {k: v for k, v in s.items() if k != "description"}, "description": s.get("description", "")}
                   for n, s in self.query.items()]
        out = {"parameters": params} if params else {}
        if self.body:
            out["requestBody"] = {"required": True, "content": {"application/json": {"schema": self.body}}}
        return out


INVESTIGATION_BODY = {
    "type": "object", "additionalProperties": False,
    "properties": {"transaction_id": TX, "request": {"type": "string", "minLength": 3, "maxLength": 2000}, "llm": LLM},
    "anyOf": [{"required": ["transaction_id"]}, {"required": ["request"]}],
}
CHAT_BODY = {
    "type": "object", "additionalProperties": False, "required": ["message"],
    "properties": {"message": {"type": "string", "minLength": 2, "maxLength": 2000}, "transaction_id": TX, "llm": LLM},
}
DECISION_BODY = {
    "type": "object", "additionalProperties": False, "required": ["decision", "analyst"],
    "properties": {"decision": {"enum": list(DECISIONS)}, "analyst": {"type": "string", "minLength": 1, "maxLength": 80},
                   "note": {"type": "string", "maxLength": 1000}},
}

ROUTES = [
    Route("GET", "/api/health", "health", "Liveness and readiness, components, versions and champion check",
          needs_ready=False),
    Route("GET", "/api/versions", "get_versions", "Model, knowledge-base index and prompt versions", needs_ready=False),
    Route("GET", "/api/alerts", "list_alerts", "Alert queue: scored transactions of the review period",
          query={"band": {"type": "string", "enum": ["alerts", "High", "Medium", "Low", "all"], "default": "alerts",
                          "description": "alerts = High and Medium"},
                 "status": {"type": "string", "enum": ["any", "open", "draft", "decided"], "default": "any"},
                 "day_from": {"type": "integer", "minimum": 0, "maximum": 400},
                 "day_to": {"type": "integer", "minimum": 0, "maximum": 400},
                 "q": {"type": "string", "pattern": "^[0-9]{1,7}$", "description": "TransactionID prefix"},
                 "sort": {"type": "string", "enum": ["score", "time"], "default": "score"},
                 "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 50},
                 "offset": {"type": "integer", "minimum": 0, "default": 0}}),
    Route("GET", "/api/transactions/{transaction_id}", "get_transaction",
          "Case view evidence: score, reasons, proxy-entity history and earlier cases", path_params={"transaction_id": TX}),
    Route("POST", "/api/investigations", "investigate", "Run an investigation and write a cited case report",
          body=INVESTIGATION_BODY),
    Route("POST", "/api/chat", "chat", "Ask the assistant (each question is answered on its own)", body=CHAT_BODY),
    Route("GET", "/api/cases", "list_cases", "Cases in FinGuard's case store, newest first",
          query={"transaction_id": TX, "status": {"type": "string", "enum": ["draft", "decided"]},
                 "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 50},
                 "offset": {"type": "integer", "minimum": 0, "default": 0}}),
    Route("GET", "/api/cases/{case_id}", "get_case", "One case: report, evidence, run record and decisions",
          path_params={"case_id": CASE_ID}),
    Route("GET", "/api/cases/{case_id}/report.md", "case_report", "The case report as a Markdown file",
          path_params={"case_id": CASE_ID}, media_type="text/markdown"),
    Route("POST", "/api/cases/{case_id}/decision", "decide", "Record an analyst's decision on a case",
          path_params={"case_id": CASE_ID}, body=DECISION_BODY),
]
ROUTE_BY_NAME = {r.name: r for r in ROUTES}


def _errors(instance, schema) -> list[str]:
    out = []
    for e in sorted(Draft202012Validator(schema).iter_errors(instance), key=lambda e: list(e.path)):
        where = "/".join(str(p) for p in e.path) or "body"
        out.append(f"{where}: {e.message}")
    return out


def _coerce(name: str, raw: str, schema: dict):
    if schema.get("type") == "integer":
        try:
            return int(raw)
        except (TypeError, ValueError):
            raise ApiError(400, "invalid_parameter", f"{name} must be an integer, got {raw!r}") from None
    return raw


def _native(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v.item() if hasattr(v, "item") else v


def available_llms() -> list[str]:
    """LLMs this server can use: the ones whose key (and model, where there is no default) is set."""
    env = lambda k: os.environ.get(k) or None                     # noqa: E731 - empty means unset
    names = ["auto"]                                             # Groq, then Ollama offline, then the template path
    if env("GROQ_API_KEY"):
        names.append("groq")
    names += ["ollama", "none", "scripted"]                       # Ollama's state is shown by /api/health
    if env("GEMINI_API_KEY") and env("GEMINI_MODEL"):
        names.append("gemini")
    if env("OPENROUTER_API_KEY") and env("OPENROUTER_MODEL"):
        names.append("openrouter")
    if env("FINGUARD_LLM_MODEL") and (env("FINGUARD_LLM_BASE_URL") or env("FINGUARD_LLM_API_KEY") or env("OPENAI_API_KEY")):
        names.append("openai")
    if env("ANTHROPIC_API_KEY"):
        names.append("anthropic")
    allowed = [n.strip() for n in os.environ.get("FINGUARD_API_LLMS", "").split(",") if n.strip()]
    return [n for n in names if not allowed or n in allowed]


class Service:
    def __init__(self, world, default_llm: str | None = None, time_budget: float | None = None,
                 max_concurrent: int | None = None, graph: str | None = None):
        self.world = world
        self.llms = available_llms()
        wanted = (default_llm or os.environ.get("FINGUARD_LLM") or "auto").lower()
        self.default_llm = wanted if wanted in self.llms else "none"
        self.default_llm_note = None if wanted == self.default_llm else (
            f"FINGUARD_LLM={wanted} is not available (missing key or model); using 'none'")
        self.time_budget = float(time_budget or os.environ.get("FINGUARD_API_TIME_BUDGET", "40"))
        self.slots = threading.BoundedSemaphore(int(max_concurrent or os.environ.get("FINGUARD_API_MAX_CONCURRENT", "2")))
        self.graph = graph or os.environ.get("FINGUARD_GRAPH")
        self.state, self.error = "starting", None
        self.started_at, self.ready_at = time.time(), None
        self.prompts = {n: p.tag for n, p in load_all().items()}
        self.prompt_lock = lock_problems()
        self._versions: dict | None = None
        self._llm_checked: tuple | None = None
        self._thread: threading.Thread | None = None

    @classmethod
    def from_env(cls) -> "Service":
        from finguard.api.world import make_world
        return cls(make_world())

    # ------------------------------------------------------------------ lifecycle
    def start(self, background: bool = True) -> "Service":
        if background:
            self._thread = threading.Thread(target=self._load, name="finguard-load", daemon=True)
            self._thread.start()
        else:
            self._load()
        return self

    def _load(self) -> None:
        try:
            t0 = time.time()
            self.world.load()
            self._versions = {**self.world.versions(), "prompts": list(self.prompts.values())}
            self.state, self.ready_at = "ready", time.time()
            log.info("FinGuard API ready in %.1f s (%s world)", time.time() - t0, self.world.name)
        except Exception as e:                                    # reported by /api/health
            self.state, self.error = "error", f"{type(e).__name__}: {e}"
            log.exception("FinGuard API failed to load")

    def wait_ready(self, timeout: float = 600) -> bool:
        if self._thread is not None:
            self._thread.join(timeout)
        return self.state == "ready"

    def stop(self) -> None:
        pass

    @property
    def ready(self) -> bool:
        return self.state == "ready"

    # ------------------------------------------------------------------ versions on every reply
    def versions(self) -> dict:
        base = {"api": __version__, "world": self.world.name, "prompts": list(self.prompts.values())}
        return {**base, **(self._versions or {})}

    def version_headers(self) -> dict:
        v = self.versions()
        h = {"X-FinGuard-API": str(v["api"]), "X-FinGuard-World": v["world"], "X-FinGuard-Prompts": ",".join(v["prompts"])}
        if v.get("model_version"):
            h["X-FinGuard-Model"] = str(v["model_version"])
        if v.get("index_version"):
            h["X-FinGuard-Index"] = str(v["index_version"])
        return h

    # ------------------------------------------------------------------ dispatch
    def handle(self, route: Route, path_params: dict, query: list[tuple[str, str]], body: bytes | dict | None) -> Reply:
        """Validate a request against its route, run it and wrap the result (errors included)."""
        try:
            if route.needs_ready and not self.ready:
                if self.state == "error":
                    raise ApiError(503, "not_ready", f"the API failed to load: {self.error}")
                raise ApiError(503, "starting", f"loading the data, model and knowledge base "
                                                f"({time.time() - self.started_at:.0f} s so far)")
            args = self._args(route, path_params, query, body)
            result = getattr(self, route.name)(**args)
            reply = result if isinstance(result, Reply) else Reply(200, result, route.media_type)
        except ApiError as e:
            reply = Reply(e.status, {"error": e.error, "message": e.message, "details": e.details})
        except Exception as e:                                    # a bug: logged, answered as 500
            log.exception("unhandled error in %s", route.name)
            reply = Reply(500, {"error": "internal", "message": f"{type(e).__name__}: {e}"})
        return self._finish(reply)

    def _finish(self, reply: Reply) -> Reply:
        if reply.media_type == "application/json" and isinstance(reply.body, dict):
            reply.body.setdefault("versions", self.versions())
        reply.headers = {**self.version_headers(), **reply.headers}
        return reply

    def dispatch(self, method: str, path: str, query: list[tuple[str, str]] | None = None,
                 body: bytes | dict | None = None) -> Reply:
        """Route by method and path (used by tests and the plain HTTP fallback)."""
        allowed = []
        for r in ROUTES:
            m = r.pattern.match(path)
            if m:
                if r.method == method:
                    return self.handle(r, m.groupdict(), query or [], body)
                allowed.append(r.method)
        if allowed:
            return self._finish(Reply(405, {"error": "method_not_allowed", "message": f"use {', '.join(allowed)}"},
                                      headers={"Allow": ", ".join(allowed)}))
        return self._finish(Reply(404, {"error": "not_found", "message": f"no route {method} {path}"}))

    def _args(self, route: Route, path_params: dict, query: list[tuple[str, str]], body) -> dict:
        args = {}
        for name, schema in route.path_params.items():
            value = _coerce(name, path_params.get(name), schema)
            problems = _errors(value, schema)
            if problems:
                raise ApiError(404 if name == "case_id" else 400, "invalid_parameter", f"{name}: {problems[0].split(': ', 1)[-1]}")
            args[name] = value
        seen = set()
        for name, raw in query:
            if name not in route.query:
                raise ApiError(400, "unknown_parameter", f"unknown query parameter '{name}'",
                               {"allowed": sorted(route.query)})
            if name in seen:
                raise ApiError(400, "repeated_parameter", f"query parameter '{name}' given twice")
            seen.add(name)
            schema = route.query[name]
            value = _coerce(name, raw, schema)
            problems = _errors(value, {k: v for k, v in schema.items() if k not in ("default", "description")})
            if problems:
                raise ApiError(400, "invalid_parameter", f"{name}: {problems[0].split(': ', 1)[-1]}")
            args[name] = value
        for name, schema in route.query.items():
            if name not in args and "default" in schema:
                args[name] = schema["default"]
        if route.body is not None:
            if isinstance(body, (bytes, bytearray, str)):
                try:
                    body = json.loads(body or b"{}")
                except (ValueError, UnicodeDecodeError):
                    raise ApiError(400, "invalid_json", "the request body is not valid JSON") from None
            problems = _errors(body if body is not None else {}, route.body)
            if problems:
                raise ApiError(400, "invalid_body", "; ".join(problems[:5]), {"schema": route.body})
            args["body"] = body
        return args

    def _llm_status(self) -> dict:
        """Groq key and Ollama server state, refreshed at most every 30 s (the health check is polled)."""
        now = time.time()
        if self._llm_checked is None or now - self._llm_checked[0] > 30:
            from finguard.agents.llm import llm_status
            self._llm_checked = (now, llm_status())
        return self._llm_checked[1]

    # ------------------------------------------------------------------ endpoints
    def get_versions(self) -> dict:
        return {"versions": self.versions(), "champion": self.world.champion() if self.ready else None}

    def health(self) -> Reply:
        body = {"status": self.state, "ready": self.ready, "world": self.world.name, "synthetic": self.world.synthetic,
                "error": self.error, "uptime_seconds": round(time.time() - self.started_at, 1),
                "load_seconds": round(self.ready_at - self.started_at, 1) if self.ready_at else None,
                "llm": {"default": self.default_llm, "available": self.llms, "note": self.default_llm_note,
                        "status": self._llm_status()},
                "time_budget_seconds": self.time_budget, "prompt_lock": self.prompt_lock or "ok",
                "disclaimer": DISCLAIMER}
        if self.ready:
            body["components"] = self.world.components()
            body["champion"] = self.world.champion()
            body["period"] = self.world.period()
        return Reply(200 if self.ready else 503, body)

    def list_alerts(self, band: str, status: str, sort: str, limit: int, offset: int, day_from: int | None = None,
                    day_to: int | None = None, q: str | None = None) -> dict:
        f = self.world.queue
        counts = {b: int((f["band"] == b).sum()) for b in ("High", "Medium", "Low")}
        if band == "alerts":
            f = f[f["band"].isin(["High", "Medium"])]
        elif band != "all":
            f = f[f["band"] == band]
        if day_from is not None:
            f = f[f["relative_day"] >= day_from]
        if day_to is not None:
            f = f[f["relative_day"] <= day_to]
        if q:
            f = f[f["transaction_id"].astype(str).str.startswith(q)]
        cases = self.world.ctx.case_store.status_by_transaction()
        if status != "any":
            def state(tid):
                c = cases.get(int(tid))
                return "open" if c is None else c["status"]
            f = f[[state(t) == status for t in f["transaction_id"]]]
        f = f.sort_values(["score", "_time"], ascending=[False, True]) if sort == "score" else f.sort_values("_time")
        page = f.iloc[offset: offset + limit]
        items = []
        for row in page.to_dict("records"):
            item = {k: _native(row.get(k)) for k in row if not k.startswith("_")}
            c = cases.get(int(item["transaction_id"]))
            item["case"] = c
            items.append(item)
        return {"total": int(len(f)), "limit": limit, "offset": offset, "items": items, "band_counts": counts,
                "period": self.world.period(), "synthetic": self.world.synthetic,
                "note": "Alerts are the High and Medium bands: flagged for review, not findings of fraud."}

    def get_transaction(self, transaction_id: int) -> dict:
        ev = Evidence([])
        out = {}
        for tool in ("fraud_score", "explain_transaction", "entity_history"):
            item = self.world.tools.call(ev, tool, {"transaction_id": transaction_id})
            if not item["ok"]:
                if item.get("retryable") is False:
                    raise ApiError(404, "unknown_transaction", item["error"])
                raise ApiError(502, "tool_failed", f"{tool}: {item['error']}")
            out[tool] = item["output"]
        out["cases"] = self.world.ctx.case_store.list(transaction_id=transaction_id, limit=20)["cases"]
        return {"transaction_id": transaction_id, **out, "synthetic": self.world.synthetic, "disclaimer": DISCLAIMER}

    def _llm(self, name: str | None):
        name = (name or self.default_llm).lower()
        if name not in self.llms:
            raise ApiError(400, "llm_not_available", f"LLM '{name}' is not available here",
                           {"available": self.llms})
        try:
            return name, make_llm(name)                           # a new client per run: its deadline is its own
        except (LLMError, ValueError) as e:
            raise ApiError(400, "llm_not_available", str(e)) from None

    def _run(self, request: str, llm_name: str | None) -> dict:
        name, llm = self._llm(llm_name)
        if not self.slots.acquire(blocking=False):
            raise ApiError(429, "busy", "investigations are already running at the limit; try again in a moment")
        try:
            orch = Orchestrator(self.world.tools, llm=llm, out_dir=self.world.case_dir, backend=self.graph,
                                time_budget=self.time_budget)
            case = orch.run(request)
        finally:
            self.slots.release()
        pol = case.get("policy") or {}
        return {"case_id": case["case_id"], "created_at": case["created_at"], "request": case["request"],
                "request_type": case.get("request_type"), "outcome": case.get("outcome"), "path": case.get("path"),
                "fallback_reason": case.get("fallback_reason"), "transaction_id": case.get("transaction_id"),
                "band": case.get("band"), "score": case.get("score"), "message": case.get("message"),
                "seconds": case.get("seconds"), "tool_calls": case.get("tool_calls"), "llm_calls": case.get("llm_calls"),
                "llm": {**case.get("llm", {}), "requested": name},
                "policy": {k: pol.get(k) for k in ("passed", "claims", "cited", "numbers", "traceable")},
                "markdown": case.get("markdown"), "report_url": f"/api/cases/{case['case_id']}/report.md",
                "synthetic": self.world.synthetic, "disclaimer": DISCLAIMER}

    def investigate(self, body: dict) -> dict:
        tid, request = body.get("transaction_id"), (body.get("request") or "").strip()
        if not request:
            request = f"Investigate alert TX-{tid}."
        elif tid is not None and tid not in transaction_ids(request):
            request = f"{request} (TX-{tid})"
        return self._run(request, body.get("llm"))

    def chat(self, body: dict) -> dict:
        message, tid = body["message"].strip(), body.get("transaction_id")
        if tid is not None and not transaction_ids(message):
            message = f"{message} (TX-{tid})"
        return self._run(message, body.get("llm"))

    def list_cases(self, limit: int, offset: int, transaction_id: int | None = None, status: str | None = None) -> dict:
        return self.world.ctx.case_store.list(limit=limit, offset=offset, transaction_id=transaction_id, status=status)

    def _case_files(self, case_id: str) -> tuple[Path, Path]:
        d = Path(self.world.case_dir) / "reports"
        js, md = d / f"{case_id}.json", d / f"{case_id}.md"
        if not js.exists():
            raise ApiError(404, "unknown_case", f"no case {case_id}")
        return js, md

    def get_case(self, case_id: str) -> dict:
        js, md = self._case_files(case_id)
        case = json.loads(js.read_text(encoding="utf-8"))
        return {"case": case, "markdown": md.read_text(encoding="utf-8") if md.exists() else None,
                "decisions": self.world.ctx.case_store.decisions(case_id), "report_url": f"/api/cases/{case_id}/report.md"}

    def case_report(self, case_id: str) -> Reply:
        _, md = self._case_files(case_id)
        return Reply(200, md.read_text(encoding="utf-8"), "text/markdown",
                     {"Content-Disposition": f'attachment; filename="{case_id}.md"'})

    def decide(self, case_id: str, body: dict) -> dict:
        try:
            rec = self.world.ctx.case_store.record_decision(case_id, body["decision"], body["analyst"], body.get("note", ""))
        except KeyError:
            raise ApiError(404, "unknown_case", f"no case {case_id}") from None
        except ValueError as e:
            raise ApiError(400, "invalid_decision", str(e)) from None
        return {**rec, "note": body.get("note", ""), "message": "Decision recorded by the analyst."}
