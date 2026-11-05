"""The orchestration graph (Project Report v3.0, section 8).

    start ─┬─ refusal ──────────────────────────────────────────────┐
           ├─ no LLM / forced ──────────────────────────► fallback ─┤
           └─► make_plan ◄─┐ (≤ 2 re-plans)                         │
                │     │                                             │
                ▼     │                                             ▼
             execute ─► check ─┬─ missing evidence ────────────► finalize
                               ├─ typology ─────────► write ◄─┐     ▲
                               └─► investigate ─────► write   │     │
                                                       │      │     │
                                                       ▼      │     │
                                                    validate ─┘ (≤ 2 drafts)
                                                       └─ fails twice ─► fallback

The LLM agents plan and write; code decides. ``start`` refuses out-of-scope requests before
any LLM call; ``check`` refuses to let an agent write until the request type's required tools
have succeeded; ``validate`` applies the evidence policy. Budgets: 8 tool calls, 2 re-plans,
2 report drafts, ``FINGUARD_AGENT_TIME_BUDGET`` seconds (default 90, room for about four LLM
calls). The time budget only stops new LLM calls and tool calls: a report already written is
still validated and kept if it passes. Past any budget the fixed fallback path runs and the
report says so.

The graph runs on LangGraph when it is installed (``FINGUARD_GRAPH=langgraph``, the default
"auto" picks it when available) and on a small built-in engine otherwise. Both run the same
node functions and routing; the state is plain JSON so LangGraph can checkpoint it.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import TypedDict

from finguard.agents import agents as A
from finguard.agents import policy, rules, schemas
from finguard.agents.case_store import CASE_DIR
from finguard.agents.evidence import Evidence
from finguard.agents.llm import LLMError
from finguard.agents.prompt_store import load_all
from finguard.agents.report import DISCLAIMER, render, template_report
from finguard.agents.tools import Tools, transaction_ids
from finguard.agents.trace import NullTracer, Tracer

END = "__end__"
TIME_BUDGET = float(os.environ.get("FINGUARD_AGENT_TIME_BUDGET", "90"))
LLM_GRACE = float(os.environ.get("FINGUARD_LLM_GRACE", "10"))   # a call started inside the budget may finish this much later
MAX_REPORT_DRAFTS = 2
MAX_SEARCHES_FROM_MAP = 3


class AgentState(TypedDict, total=False):
    request: str
    started: float
    transaction_ids: list
    request_type: str
    transaction_id: int
    plan: dict
    plan_problems: list
    replans: int
    evidence: list
    investigation: dict
    inv_attempts: int
    report: dict
    drafts: int
    feedback: str
    policy_first: dict
    policy: dict
    path: str
    fallback_reason: str
    outcome: str
    message: str
    llm_calls: int
    llm_down: bool


def _llm_down(e: Exception) -> bool:
    """The LLM could not answer at all (unreachable, out of quota, all of a chain failed): asking the
    same LLM again in this run will not help, so the run goes to the fallback path at once."""
    return isinstance(e.__cause__, LLMError)


# --------------------------------------------------------------------------- built-in engine
class SimpleGraph:
    """The subset of LangGraph's StateGraph used here: nodes return partial updates, edges and
    conditional edges route, END stops; a step limit guards against loops."""

    def __init__(self):
        self.nodes, self.edges, self.cond, self.entry = {}, {}, {}, None

    def add_node(self, name, fn):
        self.nodes[name] = fn

    def add_edge(self, a, b):
        self.edges[a] = b

    def add_conditional_edges(self, a, fn, mapping):
        self.cond[a] = (fn, mapping)

    def set_entry_point(self, name):
        self.entry = name

    def invoke(self, state: dict, max_steps: int = 40) -> dict:
        node = self.entry
        for _ in range(max_steps):
            state = {**state, **(self.nodes[node](state) or {})}
            if node in self.cond:
                fn, mapping = self.cond[node]
                node = mapping[fn(state)]
            else:
                node = self.edges[node]
            if node == END:
                return state
        raise RuntimeError("graph step limit reached")


def _langgraph(nodes: dict, edges: dict, cond: dict, entry: str):
    from langgraph.graph import END as LG_END
    from langgraph.graph import START, StateGraph
    g = StateGraph(AgentState)
    for name, fn in nodes.items():
        g.add_node(name, fn)
    g.add_edge(START, entry)
    for a, b in edges.items():
        g.add_edge(a, LG_END if b == END else b)
    for a, (fn, mapping) in cond.items():
        g.add_conditional_edges(a, fn, {k: (LG_END if v == END else v) for k, v in mapping.items()})
    app = g.compile()

    class _Runner:
        backend = "langgraph"

        def invoke(self, state, max_steps: int = 40):
            return app.invoke(state, config={"recursion_limit": max_steps})
    return _Runner()


def graph_backend(name: str | None = None) -> str:
    name = (name or os.environ.get("FINGUARD_GRAPH") or "auto").lower()
    if name == "auto":
        try:
            import langgraph  # noqa: F401
            return "langgraph"
        except ImportError:
            return "builtin"
    return name


# --------------------------------------------------------------------------- the orchestrator
class Orchestrator:
    def __init__(self, tools: Tools, llm=None, out_dir: Path | None = None, backend: str | None = None,
                 force_fallback: bool = False, time_budget: float = TIME_BUDGET, save: bool = True):
        self.tools, self.llm = tools, llm
        self.prompts = load_all()
        self.out_dir = Path(out_dir or CASE_DIR)
        self.force_fallback, self.time_budget, self.save = force_fallback, time_budget, save
        self.backend = graph_backend(backend)
        self.tracer: Tracer = NullTracer()
        self.app = self._build()

    # ------------------------------------------------------------------ graph
    def _build(self):
        # node names must differ from state keys (LangGraph refuses a node called "plan")
        nodes = {"start": self.n_start, "make_plan": self.n_plan, "execute": self.n_execute, "check": self.n_check,
                 "investigate": self.n_investigate, "write": self.n_write, "validate": self.n_validate,
                 "fallback": self.n_fallback, "finalize": self.n_finalize}
        edges = {"execute": "check", "fallback": "finalize", "finalize": END}
        cond = {"start": (self.r_start, {"plan": "make_plan", "fallback": "fallback", "finalize": "finalize"}),
                "make_plan": (self.r_plan, {"plan": "make_plan", "execute": "execute", "fallback": "fallback",
                                            "finalize": "finalize"}),
                "check": (self.r_check, {"plan": "make_plan", "investigate": "investigate", "write": "write",
                                         "fallback": "fallback", "finalize": "finalize"}),
                "investigate": (self.r_investigate, {"investigate": "investigate", "write": "write", "fallback": "fallback"}),
                "write": (self.r_write, {"write": "write", "validate": "validate", "fallback": "fallback"}),
                "validate": (self.r_validate, {"write": "write", "fallback": "fallback", "finalize": "finalize"})}
        if self.backend == "langgraph":
            return _langgraph(nodes, edges, cond, "start")
        g = SimpleGraph()
        for name, fn in nodes.items():
            g.add_node(name, fn)
        for a, b in edges.items():
            g.add_edge(a, b)
        for a, (fn, mapping) in cond.items():
            g.add_conditional_edges(a, fn, mapping)
        g.set_entry_point("start")
        return g

    def _over_budget(self, s: dict) -> bool:
        return time.time() - s["started"] > self.time_budget

    # ------------------------------------------------------------------ nodes
    def n_start(self, s):
        what = rules.out_of_scope(s["request"])
        ids = transaction_ids(s["request"])
        self.tracer.event("start", request=s["request"], transaction_ids=ids, scope_check=what)
        if what:
            return {"transaction_ids": ids, "request_type": "out_of_scope", "outcome": "refusal", "path": "refusal",
                    "message": rules.REFUSAL.format(what=what)}
        if self.llm is None:
            return {"transaction_ids": ids, "path": "template", "fallback_reason": "no LLM configured"}
        if self.force_fallback:
            return {"transaction_ids": ids, "path": "fallback", "fallback_reason": "fallback path requested"}
        return {"transaction_ids": ids, "replans": 0, "llm_calls": 0, "path": "orchestrated"}

    def r_start(self, s):
        if s.get("outcome") == "refusal":
            return "finalize"
        return "fallback" if s.get("path") in ("template", "fallback") else "plan"

    def n_plan(self, s):
        feedback = "\n".join(s.get("plan_problems") or []) or None
        try:
            plan = A.orchestrate(self.llm, self.prompts["orchestrator"], s["request"], s["transaction_ids"], self.tracer, feedback)
        except A.AgentOutputError as e:
            self.tracer.event("plan_rejected", problems=[str(e)])
            if _llm_down(e):
                return {"plan": None, "plan_problems": [str(e)], "llm_down": True, "llm_calls": s.get("llm_calls", 0) + 1,
                        "fallback_reason": f"LLM unavailable: {e}"[:300]}
            return {"plan": None, "plan_problems": [str(e)], "replans": s.get("replans", 0) + 1,
                    "llm_calls": s.get("llm_calls", 0) + 1}
        problems, updates = self._validate_plan(plan, s)
        self.tracer.event("plan", plan=plan, problems=problems)
        return {"plan": plan if not problems else None, "plan_problems": problems,
                "replans": s.get("replans", 0) + (1 if problems else 0), "llm_calls": s.get("llm_calls", 0) + 1, **updates}

    def _validate_plan(self, plan: dict, s: dict) -> tuple[list[str], dict]:
        rtype, ids = plan["request_type"], s["transaction_ids"]
        if rtype == "out_of_scope":                              # the model may refuse; code never overrides a refusal
            return [], {"request_type": rtype, "outcome": "refusal",
                        "message": rules.REFUSAL.format(what="carry out this request")}
        problems = []
        tools = [c["tool"] for c in plan["tool_calls"]]
        if rtype in schemas.TRANSACTION_TYPES:
            if not ids:
                return [], {"request_type": rtype, "outcome": "missing_transaction",
                            "message": "Please give the 7-digit TransactionID of the alert (for example TX-3548165)."}
            if plan.get("transaction_id") not in ids:
                problems.append(f"transaction_id must be one of the IDs in the request: {ids}")
        if rtype == "typology" and ids:
            problems.append("the request names a transaction: use a transaction request type")
        bad = [t for t in tools if t not in schemas.ALLOWED_TOOLS[rtype]]
        if bad:
            problems.append(f"tools not allowed for {rtype}: {bad}")
        missing = [t for t in schemas.REQUIRED_TOOLS[rtype] if t not in tools]
        if missing:
            problems.append(f"required tools missing for {rtype}: {missing}")
        return problems, {"request_type": rtype, "transaction_id": plan.get("transaction_id")}

    def r_plan(self, s):
        if s.get("outcome"):
            return "finalize"
        if s.get("plan") is not None:
            return "execute"
        if s.get("replans", 0) <= schemas.MAX_REPLANS and not self._over_budget(s) and not s.get("llm_down"):
            return "plan"
        return "fallback"

    def n_execute(self, s):
        ev = self._run_calls(s["plan"]["tool_calls"], s["request_type"], s.get("transaction_id"), s["started"])
        return {"evidence": ev.items}

    def _run_calls(self, calls: list[dict], rtype: str, tid: int | None, started: float, timed: bool = True) -> Evidence:
        ev = Evidence()
        for call in calls:
            if ev.fatal_error():                                  # unknown transaction: nothing else can succeed
                break
            if ev.calls() >= schemas.MAX_TOOL_CALLS or (timed and time.time() - started > self.time_budget):
                self.tracer.event("budget", calls=ev.calls(), seconds=round(time.time() - started, 1))
                break
            tool, args = call["tool"], dict(call.get("args") or {})
            if tool in ("fraud_score", "explain_transaction", "entity_history") and tid is not None:
                args["transaction_id"] = int(tid)
            if tool == "indicator_map":
                args = self._map_args(ev, rtype)
            if tool == "case_store":
                h = ev.output("entity_history") or {}
                args = {"action": "read", **({"transaction_id": int(tid)} if tid is not None else {}),
                        **({"entity_key": str(h["entity_key"])} if h.get("available") and h.get("entity_key") else {})}
            if tool == "policy_search" and args.pop("from_indicator_map", False):
                m = ev.output("indicator_map")
                queries = (m or {}).get("queries", [])[:MAX_SEARCHES_FROM_MAP]
                for q in queries:
                    if ev.calls() >= schemas.MAX_TOOL_CALLS:
                        break
                    qa = {"query": q["query"], "k": 3, **({"tier": q["tier"]} if q.get("tier") else {})}
                    item = self.tools.call(ev, "policy_search", qa)
                    self.tracer.event("tool_call", **item)
                if not queries:
                    item = self.tools.call(ev, "policy_search", {"query": "card-not-present fraud investigation steps", "k": 3})
                    self.tracer.event("tool_call", **item)
                continue
            if tool == "policy_search":
                args = self._search_args(args, rtype)
            item = self.tools.call(ev, tool, args)
            self.tracer.event("tool_call", **item)
        return ev

    @staticmethod
    def _search_args(args: dict, rtype: str) -> dict:
        """Keep the arguments policy_search accepts, coerced to their types (models add stray keys)."""
        out = {"query": str(args.get("query", "")).strip()[:400]}
        tier = args.get("tier")
        if isinstance(tier, str) and tier.strip().isdigit():
            tier = int(tier)
        if isinstance(tier, int) and 1 <= tier <= 3:
            out["tier"] = tier
        if isinstance(args.get("jurisdiction"), str) and args["jurisdiction"].strip():
            out["jurisdiction"] = args["jurisdiction"].strip()
        k = args.get("k")
        out["k"] = int(k) if isinstance(k, int) and 1 <= k <= 5 else (5 if rtype == "typology" else 3)
        return out

    @staticmethod
    def _map_args(ev: Evidence, rtype: str) -> dict:
        expl, score, hist = ev.output("explain_transaction"), ev.output("fraud_score"), ev.output("entity_history")
        families = [r["family"] for r in (expl or {}).get("reasons", [])]
        flags = [f["flag"] for f in (score or {}).get("novelty_flags", []) if f.get("on")]
        return {"reason_families": families, "novelty_flags": flags,
                "entity_available": bool((hist or {}).get("available", True)), "request_type": rtype}

    def n_check(self, s):
        ev = Evidence(s["evidence"])
        fatal = ev.fatal_error()
        if fatal:
            return {"outcome": "not_found", "message": f"{fatal['error'][0].upper()}{fatal['error'][1:]}."}
        missing = policy.missing_evidence(ev, s["request_type"])
        self.tracer.event("evidence_check", missing=missing, tool_calls=ev.calls())
        if missing:
            errors = [f"{i['tool']}: {i['error']}" for i in ev.items if not i["ok"]]
            return {"plan": None, "replans": s.get("replans", 0) + 1,
                    "plan_problems": [f"required tools did not succeed: {missing}"] + errors}
        return {}

    def r_check(self, s):
        if s.get("outcome"):
            return "finalize"
        if s.get("plan") is None:
            return "plan" if s.get("replans", 0) <= schemas.MAX_REPLANS and not self._over_budget(s) else "fallback"
        if self._over_budget(s):
            return "fallback"
        if s["request_type"] == "typology" or getattr(self.llm, "lean", False):
            return "write"                            # lean mode: the report agent works from the evidence directly
        return "investigate"

    def n_investigate(self, s):
        ev = Evidence(s["evidence"])
        try:
            inv = A.investigate(self.llm, self.prompts["investigation"], s["request"], s["request_type"], ev, self.tracer)
            return {"investigation": inv, "inv_attempts": s.get("inv_attempts", 0) + 1, "llm_calls": s.get("llm_calls", 0) + 1}
        except A.AgentOutputError as e:
            self.tracer.event("investigation_rejected", error=str(e))
            return {"investigation": None, "inv_attempts": s.get("inv_attempts", 0) + 1, "llm_calls": s.get("llm_calls", 0) + 1,
                    "fallback_reason": (f"LLM unavailable: {e}" if _llm_down(e) else str(e))[:300], "llm_down": _llm_down(e)}

    def r_investigate(self, s):
        if s.get("investigation") is not None and not self._over_budget(s):
            return "write"
        retry = s.get("inv_attempts", 0) < 2 and not self._over_budget(s) and not s.get("llm_down")
        return "investigate" if retry else "fallback"

    def n_write(self, s):
        ev = Evidence(s["evidence"])
        try:
            rep = A.write_report(self.llm, self.prompts["report"], s["request"], s["request_type"], ev, s.get("investigation"),
                                 self.tracer, feedback=s.get("feedback"), previous=s.get("report"))
            return {"report": rep, "drafts": s.get("drafts", 0) + 1, "llm_calls": s.get("llm_calls", 0) + 1}
        except A.AgentOutputError as e:
            self.tracer.event("report_rejected", error=str(e))
            return {"report": None, "drafts": s.get("drafts", 0) + 1, "llm_calls": s.get("llm_calls", 0) + 1,
                    "feedback": None, "fallback_reason": (f"LLM unavailable: {e}" if _llm_down(e) else str(e))[:300],
                    "llm_down": _llm_down(e)}

    def r_write(self, s):
        if s.get("report") is not None:
            return "validate"                     # a finished draft is always checked, even past the budget
        retry = s.get("drafts", 0) < MAX_REPORT_DRAFTS and not self._over_budget(s) and not s.get("llm_down")
        return "write" if retry else "fallback"

    def n_validate(self, s):
        res = policy.check(s["report"], Evidence(s["evidence"]), s["request_type"])
        self.tracer.event("policy", draft=s.get("drafts"), passed=res.ok, metrics=res.metrics(), violations=res.violations)
        out = {"policy": {**res.metrics(), "passed": res.ok}}
        if s.get("policy_first") is None:
            out["policy_first"] = res.metrics()
        if not res.ok:
            out["feedback"] = res.feedback()
            out["fallback_reason"] = "evidence policy failed on two drafts: " + ", ".join(res.metrics()["codes"])
        return out

    def r_validate(self, s):
        if s["policy"]["passed"]:
            return "finalize"
        return "write" if s.get("drafts", 0) < MAX_REPORT_DRAFTS and not self._over_budget(s) else "fallback"

    def n_fallback(self, s):
        """Fixed sequence (score, explain, entity history, indicator map, search, report by template)."""
        ids = s.get("transaction_ids") or []
        rtype = s.get("request_type") if s.get("request_type") in schemas.REQUEST_TYPES else None
        rtype = rtype or rules.classify(s["request"], bool(ids))
        if rtype == "out_of_scope":
            return {"request_type": rtype, "outcome": "refusal", "message": rules.REFUSAL.format(what="carry out this request")}
        tid = s.get("transaction_id") or (ids[0] if ids else None)
        path = "template" if s.get("path") == "template" else "fallback"
        reason = s.get("fallback_reason") or ("time budget exceeded" if self._over_budget(s) else "orchestrated path failed")
        if rtype in schemas.TRANSACTION_TYPES and tid is None:
            return {"request_type": rtype, "path": path, "fallback_reason": reason, "outcome": "missing_transaction",
                    "message": "Please give the 7-digit TransactionID of the alert (for example TX-3548165)."}
        self.tracer.event("fallback", reason=reason, request_type=rtype)
        # the fallback is the last resort: fixed, fast, not subject to the time budget
        ev = self._run_calls(rules.default_plan(rtype, tid, s["request"]), rtype, tid, time.time(), timed=False)
        fatal = ev.fatal_error()
        if fatal:
            return {"request_type": rtype, "transaction_id": tid, "evidence": ev.items, "path": path, "fallback_reason": reason,
                    "outcome": "not_found", "message": f"{fatal['error'][0].upper()}{fatal['error'][1:]}."}
        missing = policy.missing_evidence(ev, rtype)
        rep = template_report(ev, rtype) if not missing else {"summary": None, "interpretation": [], "guidance": [],
                                                                "suggested_checks": []}
        res = policy.check(rep, ev, rtype) if not missing else None
        out = {"request_type": rtype, "transaction_id": tid, "evidence": ev.items, "report": rep, "path": path,
               "fallback_reason": reason}
        if missing:
            out.update({"outcome": "tools_failed", "message": "FinGuard could not gather the required evidence "
                        f"({', '.join(missing)} failed); no report can be written. See the Sources section."})
        else:
            out["policy"] = {**res.metrics(), "passed": res.ok}
        return out

    def n_finalize(self, s):
        out = {}
        if not s.get("outcome"):
            ev = Evidence(s.get("evidence", []))
            search = ev.latest("policy_search")
            covered = search is not None and search["output"].get("status") == "ok"
            out["outcome"] = "not_covered" if s.get("request_type") == "typology" and not covered else "report"
        self.tracer.event("finalize", outcome=s.get("outcome") or out["outcome"], path=s.get("path"))
        return out

    # ------------------------------------------------------------------ public
    def run(self, request: str, case_id: str | None = None) -> dict:
        created = time.strftime("%Y-%m-%d %H:%M:%S")
        case_id = case_id or f"C{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        self.tracer = Tracer(self.out_dir / "traces" / f"{case_id}.jsonl") if self.save else NullTracer()
        t0 = time.time()
        if hasattr(self.llm, "begin_run"):                        # a chain records which model answered this run
            self.llm.begin_run()
        has_deadline = hasattr(self.llm, "deadline")
        if has_deadline:                                          # HTTP clients stop waiting at the deadline
            self.llm.deadline = t0 + self.time_budget + LLM_GRACE
        try:
            state = self.app.invoke({"request": request, "started": t0}, max_steps=40)
        finally:
            if has_deadline:
                self.llm.deadline = None
        seconds = round(time.time() - t0, 2)
        ev = Evidence(state.get("evidence", []))
        try:
            versions = self.tools.ctx.versions()
        except Exception as e:                                    # versions are a record, never a reason to fail a case
            versions = {"error": str(e)}
        llm_meta = {"name": getattr(self.llm, "name", None), "model": getattr(self.llm, "model", None),
                    "lean": bool(getattr(self.llm, "lean", False)), "passage_chars": getattr(self.llm, "passage_chars", None)}
        if hasattr(self.llm, "served"):                           # chain: who answered, who failed, who was not configured
            llm_meta.update({"served": dict(self.llm.served), "fallbacks": list(self.llm.failures),
                             "not_configured": dict(getattr(self.llm, "skipped", {}))})
        if state.get("path") in ("template", "refusal"):
            llm_meta = {"name": None, "model": None}
        hist = ev.output("entity_history") or {}
        score = ev.output("fraud_score") or {}
        case = {"case_id": case_id, "created_at": created, "request": request, "request_type": state.get("request_type"),
                "transaction_id": state.get("transaction_id") or (score.get("transaction_id")),
                "entity_key": hist.get("entity_key"), "outcome": state.get("outcome"), "message": state.get("message"),
                "path": state.get("path"), "fallback_reason": state.get("fallback_reason") if state.get("path") != "orchestrated" else None,
                "report": state.get("report"), "evidence": ev.items, "plan": state.get("plan"),
                "policy": {**(state.get("policy") or {}), "first_attempt": state.get("policy_first")},
                "versions": versions, "llm": llm_meta, "graph": self.backend,
                "prompts": [p.tag for p in self.prompts.values()] if llm_meta["name"] else [],
                "tool_calls": ev.calls(), "llm_calls": state.get("llm_calls", 0), "replans": max(0, state.get("replans", 0)),
                "drafts": state.get("drafts", 0), "seconds": seconds,
                "band": score.get("band"), "score": score.get("score"), "disclaimer": DISCLAIMER}
        case["markdown"] = render(case)
        self.tracer.event("case", case_id=case_id, outcome=case["outcome"], path=case["path"], seconds=seconds)
        if self.save:
            d = self.out_dir / "reports"
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{case_id}.md").write_text(case["markdown"], encoding="utf-8")
            (d / f"{case_id}.json").write_text(json.dumps({k: v for k, v in case.items() if k != "markdown"},
                                                          indent=1, ensure_ascii=False, default=str), encoding="utf-8")
            try:
                self.tools.ctx.case_store.save_draft({**case, "model_version": versions.get("model_version"),
                                                      "index_version": versions.get("index_version")})
            except Exception as e:                                # the report is already on disk
                self.tracer.event("case_store_error", error=str(e))
        return case
