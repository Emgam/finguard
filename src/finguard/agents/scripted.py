"""A deterministic stand-in for the LLM, used by the regression tests and CI, and to run the
pipeline without an API key.

It answers through exactly the same contract as a real model (one JSON object per agent,
validated by the same schemas and the same evidence policy), reading the structured
``context`` the agents pass alongside the prompt. Its plans are the fixed plans, and its texts
are built from the evidence, so it shows that the pipeline works; it says nothing about how
well an LLM plans or writes. Reports it produces are labelled "scripted test model".
"""
from __future__ import annotations

import json
import time

from finguard.agents import rules
from finguard.agents.evidence import Evidence
from finguard.agents.llm import LLMResponse
from finguard.agents.report import template_report


class ScriptedLLM:
    name = "scripted"
    model = "scripted-test-model"

    def complete(self, agent: str, system: str, user: str, context: dict | None = None) -> LLMResponse:
        t0 = time.time()
        ctx = context or {}
        out = getattr(self, f"_{agent}")(ctx)
        return LLMResponse(json.dumps(out, ensure_ascii=False), self.model, time.time() - t0, {})

    # ------------------------------------------------------------------ agents
    def _orchestrator(self, ctx: dict) -> dict:
        request, ids = ctx["request"], ctx.get("transaction_ids") or []
        rtype = rules.classify(request, bool(ids))
        tid = ids[0] if ids else None
        return {"request_type": rtype, "transaction_id": tid, "tool_calls": rules.default_plan(rtype, tid, request),
                "rationale": "fixed plan for the request type (scripted test model)"}

    def _investigation(self, ctx: dict) -> dict:
        rep = template_report(Evidence(ctx["evidence"]), ctx["request_type"])
        return {"interpretations": rep["interpretation"][:6], "suggested_checks": rep["suggested_checks"][:6],
                "open_questions": []}

    def _report(self, ctx: dict) -> dict:
        ev = Evidence(ctx["evidence"])
        rep = template_report(ev, ctx["request_type"])
        inv = ctx.get("investigation")
        if inv:
            rep["interpretation"], rep["suggested_checks"] = inv["interpretations"], inv["suggested_checks"]
        if ctx.get("feedback") and ctx.get("previous"):               # revision: drop what code flagged
            flagged = {(line.split("]")[0].lstrip("- ")) for line in ctx["feedback"].splitlines()}
            prev = ctx["previous"]
            for section in ("interpretation", "guidance", "suggested_checks"):
                prev[section] = [c for n, c in enumerate(prev.get(section, [])) if f"{section}[{n}" not in flagged]
            return prev
        return rep
