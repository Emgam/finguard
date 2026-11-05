"""The three LLM agents: each builds its message, calls the model, parses one JSON object and
validates it against its schema. Anything else raises ``AgentOutputError``, which the graph
routes (re-plan, retry or fallback); no agent output reaches code unvalidated.
"""
from __future__ import annotations

import json

from finguard.agents import schemas
from finguard.agents.evidence import Evidence
from finguard.agents.llm import LLMError, parse_json
from finguard.agents.prompt_store import Prompt


class AgentOutputError(Exception):
    pass


def tool_table() -> str:
    return "\n".join(f"- {name}: {desc}" for name, desc in schemas.TOOL_DESCRIPTION.items())


def _call(llm, prompt: Prompt, user: str, context: dict, schema: dict, tracer, agent: str, system: str | None = None) -> dict:
    system = system or prompt.text
    try:
        resp = llm.complete(agent=agent, system=system, user=user, context=context)
    except LLMError as e:
        tracer.event("llm_error", agent=agent, prompt=prompt.tag, error=str(e))
        raise AgentOutputError(f"{agent}: {e}") from e
    tracer.event("llm_call", agent=agent, prompt=prompt.tag, prompt_sha256=prompt.sha256, model=resp.model,
                 settings=getattr(resp, "settings", {}), seconds=round(resp.seconds, 3), usage=resp.usage, user=user,
                 reply=resp.text)
    try:
        obj = parse_json(resp.text)
    except (ValueError, json.JSONDecodeError) as e:
        raise AgentOutputError(f"{agent}: reply is not one JSON object ({e})") from e
    problems = schemas.errors(obj, schema)
    if problems:
        raise AgentOutputError(f"{agent}: output does not match its schema: " + "; ".join(problems[:6]))
    return obj


def orchestrate(llm, prompt: Prompt, request: str, transaction_ids: list[int], tracer, feedback: str | None = None) -> dict:
    system = prompt.text.replace("{tools}", tool_table())
    user = (f"Analyst request:\n{request}\n\nTransactionIDs found in the request by code: "
            f"{', '.join(map(str, transaction_ids)) or 'none'}")
    if feedback:
        user += f"\n\nYour previous plan could not be used:\n{feedback}\nReturn the whole corrected plan."
    ctx = {"request": request, "transaction_ids": transaction_ids, "feedback": feedback}
    return _call(llm, prompt, user, ctx, schemas.PLAN, tracer, "orchestrator", system=system)


def _evidence_text(llm, ev: Evidence) -> str:
    """The evidence pack; ``passage_chars`` (compact mode for free tiers and small models) shortens passages."""
    return ev.prompt_text(max_passage_chars=int(getattr(llm, "passage_chars", 1500) or 1500))


def investigate(llm, prompt: Prompt, request: str, request_type: str, ev: Evidence, tracer) -> dict:
    user = f"Request type: {request_type}\nAnalyst request:\n{request}\n\nEvidence:\n{_evidence_text(llm, ev)}"
    ctx = {"request": request, "request_type": request_type, "evidence": ev.items}
    return _call(llm, prompt, user, ctx, schemas.INVESTIGATION, tracer, "investigation")


def write_report(llm, prompt: Prompt, request: str, request_type: str, ev: Evidence, investigation: dict | None, tracer,
                 feedback: str | None = None, previous: dict | None = None) -> dict:
    user = f"Request type: {request_type}\nAnalyst request:\n{request}\n\nEvidence:\n{_evidence_text(llm, ev)}"
    if investigation is not None:
        user += f"\n\nInvestigation agent output:\n{json.dumps(investigation, ensure_ascii=False)}"
    elif request_type != "typology":
        user += ("\n\nNo investigation agent output (lean mode): interpret the evidence yourself, with the same "
                 "rules for citations, numbers and hedging.")
    if feedback:
        user += (f"\n\nYour previous report:\n{json.dumps(previous, ensure_ascii=False)}\n\nProblems found by code:\n"
                 f"{feedback}\nReturn the whole corrected report JSON.")
    ctx = {"request": request, "request_type": request_type, "evidence": ev.items, "investigation": investigation,
           "feedback": feedback, "previous": previous}
    return _call(llm, prompt, user, ctx, schemas.REPORT, tracer, "report")
