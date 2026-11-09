"""Agent evaluation on the 20 scripted cases (Project Report v3.0, sections 8.5 and 11).

Each case runs twice: through the orchestrated path (the configured LLM) and through the fixed
fallback path. Metrics:

* request-type accuracy, outcome accuracy and correct refusals;
* outcome against the evidence: a knowledge-base question must end as a report when the deployed
  retriever found passages and as "not covered by the sources" when it declined. Whether the
  retriever itself agrees with the case label is a knowledge-base (Phase 3) result: it depends on
  the deployed embedder, re-ranker and threshold, so it is listed apart and is not an agent error;
* tool-plan accuracy: the plan contains the request type's required tools and only allowed ones;
* evidence-policy compliance of the final reports (exit criterion: 100%);
* groundedness: numbers traceable in the final reports (100%) and claims cited in the agents'
  FIRST draft, before any revision or fallback (exit criterion: at least 90%);
* fallback rate, tool calls, LLM calls and latency per case.
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import pandas as pd

from finguard import config
from finguard.agents import cases as C
from finguard.agents.evidence import Evidence
from finguard.agents.graph import Orchestrator
from finguard.agents.schemas import ALLOWED_TOOLS, REQUIRED_TOOLS

WRITTEN = ("report", "not_covered")


def case_row(case_spec: dict, case: dict, mode: str) -> dict:
    exp_type, exp_out = case_spec["expected_type"], case_spec["expected_outcome"]
    plan_tools = [c["tool"] for c in (case.get("plan") or {}).get("tool_calls", [])]
    plan_ok = None
    if mode == "orchestrated" and case.get("plan"):
        rtype = case["plan"]["request_type"]
        plan_ok = (rtype == exp_type and set(REQUIRED_TOOLS[exp_type]) <= set(plan_tools)
                   and set(plan_tools) <= set(ALLOWED_TOOLS[exp_type]))
    pol = case.get("policy") or {}
    first = pol.get("first_attempt") or {}
    search = Evidence(case.get("evidence", [])).latest("policy_search")
    search_status = None if search is None else search["output"].get("status")
    if exp_type == "typology" and exp_out in WRITTEN:
        evidence_outcome = "report" if search_status == "ok" else "not_covered"   # what retrieval decided
        retrieval_agrees = evidence_outcome == exp_out
    else:
        evidence_outcome, retrieval_agrees = exp_out, None
    return {"case": case_spec["id"], "mode": mode, "stratum": case_spec["stratum"], "request": case_spec["request"],
            "expected_type": exp_type, "request_type": case.get("request_type"),
            "type_correct": case.get("request_type") == exp_type,
            "expected_outcome": exp_out, "outcome": case.get("outcome"), "outcome_correct": case.get("outcome") == exp_out,
            "search_status": search_status, "evidence_outcome": evidence_outcome,
            "outcome_follows_evidence": case.get("outcome") == evidence_outcome, "retrieval_agrees_with_label": retrieval_agrees,
            "refusal_correct": (case.get("outcome") == "refusal") == (exp_out == "refusal"),
            "plan_tools": " ".join(plan_tools), "plan_correct": plan_ok, "path": case.get("path"),
            "fallback_reason": case.get("fallback_reason"),
            "written": case.get("outcome") in WRITTEN, "policy_passed": pol.get("passed"),
            "claims": pol.get("claims"), "cited": pol.get("cited"), "numbers": pol.get("numbers"), "traceable": pol.get("traceable"),
            "first_claims": first.get("claims"), "first_cited": first.get("cited"), "first_numbers": first.get("numbers"),
            "first_traceable": first.get("traceable"), "first_violations": first.get("violations"),
            "first_codes": " ".join(first.get("codes", [])) if first else None,
            "served_by": "; ".join(f"{k} x{n}" for k, n in (case.get("llm") or {}).get("served", {}).items()) or None,
            "tool_calls": case.get("tool_calls"), "llm_calls": case.get("llm_calls"), "replans": case.get("replans"),
            "drafts": case.get("drafts"), "seconds": case.get("seconds"), "case_id": case.get("case_id")}


def _share(num, den):
    return None if not den else float(round(float(num) / float(den), 4))


def summarise(rows: pd.DataFrame) -> dict:
    out = {}
    for mode, d in rows.groupby("mode"):
        w = d[d.written]
        sec = d.seconds.dropna().tolist()
        out[mode] = {
            "cases": int(len(d)),
            "request_type_accuracy": _share(int(d.type_correct.sum()), len(d)),
            "outcome_accuracy": _share(int(d.outcome_correct.sum()), len(d)),
            "outcome_follows_evidence": _share(int(d.outcome_follows_evidence.sum()), len(d)),
            "retrieval_disagrees_with_label": sorted(d[d.retrieval_agrees_with_label.eq(False)].case.tolist()),
            "refusals_correct": _share(int(d.refusal_correct.sum()), len(d)),
            "plan_accuracy": _share(int(d.plan_correct.fillna(False).astype(bool).sum()), int(d.plan_correct.notna().sum())),
            "written_reports": int(len(w)),
            "policy_compliance": _share(int(w.policy_passed.fillna(False).astype(bool).sum()), len(w)),
            "numbers_traceable_final": _share(w.traceable.fillna(0).sum(), w.numbers.fillna(0).sum()) if w.numbers.fillna(0).sum() else 1.0,
            "claims_cited_final": _share(w.cited.fillna(0).sum(), w.claims.fillna(0).sum()),
            "claims_cited_first_draft": _share(d.first_cited.fillna(0).sum(), d.first_claims.fillna(0).sum()),
            "numbers_traceable_first_draft": _share(d.first_traceable.fillna(0).sum(), d.first_numbers.fillna(0).sum()),
            "fallback_rate": _share(int((d.path == "fallback").sum()), len(d)),
            "mean_tool_calls": round(float(d.tool_calls.mean()), 2), "mean_llm_calls": round(float(d.llm_calls.mean()), 2),
            "median_seconds": round(statistics.median(sec), 2) if sec else None,
            "max_seconds": round(max(sec), 2) if sec else None,
        }
    return out


def exit_criteria(summary: dict) -> dict:
    s = summary.get("orchestrated") or {}
    return {"policy_compliance_100": bool(s.get("policy_compliance") == 1.0),
            "numbers_traceable_100": bool(s.get("numbers_traceable_final") == 1.0),
            "claims_cited_first_draft_ge_90": bool((s.get("claims_cited_first_draft") or 0) >= 0.90)}


def select_cases(spec: str | None, cases: list[dict] | None = None) -> list[dict]:
    """``c01,c05,c13-c16`` -> those cases, in file order (all of them when ``spec`` is empty)."""
    cases = cases or C.load()
    if not spec:
        return cases
    wanted = set()
    for part in (p.strip().lower() for p in spec.split(",") if p.strip()):
        if "-" in part:
            lo, hi = (int(x.strip().lstrip("c")) for x in part.split("-", 1))
            wanted |= {f"c{n:02d}" for n in range(lo, hi + 1)}
        else:
            wanted.add(f"c{int(part.lstrip('c')):02d}")
    unknown = wanted - {c["id"] for c in cases}
    if unknown:
        raise ValueError(f"unknown case(s): {', '.join(sorted(unknown))}")
    return [c for c in cases if c["id"] in wanted]


def run(orchestrated: Orchestrator, fallback: Orchestrator | None, cases: list[dict] | None = None,
        out_dir: Path = config.TABLE_DIR, label: str = "", log=print, resume: bool = False, pause: float = 0.0) -> dict:
    """Run the cases on both paths and write the table and summary. ``resume`` keeps the rows already
    in this LLM's table and runs only the missing cases (for free tiers with daily quotas); ``pause``
    waits between cases (for per-minute limits)."""
    cases = cases or C.load()
    out_dir = Path(out_dir)
    llm = getattr(orchestrated.llm, "name", None)
    tag = f"_{llm}" if llm else ""
    table = out_dir / f"phase4_agent_cases{tag}.csv"
    modes = [m for m, o in (("orchestrated", orchestrated), ("fallback", fallback)) if o is not None]
    previous = pd.read_csv(table) if resume and table.exists() else None
    if previous is not None:
        done = {c for c, g in previous.groupby("case") if set(modes) <= set(g["mode"])}
        log(f"  resuming: {len(done)} case(s) already in {table.name}")
        cases = [c for c in cases if c["id"] not in done]
    rows = []
    for n, spec in enumerate(cases):
        if n and pause:
            time.sleep(pause)
        for mode, orch in (("orchestrated", orchestrated), ("fallback", fallback)):
            if orch is None:
                continue
            case = orch.run(spec["request"], case_id=f"eval-{spec['id']}-{mode}")
            rows.append(case_row(spec, case, mode))
            r = rows[-1]
            flag = "  (retriever differs from the case label)" if r["retrieval_agrees_with_label"] is False else ""
            log(f"  {spec['id']} {mode:<12} {r['request_type'] or '-':<14} {r['outcome']:<12} path={r['path']:<12} "
                f"policy={r['policy_passed']} tools={r['tool_calls']} llm={r['llm_calls']} {r['seconds']}s{flag}")
    df = pd.DataFrame(rows)
    if previous is not None:
        df = pd.concat([previous, df], ignore_index=True).drop_duplicates(["case", "mode"], keep="last")
    summary = summarise(df)
    payload = {"run_at": time.strftime("%Y-%m-%d %H:%M:%S"), "label": label, "llm": llm,
               "cases_done": int(df["case"].nunique()), "cases_total": len(C.load()),
               "llm_model": getattr(orchestrated.llm, "model", None), "graph": orchestrated.backend,
               "prompts": [p.tag for p in orchestrated.prompts.values()], "versions": orchestrated.tools.ctx.versions(),
               "summary": summary, "exit_criteria": exit_criteria(summary),
               "note": ("The scripted test model is deterministic and follows the fixed plans: these figures check the "
                        "pipeline, not an LLM." if llm == "scripted" else "")}
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(table, index=False)
    (out_dir / f"phase4_agent_summary{tag}.json").write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    if Path(out_dir).resolve() == Path(config.TABLE_DIR).resolve():
        from finguard.mlops import tracking             # MLflow run when installed; never fails the evaluation
        tracking.log_agents(payload, table)
    return payload
