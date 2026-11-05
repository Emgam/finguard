"""FinGuard agents from the command line (the Phase 4 deliverable: investigations producing reports).

    python -m finguard.agents investigate "Investigate alert TX-3554656" [--llm auto|groq|ollama|scripted|none]
    python -m finguard.agents investigate "What does Visa dispute condition 10.4 cover?"
    python -m finguard.agents investigate "..." --fallback          # the fixed fallback path
    python -m finguard.agents evaluate [--llm ...]                  # the 20 scripted cases, both paths
    python -m finguard.agents evaluate --llm groq --cases c01-c10   # part of them (free-tier quotas) ...
    python -m finguard.agents evaluate --llm groq --resume          # ... and the rest later, one table
    python -m finguard.agents decide CASE_ID --decision false_positive --analyst "L1 analyst" [--note "..."]
    python -m finguard.agents show CASE_ID                          # print a saved report
    python -m finguard.agents prompts [--lock]                      # prompt versions and lock status
    python -m finguard.agents cases --select                        # re-select the 20 scripted cases

The LLM comes from --llm or FINGUARD_LLM. The default, auto, is Groq openai/gpt-oss-120b (GROQ_API_KEY)
with Ollama qwen2.5:3b-instruct-q4_K_M on this computer as the offline fallback, and the template
report when neither answers; see ``finguard.agents.llm``. Scores and SHAP reasons are always computed
in Python by the tools, never by an LLM. Reports, traces and the case database are written to
``data/cases/`` (or FINGUARD_CASE_DIR).
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from finguard.agents import cases as C
from finguard.agents import prompt_store
from finguard.agents.case_store import CASE_DIR, DECISIONS, CaseStore


def _orchestrators(a, both: bool = False):
    from finguard.agents.graph import Orchestrator
    from finguard.agents.llm import make_llm
    from finguard.agents.tools import DataContext, Tools
    t0 = time.time()
    tools = Tools(DataContext().warm_up())
    print(f"(data, model and knowledge base loaded in {time.time() - t0:.0f} s)", file=sys.stderr)
    llm = make_llm(a.llm)
    main = Orchestrator(tools, llm=llm, backend=a.graph, force_fallback=getattr(a, "fallback", False))
    fb = Orchestrator(tools, llm=llm, backend=a.graph, force_fallback=True) if both else None
    return main, fb


def cmd_investigate(a) -> None:
    problems = prompt_store.lock_problems()
    if problems:
        print("WARNING: " + "; ".join(problems), file=sys.stderr)
    orch, _ = _orchestrators(a)
    case = orch.run(a.request)
    print(case["markdown"])
    print(f"\nsaved: {CASE_DIR / 'reports' / (case['case_id'] + '.md')}  (trace: traces/{case['case_id']}.jsonl)",
          file=sys.stderr)


def cmd_evaluate(a) -> None:
    from finguard.agents.evaluate import run, select_cases
    cases = select_cases(a.cases)
    orch, fb = _orchestrators(a, both=not a.no_fallback)
    res = run(orch, fb, cases=cases, label=a.note or "", resume=a.resume, pause=a.pause)
    print(json.dumps({"summary": res["summary"], "exit_criteria": res["exit_criteria"], "note": res["note"]}, indent=1, default=str))


def cmd_decide(a) -> None:
    print(json.dumps(CaseStore().record_decision(a.case_id, a.decision, a.analyst, a.note or ""), indent=1))


def cmd_show(a) -> None:
    p = CASE_DIR / "reports" / f"{a.case_id}.md"
    print(p.read_text(encoding="utf-8") if p.exists() else f"no report {p}")


def cmd_prompts(a) -> None:
    if a.lock:
        prompt_store.write_lock()
    for p in prompt_store.load_all().values():
        print(f"{p.tag:<26} sha256 {p.sha256[:16]}...")
    problems = prompt_store.lock_problems()
    print("lock: " + ("ok" if not problems else "; ".join(problems)))


def cmd_cases(a) -> None:
    if a.select:
        print(f"written {C.write(C.select())}")
    for c in C.load():
        print(f"{c['id']} {c['stratum']:<38} {c['expected_type']:<14} {c['expected_outcome']:<12} {c['request']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn, help_ in (("investigate", cmd_investigate, "run one request"), ("evaluate", cmd_evaluate, "the 20 cases")):
        s = sub.add_parser(name, help=help_)
        if name == "investigate":
            s.add_argument("request")
            s.add_argument("--fallback", action="store_true", help="run the fixed fallback path")
        else:
            s.add_argument("--no-fallback", action="store_true", help="skip the fallback-path comparison")
            s.add_argument("--note", default="")
            s.add_argument("--cases", default=None, help="subset, e.g. c01,c05,c13-c16 (default: all 20)")
            s.add_argument("--resume", action="store_true", help="keep this LLM's earlier rows; run only missing cases")
            s.add_argument("--pause", type=float, default=0.0, help="seconds to wait between cases (rate limits)")
        s.add_argument("--llm", default=None, help="auto (Groq, then Ollama offline), groq, ollama, none, scripted, gemini, "
                                                  "openrouter, openai or anthropic (default: FINGUARD_LLM or auto)")
        s.add_argument("--graph", default=None, help="langgraph, builtin or auto (default: FINGUARD_GRAPH or auto)")
        s.set_defaults(fn=fn)
    d = sub.add_parser("decide", help="record an analyst decision (people only)")
    d.add_argument("case_id")
    d.add_argument("--decision", required=True, choices=DECISIONS)
    d.add_argument("--analyst", required=True)
    d.add_argument("--note", default="")
    d.set_defaults(fn=cmd_decide)
    sh = sub.add_parser("show")
    sh.add_argument("case_id")
    sh.set_defaults(fn=cmd_show)
    p = sub.add_parser("prompts")
    p.add_argument("--lock", action="store_true")
    p.set_defaults(fn=cmd_prompts)
    c = sub.add_parser("cases")
    c.add_argument("--select", action="store_true")
    c.set_defaults(fn=cmd_cases)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
