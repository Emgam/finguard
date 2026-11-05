"""Agent regression test on the 20 scripted cases with the real tools (skipped without the data,
the Phase 2 artifacts or the knowledge-base index). It runs the scripted model, so it checks the
pipeline: outcomes, refusals, the evidence policy and the fallback path, on real evidence.

Knowledge-base questions are checked against what the deployed retriever returned: a report when
it found passages, "not covered by the sources" when it declined. Whether the retriever agrees
with a case's label depends on the embedder, re-ranker and threshold in ``abstention.json``
(a Phase 3 result, measured by ``python -m finguard.rag evaluate``), so a disagreement is shown
as a warning, not a failure."""
import tempfile
import warnings
from pathlib import Path
from unittest import SkipTest

import pandas as pd

from finguard import config


def test_twenty_scripted_cases_on_real_evidence():
    needed = [config.INTERIM_DIR / "train.pkl", config.ROOT / "artifacts" / "phase2" / "model_bundle.pkl",
              config.ROOT / "data" / "corpus" / "index" / "abstention.json"]
    if not all(Path(p).exists() or Path(str(p).replace(".pkl", ".parquet")).exists() for p in needed):
        raise SkipTest("data, Phase 2 artifacts or knowledge-base index not present")

    from finguard.agents.evaluate import run
    from finguard.agents.graph import Orchestrator
    from finguard.agents.scripted import ScriptedLLM
    from finguard.agents.tools import DataContext, Tools
    tools = Tools(DataContext().warm_up())
    out = Path(tempfile.mkdtemp())
    # no time budget here: a cross-encoder on a CPU is slow, and this test checks behaviour, not speed
    main = Orchestrator(tools, llm=ScriptedLLM(), out_dir=out, backend="builtin", time_budget=1e9)
    fb = Orchestrator(tools, llm=ScriptedLLM(), out_dir=out, backend="builtin", force_fallback=True)
    res = run(main, fb, out_dir=out, log=lambda *_: None)
    rows = pd.read_csv(out / "phase4_agent_cases_scripted.csv")

    problems = []
    for _, r in rows.iterrows():
        if not r.outcome_follows_evidence:
            problems.append(f"{r.case} {r['mode']}: outcome {r.outcome}, expected {r.evidence_outcome} "
                            f"(policy_search status {r.search_status}, path {r.path}, reason {r.fallback_reason})")
        if r.outcome in ("report", "not_covered") and r.policy_passed is not True and str(r.policy_passed) != "True":
            problems.append(f"{r.case} {r['mode']}: evidence policy failed ({r.first_codes})")
    for mode in ("orchestrated", "fallback"):
        s = res["summary"][mode]
        for key in ("refusals_correct", "request_type_accuracy", "policy_compliance", "numbers_traceable_final"):
            if s[key] != 1.0:
                problems.append(f"{mode}: {key} = {s[key]}")
    s = res["summary"]["orchestrated"]
    if s["plan_accuracy"] != 1.0 or s["fallback_rate"] != 0.0:
        problems.append(f"orchestrated: plan_accuracy {s['plan_accuracy']}, fallback_rate {s['fallback_rate']}")
    if not all(res["exit_criteria"].values()):
        problems.append(f"exit criteria: {res['exit_criteria']}")
    assert not problems, "\n".join(problems)

    differ = rows[rows.retrieval_agrees_with_label.eq(False) & rows["mode"].eq("orchestrated")]
    for _, r in differ.iterrows():
        warnings.warn(f"{r.case} ({r.request!r}): the deployed retriever returned {r.search_status}, the case label "
                      f"expects {r.expected_outcome}. This is a knowledge-base result, not an agent error.")
