"""Agents (Phase 4) on synthetic tools: no data, model or index needed, so these run in CI.

The fake tools return fixed outputs shaped like the real ones; the scripted model and a set
of deliberately faulty models exercise every route of the graph: refusal, missing
transaction, unknown transaction, re-plan, report revision, fallback, time budget.
"""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest import SkipTest

from finguard.agents import cases as C
from finguard.agents import policy, rules
from finguard.agents.case_store import CaseStore
from finguard.agents.evidence import Evidence
from finguard.agents.graph import Orchestrator
from finguard.agents.llm import AnthropicLLM, LLMResponse, OpenAICompatibleLLM, parse_json
from finguard.agents.prompt_store import lock_problems
from finguard.agents.scripted import ScriptedLLM
from finguard.agents.tools import IndicatorMap, ToolError, Tools

TID = 3554656
PASSAGES = {
    "card": {"chunk_id": "visa:0022", "doc_id": "visa", "title": "Visa Dispute Management Guide", "publisher": "Visa",
             "published": "2026-10", "jurisdiction": "Card network", "tier": 1, "language": "en", "authority": "x",
             "url": "https://example.org/visa.pdf", "section": "Condition 10.4", "subsections": [], "pages": "28-29",
             "kind": "text", "score": 1.0, "fictional": False,
             "text": "Condition 10.4 applies when a cardholder claims they did not authorize a transaction in a card-absent "
                     "environment. Merchants may respond with compelling evidence."},
    "email": {"chunk_id": "pb:0003", "doc_id": "playbook", "title": "FinGuard investigation playbook (fictional)",
              "publisher": "FinGuard project team", "published": "2026-10-29", "jurisdiction": "Project", "tier": 3,
              "language": "en", "authority": "x", "url": "corpus/docs/playbook.md", "section": "5. Indicators",
              "subsections": [], "pages": None, "kind": "text", "score": 1.0, "fictional": True,
              "text": "Purchaser and recipient email domains that differ are worth checking against the domain history. "
                      "This is consistent with stolen card details used by a third party."},
}


def fake_functions(known=(TID,), search_status="ok"):
    def need(tid):
        if tid not in known:
            raise ToolError(f"transaction {tid} is not in the labelled IEEE-CIS data loaded in FinGuard", retryable=False)

    def fraud_score(ctx, transaction_id):
        need(transaction_id)
        return {"transaction_id": transaction_id, "score": 93.4, "probability_displayed": "93%", "band": "High",
                "score_scale": "0-100", "model_version": "lgbm_numeric-v1", "period": "test", "relative_day": 170,
                "novelty_flags": [{"flag": "flag_new_device", "on": True, "rule": "New device brand for this proxy entity"}],
                "transaction": {"amount": 117.5, "product_code": "W", "card_network": "visa", "card_type": "debit",
                                "has_identity_record": False, "relative_hour": 14}, "notes": []}

    def explain_transaction(ctx, transaction_id):
        need(transaction_id)
        return {"transaction_id": transaction_id, "top": 5, "reasons": [
            {"family": "vesta_V", "label": "Vesta engineered features (V1-V339)", "anonymized": True, "checkable": False,
             "contribution_log_odds": 2.1, "signal_group": "V217-V278", "text": "Vesta engineered features: meaning not disclosed."},
            {"family": "email", "label": "Email domains", "anonymized": False, "checkable": True, "contribution_log_odds": 1.2,
             "signal_group": None, "text": "Email domains: purchaser anonymous.com, recipient gmail.com."},
            {"family": "card", "label": "Card attributes", "anonymized": False, "checkable": True, "contribution_log_odds": 0.8,
             "signal_group": None, "text": "Card attributes: network visa, type debit; card1 code seen 412 time(s)."}]}

    def entity_history(ctx, transaction_id, window_days=7):
        need(transaction_id)
        return {"window_days": float(window_days), "short_window_hours": 24, "entity": "proxy entity (heuristic)",
                "available": True, "entity_key": "1000_300_10", "earlier_transactions": 6, "earlier_in_24h": 4,
                "earlier_in_7d": 5, "amount": 117.5, "earlier_amount_mean": 39.2, "earlier_amount_max": 50.0,
                "amount_vs_earlier_mean": 2.997, "device_brands_seen": ["samsung"], "purchaser_email_domains_seen": ["gmail.com"]}

    imap = IndicatorMap.load()

    def indicator_map(ctx, reason_families, novelty_flags, entity_available=True, request_type=None):
        return imap.lookup(reason_families, novelty_flags, entity_available, request_type)

    def policy_search(ctx, query, tier=None, jurisdiction=None, k=3):
        if search_status != "ok" or "mastercard" in query.lower():
            return {"status": "not_covered", "message": "Not covered by the sources.", "passages": [],
                    "ungrounded_names": ["Mastercard"], "query": query}
        p = PASSAGES["email"] if "email" in query.lower() else PASSAGES["card"]
        return {"status": "ok", "passages": [dict(p)], "query": query}

    def case_store(ctx, action="read", transaction_id=None, entity_key=None):
        return {"cases": []}

    return {"fraud_score": fraud_score, "explain_transaction": explain_transaction, "entity_history": entity_history,
            "indicator_map": indicator_map, "policy_search": policy_search, "case_store": case_store}


class _Ctx:
    def versions(self):
        return {"model_version": "lgbm_numeric-v1", "index_version": "test"}


def orchestrator(llm=None, tmp=None, **kw):
    import tempfile
    tools = Tools(_Ctx(), functions=fake_functions(**{k: kw.pop(k) for k in ("known", "search_status") if k in kw}))
    return Orchestrator(tools, llm=llm, out_dir=tmp or tempfile.mkdtemp(), backend="builtin", save=False, **kw)


# --------------------------------------------------------------------------- scope and rules
def test_scope_check_refuses_actions_on_people_cards_and_reports_but_not_questions():
    for r in ("Block the card used in TX-3554656.", "Is the customer behind TX-3554656 a criminal?",
              "File a suspicious activity report for this customer.", "Please contact the cardholder of TX-3554656",
              "Can you freeze the account for 3554656?"):
        assert rules.out_of_scope(r), r
    for r in ("When must a suspicious transaction report be filed with the financial intelligence unit?",
              "Which practices reduce card testing?", "Why is TX-3554656 high risk?",
              "What does strong customer authentication require?", "How should merchants respond to a Visa 10.4 dispute?"):
        assert rules.out_of_scope(r) is None, r
    assert rules.classify("Why is TX-3554656 high risk, and what should I check?", True) == "what_to_check"
    assert rules.classify("Explain the score of TX-3554656", True) == "why_high_risk"
    assert rules.classify("Investigate alert TX-3554656", True) == "investigate"
    assert rules.classify("What does Visa 10.4 cover?", False) == "typology"


# --------------------------------------------------------------------------- the evidence policy
def _evidence():
    ev = Evidence()
    tools = Tools(_Ctx(), functions=fake_functions())
    for tool, args in (("fraud_score", {"transaction_id": TID}), ("explain_transaction", {"transaction_id": TID}),
                       ("entity_history", {"transaction_id": TID}), ("policy_search", {"query": "Visa 10.4"}),
                       ("policy_search", {"query": "email domains"})):
        tools.call(ev, tool, args)
    return ev


def test_evidence_ids_are_global_across_searches():
    ev = _evidence()
    assert [i["id"] for i in ev.items] == ["T1", "T2", "T3", "T4", "T5"]
    assert list(ev.passages()) == ["S1", "S2"] and ev.ids() >= {"T1", "T5", "S1", "S2"}
    assert "[S2] FinGuard investigation playbook (fictional)" in ev.prompt_text() and "FICTIONAL" in ev.prompt_text()


def test_policy_accepts_grounded_cited_hedged_text():
    ev = _evidence()
    rep = {"summary": {"text": "TX-3554656 is in the High band (score 93.4, about 93%).", "cites": ["T1"]},
           "interpretation": [{"text": "The amount is about 3.0 times the proxy entity's earlier mean, 4 transactions in "
                                       "24 h; this is consistent with patterns worth checking.", "cites": ["T3"]},
                              {"text": "The playbook (fictional) lists differing email domains as worth checking.", "cites": ["S2"]}],
           "guidance": [{"quote": "Merchants may respond with compelling evidence.", "cites": ["S1"],
                         "note": "Relevant if the cardholder disputes the payment."}],
           "suggested_checks": [{"text": "Do not block the card on the score alone; compare the email domains first.",
                                 "cites": ["T2"]}]}
    res = policy.check(rep, ev, "investigate")
    assert res.ok, res.violations
    assert res.cited == res.claims == 5 and res.traceable == res.numbers > 0


def test_policy_flags_every_kind_of_violation():
    ev = _evidence()
    rep = {"summary": {"text": "The transaction is fraudulent.", "cites": ["T1"]},
           "interpretation": [{"text": "No citation here.", "cites": []},
                              {"text": "Seen 999 times.", "cites": ["T2"]},
                              {"text": "Cites a ghost.", "cites": ["S9"]},
                              {"text": "The customer used a new device, which proves account takeover.", "cites": ["T1"]},
                              {"text": "V258 measures the device age.", "cites": ["T2"]}],
           "guidance": [{"quote": "Merchants must always refund the cardholder.", "cites": ["S1"]},
                        {"quote": "Not a passage, a tool output text here.", "cites": ["T1"]}],
           "suggested_checks": [{"text": "Block the card now.", "cites": ["T1"]}]}
    codes = {v["code"] for v in policy.check(rep, ev, "investigate").violations}
    assert codes == {"forbidden_wording", "uncited", "untraceable_number", "unknown_citation", "customer_word",
                     "anonymized_meaning", "quote_not_found", "guidance_not_passage"}
    general = {"summary": {"text": "Transaction 3554656 shows it.", "cites": ["T1"]}, "interpretation": [], "guidance": [],
               "suggested_checks": []}
    assert "transaction_in_general_answer" in {v["code"] for v in policy.check(general, ev, "typology").violations}


def test_numbers_may_be_rounded_or_percentages_of_fractions():
    idx = policy.NumberIndex(["0.4321", "93.4", "2,997.5"])
    assert idx.traceable("93") and idx.traceable("93.4") and idx.traceable("43%") and idx.traceable("43.2%")
    assert idx.traceable("2,997.5") and idx.traceable("2998") and not idx.traceable("94") and not idx.traceable("12")


def test_missing_evidence_names_the_required_tools():
    ev = Evidence()
    assert policy.missing_evidence(ev, "why_high_risk") == ["fraud_score", "explain_transaction", "entity_history"]
    assert policy.missing_evidence(ev, "out_of_scope") == []


# --------------------------------------------------------------------------- graph routes
def test_scripted_investigation_is_orchestrated_and_compliant():
    case = orchestrator(ScriptedLLM()).run(f"Investigate alert TX-{TID}")
    assert case["outcome"] == "report" and case["path"] == "orchestrated" and case["policy"]["passed"]
    assert case["llm_calls"] == 3 and case["tool_calls"] <= 8
    tools = [i["tool"] for i in case["evidence"]]
    assert tools[:4] == ["fraud_score", "explain_transaction", "entity_history", "indicator_map"] and "policy_search" in tools
    md = case["markdown"]
    for part in ("Flagged for review, not a finding of fraud.", "## Facts", "## Interpretation", "## Guidance",
                 "## Suggested checks", "## Limitations", "## Sources", "fictional", "scripted test model"):
        assert part in md, part


def test_refusal_happens_before_any_llm_or_tool_call():
    class Boom:
        name, model = "boom", "boom"

        def complete(self, **kw):
            raise AssertionError("no LLM call expected")
    case = orchestrator(Boom()).run(f"Block the card used in TX-{TID}.")
    assert case["outcome"] == "refusal" and case["tool_calls"] == 0 and case["llm_calls"] == 0
    assert "cannot act on a card" in case["markdown"]


def test_unknown_and_missing_transactions():
    assert orchestrator(ScriptedLLM()).run("Investigate TX-1234567.")["outcome"] == "not_found"
    case = orchestrator(ScriptedLLM()).run("Investigate TX-1234567.")
    assert case["tool_calls"] == 1                                       # stops after the fatal error
    assert orchestrator(None).run("Why is this alert high risk?")["outcome"] in ("report", "not_covered")  # typology


def test_typology_question_and_not_covered():
    case = orchestrator(ScriptedLLM()).run("What does Visa dispute condition 10.4 cover?")
    assert case["request_type"] == "typology" and case["outcome"] == "report" and case["report"]["guidance"]
    assert [i["tool"] for i in case["evidence"]] == ["policy_search"]
    nc = orchestrator(ScriptedLLM()).run("What is the Mastercard reason code for card-not-present fraud?")
    assert nc["outcome"] == "not_covered" and "Not covered by the sources" in nc["markdown"]


class Faulty(ScriptedLLM):
    """Scripted model with injected faults: bad plans, uncited report drafts."""
    name = "faulty"

    def __init__(self, bad_plans=0, bad_drafts=0, drop_required=False):
        self.bad_plans, self.bad_drafts, self.drop_required = bad_plans, bad_drafts, drop_required
        self.calls = []

    def complete(self, agent, system, user, context=None):
        self.calls.append(agent)
        if agent == "orchestrator" and self.bad_plans > 0:
            self.bad_plans -= 1
            return LLMResponse("I think we should look at the score.", "faulty", 0.0)
        resp = super().complete(agent, system, user, context)
        if agent == "orchestrator" and self.drop_required:
            self.drop_required = False
            plan = json.loads(resp.text)
            plan["tool_calls"] = plan["tool_calls"][:1]
            return LLMResponse(json.dumps(plan), "faulty", 0.0)
        if agent == "report" and self.bad_drafts > 0 and not (context or {}).get("feedback"):
            self.bad_drafts -= 1
            rep = json.loads(resp.text)
            rep["interpretation"].append({"text": "The transaction is fraudulent, seen 777 times.", "cites": []})
            return LLMResponse(json.dumps(rep), "faulty", 0.0)
        if agent == "report" and self.bad_drafts > 1:
            rep = json.loads(resp.text)
            rep["interpretation"].append({"text": "Uncited again.", "cites": []})
            return LLMResponse(json.dumps(rep), "faulty", 0.0)
        return resp


def test_invalid_plans_are_replanned_then_fall_back():
    llm = Faulty(bad_plans=1)
    case = orchestrator(llm).run(f"Investigate alert TX-{TID}")
    assert case["path"] == "orchestrated" and case["replans"] == 1 and llm.calls.count("orchestrator") == 2
    llm = Faulty(bad_plans=5)
    case = orchestrator(llm).run(f"Investigate alert TX-{TID}")
    assert case["path"] == "fallback" and llm.calls.count("orchestrator") == 3      # 1 plan + 2 re-plans
    assert case["policy"]["passed"] and "Fallback path" in case["markdown"]


def test_plan_without_required_tools_is_sent_back():
    llm = Faulty(drop_required=True)
    case = orchestrator(llm).run(f"Why is TX-{TID} high risk?")
    assert case["path"] == "orchestrated" and case["replans"] == 1 and case["policy"]["passed"]


def test_policy_violation_is_revised_once_then_falls_back():
    llm = Faulty(bad_drafts=1)
    case = orchestrator(llm).run(f"Investigate alert TX-{TID}")
    assert case["path"] == "orchestrated" and case["drafts"] == 2 and case["policy"]["passed"]
    assert case["policy"]["first_attempt"]["violations"] >= 2
    llm = Faulty(bad_drafts=3)
    case = orchestrator(llm).run(f"Investigate alert TX-{TID}")
    assert case["path"] == "fallback" and case["policy"]["passed"] and "evidence policy failed" in case["fallback_reason"]


def test_time_budget_sends_the_request_to_the_fallback_path():
    case = orchestrator(ScriptedLLM(), time_budget=0.0).run(f"Investigate alert TX-{TID}")
    assert case["path"] == "fallback" and case["policy"]["passed"]


class SlowReport(ScriptedLLM):
    """The report call finishes after the time budget has run out."""
    def __init__(self, orch_ref):
        super().__init__()
        self.orch_ref = orch_ref

    def complete(self, agent, system, user, context=None):
        if agent == "report":
            self.orch_ref[0].time_budget = 0.0
        return super().complete(agent, system, user, context)


def test_a_draft_finished_after_the_budget_is_still_validated_and_kept():
    ref = [None]
    orch = orchestrator(SlowReport(ref))
    ref[0] = orch
    case = orch.run(f"Investigate alert TX-{TID}")
    assert case["path"] == "orchestrated" and case["drafts"] == 1 and case["policy"]["passed"]


def test_no_llm_runs_the_template_path():
    case = orchestrator(None).run(f"What should I check next on TX-{TID}?")
    assert case["path"] == "template" and case["outcome"] == "report" and case["llm_calls"] == 0 and case["policy"]["passed"]


def test_langgraph_backend_matches_the_builtin_engine():
    try:
        import langgraph  # noqa: F401
    except ImportError:
        raise SkipTest("langgraph not installed")
    import tempfile
    tools = Tools(_Ctx(), functions=fake_functions())
    a = Orchestrator(tools, llm=ScriptedLLM(), out_dir=tempfile.mkdtemp(), backend="langgraph", save=False).run(f"Investigate TX-{TID}")
    b = orchestrator(ScriptedLLM()).run(f"Investigate TX-{TID}")
    assert (a["outcome"], a["path"], a["tool_calls"], a["llm_calls"]) == (b["outcome"], b["path"], b["tool_calls"], b["llm_calls"])


# --------------------------------------------------------------------------- LLM clients, prompts, cases, store
class _Handler(BaseHTTPRequestHandler):
    seen = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        _Handler.seen.append((self.path, {k.lower(): v for k, v in self.headers.items()}, body))
        if body["model"] == "reject-temp" and "temperature" in body:
            data = json.dumps({"error": {"message": "temperature is not supported for this model"}}).encode()
            self.send_response(400)
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        reply = {"plan": 1}
        if self.path.endswith("/v1/messages"):
            out = {"model": body["model"], "content": [{"type": "text", "text": "```json\n" + json.dumps(reply) + "\n```"}],
                   "usage": {"input_tokens": 10, "output_tokens": 5}}
        else:
            out = {"model": body["model"], "choices": [{"message": {"content": json.dumps(reply)}}], "usage": {}}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def test_http_clients_send_temperature_zero_and_parse_replies():
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1,localhost"
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        a = AnthropicLLM(model="claude-test", api_key="k", base_url=base).complete("orchestrator", "SYS", "USER")
        o = OpenAICompatibleLLM(model="local-model", base_url=base + "/v1").complete("report", "SYS", "USER")
        (p1, h1, b1), (p2, h2, b2) = _Handler.seen[-2:]
        s5 = AnthropicLLM(model="claude-sonnet-5-5", api_key="k", base_url=base).complete("report", "SYS", "USER")
        b3 = _Handler.seen[-1][2]
        r = OpenAICompatibleLLM(model="reject-temp", base_url=base + "/v1").complete("report", "SYS", "USER")
    finally:
        srv.shutdown()
    assert parse_json(a.text) == {"plan": 1} and parse_json(o.text) == {"plan": 1} and parse_json(r.text) == {"plan": 1}
    assert p1 == "/v1/messages" and b1["temperature"] == 0 and b1["system"] == "SYS" and h1["x-api-key"] == "k"
    assert p2 == "/v1/chat/completions" and b2["temperature"] == 0 and b2["messages"][0]["role"] == "system"
    assert b2["response_format"] == {"type": "json_object"}
    # Claude 5.x refuses temperature: none is sent, effort is set and max_tokens leaves room for thinking
    assert "temperature" not in b3 and b3["output_config"] == {"effort": "low"} and b3["max_tokens"] >= 16000
    assert s5.settings["output_config"] == {"effort": "low"}
    # an endpoint that refuses temperature gets the request again without it
    assert r.settings["dropped"] == ["temperature"] and "temperature" not in _Handler.seen[-1][2]


def test_prompts_are_locked_at_their_versions():
    assert lock_problems() == []


def test_scripted_cases_cover_the_design_strata():
    cases = C.load()
    assert len(cases) == 20 and len({c["id"] for c in cases}) == 20
    strata = " ".join(c["stratum"] for c in cases)
    for part in ("High", "Medium", "Low", "known entity", "new entity", "no identity", "no entity key", "out of scope",
                 "general question", "not covered", "unknown transaction"):
        assert part in strata, part


def test_case_store_reads_saves_and_records_people_decisions():
    import tempfile
    store = CaseStore(os.path.join(tempfile.mkdtemp(), "cases.sqlite"))
    store.save_draft({"case_id": "C1", "created_at": "2026-11-04 10:00:00", "transaction_id": TID, "entity_key": "k",
                      "request": "r", "band": "High", "report": {"x": 1}})
    assert [c["case_id"] for c in store.read(transaction_id=TID)["cases"]] == ["C1"]
    assert store.read(entity_key="k")["cases"][0]["decision"] is None
    store.record_decision("C1", "false_positive", "L1 analyst", "legit")
    assert store.read(transaction_id=TID)["cases"][0]["decision"] == "false_positive"
    for bad in (("C1", "fraud", "x"), ("C1", "false_positive", " "), ("C9", "false_positive", "x")):
        try:
            store.record_decision(*bad)
            raise AssertionError(bad)
        except (ValueError, KeyError):
            pass
