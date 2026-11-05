"""Phase 5: the API contract, the HTTP layer and the alert-to-report path.

Most tests use the synthetic demo world and the scripted test model, so they run in CI without
the Kaggle data. The HTTP tests run on FastAPI when it is installed (Starlette otherwise) and skip
without ``httpx``/``uvicorn``; the last test runs the real world and skips without the data.
"""
import json
import os
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest import SkipTest

from finguard import config
from finguard.api.service import ROUTES, Service
from finguard.api.world import SyntheticWorld

_SERVICE = None


def service() -> Service:
    """One loaded synthetic service for the module (building its small index takes about a second)."""
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = Service(SyntheticWorld(Path(tempfile.mkdtemp())), default_llm="scripted").start(background=False)
        assert _SERVICE.ready, _SERVICE.error
    return _SERVICE


def call(svc, method, path, query=None, body=None):
    q = list((query or {}).items())
    return svc.dispatch(method, path, [(k, str(v)) for k, v in q], body)


# --------------------------------------------------------------------------- contract
def test_every_route_documents_its_parameters():
    names = set()
    for r in ROUTES:
        assert r.name not in names and r.summary
        names.add(r.name)
        for p in r.pattern.groupindex:
            assert p in r.path_params, (r.path, p)
        spec = r.openapi()
        assert all(p["schema"] for p in spec.get("parameters", []))
        assert (r.body is None) == ("requestBody" not in spec)


def test_starting_service_answers_503_and_still_carries_versions():
    svc = Service(SyntheticWorld(Path(tempfile.mkdtemp())), default_llm="scripted")   # not started
    r = call(svc, "GET", "/api/health")
    assert r.status == 503 and r.body["status"] == "starting" and r.body["versions"]["prompts"]
    r = call(svc, "GET", "/api/alerts")
    assert r.status == 503 and r.body["error"] == "starting" and "X-FinGuard-Prompts" in r.headers


def test_every_reply_carries_model_index_and_prompt_versions():
    svc = service()
    for method, path, q, b in [("GET", "/api/health", None, None), ("GET", "/api/alerts", {"limit": 2}, None),
                               ("GET", "/api/transactions/9000001", None, None), ("GET", "/api/nope", None, None),
                               ("POST", "/api/chat", None, {"message": "How is the proxy entity formed?"})]:
        r = call(svc, method, path, q, b)
        assert {"X-FinGuard-Model", "X-FinGuard-Index", "X-FinGuard-Prompts", "X-FinGuard-World"} <= set(r.headers), path
        assert r.body["versions"]["index_version"] and r.body["versions"]["prompts"], path


# --------------------------------------------------------------------------- endpoints
def test_alert_queue_filters_sorts_and_pages():
    svc = service()
    r = call(svc, "GET", "/api/alerts")
    assert r.status == 200 and r.body["total"] == 18 and r.body["band_counts"] == {"High": 6, "Medium": 12, "Low": 30}
    scores = [i["score"] for i in r.body["items"]]
    assert scores == sorted(scores, reverse=True) and {i["band"] for i in r.body["items"]} == {"High", "Medium"}
    r = call(svc, "GET", "/api/alerts", {"band": "High", "limit": 4, "offset": 4})
    assert r.body["total"] == 6 and len(r.body["items"]) == 2
    r = call(svc, "GET", "/api/alerts", {"band": "all", "sort": "time", "day_from": 160, "day_to": 170})
    days = [i["relative_day"] for i in r.body["items"]]
    assert days and all(160 <= d <= 170 for d in days) and days == sorted(days)
    r = call(svc, "GET", "/api/alerts", {"band": "all", "q": "900000"})
    assert all(str(i["transaction_id"]).startswith("900000") for i in r.body["items"])


def test_case_view_and_unknown_or_invalid_transactions():
    svc = service()
    r = call(svc, "GET", "/api/transactions/9000002")
    assert r.status == 200 and r.body["fraud_score"]["band"] == "High" and len(r.body["explain_transaction"]["reasons"]) == 5
    assert r.body["entity_history"]["entity"].startswith("proxy entity") and isinstance(r.body["cases"], list)
    assert call(svc, "GET", "/api/transactions/9999999").status == 404
    assert call(svc, "GET", "/api/transactions/123").status == 400
    assert call(svc, "GET", "/api/transactions/abc").status == 400


def test_investigation_writes_a_cited_report_that_an_analyst_can_decide():
    svc = service()
    r = call(svc, "POST", "/api/investigations", body={"transaction_id": 9000004})
    assert r.status == 200, r.body
    case = r.body
    assert case["outcome"] == "report" and case["path"] == "orchestrated" and case["policy"]["passed"]
    assert case["policy"]["cited"] == case["policy"]["claims"] and case["policy"]["traceable"] == case["policy"]["numbers"]
    assert "Flagged for review, not a finding of fraud." in case["markdown"]
    cid = case["case_id"]
    full = call(svc, "GET", f"/api/cases/{cid}")
    assert full.status == 200 and full.body["case"]["case_id"] == cid and full.body["decisions"] == []
    md = call(svc, "GET", f"/api/cases/{cid}/report.md")
    assert md.status == 200 and md.media_type == "text/markdown" and "attachment" in md.headers["Content-Disposition"]
    assert md.body.startswith("# FinGuard case report")
    queue = {i["transaction_id"]: i for i in call(svc, "GET", "/api/alerts", {"band": "High"}).body["items"]}
    assert queue[9000004]["case"]["status"] == "draft"
    bad = call(svc, "POST", f"/api/cases/{cid}/decision", body={"decision": "fraud", "analyst": "A1"})
    assert bad.status == 400
    assert call(svc, "POST", f"/api/cases/{cid}/decision", body={"decision": "false_positive", "analyst": " "}).status == 400
    ok = call(svc, "POST", f"/api/cases/{cid}/decision", body={"decision": "false_positive", "analyst": "Analyst A1", "note": "known pattern"})
    assert ok.status == 200 and ok.body["decision"] == "false_positive"
    assert call(svc, "GET", f"/api/cases/{cid}").body["decisions"][0]["analyst"] == "Analyst A1"
    queue = {i["transaction_id"]: i for i in call(svc, "GET", "/api/alerts", {"band": "High", "status": "decided"}).body["items"]}
    assert queue[9000004]["case"]["decision"] == "false_positive"
    listed = call(svc, "GET", "/api/cases", {"transaction_id": 9000004})
    assert listed.body["total"] >= 1 and listed.body["cases"][0]["decision"] == "false_positive"


def test_chat_answers_from_the_knowledge_base_and_declines_out_of_scope_requests():
    svc = service()
    r = call(svc, "POST", "/api/chat", body={"message": "How is the proxy entity formed?"})
    assert r.status == 200 and r.body["request_type"] == "typology" and r.body["policy"]["passed"]
    r = call(svc, "POST", "/api/chat", body={"message": "Block the card used in this payment.", "transaction_id": 9000001})
    assert r.status == 200 and r.body["outcome"] == "refusal" and r.body["llm_calls"] == 0
    r = call(svc, "POST", "/api/chat", body={"message": "Why is it high risk?", "transaction_id": 9000001})
    assert r.body["transaction_id"] == 9000001 and r.body["request_type"] == "why_high_risk"


def test_requests_are_validated_against_the_contract():
    svc = service()
    assert call(svc, "GET", "/api/alerts", {"bnad": "High"}).body["error"] == "unknown_parameter"
    assert call(svc, "GET", "/api/alerts", {"limit": 0}).status == 400
    assert call(svc, "GET", "/api/alerts", {"band": "Critical"}).status == 400
    assert call(svc, "POST", "/api/investigations", body=b"{not json").body["error"] == "invalid_json"
    assert call(svc, "POST", "/api/investigations", body={}).body["error"] == "invalid_body"
    assert call(svc, "POST", "/api/investigations", body={"transaction_id": 9000001, "extra": 1}).status == 400
    r = call(svc, "POST", "/api/investigations", body={"transaction_id": 9000001, "llm": "anthropic"})
    assert r.status == 400 and r.body["error"] == "llm_not_available"
    assert call(svc, "GET", "/api/cases/..%2F..%2Fetc").status == 404
    assert call(svc, "GET", "/api/cases/no_such_case").status == 404
    assert call(svc, "DELETE", "/api/cases").status == 405


def test_concurrency_limit_answers_429():
    svc = Service(SyntheticWorld(Path(tempfile.mkdtemp())), default_llm="scripted", max_concurrent=1)
    svc.world, svc.state = service().world, "ready"
    assert svc.slots.acquire(blocking=False)
    try:
        r = call(svc, "POST", "/api/investigations", body={"transaction_id": 9000001})
        assert r.status == 429 and r.body["error"] == "busy"
    finally:
        svc.slots.release()


def test_time_budget_of_the_api_routes_to_the_fallback_path():
    svc = Service(SyntheticWorld(Path(tempfile.mkdtemp())), default_llm="scripted", time_budget=1e-9)
    svc.world, svc.state = service().world, "ready"
    r = call(svc, "POST", "/api/investigations", body={"transaction_id": 9000002})
    assert r.status == 200 and r.body["path"] == "fallback" and r.body["policy"]["passed"]


def test_llm_calls_stop_at_the_deadline():
    from finguard.agents.llm import LLMError, OpenAICompatibleLLM

    class Slow(BaseHTTPRequestHandler):
        def do_POST(self):
            time.sleep(4)
            self.send_response(200)
            self.end_headers()

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Slow)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        llm = OpenAICompatibleLLM(model="m", base_url=f"http://127.0.0.1:{srv.server_port}/v1")
        llm.deadline = time.time() + 1.5
        t0 = time.time()
        try:
            llm.complete("report", "s", "u")
            raise AssertionError("the call should have stopped at the deadline")
        except LLMError:
            assert time.time() - t0 < 3.5
        llm.deadline = time.time() - 1
        try:
            llm.complete("report", "s", "u")
            raise AssertionError("no call may start past the deadline")
        except LLMError as e:
            assert "time budget" in str(e)
    finally:
        srv.shutdown()


# --------------------------------------------------------------------------- HTTP layer
def _client(svc):
    try:
        from starlette.testclient import TestClient
    except Exception as e:                                       # httpx missing
        raise SkipTest(f"starlette test client unavailable: {e}")
    from finguard.api.app import create_app
    return TestClient(create_app(svc, start=False))


def test_http_layer_serves_the_contract_and_the_dashboard():
    svc = service()
    with _client(svc) as c:
        r = c.get("/")
        assert r.status_code == 200 and "FinGuard AI" in r.text and r.headers["x-finguard-world"] == "synthetic"
        r = c.get("/api/alerts", params={"band": "High", "limit": 3})
        assert r.status_code == 200 and len(r.json()["items"]) == 3 and r.headers["x-finguard-index"]
        r = c.post("/api/investigations", json={"transaction_id": 9000003})
        assert r.status_code == 200 and r.json()["policy"]["passed"]
        r = c.get(r.json()["report_url"])
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/markdown")
        assert c.post("/api/investigations", content=b"{", headers={"content-type": "application/json"}).status_code == 400
        r = c.delete("/api/alerts")
        assert r.status_code == 405 and r.json()["error"] == "method_not_allowed" and r.headers["x-finguard-prompts"]
        assert c.get("/api/unknown").status_code == 404
        from finguard.api.app import framework
        if framework() == "fastapi":
            spec = c.get("/openapi.json").json()
            ops = {op["operationId"]: op for p in spec["paths"].values() for op in p.values()}
            assert {r.name for r in ROUTES} <= set(ops)
            assert any(p["name"] == "band" for p in ops["list_alerts"]["parameters"])
            assert "requestBody" in ops["investigate"]


def test_alert_to_report_demo_on_a_live_server_is_under_60_seconds():
    try:
        import uvicorn
    except ImportError:
        raise SkipTest("uvicorn not installed")
    from finguard.api.app import create_app
    from finguard.api.client import demo
    svc = service()
    server = uvicorn.Server(uvicorn.Config(create_app(svc, start=False), host="127.0.0.1", port=0, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.1)
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        out = Path(tempfile.mkdtemp()) / "demo.json"
        res = demo(f"http://127.0.0.1:{port}", llm="scripted", out=out, log=lambda *_: None)
        assert res["passed"] and res["alert_to_report_seconds"] < 60 and json.loads(out.read_text())["passed"]
        assert {"x-finguard-model", "x-finguard-index", "x-finguard-prompts"} <= {k.lower() for k in res["version_headers"]}
    finally:
        server.should_exit = True
        thread.join(10)


# --------------------------------------------------------------------------- real world
def test_real_world_queue_matches_the_scorer_and_reports_pass_the_policy():
    needed = [config.INTERIM_DIR / "train.pkl", config.ROOT / "artifacts" / "phase2" / "model_bundle.pkl",
              config.ROOT / "data" / "corpus" / "index" / "abstention.json"]
    if not all(Path(p).exists() or Path(str(p).replace(".pkl", ".parquet")).exists() for p in needed):
        raise SkipTest("data, Phase 2 artifacts or knowledge-base index not present")
    from finguard.api.world import RealWorld
    svc = Service(RealWorld(Path(tempfile.mkdtemp())), default_llm="scripted", time_budget=1e9).start(background=False)
    assert svc.ready, svc.error
    q = call(svc, "GET", "/api/alerts", {"band": "High", "limit": 5}).body
    assert q["total"] > 0 and q["band_counts"]["High"] > 0
    for item in q["items"][:3]:
        t = call(svc, "GET", f"/api/transactions/{item['transaction_id']}").body
        assert abs(t["fraud_score"]["score"] - item["score"]) < 0.05 and t["fraud_score"]["band"] == item["band"]
    r = call(svc, "POST", "/api/investigations", body={"transaction_id": q["items"][0]["transaction_id"]})
    assert r.status == 200 and r.body["outcome"] == "report" and r.body["policy"]["passed"], r.body
    h = call(svc, "GET", "/api/health").body
    assert h["components"]["alert_queue"] == q["band_counts"]["High"] + q["band_counts"]["Medium"]
    assert os.path.exists(Path(svc.world.case_dir) / "reports" / f"{r.body['case_id']}.md")
