"""Standard-library HTTP client for the running API: the health check used by Docker and the
alert-to-report demo that measures the Phase 5 exit criterion (under 60 seconds).

The demo does what an analyst does in the dashboard: open the alert queue, open the top alert's
case view, run the investigation, open the report. It times each step and the whole path.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

LOCAL = ("localhost", "127.0.0.1", "0.0.0.0", "host.docker.internal", "api")


class ApiClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000", timeout: float = 180.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        host = urllib.parse.urlparse(self.base_url).hostname or ""
        handlers = [urllib.request.ProxyHandler({})] if host in LOCAL else []   # never send local calls to a proxy
        self.opener = urllib.request.build_opener(*handlers)

    def request(self, method: str, path: str, body: dict | None = None, query: dict | None = None):
        url = self.base_url + path + (f"?{urllib.parse.urlencode(query)}" if query else "")
        data = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, method=method, headers={"content-type": "application/json"})
        try:
            with self.opener.open(req, timeout=self.timeout) as r:
                raw, status, headers = r.read(), r.status, dict(r.headers)
        except urllib.error.HTTPError as e:
            raw, status, headers = e.read(), e.code, dict(e.headers)
        ctype = headers.get("content-type", headers.get("Content-Type", ""))
        payload = json.loads(raw.decode("utf-8")) if "json" in ctype else raw.decode("utf-8")
        return status, payload, headers

    def health(self) -> tuple[int, dict]:
        try:
            status, body, _ = self.request("GET", "/api/health")
            return status, body if isinstance(body, dict) else {}
        except (urllib.error.URLError, OSError, ValueError) as e:
            return 0, {"status": "unreachable", "error": str(e)}

    def wait_ready(self, wait: float = 600.0, every: float = 3.0, log=print) -> dict:
        t0, last = time.time(), None
        while True:
            status, body = self.health()
            if status == 200 and body.get("ready"):
                return body
            if body.get("status") == "error":
                raise RuntimeError(f"the API failed to load: {body.get('error')}")
            if body.get("status") != last:
                log(f"  waiting for the API: {body.get('status')} {body.get('error') or ''}".rstrip())
                last = body.get("status")
            if time.time() - t0 > wait:
                raise TimeoutError(f"the API was not ready after {wait:.0f} s (last status: {body.get('status')})")
            time.sleep(every)


def healthcheck(base_url: str, ready: bool = True) -> int:
    status, body = ApiClient(base_url, timeout=5).health()
    ok = status == 200 and body.get("ready") if ready else status in (200, 503) and body.get("status") != "error"
    print(json.dumps({"status": body.get("status"), "ready": body.get("ready"), "world": body.get("world")}))
    return 0 if ok else 1


def demo(base_url: str, llm: str | None = None, transaction_id: int | None = None, max_seconds: float = 60.0,
         wait: float = 900.0, out: Path | None = None, log=print) -> dict:
    """Alert queue -> case view -> investigation -> report, timed. Exit criterion: under ``max_seconds``."""
    c = ApiClient(base_url)
    health = c.wait_ready(wait, log=log)
    steps = []

    def step(name, method, path, body=None, query=None, expect=200):
        t = time.time()
        status, payload, headers = c.request(method, path, body, query)
        steps.append({"step": name, "status": status, "seconds": round(time.time() - t, 3)})
        log(f"  {name:<14} HTTP {status}  {steps[-1]['seconds']:.2f} s")
        if status != expect:
            raise RuntimeError(f"{name}: HTTP {status}: {payload}")
        return payload, headers

    t0 = time.time()
    if transaction_id is None:
        queue, _ = step("alert queue", "GET", "/api/alerts", query={"band": "High", "limit": 1, "sort": "score"})
        if not queue["items"]:
            raise RuntimeError("the alert queue is empty")
        transaction_id = queue["items"][0]["transaction_id"]
    step("case view", "GET", f"/api/transactions/{transaction_id}")
    body = {"transaction_id": transaction_id, **({"llm": llm} if llm else {})}
    case, _ = step("investigation", "POST", "/api/investigations", body=body)
    report, headers = step("report", "GET", case["report_url"])
    total = round(time.time() - t0, 2)
    versions = case.get("versions", {})
    result = {"run_at": time.strftime("%Y-%m-%d %H:%M:%S"), "base_url": base_url, "world": health.get("world"),
              "transaction_id": transaction_id, "case_id": case["case_id"], "llm": case.get("llm"),
              "path": case.get("path"), "fallback_reason": case.get("fallback_reason"), "outcome": case.get("outcome"),
              "policy": case.get("policy"), "investigation_seconds": case.get("seconds"),
              "tool_calls": case.get("tool_calls"), "llm_calls": case.get("llm_calls"),
              "alert_to_report_seconds": total, "max_seconds": max_seconds, "steps": steps,
              "versions": versions, "version_headers": {k: v for k, v in headers.items() if k.lower().startswith("x-finguard")},
              "report_chars": len(report) if isinstance(report, str) else None}
    result["passed"] = bool(total < max_seconds and case.get("outcome") == "report"
                            and (case.get("policy") or {}).get("passed") and result["version_headers"])
    log(f"  alert-to-report {total:.2f} s (limit {max_seconds:g} s), path {result['path']}, "
        f"policy {'passed' if (case.get('policy') or {}).get('passed') else 'FAILED'}: {'PASS' if result['passed'] else 'FAIL'}")
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
        log(f"  written: {out}")
    return result
