"""LLM clients behind one interface: ``complete(agent, system, user, context) -> LLMResponse``.

Select with ``FINGUARD_LLM`` (or ``--llm``):

* ``anthropic``  Anthropic Messages API. Key: ``ANTHROPIC_API_KEY``; model: ``FINGUARD_LLM_MODEL``
                 (default ``claude-sonnet-5-5``).
* ``openai``     any OpenAI-compatible chat-completions endpoint: OpenAI, Groq, Mistral, a local
                 Ollama (``FINGUARD_LLM_BASE_URL=http://localhost:11434/v1``) ... Key:
                 ``FINGUARD_LLM_API_KEY`` or ``OPENAI_API_KEY`` (none for a local server); model:
                 ``FINGUARD_LLM_MODEL`` (required).
* ``scripted``   a deterministic stand-in that follows the same contract. It is used by the
                 regression tests and CI, and to run the whole pipeline without an API key.
                 Reports it writes are labelled as such; it is not an evaluation of an LLM.
* ``none``       no LLM: every request goes straight to the fixed fallback path (template report).

Sampling. Project Report v3.0 asks for temperature 0. Claude 5.x models (Sonnet 5.5, Opus 5.5 ...)
refuse ``temperature`` with a 400 error and think adaptively instead, so for them the client sends no
temperature and sets ``output_config.effort`` (``FINGUARD_LLM_EFFORT``, default ``low``: fast, little
thinking) with a ``max_tokens`` large enough for thinking plus the answer. For other models it sends
temperature 0, and if an endpoint refuses a parameter (temperature, effort, JSON mode) the client
drops it and retries once. What was actually sent is recorded in every trace. Determinism does not
rest on sampling anyway: plans are checked against the required tools and every report against the
evidence policy, in code.

The HTTP clients use only the standard library (``urllib``), so no SDK has to be installed;
``HTTPS_PROXY`` is honoured.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

DEFAULT_MODELS = {"anthropic": "claude-sonnet-5-5"}


@dataclass
class LLMResponse:
    text: str
    model: str
    seconds: float
    usage: dict = field(default_factory=dict)
    settings: dict = field(default_factory=dict)


class LLMError(Exception):
    """The LLM could not be reached or returned an unusable answer."""

    def __init__(self, message: str, status: int | None = None, detail: str = ""):
        super().__init__(message)
        self.status, self.detail = status, detail


ADAPTIVE = re.compile(r"^claude-(sonnet|opus|haiku|fable|mythos)-5", re.I)     # Claude 5.x: no temperature, effort instead


def _refused_param(err: LLMError, body: dict) -> str | None:
    """The request parameter a 400 error complains about, if the body has it."""
    if err.status != 400:
        return None
    text = err.detail.lower()
    for key in ("temperature", "output_config", "thinking", "response_format"):
        if key in body and (key in text or (key == "output_config" and "effort" in text)
                            or (key == "response_format" and ("json" in text or "format" in text))):
            return key
    return None


def parse_json(text: str) -> dict:
    """The first JSON object in a reply (code fences and surrounding prose are tolerated)."""
    text = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fenced:
        text = fenced.group(1)
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in the reply")
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                obj = json.loads(text[start:i + 1])
                if not isinstance(obj, dict):
                    raise ValueError("the reply is not a JSON object")
                return obj
    raise ValueError("unterminated JSON object in the reply")


def _post_adapting(url: str, body: dict, headers: dict, timeout: float) -> tuple[dict, dict]:
    """POST; if the endpoint refuses a sampling or format parameter, drop it and retry once per parameter."""
    body, dropped = dict(body), []
    for _ in range(3):
        try:
            return _post(url, body, headers, timeout), {"dropped": dropped}
        except LLMError as e:
            key = _refused_param(e, body)
            if key is None:
                raise
            body.pop(key)
            dropped.append(key)
    return _post(url, body, headers, timeout), {"dropped": dropped}


def _post(url: str, body: dict, headers: dict, timeout: float, retries: int = 2) -> dict:
    data = json.dumps(body).encode("utf-8")
    last = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, method="POST",
                                     headers={"content-type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:600]
            last = LLMError(f"HTTP {e.code} from {url}: {detail}", status=e.code, detail=detail)
            if e.code not in (408, 409, 429, 500, 502, 503, 504, 529):
                raise last from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = LLMError(f"cannot reach {url}: {e}")
        if attempt < retries:
            time.sleep(2.0 * (attempt + 1))
    raise last


class AnthropicLLM:
    name = "anthropic"

    def __init__(self, model: str | None = None, api_key: str | None = None, timeout: float = 60.0,
                 max_tokens: int | None = None, base_url: str = "https://api.anthropic.com", effort: str | None = None):
        self.model = model or os.environ.get("FINGUARD_LLM_MODEL") or DEFAULT_MODELS["anthropic"]
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set")
        self.adaptive = bool(ADAPTIVE.match(self.model))
        self.effort = effort or os.environ.get("FINGUARD_LLM_EFFORT", "low")
        self.max_tokens = max_tokens or (16000 if self.adaptive else 4000)
        self.timeout, self.base_url = timeout, base_url.rstrip("/")

    def request_body(self, system: str, user: str) -> dict:
        body = {"model": self.model, "max_tokens": self.max_tokens, "system": system,
                "messages": [{"role": "user", "content": user}]}
        if self.adaptive:
            body["output_config"] = {"effort": self.effort}
        else:
            body["temperature"] = 0
        return body

    def complete(self, agent: str, system: str, user: str, context: dict | None = None) -> LLMResponse:
        t0 = time.time()
        body = self.request_body(system, user)
        out, info = _post_adapting(f"{self.base_url}/v1/messages", body,
                                   {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}, self.timeout)
        text = "".join(b.get("text", "") for b in out.get("content", []) if b.get("type") == "text")
        if out.get("stop_reason") == "max_tokens":
            raise LLMError(f"reply cut at max_tokens={self.max_tokens}; raise it or lower FINGUARD_LLM_EFFORT")
        settings = {k: body[k] for k in ("temperature", "output_config") if k in body and k not in info["dropped"]}
        return LLMResponse(text, out.get("model", self.model), time.time() - t0, out.get("usage", {}),
                           {**settings, "dropped": info["dropped"]})


class OpenAICompatibleLLM:
    name = "openai"

    def __init__(self, model: str | None = None, api_key: str | None = None, base_url: str | None = None,
                 timeout: float = 60.0, json_mode: bool | None = None):
        self.base_url = (base_url or os.environ.get("FINGUARD_LLM_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.model = model or os.environ.get("FINGUARD_LLM_MODEL")
        if not self.model:
            raise LLMError("set FINGUARD_LLM_MODEL to the model name of the OpenAI-compatible endpoint")
        self.api_key = api_key or os.environ.get("FINGUARD_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
        local = "localhost" in self.base_url or "127.0.0.1" in self.base_url
        if not self.api_key and not local:
            raise LLMError("FINGUARD_LLM_API_KEY (or OPENAI_API_KEY) is not set")
        self.timeout = timeout
        self.json_mode = (os.environ.get("FINGUARD_LLM_JSON_MODE", "1") != "0") if json_mode is None else json_mode

    def request_body(self, system: str, user: str) -> dict:
        body = {"model": self.model, "temperature": 0,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
        return body

    def complete(self, agent: str, system: str, user: str, context: dict | None = None) -> LLMResponse:
        t0 = time.time()
        headers = {"authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        body = self.request_body(system, user)
        out, info = _post_adapting(f"{self.base_url}/chat/completions", body, headers, self.timeout)
        try:
            text = out["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"unexpected response shape: {str(out)[:300]}") from e
        settings = {k: body[k] for k in ("temperature", "response_format") if k in body and k not in info["dropped"]}
        return LLMResponse(text, out.get("model", self.model), time.time() - t0, out.get("usage", {}),
                           {**settings, "dropped": info["dropped"]})


def make_llm(name: str | None = None):
    """The configured client, or None for the template path (``none``)."""
    name = (name or os.environ.get("FINGUARD_LLM") or "none").lower()
    if name == "none":
        return None
    if name == "anthropic":
        return AnthropicLLM()
    if name in ("openai", "openai-compatible", "ollama", "groq"):
        return OpenAICompatibleLLM()
    if name in ("scripted", "mock"):
        from finguard.agents.scripted import ScriptedLLM
        return ScriptedLLM()
    raise ValueError(f"unknown LLM '{name}': use anthropic, openai, scripted or none")
