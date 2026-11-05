"""LLM clients behind one interface: ``complete(agent, system, user, context) -> LLMResponse``.

The LLMs only plan, interpret and write. Every score, probability, band and SHAP contribution is
computed in Python by the tools (LightGBM, the isotonic calibrator, TreeSHAP) before any LLM sees
it; the evidence policy rejects any number in an agent's text that is not in a tool output.

Select with ``FINGUARD_LLM`` (or ``--llm``); the default is ``auto``:

* ``auto``       Groq ``openai/gpt-oss-120b`` first; when Groq cannot answer (no key, offline, rate
                 limit), Ollama ``qwen2.5:3b-instruct-q4_K_M`` (Q4_K_M quantization) on this computer;
                 when neither can, the orchestrator's fixed fallback path writes a template report.
                 Every report's run record says which model wrote it. Any order works:
                 ``FINGUARD_LLM=groq,ollama``.
* ``ollama``     Ollama's native API on this computer (``OLLAMA_BASE_URL``, default
                 http://localhost:11434; model ``OLLAMA_MODEL``). It sets the context window
                 (``FINGUARD_OLLAMA_NUM_CTX``, default 8192), because Ollama's default of 4096 tokens
                 would silently cut the evidence.

* ``anthropic``  Anthropic Messages API. Key: ``ANTHROPIC_API_KEY``; model: ``FINGUARD_LLM_MODEL``
                 (default ``claude-sonnet-5-5``).
* ``openai``     any OpenAI-compatible chat-completions endpoint (``FINGUARD_LLM_BASE_URL``, default
                 OpenAI). Key: ``FINGUARD_LLM_API_KEY`` or ``OPENAI_API_KEY`` (none for a local
                 server); model: ``FINGUARD_LLM_MODEL`` (required).
* free presets   the same client with the address filled in (``PROVIDERS``): ``groq`` (free tier;
                 ``GROQ_API_KEY``, model ``GROQ_MODEL``, default openai/gpt-oss-120b), ``gemini``
                 (free tier; ``GEMINI_API_KEY``, ``GEMINI_MODEL``), ``openrouter`` (free models;
                 ``OPENROUTER_API_KEY``, ``OPENROUTER_MODEL``). ``groq`` and ``ollama`` default to the
                 lean, compact mode below.
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

Deadline. A client's ``deadline`` (a ``time.time()`` value, set by the orchestrator for each run)
caps every attempt's timeout at the time left and stops retries once it has passed, so one slow
call cannot hold a request past its time budget.

Free tiers. A 429 (rate limit) is retried after the provider's ``retry-after`` (at most
``FINGUARD_LLM_MAX_WAIT`` seconds, default 65, and never past the deadline). Two settings make an
investigation cheaper in tokens, for per-minute token limits and small local models:
``FINGUARD_PROMPT_PASSAGE_CHARS`` (default 1500: characters of each passage shown to the agents;
quotes must still be verbatim) and ``FINGUARD_AGENT_LEAN=1`` (the report agent works from the
evidence directly, without the investigation agent: two LLM calls instead of three).
``FINGUARD_LLM_REASONING_EFFORT`` (low, medium, high) is sent as ``reasoning_effort`` to models that
reason (gpt-oss on Groq defaults to low).
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

# OpenAI-compatible providers: address, key variables, model variable and default model, defaults
PROVIDERS = {
    "openai": {"base_url": "https://api.openai.com/v1", "keys": ("FINGUARD_LLM_API_KEY", "OPENAI_API_KEY"),
               "model_env": "FINGUARD_LLM_MODEL", "model": None},
    "groq": {"base_url": "https://api.groq.com/openai/v1", "keys": ("GROQ_API_KEY",), "model_env": "GROQ_MODEL",
             "model": "openai/gpt-oss-120b", "lean": True, "passage_chars": 700},
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai", "keys": ("GEMINI_API_KEY",),
               "model_env": "GEMINI_MODEL", "model": None},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "keys": ("OPENROUTER_API_KEY",),
                   "model_env": "OPENROUTER_MODEL", "model": None},
}
OLLAMA_DEFAULT_MODEL = "qwen2.5:3b-instruct-q4_K_M"           # Qwen2.5 3B instruct, Q4_K_M quantization (1.9 GB)
CHAINS = {"auto": ("groq", "ollama")}                       # primary first, then the offline fallback


def _env(name: str) -> str | None:
    """An environment variable, with empty meaning unset (Docker Compose passes empty strings)."""
    v = os.environ.get(name)
    return v if v not in (None, "") else None


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
    for key in ("temperature", "output_config", "thinking", "reasoning_effort", "response_format"):
        if key in body and (key in text or (key == "output_config" and "effort" in text)
                            or (key == "reasoning_effort" and "reasoning" in text)
                            or (key == "response_format" and ("json" in text or "format" in text))):
            return key
    return None


def _retry_after(err) -> float | None:
    try:
        v = float(err.headers.get("retry-after"))
    except (TypeError, ValueError, AttributeError):
        return None
    return max(0.0, v)


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


def _post_adapting(url: str, body: dict, headers: dict, timeout: float, deadline: float | None = None,
                   retries: int = 3, max_wait: float | None = None) -> tuple[dict, dict]:
    """POST; if the endpoint refuses a sampling or format parameter, drop it and retry once per parameter."""
    body, dropped = dict(body), []
    for _ in range(3):
        try:
            return _post(url, body, headers, timeout, retries=retries, deadline=deadline, max_wait=max_wait), {"dropped": dropped}
        except LLMError as e:
            key = _refused_param(e, body)
            if key is None:
                raise
            body.pop(key)
            dropped.append(key)
    return _post(url, body, headers, timeout, retries=retries, deadline=deadline, max_wait=max_wait), {"dropped": dropped}


def _post(url: str, body: dict, headers: dict, timeout: float, retries: int = 3, deadline: float | None = None,
          max_wait: float | None = None) -> dict:
    data = json.dumps(body).encode("utf-8")
    last = None
    for attempt in range(retries + 1):
        wait = timeout
        if deadline is not None:
            wait = min(timeout, deadline - time.time())
            if wait < 1.0:
                raise last or LLMError("time budget used up before the LLM call could start")
        req = urllib.request.Request(url, data=data, method="POST",
                                     headers={"content-type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=wait) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:600]
            hint = (" (the request is larger than the provider's per-minute token limit: lower "
                    "FINGUARD_PROMPT_PASSAGE_CHARS or set FINGUARD_AGENT_LEAN=1)" if e.code == 413 else "")
            last = LLMError(f"HTTP {e.code} from {url}: {detail}{hint}", status=e.code, detail=detail)
            if e.code not in (408, 409, 429, 500, 502, 503, 504, 529):
                raise last from e
            wait_for = _retry_after(e) if e.code in (429, 503) else None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = LLMError(f"cannot reach {url}: {e}")
            wait_for = None
        if attempt < retries:
            pause = 2.0 * (attempt + 1)
            if wait_for is not None:                              # rate limit: wait as long as the provider asks
                limit = max_wait if max_wait is not None else float(_env("FINGUARD_LLM_MAX_WAIT") or 65)
                pause = min(max(pause, wait_for + 0.5), limit)
            if deadline is not None and time.time() + pause + 1.0 >= deadline:
                break
            time.sleep(pause)
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
        self.deadline: float | None = None
        self.retries, self.max_wait = 3, None

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
                                   {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}, self.timeout,
                                   deadline=self.deadline, retries=self.retries, max_wait=self.max_wait)
        text = "".join(b.get("text", "") for b in out.get("content", []) if b.get("type") == "text")
        if out.get("stop_reason") == "max_tokens":
            raise LLMError(f"reply cut at max_tokens={self.max_tokens}; raise it or lower FINGUARD_LLM_EFFORT")
        settings = {k: body[k] for k in ("temperature", "output_config") if k in body and k not in info["dropped"]}
        return LLMResponse(text, out.get("model", self.model), time.time() - t0, out.get("usage", {}),
                           {**settings, "dropped": info["dropped"]})


class OpenAICompatibleLLM:
    name = "openai"

    def __init__(self, model: str | None = None, api_key: str | None = None, base_url: str | None = None,
                 timeout: float = 60.0, json_mode: bool | None = None, provider: str = "openai"):
        if provider not in PROVIDERS:
            raise LLMError(f"unknown provider '{provider}': use one of {', '.join(PROVIDERS)}")
        cfg = PROVIDERS[provider]
        self.name = provider
        env_url = _env("FINGUARD_LLM_BASE_URL") if provider == "openai" else _env(f"{provider.upper()}_BASE_URL")
        self.base_url = (base_url or env_url or cfg["base_url"]).rstrip("/")
        self.model = model or _env(cfg["model_env"]) or cfg["model"]
        if not self.model:
            raise LLMError(f"set {cfg['model_env']} to the model to use with {provider}")
        self.api_key = api_key or next((_env(k) for k in cfg["keys"] if _env(k)), "") or ""
        local = any(h in self.base_url for h in ("localhost", "127.0.0.1", "host.docker.internal"))
        if not self.api_key and not local:
            raise LLMError(f"{' or '.join(cfg['keys'])} is not set")
        self.timeout = timeout
        self.deadline: float | None = None
        self.retries, self.max_wait = 3, None
        self.json_mode = (_env("FINGUARD_LLM_JSON_MODE") != "0") if json_mode is None else json_mode
        lean = _env("FINGUARD_AGENT_LEAN")
        self.lean = cfg.get("lean", False) if lean is None else lean == "1"
        self.passage_chars = int(_env("FINGUARD_PROMPT_PASSAGE_CHARS") or cfg.get("passage_chars", 1500))
        effort = _env("FINGUARD_LLM_REASONING_EFFORT")
        if effort is None and provider == "groq" and self.model.startswith("openai/gpt-oss"):
            effort = "low"
        self.reasoning_effort = effort

    def request_body(self, system: str, user: str) -> dict:
        body = {"model": self.model, "temperature": 0,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        return body

    def complete(self, agent: str, system: str, user: str, context: dict | None = None) -> LLMResponse:
        t0 = time.time()
        headers = {"authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        body = self.request_body(system, user)
        out, info = _post_adapting(f"{self.base_url}/chat/completions", body, headers, self.timeout, deadline=self.deadline,
                                   retries=self.retries, max_wait=self.max_wait)
        try:
            text = out["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"unexpected response shape: {str(out)[:300]}") from e
        settings = {k: body[k] for k in ("temperature", "response_format", "reasoning_effort")
                    if k in body and k not in info["dropped"]}
        return LLMResponse(text, out.get("model", self.model), time.time() - t0, out.get("usage", {}),
                           {**settings, "dropped": info["dropped"], "lean": self.lean, "passage_chars": self.passage_chars})


def ollama_base_url(base_url: str | None = None) -> str:
    url = (base_url or _env("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
    for suffix in ("/v1", "/api"):                                # accept the OpenAI-style address too
        if url.endswith(suffix):
            url = url[: -len(suffix)]
    return url


class OllamaLLM:
    """Ollama's native chat API, on this computer (offline). JSON mode, temperature 0, and an explicit
    context window: Ollama's default of 4096 tokens would silently drop the start of the prompt."""
    name = "ollama"

    def __init__(self, model: str | None = None, base_url: str | None = None, timeout: float = 300.0,
                 num_ctx: int | None = None):
        self.model = model or _env("OLLAMA_MODEL") or OLLAMA_DEFAULT_MODEL
        self.base_url = ollama_base_url(base_url)
        self.num_ctx = int(num_ctx or _env("FINGUARD_OLLAMA_NUM_CTX") or 8192)
        self.keep_alive = _env("FINGUARD_OLLAMA_KEEP_ALIVE") or "30m"         # stay loaded between requests
        self.timeout = timeout
        self.deadline: float | None = None
        self.retries, self.max_wait = 1, None
        lean = _env("FINGUARD_AGENT_LEAN")
        self.lean = True if lean is None else lean == "1"                     # a 3B model: two calls, short passages
        self.passage_chars = int(_env("FINGUARD_PROMPT_PASSAGE_CHARS") or 600)

    def request_body(self, system: str, user: str) -> dict:
        return {"model": self.model, "stream": False, "format": "json", "keep_alive": self.keep_alive,
                "options": {"temperature": 0, "num_ctx": self.num_ctx},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}

    def complete(self, agent: str, system: str, user: str, context: dict | None = None) -> LLMResponse:
        estimate = (len(system) + len(user)) / 3.2                            # tokens, on the safe side
        if estimate > self.num_ctx - 1024:
            raise LLMError(f"the prompt (about {estimate:.0f} tokens) does not fit Ollama's context window of "
                           f"{self.num_ctx} with room for the answer: raise FINGUARD_OLLAMA_NUM_CTX or lower "
                           "FINGUARD_PROMPT_PASSAGE_CHARS")
        t0 = time.time()
        try:
            out = _post(f"{self.base_url}/api/chat", self.request_body(system, user), {}, self.timeout,
                        retries=self.retries, deadline=self.deadline, max_wait=self.max_wait)
        except LLMError as e:
            if e.status == 404:
                raise LLMError(f"Ollama has no model '{self.model}': run `ollama pull {self.model}`",
                               status=404, detail=e.detail) from e
            raise
        used = out.get("prompt_eval_count")
        if used is not None and used >= self.num_ctx - 8:
            raise LLMError(f"the prompt filled Ollama's context window ({used} of {self.num_ctx} tokens) and was cut")
        if out.get("done_reason") == "length":
            raise LLMError("Ollama stopped at its length limit: the reply is incomplete")
        text = (out.get("message") or {}).get("content") or ""
        return LLMResponse(text, out.get("model", self.model), time.time() - t0,
                           {"input_tokens": used, "output_tokens": out.get("eval_count")},
                           {"temperature": 0, "format": "json", "num_ctx": self.num_ctx, "lean": self.lean,
                            "passage_chars": self.passage_chars})


_COOLDOWN: dict[str, float] = {}       # chain member -> time until which it is skipped (it could not be reached)


class ChainLLM:
    """A primary LLM with ordered fallbacks (``auto``: Groq gpt-oss-120b, then Ollama on this computer).

    The next member answers when one fails: no key, no network, rate limit, server error, bad reply
    size. A member that cannot be reached is skipped for ``FINGUARD_LLM_COOLDOWN`` seconds (default
    60). A primary with a fallback retries once and waits at most ``FINGUARD_CHAIN_MAX_WAIT`` seconds
    (default 15) on a rate limit before handing over. ``model`` names the models that answered in the
    current run, for the run record."""
    name = "auto"

    def __init__(self, members: list, skipped: dict | None = None):
        self.members, self.skipped = list(members), dict(skipped or {})
        for m in self.members[:-1]:
            m.retries = min(getattr(m, "retries", 3), 1)
            m.max_wait = float(_env("FINGUARD_CHAIN_MAX_WAIT") or 15)
        self._deadline: float | None = None
        self.begin_run()

    def begin_run(self) -> None:
        self.served: dict[str, int] = {}
        self.failures: list[dict] = []

    @property
    def deadline(self):
        return self._deadline

    @deadline.setter
    def deadline(self, value):
        self._deadline = value
        for m in self.members:
            m.deadline = value

    @property
    def lean(self) -> bool:
        return all(getattr(m, "lean", False) for m in self.members)

    @property
    def passage_chars(self) -> int:
        return min(int(getattr(m, "passage_chars", 1500) or 1500) for m in self.members)

    @property
    def model(self) -> str:
        if self.served:
            return "; ".join(f"{who} x{n}" for who, n in self.served.items())
        return " -> ".join(f"{m.name} {m.model}" for m in self.members)

    def complete(self, agent: str, system: str, user: str, context: dict | None = None) -> LLMResponse:
        errors = []
        for m in self.members:
            key = f"{m.name}:{getattr(m, 'base_url', '')}"
            if _COOLDOWN.get(key, 0) > time.time():
                errors.append(f"{m.name}: skipped (could not be reached a moment ago)")
                continue
            try:
                r = m.complete(agent, system, user, context)
            except LLMError as e:
                errors.append(f"{m.name}: {e}"[:300])
                if str(e).startswith("cannot reach"):
                    _COOLDOWN[key] = time.time() + float(_env("FINGUARD_LLM_COOLDOWN") or 60)
                continue
            _COOLDOWN.pop(key, None)
            who = f"{m.name} {r.model}"
            self.served[who] = self.served.get(who, 0) + 1
            if errors:
                self.failures.append({"agent": agent, "errors": errors})
            r.settings = {**(r.settings or {}), "served_by": m.name, "fallback_from": errors}
            return r
        raise LLMError("no LLM could answer: " + " | ".join(errors))


def llm_status(timeout: float = 1.5) -> dict:
    """What the ``auto`` chain can use right now (for /api/health): Groq key, Ollama server and model."""
    groq_model = _env("GROQ_MODEL") or PROVIDERS["groq"]["model"]
    groq = {"model": groq_model, "status": "key set" if _env("GROQ_API_KEY") else "GROQ_API_KEY not set"}
    base, model = ollama_base_url(), _env("OLLAMA_MODEL") or OLLAMA_DEFAULT_MODEL
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f"{base}/api/tags", timeout=timeout) as r:
            names = {m.get("name") for m in json.loads(r.read().decode("utf-8")).get("models", [])}
        pulled = model in names or f"{model}:latest" in names
        status = "ready" if pulled else f"running, but the model is not pulled: ollama pull {model}"
    except (urllib.error.URLError, OSError, ValueError):
        status = f"not reachable at {base} (start Ollama for the offline fallback)"
    return {"groq": groq, "ollama": {"model": model, "url": base, "status": status}}


def make_llm(name: str | None = None):
    """The configured client, or None for the template path (``none``)."""
    name = (name or _env("FINGUARD_LLM") or "auto").lower().replace(" ", "")
    if name == "none":
        return None
    if name in CHAINS or "," in name:
        members, skipped = [], {}
        for part in CHAINS.get(name) or tuple(p for p in name.split(",") if p):
            try:
                m = make_llm(part)
            except (LLMError, ValueError) as e:                   # e.g. no Groq key: the chain starts at Ollama
                skipped[part] = str(e)
                continue
            if m is not None:
                members.append(m)
        if not members:
            raise LLMError("no LLM of the chain is configured: " + "; ".join(f"{k}: {v}" for k, v in skipped.items()))
        chain = ChainLLM(members, skipped)
        chain.name = name if name in CHAINS else "chain"
        return chain
    if name == "anthropic":
        return AnthropicLLM()
    if name in ("openai", "openai-compatible"):
        return OpenAICompatibleLLM()
    if name == "ollama":
        return OllamaLLM()
    if name in PROVIDERS:
        return OpenAICompatibleLLM(provider=name)
    if name in ("scripted", "mock"):
        from finguard.agents.scripted import ScriptedLLM
        return ScriptedLLM()
    raise ValueError(f"unknown LLM '{name}': use auto, none, scripted, ollama, anthropic, {', '.join(PROVIDERS)}")
