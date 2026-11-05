"""Versioned prompts (Project Report v3.0, section 9: prompt versioning).

Each prompt is a Markdown file with a header (``id``, ``version``). ``prompts.lock.json`` records
the SHA-256 of every version. A prompt whose text changed while its version did not is refused
by the agent regression test, so every behaviour change is a new, traceable version. The trace
of every LLM call records ``id@version`` and the hash.

    python -m finguard.agents prompts           # show versions and lock status
    python -m finguard.agents prompts --lock    # record the current versions after a deliberate change
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

PROMPT_DIR = Path(__file__).parent / "prompts"
LOCK = PROMPT_DIR / "prompts.lock.json"
NAMES = ("orchestrator", "investigation", "report")


@dataclass(frozen=True)
class Prompt:
    id: str
    version: str
    text: str
    sha256: str

    @property
    def tag(self) -> str:
        return f"{self.id}@{self.version}"


def load(name: str) -> Prompt:
    raw = (PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8").replace("\r\n", "\n")
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", raw, re.S)
    if not m:
        raise ValueError(f"prompt {name}: missing the --- header ---")
    head = dict(line.split(":", 1) for line in m.group(1).splitlines() if ":" in line)
    head = {k.strip(): v.strip() for k, v in head.items()}
    if head.get("id") != name or not head.get("version"):
        raise ValueError(f"prompt {name}: header needs id: {name} and a version")
    return Prompt(name, head["version"], m.group(2).strip(), hashlib.sha256(raw.encode("utf-8")).hexdigest())


def load_all() -> dict[str, Prompt]:
    return {n: load(n) for n in NAMES}


def lock_problems() -> list[str]:
    lock = json.loads(LOCK.read_text(encoding="utf-8")) if LOCK.exists() else {}
    problems = []
    for name, p in load_all().items():
        rec = lock.get(name, {}).get(p.version)
        if rec is None:
            problems.append(f"{p.tag} is not in the lock file: run `python -m finguard.agents prompts --lock`")
        elif rec != p.sha256:
            problems.append(f"{p.tag} changed after it was locked: bump its version, then lock it")
    return problems


def write_lock() -> dict:
    lock = json.loads(LOCK.read_text(encoding="utf-8")) if LOCK.exists() else {}
    for name, p in load_all().items():
        versions = lock.setdefault(name, {})
        if p.version in versions and versions[p.version] != p.sha256:
            raise ValueError(f"{p.tag} is locked with other text: bump its version first")
        versions[p.version] = p.sha256
    LOCK.write_text(json.dumps(lock, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return lock
