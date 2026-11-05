"""The evidence gathered for one request: every tool result gets an ID (T1, T2, ...) and every
retrieved passage a global ID (S1, S2, ... across all searches; a passage found again by another
search keeps its first ID). Agents may cite only these IDs.

The evidence is kept as a plain list of dicts so it can live in the graph state (LangGraph
checkpoints it) and in the trace; ``Evidence`` is a thin view over that list.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

NUMBER = re.compile(r"(?<![\w.])[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?%?|(?<![\w.])[-+]?\d+(?:\.\d+)?%?")


@dataclass
class Evidence:
    items: list[dict] = field(default_factory=list)

    # ------------------------------------------------------------------ building
    def add(self, tool: str, args: dict, output: dict | None = None, error: str | None = None,
            retryable: bool = True, seconds: float = 0.0) -> dict:
        item = {"id": f"T{sum(1 for i in self.items) + 1}", "tool": tool, "args": args, "ok": error is None,
                "output": output, "error": error, "retryable": retryable if error else None, "seconds": round(seconds, 3)}
        if error is None and tool == "policy_search":
            seen = self.passages()
            known = {p.get("chunk_id"): sid for sid, p in seen.items() if p.get("chunk_id")}
            n, ids = len(seen), []
            for p in output.get("passages", []):
                cid = p.get("chunk_id")
                if cid and cid in known:                          # the same passage from another search keeps its ID
                    p["label"] = known[cid]
                else:
                    n += 1
                    p["label"] = f"S{n}"
                    if cid:
                        known[cid] = p["label"]
                ids.append(p["label"])
            item["passage_ids"] = ids
        self.items.append(item)
        return item

    # ------------------------------------------------------------------ reading
    def succeeded(self, tool: str) -> bool:
        return any(i["ok"] and i["tool"] == tool for i in self.items)

    def latest(self, tool: str) -> dict | None:
        for i in reversed(self.items):
            if i["ok"] and i["tool"] == tool:
                return i
        return None

    def output(self, tool: str) -> dict | None:
        i = self.latest(tool)
        return None if i is None else i["output"]

    def calls(self) -> int:
        return len(self.items)

    def passages(self) -> dict[str, dict]:
        out = {}
        for i in self.items:
            if i["ok"] and i["tool"] == "policy_search":
                for p in i["output"].get("passages", []):
                    out[p["label"]] = p
        return out

    def ids(self) -> set[str]:
        return {i["id"] for i in self.items if i["ok"]} | set(self.passages())

    def fatal_error(self) -> dict | None:
        """A non-retryable error (unknown transaction): re-planning cannot fix it."""
        return next((i for i in self.items if not i["ok"] and not i["retryable"]), None)

    def numbers(self) -> list[str]:
        """Every number that appears in a tool output or a passage (as text), for the traceability check."""
        out = []
        for i in self.items:
            if i["ok"]:
                out.extend(NUMBER.findall(json.dumps(i["output"], ensure_ascii=False)))
        for p in self.passages().values():
            out.extend(NUMBER.findall(p.get("text", "")))
        return out

    # ------------------------------------------------------------------ for the agents
    def prompt_text(self, max_passage_chars: int = 1500) -> str:
        """Compact evidence pack: one block per tool result and per passage, each with its ID."""
        lines = []
        for i in self.items:
            if not i["ok"]:
                lines.append(f"[{i['id']}] {i['tool']} FAILED: {i['error']}")
                continue
            out = dict(i["output"])
            if i["tool"] == "policy_search":
                out = {k: v for k, v in out.items() if k in ("status", "message", "ungrounded_names", "query")}
                out["passages"] = i.get("passage_ids", [])
            lines.append(f"[{i['id']}] {i['tool']} {json.dumps(out, ensure_ascii=False, default=str)}")
        for sid, p in self.passages().items():
            head = (f"[{sid}] {p.get('title')} ({p.get('publisher')}, {p.get('published')}; {p.get('jurisdiction')}; "
                    f"tier {p.get('tier')}) | section: {p.get('section')}"
                    + (f" (also covers: {'; '.join(p['subsections'])})" if p.get("subsections") else "")
                    + f" | pages {p.get('pages')}" + (" | FICTIONAL document" if p.get("fictional") else ""))
            text = p.get("text", "")
            if len(text) > max_passage_chars:
                text = text[:max_passage_chars] + " [...]"
            lines.append(f"{head}\n\"\"\"{text}\"\"\"")
        return "\n".join(lines)
