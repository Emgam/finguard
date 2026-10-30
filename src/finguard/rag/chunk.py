"""Heading-based chunking (Project Report v3.0, section 7.2).

* Split on headings: a chunk never crosses a top-level (level 1) section.
* 300-500 tokens per chunk (target 400, approximate LLM tokens). A sub-section shorter than
  300 tokens is not emitted alone: the following sub-sections join it, their headings kept
  inline ("Heading. text ..."), until the chunk reaches 300 tokens; a short tail is merged into
  the chunk before it.
* 12% overlap between consecutive chunks of the same section only.
* Tables of at least 120 tokens are chunks of their own (split by rows if longer than 500
  tokens); smaller tables (often a header fragment found by the PDF table detector) are kept
  in the surrounding text.
* The section path is stored with each chunk and prepended to the text that is indexed.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from finguard.rag.parse import Block
from finguard.rag.text import approx_tokens, join_hyphenation

TARGET, MAX_TOKENS, MIN_TOKENS, OVERLAP = 400, 500, 120, 0.12
MERGE_BELOW = 300                    # sub-sections shorter than this absorb the next one
_SENT = re.compile(r"(?<=[.!?;:])\s+(?=[\"'(«A-ZÀ-Ý0-9•\-])")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    section: str
    page_start: int | None
    page_end: int | None
    kind: str
    text: str
    subsections: tuple[str, ...] = ()        # sub-headings kept inline in this chunk (cited with the section)

    def indexed_text(self, title: str) -> str:
        section = self.section
        if section.lower().startswith(title.lower()):                 # a Markdown H1 that repeats the title
            section = section[len(title):].lstrip(" >")
        path = f"{title} > {section}" if section else title
        return f"{path}\n{self.text}"

    def to_dict(self) -> dict:
        return asdict(self)


def _sentences(text: str, page: int | None) -> list[tuple[str, int | None]]:
    flat = re.sub(r"\s*\n\s*", " ", join_hyphenation(text)).strip()
    out = []
    for s in _SENT.split(flat):
        words = s.split()
        while len(words) > 300:                                   # pathological run-on text
            out.append((" ".join(words[:300]), page))
            words = words[300:]
        if words:
            out.append((" ".join(words), page))
    return out


def _pack(sents: list[tuple[str, int | None]]) -> list[list[tuple[str, int | None]]]:
    """Group consecutive sentences into chunks of TARGET..MAX_TOKENS with OVERLAP between neighbours."""
    tok = [approx_tokens(s) for s, _ in sents]
    spans: list[tuple[int, int]] = []
    i, n = 0, len(sents)
    while i < n:
        j, size = i, 0
        while j < n and (size == 0 or size + tok[j] <= MAX_TOKENS):
            size += tok[j]
            j += 1
            if size >= TARGET:
                break
        spans.append((i, j))
        if j >= n:
            break
        k, o = j, 0                                               # step back for the overlap, but always advance
        while k - 1 > i and o + tok[k - 1] <= OVERLAP * TARGET:
            k -= 1
            o += tok[k]
        i = k
    if len(spans) >= 2:                                           # merge a short tail into the chunk before it
        (a, b), (c, d) = spans[-2], spans[-1]
        new_part = sum(tok[max(b, c):d])
        if new_part < MIN_TOKENS and sum(tok[a:d]) <= MAX_TOKENS * 1.3:
            spans[-2:] = [(a, d)]
    return [sents[a:b] for a, b in spans]


def chunk_blocks(blocks: list[Block], doc_id: str) -> list[Chunk]:
    """Chunks that never cross a top-level section. A section whose text is shorter than
    MERGE_BELOW is not emitted on its own: the next sub-heading is kept inline in the text
    ("Heading. text ...") and the chunk keeps the path where its text started."""
    chunks: list[Chunk] = []
    path: list[tuple[int, str]] = []
    pending: list[tuple[str, int | None]] = []
    inline: set[str] = set()                 # sentences of `pending` that are sub-headings kept inline
    pending_path = ""

    def section() -> str:
        return " > ".join(t for _, t in path)

    def add(text: str, pages: list[int | None], kind: str, sec: str, subs: tuple[str, ...] = ()) -> None:
        pg = [p for p in pages if p is not None]
        chunks.append(Chunk(f"{doc_id}:{len(chunks):04d}", doc_id, sec, min(pg) if pg else None,
                            max(pg) if pg else None, kind, text, subs))

    def flush() -> None:
        for group in _pack(pending):
            subs = tuple(dict.fromkeys(s[:-1] for s, _ in group if s in inline))
            add(" ".join(s for s, _ in group), [p for _, p in group], "text", pending_path, subs)
        pending.clear()
        inline.clear()

    for b in blocks:
        if b.kind == "heading":
            small = sum(approx_tokens(t) for t, _ in pending) < MERGE_BELOW
            if pending and small and b.level > 1:
                pending.append((b.text.rstrip(".:") + ".", b.page))            # keep the sub-heading inline
                inline.add(pending[-1][0])
            else:
                flush()
            while path and path[-1][0] >= b.level:
                path.pop()
            path.append((b.level, b.text))
        elif b.kind == "text":
            if not pending:
                pending_path = section()
            pending.extend(_sentences(b.text, b.page))
        elif approx_tokens(b.text) < MIN_TOKENS:                             # small table: keep it in the text
            if not pending:
                pending_path = section()
            rows = [r.strip(" |") for r in b.text.split("\n") if r.strip(" |")]
            pending.append(("; ".join(re.sub(r"\s*\|\s*(\|\s*)*", " | ", r) for r in rows) + ".", b.page))
        else:
            flush()
            rows, cur = b.text.split("\n"), []
            for r in rows:
                if cur and approx_tokens("\n".join(cur + [r])) > MAX_TOKENS:
                    add("\n".join(cur), [b.page], "table", section())
                    cur = [rows[0]] if rows[0] != r else []              # repeat the header row
                cur.append(r)
            if cur:
                add("\n".join(cur), [b.page], "table", section())
    flush()
    return chunks
