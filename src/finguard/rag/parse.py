"""Parse PDF and Markdown documents into ordered blocks (heading, text, table) with page numbers.

PDF (pdfplumber):
* text lines with their font size and boldness;
* running headers and footers removed: lines that repeat (digits ignored) in the first or
  last three lines of at least 30% of pages, plus bare page numbers;
* headings: lines clearly larger than body text, short bold lines, or lines that start with a
  structural keyword (Article, Recommendation, Guideline, Chapter, Section, Annex and their
  French forms); level from font size, then numbering depth;
* tables found by pdfplumber become their own blocks, and their lines are removed from the text.

Markdown (project documents): '#' headings, pipe tables, paragraphs; no page numbers.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from finguard.rag.text import clean, fold

STRUCTURAL = re.compile(
    r"^(article|recommendation|guideline|orientation|chapter|chapitre|section|annex|annexe|title|titre|part|partie|appendix)"
    r"\s+[0-9ivxlc]+\b", re.I)
NUMBERED = re.compile(r"^((\d+)(\.\d+){0,3})\.?\s+\S")
ARTICLE = re.compile(r"^((?:article|art\.|recommendation|guideline|orientation)\s*(?:\d+|premier|first|[ivxlc]+)(?:\s*\(?\s*(?:bis|ter|nouveau|new)\s*\)?)*)\s*(.*)$", re.I)
BULLET = ("−", "•", "-", "–", "▪", "◦", "*", "o ")


def _structural_split(text: str) -> tuple[str, str] | None:
    """'Article 3 : Les banques ...' -> ('Article 3', 'Les banques ...'); a sentence that only cites
    an article ('Article 18 of the RTS ...') is not a heading and returns None."""
    m = ARTICLE.match(text)
    if not m:
        return None
    head, rest = m.group(1), m.group(2).strip()
    if not rest:
        return head, ""
    if re.match(r"[.,]\d", rest):                                    # 'Guideline 1.1, ...' cites a paragraph
        return None
    if rest[0] in ":.-–—":
        return head, rest[1:].strip()
    if len(rest.split()) <= 8 and rest[:1].isupper() and not rest.endswith((".", ";", ",")):
        return f"{head} {rest}", ""
    return None
PAGE_NUMBER = re.compile(r"^(page\s*)?[-–|\s]*\d{1,4}\s*((/|of|sur|de)\s*\d{1,4})?[-–|\s]*$", re.I)
DOT_LEADER = re.compile(r"(\.\s?){4,}|…{2,}")
TOC_LINE = re.compile(r"[A-Za-zÀ-ÿ].*\s\d{1,3}$")


@dataclass
class Block:
    kind: str                 # "heading", "text" or "table"
    text: str
    page: int | None
    level: int = 0            # heading level (1 = top); 0 for text and tables


# --------------------------------------------------------------------------- PDF
def _runs(chars: list[dict]) -> list[list[dict]]:
    """Split a line's characters (stream order) where a new run jumps back to the left of the
    previous one: two text objects overlaid on one baseline (a sidebar label next to body text)."""
    out: list[list[dict]] = []
    cur: list[dict] = []
    right = -1e9
    for c in chars:
        if cur and c["x0"] < right - 15 and c["text"].strip():           # a real jump back, not a ligature or superscript
            out.append(cur)
            cur, right = [], -1e9
        cur.append(c)
        right = max(right, c["x1"])
    if cur:
        out.append(cur)
    return out


def _drop_superscripts(chars: list[dict]) -> list[dict]:
    """Remove raised small characters (footnote markers, service marks) so 'transactions9' and
    'FraudClassifierSM' index as 'transactions' and 'FraudClassifier'."""
    vis = [c for c in chars if c["text"].strip()]
    if len(vis) < 4:
        return chars
    sizes = sorted(c["size"] for c in vis)
    med = sizes[len(sizes) // 2]
    base = sorted(c["bottom"] for c in vis if c["size"] >= 0.9 * med)
    if not base:
        return chars
    baseline = base[len(base) // 2]
    return [c for c in chars if not (c["text"].strip() and c["size"] <= 0.72 * med and c["bottom"] <= baseline - 0.2 * med)]


def _join(chars: list[dict]) -> str:
    """Characters to text, inserting a space where the gap exceeds 12% of the font size."""
    out, prev = [], None
    for c in chars:
        if prev is not None and c["x0"] - prev["x1"] > max(1.0, 0.12 * c["size"]) and not c["text"].isspace():
            out.append(" ")
        out.append(c["text"])
        prev = c
    return "".join(out)


def _line_records(pdf) -> list[list[dict]]:
    pages = []
    for page in pdf.pages:
        try:
            page = page.dedupe_chars(tolerance=1)                # shadow text drawn twice ("fake bold")
        except Exception:                                         # pragma: no cover
            pass
        try:
            tables = page.find_tables()
        except Exception:                                         # pragma: no cover - malformed pages
            tables = []
        boxes = [t.bbox for t in tables]
        recs = []
        for ln in page.extract_text_lines(return_chars=True, strip=True, use_text_flow=True):
            if any(b[1] - 1 <= ln["top"] <= b[3] + 1 for b in boxes):
                continue
            raw = [c for c in ln["chars"] if c["text"] != "\n"]
            kept = _drop_superscripts(raw)
            runs = _runs(kept)
            for chars in runs:
                text = clean(ln["text"] if len(runs) == 1 and len(kept) == len(raw) else _join(chars)).strip()
                vis = [c for c in chars if c["text"].strip()]
                if not text or not vis:
                    continue
                sizes = sorted(round(c["size"] * 2) / 2 for c in vis)
                bold = sum("bold" in c.get("fontname", "").lower() or "black" in c.get("fontname", "").lower() for c in vis)
                recs.append({"text": re.sub(r"\s+", " ", text), "size": sizes[len(sizes) // 2], "bold": bold / len(vis),
                             "top": ln["top"]})
        table_rows = []
        for t in tables:
            try:
                rows = t.extract()
            except Exception:                                     # pragma: no cover
                continue
            lines = [" | ".join(clean(c or "").replace("\n", " ").strip() for c in row) for row in rows if row]
            lines = [x for x in lines if x.strip(" |")]
            if lines:
                table_rows.append("\n".join(lines))
        pages.append({"lines": recs, "tables": table_rows})
    return pages


def _body_size(pages: list[dict]) -> float:
    weights: Counter = Counter()
    for p in pages:
        for x in p["lines"]:
            weights[x["size"]] += len(x["text"])
    return weights.most_common(1)[0][0] if weights else 10.0


def _running_keys(pages: list[dict], body: float, edge: int = 3, share: float = 0.3) -> tuple[set[str], set[str]]:
    """Running headers and footers: lines repeated near the top or bottom of many pages.
    Exact repeats are removed at any size; repeats that differ only in digits ('Page 3 of 16')
    only in type no larger than body text, so numbered headings ('Article 3') are kept."""
    exact: Counter = Counter()
    digits: Counter = Counter()
    for p in pages:
        lines = p["lines"]
        edge_lines = lines[:edge] + lines[-edge:]
        exact.update({x["text"] for x in edge_lines})
        digits.update({re.sub(r"\d+", "#", fold(x["text"])) for x in edge_lines if x["size"] <= body + 0.25})
    anywhere: Counter = Counter()
    for p in pages:
        anywhere.update({x["text"] for x in p["lines"]})
    small: Counter = Counter()
    for p in pages:
        lines = p["lines"]
        small.update({x["text"] for x in lines[:edge] + lines[-edge:] if x["size"] < body - 0.5})
    need = max(3, int(share * len(pages)))
    if len(pages) < 4:
        return set(), set()
    boiler = {k for k, n in anywhere.items() if n >= max(4, int(0.4 * len(pages))) and len(k) >= 8}
    # a line in small type repeated at the page edge is a running header even when the document
    # changes its header part-way (consolidated versions): 10% of pages is enough
    small_rep = {k for k, n in small.items() if n >= max(3, int(0.1 * len(pages))) and len(k) >= 8}
    return {k for k, n in exact.items() if n >= need} | boiler | small_rep, {k for k, n in digits.items() if n >= need}


def _is_toc_page(lines: list[dict]) -> bool:
    texts = [x["text"] for x in lines if len(x["text"]) > 3]
    if len(texts) < 6:
        return False
    hits = sum(bool(DOT_LEADER.search(t)) or bool(TOC_LINE.search(t)) and len(t) < 120 for t in texts)
    return hits / len(texts) >= 0.4


def _is_diagram_page(lines: list[dict]) -> bool:
    """A flowchart or infographic: many short labels and almost no running prose. Its text is
    kept for search, but its labels are not used as section headings."""
    if len(lines) < 15:
        return False
    short = sum(len(x["text"].split()) <= 4 for x in lines) / len(lines)
    prose = sum(len(x["text"].split()) >= 8 for x in lines)
    return short >= 0.6 and prose < 3


def _letterless(text: str) -> bool:
    return not re.search(r"[A-Za-zÀ-ÿ]", text) and len(text) <= 14


def parse_pdf(path: str | Path) -> list[Block]:
    import pdfplumber
    with pdfplumber.open(str(path)) as pdf:
        pages = _line_records(pdf)
    body = _body_size(pages)
    running_exact, running_digits = _running_keys(pages, body)
    head_sizes = sorted({x["size"] for p in pages for x in p["lines"] if x["size"] >= body + 1.0}, reverse=True)
    size_level = {s: min(i + 1, 3) for i, s in enumerate(head_sizes)}

    blocks: list[Block] = []
    for pno, p in enumerate(pages, 1):
        lines = p["lines"]
        if _is_toc_page(lines):
            continue
        diagram = _is_diagram_page(lines)
        if sum(_letterless(x["text"]) for x in lines) >= 10:       # chart axis ticks and data labels
            lines = [x for x in lines if not _letterless(x["text"])]
        for i, x in enumerate(lines):
            key = re.sub(r"\d+", "#", fold(x["text"]))
            at_edge = i < 3 or i >= len(lines) - 3
            if x["text"] in running_exact and (at_edge or len(x["text"]) >= 8) \
                    or (at_edge and key in running_digits and x["size"] <= body + 0.25):
                continue
            if PAGE_NUMBER.match(x["text"]) or DOT_LEADER.search(x["text"]):
                continue
            text = x["text"]
            letters = len(re.findall(r"[A-Za-zÀ-ÿ]", text))
            short = (len(text) <= 120 and not text.endswith((",", ";")) and letters >= 3
                     and not re.fullmatch(r"[ivxlcIVXLC]+\.?|[a-z]\)|\(?[a-z0-9]{1,2}\)", text))
            junk = (text.startswith(BULLET) or len(re.findall(r"\b\d+\b", text)) >= 3
                    or len(re.findall(r"\bR\.\s?\d+", text)) >= 2)
            level = 0
            large = short and x["size"] >= body + 1.0
            split = _structural_split(text)
            if split and split[1] and not large:                       # inline "Article N : text" in body type
                blocks.append(Block("heading", split[0], pno, 3))
                blocks.append(Block("text", split[1], pno))
                continue
            if junk or (diagram and not NUMBERED.match(text) and not split):
                level = 0
            elif large:                                                # 'Guideline 3: Frequency, ...' set large
                level = size_level.get(x["size"], 3)
            elif split:
                level, text = 3, split[0]
            elif short and x["bold"] >= 0.8 and len(text) <= 100 and x["size"] >= body - 0.5:
                level = 3
            if level:
                m = NUMBERED.match(text)
                if m and level == 3:
                    level = min(3, m.group(1).count(".") + 1 + (1 if head_sizes else 0))
                prev = blocks[-1] if blocks else None
                if prev and prev.kind == "heading" and prev.page == pno and prev.level == level and len(prev.text) < 120:
                    prev.text = f"{prev.text} {text}"                       # multi-line heading
                    continue
                blocks.append(Block("heading", text, pno, level))
            else:
                prev = blocks[-1] if blocks else None
                if prev and prev.kind == "text" and prev.page == pno:
                    prev.text += "\n" + text
                else:
                    blocks.append(Block("text", text, pno))
        for t in p["tables"]:
            blocks.append(Block("table", t, pno))
    return blocks


# --------------------------------------------------------------------------- Markdown
def parse_markdown(path: str | Path) -> list[Block]:
    blocks: list[Block] = []
    buf: list[str] = []
    table: list[str] = []

    def flush_text():
        if buf and "".join(buf).strip():
            blocks.append(Block("text", clean("\n".join(buf)).strip(), None))
        buf.clear()

    def flush_table():
        rows = [r for r in table if not re.match(r"^\|\s*:?-{3,}", r)]
        if rows:
            blocks.append(Block("table", clean("\n".join(r.strip().strip("|").strip() for r in rows)), None))
        table.clear()

    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        h = re.match(r"^(#{1,6})\s+(.*)$", line)
        if line.startswith("|"):
            flush_text()
            table.append(line)
            continue
        if table:
            flush_table()
        if h:
            flush_text()
            blocks.append(Block("heading", clean(h.group(2)).strip(), None, min(len(h.group(1)), 3)))
        elif not line.strip():
            flush_text()
        else:
            buf.append(line)
    flush_table()
    flush_text()
    return blocks


def parse(path: str | Path) -> list[Block]:
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(path)
    if suffix in (".md", ".markdown", ".txt"):
        return parse_markdown(path)
    raise ValueError(f"unsupported document type: {path}")
