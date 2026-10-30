"""Text normalisation and tokenisation shared by parsing, indexing and evaluation.

Regulatory text depends on exact identifiers ("10.4", "R.16", "EBA/GL/2018/05",
"2017-08"), so the tokenizer keeps dotted, slashed and hyphenated identifiers whole and also
emits their parts. Accents are removed for matching only (English and French documents).
"""
from __future__ import annotations

import re
import unicodedata

LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "­": ""}
QUOTES = {"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", " ": " "}
MARKS = {"\u2122": "", "\u2120": "", "\u00ae": ""}           # trade and service marks (NFKC would turn them into "TM", "SM")
PUA_BULLETS = {"\uf0b7": "\u2022", "\uf0a7": "\u2022", "\uf076": "\u2022", "\uf0d8": "\u2022", "\uf0fc": "\u2022", "\uf06e": "\u2022"}
_PUA = re.compile("[\ue000-\uf8ff]")                    # other private-use glyphs (symbol-font copyright signs, icons)
_GLUED_MARK = re.compile(r"(?<=[a-z])(?:SM|TM)\b")      # "FraudClassifierSM": a superscript mark glued to a name

STOP_EN = set("""a about above after again against all am an and any are as at be because been before being below between both
but by can could did do does doing down during each few for from further had has have having he her here hers him his how i if
in into is it its itself just me more most my no nor not now of off on once only or other our ours out over own same she should
so some such than that the their theirs them then there these they this those through to too under until up very was we were
what when where which while who whom why will with would you your yours shall may must also per via""".split())
STOP_FR = set("""au aux avec ce ces cette dans de des du elle en et eux il ils je la le les leur leurs lui ma mais me meme mes moi
mon ne nos notre nous on ou par pas pour qu que qui sa se ses son sur ta te tes toi ton tu un une vos votre vous c d j l m n s t y
ete etre etait sont est ont a sans sous selon lors dont ainsi entre vers chez""".split())
STOPWORDS = STOP_EN | STOP_FR

_TOKEN = re.compile(r"[a-z0-9]+(?:[./-][a-z0-9]+)*")


def clean(text: str) -> str:
    """Unicode clean-up that keeps meaning: ligatures, typographic quotes and dashes, spaces."""
    for a, b in {**MARKS, **PUA_BULLETS}.items():
        text = text.replace(a, b)
    text = _PUA.sub("", text)
    text = unicodedata.normalize("NFKC", text)
    for a, b in {**LIGATURES, **QUOTES}.items():
        text = text.replace(a, b)
    text = _GLUED_MARK.sub("", text)
    return re.sub(r"[ \t]+", " ", text)


def fold(text: str) -> str:
    """Lower-case and strip accents (for matching, never for display)."""
    text = unicodedata.normalize("NFKD", clean(text).lower())
    return "".join(ch for ch in text if not unicodedata.combining(ch))


_KEEP_S = ("ss", "us", "is", "ous", "sis")


def light_stem(word: str) -> str:
    """Fold English and French plurals: 'units' -> 'unit', 'policies' -> 'policy', 'PSPs' -> 'psp'.
    Only plain alphabetic words of 4+ letters; identifiers and numbers are never changed."""
    if len(word) < 4 or not word.isalpha() or not word.endswith(("s", "x")):
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("aux") and len(word) > 5:                    # French: 'principaux' -> 'principal'
        return word[:-3] + "al"
    if word.endswith("x") or word.endswith(_KEEP_S):
        return word
    return word[:-1]


def tokens(text: str, keep_stopwords: bool = False) -> list[str]:
    """Search tokens: whole identifiers plus their parts ('r.16' -> 'r.16', 'r', '16'); plain
    words have their plural folded ('transactions' -> 'transaction')."""
    out: list[str] = []
    for t in _TOKEN.findall(fold(text)):
        if keep_stopwords or t not in STOPWORDS:
            out.append(light_stem(t))
        if any(sep in t for sep in "./-"):
            out.extend(p for p in re.split(r"[./-]", t) if p and (keep_stopwords or p not in STOPWORDS))
    return out


def approx_tokens(text: str) -> int:
    """Approximate LLM token count (about 1.3 tokens per word), used for chunk budgets."""
    return int(round(len(text.split()) * 1.3))


def join_hyphenation(text: str) -> str:
    """Join words broken across lines ('regula-\\ntion' -> 'regulation'); keep real hyphens inline."""
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


def normalize_for_match(text: str) -> str:
    """Canonical form used to find a gold quote inside a chunk: folded, punctuation-light, one space."""
    t = fold(join_hyphenation(text))
    t = re.sub(r"[^a-z0-9%.]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()
