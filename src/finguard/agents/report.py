"""Case reports: the template parts, the fallback report and the Markdown rendering.

The five sections of Project Report v3.0 (section 8.4): Facts (template, from tool outputs),
Interpretation (cited), Guidance (quoted passages), Suggested checks (labelled as suggestions)
and Limitations (fixed). Agents write only the summary, interpretation, guidance and checks;
every number in Facts comes straight from a tool.

``template_report`` builds those agent sections deterministically from the evidence alone. It
is the fallback path's report, and the report when no LLM is configured.
"""
from __future__ import annotations

import re

from finguard.agents.evidence import Evidence
from finguard.rag.text import tokens

DISCLAIMER = "Flagged for review, not a finding of fraud."
ANONYMIZED_FAMILIES = {"identity_masked", "counting_C", "timedelta_D", "match_M", "vesta_V"}

LIMITATIONS_TX = [
    "The proxy entity (card1 + addr1 + first-seen day) is a heuristic grouping: it can merge or split real people "
    "and is never a customer or a verified identity.",
    "Reasons explain the model's score, not the cause of fraud. Anonymized signal groups (V, C, D, M, id) have no "
    "disclosed meaning and cannot be checked against the transaction.",
    "The band leads; the calibrated probability is secondary and capped at \"<1%\" and \">99%\". In the alert range it "
    "overstated the test-month fraud rate by 4-5 points.",
    "Guidance comes from 11 registered documents; a passage is context for the analyst's decision, not legal advice. "
    "Passages marked fictional come from FinGuard's fictional playbook.",
]
LIMITATIONS_GENERAL = [
    "Answers come only from the 11 registered documents; anything else is \"not covered by the sources\".",
    "A passage is context for the analyst's decision, not legal advice. Passages marked fictional come from "
    "FinGuard's fictional playbook.",
]


# --------------------------------------------------------------------------- helpers
def first_sentence(text: str, min_chars: int = 40, max_chars: int = 300) -> str:
    """A verbatim prefix of the passage: its first sentence, or a cut at a word boundary."""
    t = " ".join((text or "").split())
    end = t.find(". ", min_chars)
    if 0 < end <= max_chars:
        return t[:end + 1]
    if len(t) <= max_chars:
        return t
    cut = t.rfind(" ", min_chars, max_chars)
    return t[:cut if cut > 0 else max_chars]


_SENT_END = re.compile(r"(?<=[a-z0-9\)\]\"”%][.!?])\s+(?=[A-Z0-9\"“(])")


def sentences(text: str) -> list[str]:
    """Verbatim sentences of a passage (whitespace collapsed); bullets and table rows are skipped."""
    t = " ".join((text or "").split())
    return [x for x in _SENT_END.split(t) if x and "|" not in x and "**" not in x and "`" not in x
            and not x.lstrip().startswith(("−", "-", "•", "Example"))]


def best_sentence(text: str, query: str, min_chars: int = 40, max_chars: int = 400) -> str | None:
    """The passage sentence that shares most search terms with the query (a verbatim span), or None."""
    q = set(tokens(query))
    best, best_score = None, 0.0
    for sent in sentences(text):
        if not (min_chars <= len(sent) <= max_chars):
            continue
        overlap = len(q & set(tokens(sent))) / max(len(q), 1)
        if overlap > best_score:
            best, best_score = sent, overlap
    return best


def _ids(ev: Evidence, tool: str) -> str | None:
    i = ev.latest(tool)
    return None if i is None else i["id"]


def _playbook_passage(ev: Evidence) -> str | None:
    return next((sid for sid, p in ev.passages().items() if p.get("fictional")), None)


# --------------------------------------------------------------------------- the deterministic report
def template_report(ev: Evidence, request_type: str) -> dict:
    """Agent-format report built from the evidence only; it passes the policy check by construction."""
    if request_type == "typology":
        search = ev.latest("policy_search")
        if search is None or search["output"].get("status") != "ok":
            missing = (search or {}).get("output", {}) or {}
            names = missing.get("ungrounded_names") or []
            text = "Not covered by the sources." + (f" The sources found do not mention: {', '.join(names)}." if names else "")
            return {"summary": {"text": text, "cites": [search["id"]] if search else []}, "interpretation": [],
                    "guidance": [], "suggested_checks": []}
        passages, query = ev.passages(), search["args"].get("query", "")
        guidance = []
        for sid in search.get("passage_ids", []):
            quote = best_sentence(passages[sid]["text"], query)
            if quote and len(guidance) < 3:
                guidance.append({"quote": quote, "cites": [sid]})
        cites = [g["cites"][0] for g in guidance[:1]] or search.get("passage_ids", [])[:1]
        return {"summary": {"text": "The quoted passages below address the question; read them with their source and date.",
                            "cites": cites},
                "interpretation": [], "guidance": guidance, "suggested_checks": []}

    score, expl, hist, imap = (ev.latest(t) for t in ("fraud_score", "explain_transaction", "entity_history", "indicator_map"))
    s = score["output"]
    report = {"summary": {"text": f"Transaction {s['transaction_id']} is in the {s['band']} band with a score of {s['score']}.",
                          "cites": [score["id"]]},
              "interpretation": [], "guidance": [], "suggested_checks": []}
    interp = report["interpretation"]
    reasons = expl["output"]["reasons"] if expl else []
    for r in [r for r in reasons if r.get("checkable")][:3]:
        interp.append({"text": r["text"], "cites": [expl["id"]]})
    anon = [r["label"] for r in reasons if r.get("anonymized")]
    if anon:
        interp.append({"text": f"Anonymized signal groups also pushed the score up ({'; '.join(anon)}); their meaning is "
                               "not disclosed, so they cannot be checked against the transaction.", "cites": [expl["id"]]})
    for f in s.get("novelty_flags", []):
        if f.get("on"):
            interp.append({"text": f"Novelty flag on: {f['rule']}.", "cites": [score["id"]]})
    if hist:
        h = hist["output"]
        if h.get("available") and not h.get("earlier_transactions"):
            interp.append({"text": "This is the proxy entity's first transaction in the data, so there is no earlier "
                                   "history to compare.", "cites": [hist["id"]]})
        elif h.get("available"):
            text = (f"The proxy entity had {h['earlier_transactions']} earlier transactions, {h['earlier_in_24h']} of them "
                    "in the 24 hours before this one.")
            if h.get("amount_vs_earlier_mean") is not None:
                text += f" The amount is {h['amount_vs_earlier_mean']:.2f} times its earlier average."
            interp.append({"text": text, "cites": [hist["id"]]})
        else:
            interp.append({"text": "No proxy entity could be formed (card1, addr1 or D1 is missing), so there is no "
                                   "history to compare.", "cites": [hist["id"]]})
    if imap:
        m = imap["output"]
        if m.get("typologies"):
            cites = [imap["id"]] + ([p] if (p := _playbook_passage(ev)) else [])
            interp.append({"text": "In FinGuard's fictional playbook, these indicators are consistent with "
                                   f"{', '.join(m['typologies'])}.", "cites": cites})
        report["suggested_checks"] = [{"text": c["text"], "cites": [imap["id"]]}
                                      for c in m.get("checks", []) if c["trigger"] not in ANONYMIZED_FAMILIES][:4]
    elif expl:
        report["suggested_checks"] = [{"text": "Verify each checkable reason above against the transaction record.",
                                       "cites": [expl["id"]]}]
    passages = ev.passages()
    for item in ev.items:
        if not (item["ok"] and item["tool"] == "policy_search") or len(report["guidance"]) >= 3:
            continue
        for sid in item.get("passage_ids", []):
            quote = best_sentence(passages[sid]["text"], item["args"].get("query", ""))
            if quote:
                report["guidance"].append({"quote": quote, "cites": [sid]})
                break
    return report


# --------------------------------------------------------------------------- Markdown
def _cite(cites: list[str]) -> str:
    return f" [{', '.join(cites)}]" if cites else ""


def _fmt(v, nd: int = 2) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:,.{nd}f}".rstrip("0").rstrip(".") if abs(v - round(v)) > 1e-9 else f"{v:,.0f}"
    return str(v)


def facts(ev: Evidence) -> list[str]:
    lines = []
    sc = ev.latest("fraud_score")
    if sc:
        s, t = sc["output"], sc["output"].get("transaction", {})
        lines.append(f"- **Transaction {s['transaction_id']}** ({s['period']} period, relative day {s['relative_day']}, "
                     f"relative hour {_fmt(t.get('relative_hour'))}): amount {_fmt(t.get('amount'))}, product code "
                     f"{t.get('product_code')}, card {t.get('card_network')} {t.get('card_type')}, identity record "
                     f"{'yes' if t.get('has_identity_record') else 'no'}. [{sc['id']}]")
        lines.append(f"- **Band {s['band']}**, score {s['score']} on a 0-100 scale (displayed probability "
                     f"{s['probability_displayed']}); model {s['model_version']}. [{sc['id']}]")
        on = [f["rule"] for f in s.get("novelty_flags", []) if f.get("on")]
        lines.append(f"- Novelty flags (shown, not part of the score): {'; '.join(on) if on else 'none on'}. [{sc['id']}]")
    ex = ev.latest("explain_transaction")
    if ex:
        lines.append(f"- Reason card (explains the score, not the cause) [{ex['id']}]:")
        for n, r in enumerate(ex["output"]["reasons"], 1):
            kind = "checkable" if r.get("checkable") else "anonymized, not checkable"
            lines.append(f"    {n}. {r['text']} *({kind}; +{r['contribution_log_odds']:.2f} log-odds)*")
    hi = ev.latest("entity_history")
    if hi:
        h = hi["output"]
        if h.get("available") and not h.get("earlier_transactions"):
            lines.append(f"- Proxy entity (heuristic, not a customer): first transaction in the data, no earlier history. "
                         f"[{hi['id']}]")
        elif h.get("available"):
            win = h.get("window_days", 7)
            k7 = h.get(f"earlier_in_{win:g}d")
            lines.append(f"- Proxy entity (heuristic, not a customer): {h['earlier_transactions']} earlier transactions, "
                         f"{h['earlier_in_24h']} in the previous 24 h, {k7} in the previous {win:g} days; "
                         f"amount {_fmt(h.get('amount'))} vs earlier mean {_fmt(h.get('earlier_amount_mean'))} and max "
                         f"{_fmt(h.get('earlier_amount_max'))}; device brands seen: "
                         f"{', '.join(h.get('device_brands_seen') or []) or 'none'}; email domains seen: "
                         f"{', '.join(h.get('purchaser_email_domains_seen') or []) or 'none'}. [{hi['id']}]")
        else:
            lines.append(f"- Proxy entity: none ({h.get('reason')}). [{hi['id']}]")
    cs = ev.latest("case_store")
    if cs:
        cases = cs["output"].get("cases", [])
        txt = "none" if not cases else "; ".join(
            f"{c['case_id']} ({c['created_at']}, band {c.get('band')}, decision {c.get('decision') or 'none yet'})" for c in cases)
        lines.append(f"- Earlier FinGuard cases for this transaction: {txt}. [{cs['id']}]")
    return lines


def _pages(p: dict, lead: str = "; ") -> str:
    """'pages 17-18' for PDFs; nothing for markdown documents, which have no pages."""
    return f"{lead}pages {p['pages']}" if p.get("pages") not in (None, "", "None") else ""


def _source_line(item_or_passage: dict, is_passage: bool) -> str:
    if is_passage:
        p = item_or_passage
        fic = " **FICTIONAL**" if p.get("fictional") else ""
        subs = f" (also: {'; '.join(p['subsections'])})" if p.get("subsections") else ""
        return (f"- [{p['label']}] *{p.get('title')}*, {p.get('publisher')}, {p.get('published')}; {p.get('jurisdiction')}; "
                f"tier {p.get('tier')}; section: {p.get('section')}{subs}{_pages(p)}; {p.get('url')}{fic}")
    i = item_or_passage
    args = ", ".join(f"{k}={v}" for k, v in i["args"].items())
    status = "ok" if i["ok"] else f"failed: {i['error']}"
    if i["ok"] and i["tool"] == "policy_search":
        status = i["output"].get("status")
    return f"- [{i['id']}] `{i['tool']}({args})`: {status}"


def _llm_line(llm: dict) -> str:
    if not llm.get("model"):
        return "none"
    if "served" in llm:                                          # a chain (auto: Groq, then Ollama offline)
        if llm["served"]:
            who = "; ".join(f"{k} ({n} call{'s' if n > 1 else ''})" for k, n in llm["served"].items())
        else:
            who = "no LLM answered"
        who += f" [{llm.get('name')}: {llm['model'] if not llm['served'] else 'Groq first, Ollama offline fallback'}]"
        if llm.get("fallbacks"):
            who += f"; {len(llm['fallbacks'])} call(s) handed to the fallback"
    else:
        who = f"{llm['model']} ({llm['name']})" if llm.get("name") and llm["name"] not in ("anthropic", "scripted") else llm["model"]
    modes = (["lean: no investigation agent"] if llm.get("lean") else []) + (
        [f"passages shown up to {llm['passage_chars']} characters"] if llm.get("passage_chars") not in (None, 1500) else [])
    return who + (f", {'; '.join(modes)}" if modes else "")


def render(case: dict) -> str:
    ev = Evidence(case.get("evidence", []))
    rep = case.get("report") or {}
    passages = ev.passages()
    title = (f"Transaction {case['transaction_id']}" if case.get("transaction_id") else "Knowledge-base question")
    L = [f"# FinGuard case report: {title}", "", f"**{DISCLAIMER}**", ""]
    path = {"orchestrated": f"orchestrated by {case.get('llm', {}).get('model')}",
            "fallback": f"fallback path ({case.get('fallback_reason')})",
            "template": "template path (no LLM configured)", "refusal": "declined by the scope check"}.get(case.get("path"), case.get("path"))
    L += [f"Case `{case['case_id']}` · {case['created_at']} · request type **{case.get('request_type')}** · {path}", "",
          f"> {case['request']}", ""]
    if case.get("llm", {}).get("name") == "scripted":
        L += ["*Written by the scripted test model (deterministic, no LLM): a check of the pipeline, not of an LLM.*", ""]
    if case["outcome"] == "refusal":
        L += ["## Answer", "", case["message"], ""]
        return "\n".join(L)
    if case["outcome"] in ("not_found", "missing_transaction"):
        L += ["## Answer", "", case["message"], "", "## Sources", ""] + [_source_line(i, False) for i in ev.items]
        return "\n".join(L)
    if rep.get("summary"):
        L += ["## Summary", "", rep["summary"]["text"] + _cite(rep["summary"].get("cites", [])), ""]
    if case.get("transaction_id"):
        L += ["## Facts", ""] + facts(ev) + [""]
    if rep.get("interpretation"):
        L += ["## Interpretation", ""] + [f"- {c['text']}{_cite(c['cites'])}" for c in rep["interpretation"]] + [""]
    if rep.get("guidance"):
        L += ["## Guidance", ""]
        for g in rep["guidance"]:
            p = passages.get(g["cites"][0], {})
            fic = " *(fictional playbook)*" if p.get("fictional") else ""
            L += [f"> \"{g['quote']}\"", ">", f"> {_cite(g['cites']).strip()} *{p.get('title')}*, {p.get('section')}"
                  f"{_pages(p, ', ')} ({p.get('publisher')}, {p.get('published')}; {p.get('jurisdiction')}){fic}", ""]
            if g.get("note"):
                L += [g["note"], ""]
    if rep.get("suggested_checks"):
        L += ["## Suggested checks", "", "*Suggestions for the analyst, not required steps.*", ""]
        L += [f"- {c['text']}{_cite(c['cites'])}" for c in rep["suggested_checks"]] + [""]
    L += ["## Limitations", ""] + [f"- {x}" for x in (LIMITATIONS_TX if case.get("transaction_id") else LIMITATIONS_GENERAL)]
    if case.get("path") == "fallback":
        L.append(f"- Fallback path: the orchestrated path did not pass ({case.get('fallback_reason')}); this report is "
                 "built by template from the tool outputs.")
    sc = ev.output("fraud_score")
    if sc and sc.get("period") == "training":
        L.append("- This transaction is in the model's training period: its score is in-sample.")
    L += ["", "## Sources", ""] + [_source_line(i, False) for i in ev.items] + [_source_line(p, True) for p in passages.values()]
    v, pol = case.get("versions", {}), case.get("policy", {})
    first = pol.get("first_attempt") or {}
    L += ["", "## Run record", "",
          f"- Model {v.get('model_version')}; index {v.get('index_version')} ({v.get('embedder')}, re-ranker "
          f"{v.get('reranker')}, variant {v.get('retrieval_variant')}); indicator map {v.get('indicator_map')}; "
          f"prompts {', '.join(case.get('prompts', [])) or 'none'}; LLM {_llm_line(case.get('llm') or {})}.",
          f"- Evidence policy: {'passed' if pol.get('passed') else 'FAILED'}; claims cited {pol.get('cited')}/{pol.get('claims')}, "
          f"numbers traceable {pol.get('traceable')}/{pol.get('numbers')}"
          + (f"; first agent draft: {first.get('cited')}/{first.get('claims')} cited, {first.get('traceable')}/"
             f"{first.get('numbers')} numbers traceable, {first.get('violations')} violation(s)" if first else "") + ".",
          f"- {case.get('tool_calls')} tool calls, {case.get('llm_calls')} LLM calls, {case.get('seconds')} s."]
    return "\n".join(L) + "\n"
