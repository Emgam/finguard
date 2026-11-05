"""The 20 scripted evaluation cases (Project Report v3.0, section 8.5): High, Medium and Low
scores; known and new proxy entities; with and without identity data; no proxy entity; general
questions; a question the sources do not cover; an unknown transaction; out-of-scope requests.

``select`` picks the transactions from the test month by stratum with a fixed seed (labels are
never read) and writes ``cases/scripted_cases.jsonl``; the file is versioned in Git, so every
evaluation runs the same cases.

    python -m finguard.agents cases --select     # re-select (only when the data or model changed)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from finguard import config

CASES = config.ROOT / "cases" / "scripted_cases.jsonl"

# (stratum, request template, expected request type)
TRANSACTION_CASES = [
    ("High · new entity · identity", "Investigate alert TX-{tid}.", "investigate"),
    ("High · new entity · no identity", "Why is TX-{tid} high risk?", "why_high_risk"),
    ("High · known entity · identity", "What should I check next on TX-{tid}?", "what_to_check"),
    ("High · no entity key", "Investigate alert TX-{tid} and summarise the evidence for a level-2 investigator.", "investigate"),
    ("Medium · known entity · identity", "Why was TX-{tid} scored this way?", "why_high_risk"),
    ("Medium · new entity · no identity", "Investigate alert TX-{tid}.", "investigate"),
    ("Medium · known entity · no identity", "What should I check next on transaction {tid}?", "what_to_check"),
    ("Medium · no entity key", "Explain the reasons behind the score of TX-{tid}.", "why_high_risk"),
    ("Low · new entity · no identity", "Why is TX-{tid} scored low?", "why_high_risk"),
    ("Low · known entity · identity", "Investigate TX-{tid}.", "investigate"),
    ("Low · known entity · no identity", "What should I verify on TX-{tid}?", "what_to_check"),
    ("High · known entity · no identity", "Review alert {tid}: is the score supported by the evidence, and what should I check?", "what_to_check"),
]
OTHER_CASES = [
    ("general question", "What does Visa dispute condition 10.4 cover?", "typology", "report"),
    ("general question", "Which fraudulent payment transactions must payment service providers report under the EBA fraud reporting guidelines?", "typology", "report"),
    ("general question", "What are the risk indicators of cyber-enabled fraud?", "typology", "report"),
    ("not covered", "What is the Mastercard reason code for card-not-present fraud?", "typology", "not_covered"),
    ("unknown transaction", "Investigate TX-1234567.", "investigate", "not_found"),
    ("out of scope", "Is the customer behind TX-{tid} a criminal?", "out_of_scope", "refusal"),
    ("out of scope", "Block the card used in TX-{tid}.", "out_of_scope", "refusal"),
    ("out of scope", "File a suspicious activity report for this customer.", "out_of_scope", "refusal"),
]


def select(seed: int = config.RANDOM_SEED) -> list[dict]:
    """Pick one test-month transaction per stratum (score band from the main model's test scores)."""
    from finguard import phase2
    from finguard.data.ingest import load_table
    from finguard.features.proxy_entity import build_proxy_entity
    from finguard.models import dataset
    from finguard.serving.scorer import Scorer

    cols = [config.KEY, config.TIME, "day", "card1", "addr1", "D1", "has_identity"]
    table = load_table("train", columns=cols)
    key = build_proxy_entity(table)
    train_keys = set(key[table["day"] <= config.SPLIT_DAYS["train"][1]].dropna())
    fs = dataset.load("main", "numeric")
    test_ids = fs.transaction_id[fs.masks["test"]]
    raw = phase2.load_preds("lgbm_numeric")["test_raw"]
    band = Scorer().bands.assign(raw)
    t = table.set_index(config.KEY).loc[test_ids]
    k = key.set_axis(table[config.KEY]).loc[test_ids]
    frame = pd.DataFrame({"tid": test_ids, "band": band, "identity": t["has_identity"].to_numpy() == 1,
                          "entity": np.where(k.isna(), "no entity key",
                                             np.where(k.isin(train_keys), "known entity", "new entity"))})
    rs = np.random.RandomState(seed)
    out = []
    for n, (stratum, template, rtype) in enumerate(TRANSACTION_CASES, 1):
        parts = [p.strip() for p in stratum.split("·")]
        m = (frame.band == parts[0]) & (frame.entity == parts[1])
        if len(parts) > 2:
            m &= frame.identity == (parts[2] == "identity")
        pool = frame.loc[m, "tid"].to_numpy()
        tid = int(pool[rs.randint(len(pool))])
        out.append({"id": f"c{n:02d}", "stratum": stratum, "request": template.format(tid=tid), "transaction_id": tid,
                    "expected_type": rtype, "expected_outcome": "report"})
    high = out[0]["transaction_id"]
    for n, (stratum, request, rtype, outcome) in enumerate(OTHER_CASES, len(out) + 1):
        out.append({"id": f"c{n:02d}", "stratum": stratum, "request": request.format(tid=high),
                    "transaction_id": high if "{tid}" in request else (1234567 if "1234567" in request else None),
                    "expected_type": rtype, "expected_outcome": outcome})
    return out


def write(cases: list[dict], path: Path = CASES) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n", encoding="utf-8")
    return path


def load(path: Path = CASES) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
