"""FinGuard's own case database (SQLite): earlier alerts, report drafts and analyst decisions.

The ``case_store`` tool only reads. Code saves the report draft at the end of a run (the one
write the allow-list permits), and only a person records a decision, through the command line
(``python -m finguard.agents decide``). The decisions come from FinGuard analysts, never from
the dataset's labels.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from finguard import config

CASE_DIR = Path(os.environ.get("FINGUARD_CASE_DIR", config.ROOT / "data" / "cases"))
DECISIONS = ("confirmed_fraud", "false_positive", "inconclusive")

SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    case_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, transaction_id INTEGER, entity_key TEXT,
    request TEXT NOT NULL, request_type TEXT, outcome TEXT, path TEXT, band TEXT, score REAL,
    model_version TEXT, index_version TEXT, status TEXT NOT NULL DEFAULT 'draft', report_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS decisions (
    case_id TEXT NOT NULL REFERENCES cases(case_id), decided_at TEXT NOT NULL, decision TEXT NOT NULL,
    analyst TEXT NOT NULL, note TEXT);
CREATE INDEX IF NOT EXISTS cases_tx ON cases(transaction_id);
CREATE INDEX IF NOT EXISTS cases_entity ON cases(entity_key);
"""


class CaseStore:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path or CASE_DIR / "case_store.sqlite")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.executescript(SCHEMA)

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        """Commit on success, roll back on error, and always close (an open handle keeps the
        file locked on Windows)."""
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    # ------------------------------------------------------------------ the tool (read only)
    def read(self, transaction_id: int | None = None, entity_key: str | None = None, limit: int = 10,
             exclude_case: str | None = None) -> dict:
        where, args = [], []
        if transaction_id is not None:
            where.append("c.transaction_id = ?")
            args.append(int(transaction_id))
        if entity_key:
            where.append("c.entity_key = ?")
            args.append(str(entity_key))
        if not where:
            return {"cases": [], "note": "no transaction or proxy entity given"}
        sql = (f"SELECT c.case_id, c.created_at, c.transaction_id, c.request_type, c.outcome, c.path, c.band, c.score, "
               f"c.status, d.decision, d.decided_at FROM cases c LEFT JOIN decisions d ON d.case_id = c.case_id "
               f"WHERE ({' OR '.join(where)})" + (" AND c.case_id <> ?" if exclude_case else "") +
               " ORDER BY c.created_at DESC LIMIT ?")
        args += ([exclude_case] if exclude_case else []) + [int(limit)]
        with self._db() as db:
            rows = [dict(r) for r in db.execute(sql, args)]
        return {"cases": rows, "note": "Earlier FinGuard cases for this transaction or proxy entity; decisions are "
                                       "analysts' records, not dataset labels."}

    # ------------------------------------------------------------------ code and people only
    def save_draft(self, case: dict) -> str:
        with self._db() as db:
            db.execute("INSERT OR REPLACE INTO cases (case_id, created_at, transaction_id, entity_key, request, request_type, "
                       "outcome, path, band, score, model_version, index_version, status, report_json) "
                       "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,'draft',?)",
                       (case["case_id"], case["created_at"], case.get("transaction_id"), case.get("entity_key"),
                        case["request"], case.get("request_type"), case.get("outcome"), case.get("path"), case.get("band"),
                        case.get("score"), case.get("model_version"), case.get("index_version"),
                        json.dumps(case.get("report", {}), ensure_ascii=False, default=str)))
        return case["case_id"]

    def record_decision(self, case_id: str, decision: str, analyst: str, note: str = "") -> dict:
        if decision not in DECISIONS:
            raise ValueError(f"decision must be one of {DECISIONS}")
        if not analyst or not analyst.strip():
            raise ValueError("a decision needs the analyst's name or role")
        with self._db() as db:
            if db.execute("SELECT 1 FROM cases WHERE case_id = ?", (case_id,)).fetchone() is None:
                raise KeyError(f"unknown case {case_id}")
            at = time.strftime("%Y-%m-%d %H:%M:%S")
            db.execute("INSERT INTO decisions VALUES (?,?,?,?,?)", (case_id, at, decision, analyst.strip(), note))
            db.execute("UPDATE cases SET status = 'decided' WHERE case_id = ?", (case_id,))
        return {"case_id": case_id, "decision": decision, "analyst": analyst.strip(), "decided_at": at}

    def get(self, case_id: str) -> dict | None:
        with self._db() as db:
            r = db.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
        return None if r is None else dict(r)
