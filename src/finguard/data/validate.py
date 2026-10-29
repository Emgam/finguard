"""Data-quality checks for the IEEE-CIS files.

Each check returns a row (check, status, observed, expected, detail). Status is one of
PASS, WARN (known issue, handled downstream), FAIL (stop the pipeline) or INFO.

Usage in CI:
    python -m finguard.data.validate      # exits with code 1 if any check FAILs
"""
from __future__ import annotations

import re
import sys

import numpy as np
import pandas as pd

from finguard import config
from finguard.data import schema

EXPECTED_ROWS = {"train_transaction": 590_540, "train_identity": 144_233,
                 "test_transaction": 506_691, "test_identity": 141_907}


def _row(check, status, observed, expected="", detail=""):
    return {"check": check, "status": status, "observed": observed, "expected": expected, "detail": detail}


def check_raw_files(frames: dict[str, pd.DataFrame]) -> list[dict]:
    """Checks on the four raw tables BEFORE any fix."""
    rows = []
    expected_cols = {"train_transaction": schema.TRAIN_TXN_COLS, "train_identity": schema.IDENTITY_COLS,
                     "test_transaction": schema.TEST_TXN_COLS, "test_identity": schema.IDENTITY_COLS}
    for name, df in frames.items():
        if name not in expected_cols:
            continue
        cols = list(df.columns)
        if cols == expected_cols[name]:
            rows.append(_row(f"{name}: column names and order", "PASS", f"{len(cols)} columns", f"{len(expected_cols[name])}"))
        else:
            fixed = [c.replace("-", "_") for c in cols]
            status = "WARN" if fixed == expected_cols[name] else "FAIL"
            n_bad = sum(a != b for a, b in zip(cols, expected_cols[name]))
            rows.append(_row(f"{name}: column names and order", status, f"{n_bad} names differ",
                             "match schema", "hyphenated id-XX names; fixed by normalize_identity_columns()"
                             if status == "WARN" else "unexpected schema"))
        rows.append(_row(f"{name}: row count", "PASS" if len(df) == EXPECTED_ROWS[name] else "WARN",
                         len(df), EXPECTED_ROWS[name]))
        dup = int(df[config.KEY].duplicated().sum())
        rows.append(_row(f"{name}: TransactionID unique", "PASS" if dup == 0 else "FAIL", f"{dup} duplicates", "0"))
    for split in ("train", "test"):
        t, i = frames[f"{split}_transaction"], frames[f"{split}_identity"]
        orphans = int((~i[config.KEY].isin(t[config.KEY])).sum())
        rows.append(_row(f"{split}: identity rows without a transaction", "PASS" if orphans == 0 else "FAIL", orphans, "0"))
        cov = float(t[config.KEY].isin(i[config.KEY]).mean())
        rows.append(_row(f"{split}: transactions with an identity row", "INFO", f"{cov:.1%}", "",
                         "identity data exists for a minority of transactions"))
    overlap = np.intersect1d(frames["train_transaction"][config.KEY], frames["test_transaction"][config.KEY]).size
    rows.append(_row("train/test TransactionID overlap", "PASS" if overlap == 0 else "FAIL", int(overlap), "0"))
    return rows


def check_joined(train: pd.DataFrame, test: pd.DataFrame) -> list[dict]:
    """Checks on the joined, fixed tables."""
    rows = []
    y = train[config.TARGET]
    ok = set(y.dropna().unique()) <= {0, 1} and not y.isna().any()
    rows.append(_row("target domain {0,1}, no missing", "PASS" if ok else "FAIL", sorted(map(int, y.unique()))))
    rows.append(_row("fraud prevalence", "INFO", f"{int(y.sum()):,} frauds = {y.mean():.3%}", "",
                     f"imbalance 1:{(1 - y.mean()) / y.mean():.1f}"))
    extra = sorted(set(train.columns) ^ set(test.columns))
    rows.append(_row("train/test column parity after fix", "PASS" if extra == [config.TARGET] else "FAIL", extra, "['isFraud']"))

    feat = [c for c in train.columns if c not in (config.KEY, config.TARGET, "has_identity", "day", "rel_hour", "rel_weekday")]
    h = pd.util.hash_pandas_object(train[feat], index=False)
    n_exact = int(h.duplicated().sum())
    rows.append(_row("exact duplicate rows (ignoring TransactionID)", "PASS" if n_exact == 0 else "WARN", n_exact, "0",
                     "keep: too few to matter; drop only if they share a time stamp"))
    feat_nt = [c for c in feat if c != config.TIME]
    h2 = pd.util.hash_pandas_object(train[feat_nt], index=False)
    grp = pd.DataFrame({"h": h2, "y": y})[h2.duplicated(keep=False)].groupby("h").y.agg(["min", "max"])
    rows.append(_row("rows identical except ID and time", "WARN" if len(grp) else "PASS",
                     f"{int(h2.duplicated().sum())} rows in {len(grp)} groups",
                     "", f"{int((grp['min'] != grp['max']).sum())} groups carry conflicting labels"))

    amt_ok = bool((train["TransactionAmt"] > 0).all() and (test["TransactionAmt"] > 0).all())
    rows.append(_row("TransactionAmt strictly positive", "PASS" if amt_ok else "FAIL", amt_ok))

    rho = train[[config.KEY, config.TIME]].corr("spearman").iloc[0, 1]
    rows.append(_row("TransactionID order vs time", "WARN", f"Spearman {rho:.4f}", "",
                     "TransactionID is a clock in disguise: exclude from features"))
    gap = (test[config.TIME].min() - train[config.TIME].max()) / config.SECONDS_PER_DAY
    span_tr = (train[config.TIME].max() - train[config.TIME].min()) / config.SECONDS_PER_DAY
    span_te = (test[config.TIME].max() - test[config.TIME].min()) / config.SECONDS_PER_DAY
    rows.append(_row("time coverage", "INFO", f"train {span_tr:.0f} d, gap {gap:.0f} d, test {span_te:.0f} d", "",
                     "TransactionDT is relative seconds; no calendar date or time zone"))
    rows.append(_row("raw TransactionDT range overlap train/test", "WARN", "disjoint", "",
                     "test values all exceed training values: never use raw TransactionDT as a feature"))

    pattern = re.compile(r"(cust|client|account|acct|user|member|person|cardholder)", re.I)
    id_like = [c for c in train.columns if pattern.search(c)]
    rows.append(_row("customer or account identifier present", "INFO", id_like or "none", "",
                     "no customer ID exists; any entity grouping is a heuristic proxy"))
    top_card1 = train["card1"].value_counts().iloc[0]
    rows.append(_row("card1 is a shared code, not a card identity", "INFO",
                     f"most frequent card1 value covers {top_card1:,} transactions", "",
                     "card1 groups many unrelated payments"))

    miss = train.isna().mean()
    very = miss[miss > 0.99].index.tolist()
    rows.append(_row("columns more than 99% missing", "WARN" if very else "PASS", len(very), "0", ", ".join(very)))
    const = [c for c in train.columns if train[c].nunique(dropna=True) <= 1]
    rows.append(_row("constant columns", "PASS" if not const else "WARN", len(const), "0", ", ".join(const)))
    return rows


def run_all(frames: dict[str, pd.DataFrame], train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(check_raw_files(frames) + check_joined(train, test))


def main() -> int:
    from finguard.data.ingest import read_raw, normalize_identity_columns, join_transaction_identity, add_time_index
    frames = {n: read_raw(n) for n in ["train_transaction", "train_identity", "test_transaction", "test_identity"]}
    raw = dict(frames)
    frames["test_identity"], _ = normalize_identity_columns(frames["test_identity"])
    train = join_transaction_identity(frames["train_transaction"], frames["train_identity"])
    test = join_transaction_identity(frames["test_transaction"], frames["test_identity"])
    origin = float(train[config.TIME].min())
    train, test = add_time_index(train, origin), add_time_index(test, origin)
    res = run_all(raw, train, test)
    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(config.TABLE_DIR / "validation_results.csv", index=False)
    print(res.to_string(index=False))
    return 1 if (res.status == "FAIL").any() else 0


if __name__ == "__main__":
    sys.exit(main())
