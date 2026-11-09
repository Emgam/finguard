"""Data contracts (Pandera) for the tables one pipeline step hands to the next.

``finguard.data.validate`` describes the data (row counts, duplicates, prevalence); the
contracts below are the promises each step makes to the next one, checked on every run:

| Contract              | Table                                   | What it guarantees                                       |
| raw_transaction_*     | train/test_transaction.csv as read      | key unique, types as ingested, amounts > 0, documented    |
|                       |                                         | domains (ProductCD, card4/6, M1-M9), counts >= 0          |
| raw_identity_*        | train/test_identity.csv, names fixed    | key unique, exact column set and order, DeviceType domain |
| interim_*             | data/interim/{train,test} (joined)      | the join (no identity values without an identity row),    |
|                       |                                         | day / rel_hour / rel_weekday derived from TransactionDT,  |
|                       |                                         | training days cover the time split, test is later        |
| features_*            | Phase 2 feature matrix (model input)    | exact column list and order of the model, float32, no     |
|                       |                                         | infinities, entity/frequency features >= 0, codes >= 0    |
| scores_*              | validation / test-month model outputs   | probability in [0, 1], score 1-99, band consistent with   |
|                       |                                         | the frozen thresholds, calibration monotone in the score  |

A failed check is an error (exit code 1, the data gate fails); checks marked ``warn`` are
known-to-move properties (a new card network, the alert share of a later month) and are
reported without failing. Results: ``reports/tables/contracts_data.json`` and
``reports/tables/contracts_model.json`` (one file per stage, so each DVC stage owns its output).

    python -m finguard.mlops contracts --stage data      # raw files and interim tables
    python -m finguard.mlops contracts --stage model     # feature matrix and model outputs
    python -m finguard.mlops contracts                   # both
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from finguard import config
from finguard.data import schema as S
from finguard.mlops import optional, params, version

OUT = {"data": config.TABLE_DIR / "contracts_data.json", "model": config.TABLE_DIR / "contracts_model.json"}
ART = Path(config.ROOT / "artifacts" / "phase2")
FEATURE_SAMPLE = int(params("contracts", {"feature_sample": 100_000})["feature_sample"])

CARD4 = ["visa", "mastercard", "american express", "discover"]
CARD6 = ["debit", "credit", "debit or credit", "charge card"]
ID_TEXT_DOMAINS = {
    "id_12": ["Found", "NotFound"], "id_15": ["Found", "New", "Unknown"], "id_16": ["Found", "NotFound"],
    "id_23": ["IP_PROXY:ANONYMOUS", "IP_PROXY:HIDDEN", "IP_PROXY:TRANSPARENT"], "id_27": ["Found", "NotFound"],
    "id_28": ["Found", "New"], "id_29": ["Found", "NotFound"],
    "id_34": ["match_status:-1", "match_status:0", "match_status:1", "match_status:2"],
    "id_35": ["F", "T"], "id_36": ["F", "T"], "id_37": ["F", "T"], "id_38": ["F", "T"],
}
DERIVED = ["has_identity", "day", "rel_hour", "rel_weekday"]


class ContractError(RuntimeError):
    """A contract failed where the pipeline must stop (e.g. before training on a bad matrix)."""


def available() -> bool:
    return optional("pandera.pandas") is not None


def _pa():
    pa = optional("pandera.pandas")
    if pa is None:
        raise ImportError("pandera is not installed: pip install \"pandera>=0.26\" (or pip install -e \".[mlops]\")")
    return pa


# --------------------------------------------------------------------------- schemas
def _finite(pa):
    return pa.Check(lambda s: ~np.isinf(s.astype("float64")), name="finite", error="no +/-inf")


def _txn_columns(pa, split: str) -> dict:
    warn = {"raise_warning": True}
    cols = {config.KEY: pa.Column("int64", unique=True, nullable=False)}
    if split == "train":
        cols[config.TARGET] = pa.Column("int8", pa.Check.isin([0, 1]), nullable=False)
    cols |= {
        config.TIME: pa.Column("float64", pa.Check.ge(0), nullable=False),
        "TransactionAmt": pa.Column("float64", [pa.Check.gt(0), _finite(pa)], nullable=False),
        "ProductCD": pa.Column(str, pa.Check.isin(list("WCRHS")), nullable=False),
        "card1": pa.Column("float32", pa.Check.in_range(1000, 99_999), nullable=True),
        "card4": pa.Column(str, pa.Check.isin(CARD4, **warn), nullable=True),
        "card6": pa.Column(str, pa.Check.isin(CARD6, **warn), nullable=True),
        "P_emaildomain": pa.Column(str, pa.Check.str_matches(r"^[a-z0-9.\-]+$", **warn), nullable=True),
        "R_emaildomain": pa.Column(str, pa.Check.str_matches(r"^[a-z0-9.\-]+$", **warn), nullable=True),
    }
    for c in ("card2", "card3", "card5", "addr1", "addr2"):
        cols[c] = pa.Column("float32", pa.Check.in_range(0, 9_999), nullable=True)
    for c in ("dist1", "dist2"):
        cols[c] = pa.Column("float32", pa.Check.ge(0), nullable=True)
    for c in S.C_COLS:
        cols[c] = pa.Column("float32", pa.Check.ge(0), nullable=True)
    for c in S.D_COLS:                                       # some D columns are legitimately negative
        cols[c] = pa.Column("float32", _finite(pa), nullable=True)
    cols["D1"] = pa.Column("float32", pa.Check.ge(0, **warn), nullable=True)   # the proxy entity uses day - D1
    for c in S.M_COLS:
        dom = ["M0", "M1", "M2"] if c == "M4" else ["T", "F"]
        cols[c] = pa.Column(str, pa.Check.isin(dom), nullable=True)
    for c in S.V_COLS:
        cols[c] = pa.Column("float32", pa.Check.ge(0, **warn), nullable=True)
    return {c: cols[c] for c in (S.TRAIN_TXN_COLS if split == "train" else S.TEST_TXN_COLS)}


def _identity_columns(pa) -> dict:
    cols = {config.KEY: pa.Column("int64", unique=True, nullable=False)}
    for c in S.ID_COLS:
        if c in S.TEXT_COLS:
            checks = [pa.Check.isin(ID_TEXT_DOMAINS[c], raise_warning=True)] if c in ID_TEXT_DOMAINS else []
            cols[c] = pa.Column(str, checks, nullable=True)
        else:
            cols[c] = pa.Column("float32", _finite(pa), nullable=True)
    cols["DeviceType"] = pa.Column(str, pa.Check.isin(["desktop", "mobile"]), nullable=True)
    cols["DeviceInfo"] = pa.Column(str, nullable=True)
    return cols


def raw_transaction_schema(split: str = "train"):
    pa = _pa()
    return pa.DataFrameSchema(
        _txn_columns(pa, split), strict=True, ordered=True, name=f"raw_transaction_{split}",
        checks=[pa.Check(lambda d: d[config.TIME].is_monotonic_increasing, name="time_sorted", raise_warning=True,
                         error="file is sorted by TransactionDT")])


def raw_identity_schema(split: str = "train"):
    pa = _pa()
    return pa.DataFrameSchema(_identity_columns(pa), strict=True, ordered=True, name=f"raw_identity_{split}")


def interim_schema(split: str = "train", origin: float | None = None):
    """The joined table written by ``finguard.data.ingest``."""
    pa = _pa()
    origin = float(origin if origin is not None else _time_origin())
    cols = _txn_columns(pa, split) | {k: v for k, v in _identity_columns(pa).items() if k != config.KEY}
    cols |= {"has_identity": pa.Column("int8", pa.Check.isin([0, 1]), nullable=False),
             "day": pa.Column("int32", pa.Check.ge(0), nullable=False),
             "rel_hour": pa.Column("int8", pa.Check.in_range(0, 23), nullable=False),
             "rel_weekday": pa.Column("int8", pa.Check.in_range(0, 6), nullable=False)}
    probe = ["DeviceType", "id_01", "id_12", "id_31"]

    def join_ok(d):            # a row without an identity record carries no identity value
        return (d["has_identity"] == 1) | d[probe].isna().all(axis=1)

    def day_ok(d):
        return d["day"] == ((d[config.TIME] - origin) // config.SECONDS_PER_DAY).astype("int64")

    def hour_ok(d):
        return d["rel_hour"] == ((d[config.TIME] // 3600) % 24).astype("int64")

    checks = [pa.Check(join_ok, name="identity_join", error="identity values only on rows with has_identity = 1"),
              pa.Check(day_ok, name="day_from_time", error=f"day = (TransactionDT - {origin:.0f}) // 86400"),
              pa.Check(hour_ok, name="rel_hour_from_time", error="rel_hour = TransactionDT // 3600 mod 24")]
    if split == "train":
        last = max(hi for _, hi in config.SPLIT_DAYS.values())
        checks.append(pa.Check(lambda d: int(d["day"].max()) >= last, name="split_coverage",
                               error=f"labelled days reach the end of the time split (day {last})"))
    return pa.DataFrameSchema(cols, strict=True, name=f"interim_{split}", checks=checks)


def feature_schema(columns: list[str], categorical: list[str] = ()):
    """The model input: exactly the model's columns, in order, float32, finite."""
    pa = _pa()
    cats = set(categorical)
    nonneg = ("entity_prev_txn", "entity_secs_since_prev", "entity_txn_24h", "entity_txn_7d", "entity_n_devices",
              "entity_n_emails", "entity_age_days", "n_missing", "amt_decimals")
    cols = {}
    for c in columns:
        checks = [_finite(pa)]
        if c in cats:
            checks.append(pa.Check(lambda s: (s >= 0) & (s == np.floor(s)), name="category_code", error="codes are integers >= 0"))
        elif c in nonneg or c.startswith("freq_"):
            checks.append(pa.Check.ge(0))
        elif c == "entity_amt_ratio" or c == "TransactionAmt":
            checks.append(pa.Check.gt(0))
        elif c.startswith("block_present_") or c in ("has_identity", "email_match"):
            checks.append(pa.Check.isin([0.0, 1.0]))
        elif c == "rel_hour":
            checks.append(pa.Check.in_range(0, 23))
        elif c == "rel_weekday":
            checks.append(pa.Check.in_range(0, 6))
        cols[c] = pa.Column("float32", checks, nullable=True)
    return pa.DataFrameSchema(cols, strict=True, ordered=True, name="features")


def score_schema(high: float, medium: float, period: str = "test"):
    """Model outputs for one period: calibrated probability, 1-99 score and band."""
    pa = _pa()

    def band_ok(d):
        want = np.where(d["raw_score"] >= high, "High", np.where(d["raw_score"] >= medium, "Medium", "Low"))
        return pd.Series(want == d["band"].to_numpy(), index=d.index)

    def monotone(d):
        s = d.sort_values("raw_score", kind="stable")["probability"].to_numpy()
        return bool((np.diff(s) >= -1e-12).all())

    def share(b: str, lo: float, hi: float):
        return pa.Check(lambda d: lo <= float((d["band"] == b).mean()) <= hi, raise_warning=True,
                        name=f"{b.lower()}_share", error=f"{b} share of {period} rows in [{lo:.1%}, {hi:.1%}]")

    return pa.DataFrameSchema({
        "transaction_id": pa.Column("int64", unique=True),
        "raw_score": pa.Column("float64", _finite(pa)),
        "probability": pa.Column("float64", pa.Check.in_range(0, 1)),
        "score": pa.Column("float64", pa.Check.in_range(1, 99)),
        "band": pa.Column(str, pa.Check.isin(["High", "Medium", "Low"])),
    }, strict=True, name=f"scores_{period}", checks=[
        pa.Check(band_ok, name="band_matches_thresholds", error="band follows the frozen raw-score thresholds"),
        pa.Check(monotone, name="calibration_monotone", error="calibrated probability never falls as the raw score rises"),
        share("High", 0.005, 0.02), share("Medium", 0.02, 0.08)])


# --------------------------------------------------------------------------- running
def validate(schema, df: pd.DataFrame, sampled: bool = False) -> dict:
    """Validate lazily: every failure is collected, warn-level checks are recorded, nothing is raised."""
    pa = _pa()
    t0 = time.time()
    res = {"contract": schema.name, "rows": int(len(df)), "columns": int(df.shape[1]), "sampled": sampled,
           "passed": True, "errors": 0, "failures": [], "warnings": []}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            schema.validate(df, lazy=True)
        except pa.errors.SchemaErrors as exc:
            fc = exc.failure_cases
            res["passed"], res["errors"] = False, int(len(fc))
            by_check = fc.groupby(["column", "check"], dropna=False).size().sort_values(ascending=False)
            for (col, chk), n in by_check.head(20).items():
                example = fc[(fc["column"].astype(str) == str(col)) & (fc["check"].astype(str) == str(chk))]["failure_case"].head(3)
                res["failures"].append({"column": None if pd.isna(col) else str(col), "check": str(chk), "cases": int(n),
                                        "examples": [str(x) for x in example]})
    res["warnings"] = sorted({str(w.message).splitlines()[0][:300] for w in caught
                              if w.category.__name__ == "SchemaWarning"})
    res["seconds"] = round(time.time() - t0, 1)
    return res


def _time_origin() -> float:
    m = config.INTERIM_DIR / "manifest.json"
    if m.exists():
        return float(json.loads(m.read_text(encoding="utf-8")).get("time_origin_seconds", config.SECONDS_PER_DAY))
    return float(config.SECONDS_PER_DAY)


def _cross(name: str, ok: bool, detail: str) -> dict:
    return {"contract": name, "rows": None, "columns": None, "sampled": False, "passed": bool(ok), "errors": 0 if ok else 1,
            "failures": [] if ok else [{"column": None, "check": name, "cases": 1, "examples": [detail]}],
            "warnings": [], "seconds": 0.0, "detail": detail}


def check_data(log=print) -> list[dict]:
    """Raw files (as read, identity names fixed) and the interim tables."""
    from finguard.data.ingest import load_table, normalize_identity_columns, read_raw
    out = []
    for split in ("train", "test"):
        t = read_raw(f"{split}_transaction")
        out.append(validate(raw_transaction_schema(split), t))
        log(_line(out[-1]))
        del t
        i, _ = normalize_identity_columns(read_raw(f"{split}_identity"))
        out.append(validate(raw_identity_schema(split), i))
        log(_line(out[-1]))
        del i
    origin = _time_origin()
    days = {}
    for split in ("train", "test"):
        df = load_table(split)
        out.append(validate(interim_schema(split, origin), df))
        log(_line(out[-1]))
        days[split] = (int(df["day"].min()), int(df["day"].max()), list(df.columns))
        del df
    later = days["test"][0] > days["train"][1]
    out.append(_cross("interim_time_disjoint", later, f"train days {days['train'][0]}-{days['train'][1]}, "
                                                      f"later file days {days['test'][0]}-{days['test'][1]}"))
    parity = sorted(set(days["train"][2]) ^ set(days["test"][2])) == [config.TARGET]
    out.append(_cross("interim_column_parity", parity, "train and later file share every column except the label"))
    for r in out[-2:]:
        log(_line(r))
    return out


def feature_frame(fs, columns: list[str], sample: int | None = FEATURE_SAMPLE, seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    n = len(fs.y)
    rows = np.arange(n) if not sample or sample >= n else np.sort(np.random.default_rng(seed).choice(n, sample, replace=False))
    return pd.DataFrame(np.asarray(fs.X[rows][:, fs.cols(columns)]), columns=columns)


def check_features(fs, columns: list[str] | None = None, categorical: list[str] | None = None,
                   sample: int | None = FEATURE_SAMPLE, raise_on_fail: bool = False) -> dict:
    """Feature-matrix contract on a row sample, plus a whole-matrix scan for infinities."""
    columns = list(columns or fs.columns)
    categorical = [c for c in (categorical if categorical is not None else fs.categorical) if c in columns]
    res = validate(feature_schema(columns, categorical), feature_frame(fs, columns, sample),
                   sampled=bool(sample and sample < len(fs.y)))
    res["contract"] = f"features_{fs.scheme}"
    n_inf, step = 0, 50_000
    for a in range(0, len(fs.y), step):
        n_inf += int(np.isinf(np.asarray(fs.X[a:a + step])).sum())
    res["whole_matrix_infinities"] = n_inf
    if n_inf:
        res["passed"] = False
        res["failures"].append({"column": None, "check": "whole_matrix_finite", "cases": n_inf, "examples": []})
    if raise_on_fail and not res["passed"]:
        raise ContractError(f"feature contract failed: {res['failures'][:3]}")
    return res


def scores_frame(preds: dict, period: str, transaction_id: np.ndarray, iso, high: float, medium: float) -> pd.DataFrame:
    raw = np.asarray(preds[f"{period}_raw"], dtype="float64")
    p = np.asarray(iso.predict(raw), dtype="float64")
    band = np.where(raw >= high, "High", np.where(raw >= medium, "Medium", "Low"))
    return pd.DataFrame({"transaction_id": np.asarray(transaction_id)[preds[f"{period}_idx"]].astype("int64"),
                         "raw_score": raw, "probability": p, "score": np.round(100 * np.clip(p, 0.01, 0.99), 1),
                         "band": band.astype(object)})


def check_model(sample: int | None = FEATURE_SAMPLE, log=print) -> list[dict]:
    """The model's input matrix and its outputs on the validation and test months."""
    import pickle

    from finguard.models import dataset
    sel = json.loads((ART / "selection.json").read_text(encoding="utf-8"))["codes_as"]
    cols = json.loads((ART / f"lgbm_{sel}" / "columns.json").read_text(encoding="utf-8"))
    fs = dataset.load("main", sel)
    out = [check_features(fs, cols["columns"], cols["categorical"], sample)]
    log(_line(out[-1]))
    with open(ART / "model_bundle.pkl", "rb") as f:
        bundle = pickle.load(f)
    with np.load(ART / f"lgbm_{sel}" / "preds.npz") as z:
        preds = {k: z[k] for k in z.files}
    b = bundle["bands"]
    for period in ("valid", "test"):
        out.append(validate(score_schema(b.high, b.medium, period),
                            scores_frame(preds, period, fs.transaction_id, bundle["isotonic"], b.high, b.medium)))
        log(_line(out[-1]))
    return out


def _line(r: dict) -> str:
    status = "PASS" if r["passed"] else "FAIL"
    warn = f", {len(r['warnings'])} warning(s)" if r["warnings"] else ""
    rows = f"{r['rows']:,} rows" if r.get("rows") is not None else r.get("detail", "")
    fails = "; ".join(f"{f['column'] or '-'}:{f['check']} x{f['cases']}" for f in r["failures"][:3])
    return f"  {status}  {r['contract']:<26} {rows}{warn}" + (f"  [{fails}]" if fails else "")


def run(stage: str = "all", sample: int | None = FEATURE_SAMPLE, out: dict[str, Path] | None = None, log=print) -> dict:
    """Run the contracts of one stage ("data", "model") or both; one result file per stage."""
    pa = _pa()
    out = out or OUT
    reports = {}
    for st in (("data", "model") if stage == "all" else (stage,)):
        results = check_data(log) if st == "data" else check_model(sample, log)
        reports[st] = {"run_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "pandera": getattr(pa, "__version__", None) or version("pandera"), "stage": st,
                       "passed": all(r["passed"] for r in results), "contracts": results}
        out[st].parent.mkdir(parents=True, exist_ok=True)
        out[st].write_text(json.dumps(reports[st], indent=1, default=str), encoding="utf-8")
    return {"passed": all(r["passed"] for r in reports.values()), "stages": reports}


def gate_check(out: dict[str, Path] | None = None) -> tuple[bool, str]:
    """For ``finguard.gate``: did the last contract runs pass, for both the data and the model?"""
    out = out or OUT
    parts, ok = [], True
    for st, path in out.items():
        if not path.exists():
            ok = False
            parts.append(f"{st}: not run (python -m finguard.mlops contracts --stage {st})")
            continue
        r = json.loads(path.read_text(encoding="utf-8"))
        failed = [c["contract"] for c in r["contracts"] if not c["passed"]]
        n_warn = sum(len(c["warnings"]) for c in r["contracts"])
        ok &= not failed
        parts.append(f"{st}: {len(r['contracts'])} contracts, {len(failed)} failed"
                     f"{' (' + ', '.join(failed[:4]) + ')' if failed else ''}, {n_warn} warning(s), {r['run_at']}")
    return ok, "; ".join(parts)
