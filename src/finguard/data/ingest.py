"""Load the raw IEEE-CIS CSV files, fix schema defects, join transaction and identity tables.

Usage
-----
    python -m finguard.data.ingest            # uses FINGUARD_RAW_DIR / FINGUARD_INTERIM_DIR
    python -m finguard.data.ingest --check-md5

Outputs (in the interim directory)
    train.parquet, test.parquet   joined tables (pickle fallback if pyarrow is missing)
    manifest.json                 shapes, checksums, schema fixes applied
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from finguard import config
from finguard.data import schema


# --------------------------------------------------------------------------- reading
def _dtype_map(columns: list[str]) -> dict[str, str]:
    """Compact dtypes: float32 for anonymised numerics, full precision where it matters."""
    dt = {}
    for c in columns:
        if c in schema.TEXT_COLS or c.replace("-", "_") in schema.TEXT_COLS:
            dt[c] = "str"
        elif c == config.KEY:
            dt[c] = "int64"
        elif c == config.TARGET:
            dt[c] = "int8"
        elif c in (config.TIME, "TransactionAmt"):
            dt[c] = "float64"      # TransactionDT exceeds 2**24; amounts keep their decimals
        else:
            dt[c] = "float32"
    return dt


def md5sum(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def read_raw(name: str, raw_dir: Path | None = None) -> pd.DataFrame:
    """Read one raw file with explicit dtypes."""
    raw_dir = Path(raw_dir or config.RAW_DIR)
    path = raw_dir / config.RAW_FILES[name]
    header = pd.read_csv(path, nrows=0).columns.tolist()
    return pd.read_csv(path, dtype=_dtype_map(header))


# --------------------------------------------------------------------------- schema fixes
def normalize_identity_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Rename `id-XX` to `id_XX`.

    test_identity.csv uses hyphens while train_identity.csv uses underscores. Without this
    fix, concatenating train and test silently creates 38 extra, half-empty columns.
    """
    renamed = [c for c in df.columns if c.startswith("id-")]
    return df.rename(columns={c: c.replace("-", "_") for c in renamed}), renamed


def join_transaction_identity(txn: pd.DataFrame, ident: pd.DataFrame) -> pd.DataFrame:
    """Left-join identity onto transactions; fails loudly on key problems or row inflation."""
    if not txn[config.KEY].is_unique or not ident[config.KEY].is_unique:
        raise ValueError("TransactionID is not unique in one of the tables")
    orphans = ~ident[config.KEY].isin(txn[config.KEY])
    if orphans.any():
        raise ValueError(f"{int(orphans.sum())} identity rows have no matching transaction")
    out = txn.merge(ident, on=config.KEY, how="left", validate="one_to_one")
    if len(out) != len(txn):
        raise AssertionError("join changed the number of transactions")
    flag = out[config.KEY].isin(ident[config.KEY]).astype("int8").rename("has_identity")
    return pd.concat([out, flag], axis=1)


def add_time_index(df: pd.DataFrame, origin: float) -> pd.DataFrame:
    """Relative day / hour / weekday from TransactionDT (seconds from an undisclosed reference).

    `origin` must be the minimum TransactionDT of the TRAINING file so train and test share
    one clock. Hour and weekday are relative: the real clock and time zone are unknown.
    """
    dt = df[config.TIME]
    extra = pd.DataFrame({
        "day": ((dt - origin) // config.SECONDS_PER_DAY).astype("int32"),
        "rel_hour": ((dt // 3600) % 24).astype("int8"),
        "rel_weekday": ((dt // config.SECONDS_PER_DAY) % 7).astype("int8"),
    }, index=df.index)
    return pd.concat([df.drop(columns=[c for c in extra.columns if c in df.columns]), extra], axis=1)


# --------------------------------------------------------------------------- persistence
def _has_pyarrow() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


def save_table(df: pd.DataFrame, name: str, out_dir: Path | None = None) -> Path:
    out_dir = Path(out_dir or config.INTERIM_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    if _has_pyarrow():
        path = out_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
    else:  # fallback for environments without pyarrow
        path = out_dir / f"{name}.pkl"
        df.to_pickle(path)
    return path


def load_table(name: str, in_dir: Path | None = None, columns: list[str] | None = None) -> pd.DataFrame:
    in_dir = Path(in_dir or config.INTERIM_DIR)
    pq, pk = in_dir / f"{name}.parquet", in_dir / f"{name}.pkl"
    if pq.exists():
        return pd.read_parquet(pq, columns=columns)
    if pk.exists():
        df = pd.read_pickle(pk)
        return df[columns] if columns else df
    raise FileNotFoundError(f"{name} not found in {in_dir}; run `python -m finguard.data.ingest` first")


# --------------------------------------------------------------------------- pipeline
def build_interim(raw_dir: Path | None = None, out_dir: Path | None = None,
                  check_md5: bool = False, verbose: bool = True) -> dict:
    t0 = time.time()
    raw_dir = Path(raw_dir or config.RAW_DIR)
    log = (lambda *a: print(f"[{time.time() - t0:5.1f}s]", *a, flush=True)) if verbose else (lambda *a: None)
    manifest: dict = {"raw_dir": str(raw_dir), "files": {}, "fixes": {}}

    frames = {}
    for name in ["train_transaction", "train_identity", "test_transaction", "test_identity"]:
        path = raw_dir / config.RAW_FILES[name]
        entry = {"bytes": path.stat().st_size}
        if check_md5:
            entry["md5"] = md5sum(path)
            entry["md5_matches_release"] = entry["md5"] == config.RAW_MD5[name]
        frames[name] = read_raw(name, raw_dir)
        entry["shape"] = list(frames[name].shape)
        manifest["files"][name] = entry
        log("read", name, frames[name].shape)

    frames["test_identity"], renamed = normalize_identity_columns(frames["test_identity"])
    manifest["fixes"]["test_identity_renamed_columns"] = len(renamed)
    log(f"renamed {len(renamed)} test identity columns id-XX -> id_XX")

    train = join_transaction_identity(frames["train_transaction"], frames["train_identity"])
    test = join_transaction_identity(frames["test_transaction"], frames["test_identity"])
    origin = float(frames["train_transaction"][config.TIME].min())
    train, test = add_time_index(train, origin), add_time_index(test, origin)
    manifest["time_origin_seconds"] = origin

    extra = sorted(set(train.columns) ^ set(test.columns))
    if extra != [config.TARGET]:
        raise AssertionError(f"train/test columns differ beyond the target: {extra}")

    manifest["train_shape"], manifest["test_shape"] = list(train.shape), list(test.shape)
    manifest["storage"] = "parquet" if _has_pyarrow() else "pickle (pyarrow not installed)"
    manifest["train_path"] = str(save_table(train, "train", out_dir))
    manifest["test_path"] = str(save_table(test, "test", out_dir))
    out = Path(out_dir or config.INTERIM_DIR)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log("saved", manifest["train_path"], manifest["test_path"])
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--check-md5", action="store_true")
    a = ap.parse_args()
    build_interim(a.raw, a.out, a.check_md5)


if __name__ == "__main__":
    main()
