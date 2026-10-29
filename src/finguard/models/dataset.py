"""Build, cache and load the Phase 2 feature matrix.

One cache per (split scheme, code encoding). The FeatureBuilder is fitted on the scheme's
training window only, so the label-delay scheme gets its own vocabularies and counts.

CLI:  python -m finguard.models.dataset --scheme main --codes numeric
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from finguard import config, split
from finguard.data.ingest import load_table
from finguard.features.build import FeatureBuilder
from finguard.features.proxy_entity import build_proxy_entity

PROCESSED_DIR = Path(config.ROOT / "data" / "processed")


@dataclass
class FeatureSet:
    """Feature matrix plus everything needed to evaluate it honestly."""
    X: np.ndarray                 # float32, rows in file order, columns = builder.columns_
    columns: list[str]
    categorical: list[str]
    y: np.ndarray                 # int8 label
    day: np.ndarray               # relative day (split only, never a feature)
    transaction_id: np.ndarray
    entity: np.ndarray            # proxy-entity key (object, None when missing) -- evaluation only
    scheme: str
    builder: FeatureBuilder

    @property
    def masks(self) -> dict[str, np.ndarray]:
        return split.masks(self.day, self.scheme)

    def cols(self, names: list[str]) -> list[int]:
        idx = {c: i for i, c in enumerate(self.columns)}
        return [idx[c] for c in names]

    def frame(self, rows: np.ndarray, names: list[str] | None = None) -> pd.DataFrame:
        names = names or self.columns
        return pd.DataFrame(self.X[np.ix_(rows, self.cols(names))] if rows.dtype != bool
                            else self.X[rows][:, self.cols(names)], columns=names)


def build(scheme: str = "main", codes_as: str = "numeric", df: pd.DataFrame | None = None) -> FeatureSet:
    df = load_table("train") if df is None else df
    split.check(df["day"].to_numpy(), scheme)
    fb = FeatureBuilder(fit_days=split.SCHEMES[scheme]["train"], codes_as=codes_as).fit(df)
    feats = fb.transform(df)
    X = np.ascontiguousarray(feats.to_numpy(dtype="float32"))
    del feats
    ent = build_proxy_entity(df)
    return FeatureSet(X=X, columns=fb.columns_, categorical=fb.categorical_,
                      y=df[config.TARGET].to_numpy(dtype="int8"), day=df["day"].to_numpy(dtype="int32"),
                      transaction_id=df[config.KEY].to_numpy(dtype="int64"),
                      entity=ent.astype(object).where(ent.notna(), None).to_numpy(), scheme=scheme, builder=fb)


def cache_dir(scheme: str, codes_as: str) -> Path:
    return PROCESSED_DIR / f"{scheme}_{codes_as}"


def save(fs: FeatureSet, codes_as: str) -> Path:
    d = cache_dir(fs.scheme, codes_as)
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "X.npy", fs.X)
    meta = {k: getattr(fs, k) for k in ("columns", "categorical", "y", "day", "transaction_id", "entity", "scheme", "builder")}
    with open(d / "meta.pkl", "wb") as f:
        pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)
    (d / "summary.json").write_text(json.dumps({
        "scheme": fs.scheme, "codes_as": codes_as, "shape": list(fs.X.shape), "n_categorical": len(fs.categorical),
        "fit_window_days": list(fs.builder.fit_days), "fit_rows": fs.builder.fit_rows_,
        "n_nan_blocks": len(fs.builder.blocks_), "n_missing_patterns": len(fs.builder.patterns_)}, indent=1))
    return d


def load(scheme: str = "main", codes_as: str = "numeric", mmap: bool = True) -> FeatureSet:
    d = cache_dir(scheme, codes_as)
    with open(d / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    X = np.load(d / "X.npy", mmap_mode="r" if mmap else None)
    return FeatureSet(X=X, **meta)


def ensure(scheme: str = "main", codes_as: str = "numeric") -> FeatureSet:
    """Load the cache, building it first if it does not exist."""
    if not (cache_dir(scheme, codes_as) / "X.npy").exists():
        save(build(scheme, codes_as), codes_as)
    return load(scheme, codes_as)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scheme", default="main", choices=list(split.SCHEMES))
    ap.add_argument("--codes", default="numeric", choices=["numeric", "category"])
    a = ap.parse_args()
    t0 = time.time()
    d = save(build(a.scheme, a.codes), a.codes)
    print(f"wrote {d} in {time.time() - t0:.0f}s")
    print((d / "summary.json").read_text())


if __name__ == "__main__":
    main()
