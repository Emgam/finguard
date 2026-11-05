"""Where the API's evidence comes from: the real data, model and index, or the synthetic demo world.

``RealWorld`` loads the same ``DataContext`` the agents use (labelled table, calibrated scorer,
proxy-entity store, deployed retriever, case store) and builds the alert queue: the test month
(relative days 152-181) scored once by the served model, from the Phase 2 predictions, with the
frozen validation-month bands. The queue is what a reviewer would have seen; labels are never read.

It also checks the registry: the served model is the one in the Phase 2 bundle (its calibrator and
bands belong to it), and the registry's "champion" alias must point at that same model file
(SHA-256). With ``FINGUARD_REQUIRE_CHAMPION=1`` the API refuses to become ready otherwise.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import numpy as np
import pandas as pd

from finguard import config
from finguard.agents.case_store import CASE_DIR, CaseStore
from finguard.agents.evidence import Evidence
from finguard.agents.tools import DataContext, Tools

QUEUE_COLUMNS = ["transaction_id", "relative_day", "relative_hour", "amount", "product_code", "card_network", "card_type",
                 "has_identity_record", "score", "probability_displayed", "band"]


class LockedTools(Tools):
    """Tool calls are serialised: the scorer and the re-ranker are shared by concurrent requests."""

    def __init__(self, ctx, functions=None):
        super().__init__(ctx, functions)
        self._lock = threading.Lock()

    def call(self, evidence: Evidence, tool: str, args: dict | None = None) -> dict:
        with self._lock:
            return super().call(evidence, tool, args)


def champion_status(model_sha12: str | None, registry_path: Path | None = None) -> dict:
    reg_path = Path(registry_path or config.ROOT / "artifacts" / "registry.json")
    if not reg_path.exists():
        return {"alias": "champion", "model_id": None, "matches_served_model": False,
                "note": "no model registry (artifacts/registry.json); run `python -m finguard.gate --register`"}
    reg = json.loads(reg_path.read_text(encoding="utf-8"))
    mid = reg.get("aliases", {}).get("champion")
    if mid is None:
        return {"alias": "champion", "model_id": None, "matches_served_model": False,
                "note": "no model is promoted to champion; serving the Phase 2 bundle model "
                        "(promote with `python -m finguard.registry promote MODEL_ID --alias champion --approved-by ...`)"}
    entry = next((m for m in reg.get("models", []) if m["model_id"] == mid), {})
    match = bool(model_sha12) and (entry.get("model_sha256") or "")[:12] == model_sha12
    return {"alias": "champion", "model_id": mid, "matches_served_model": match,
            "note": "the champion is the served model" if match else
                    "the champion in the registry is NOT the model in the Phase 2 bundle; rebuild the bundle or promote it"}


class RealWorld:
    name = "real"
    synthetic = False

    def __init__(self, case_dir: Path | None = None, require_champion: bool | None = None):
        self.case_dir = Path(case_dir or CASE_DIR)
        self.require_champion = (os.environ.get("FINGUARD_REQUIRE_CHAMPION", "0") == "1"
                                 if require_champion is None else require_champion)
        self.ctx = self.tools = self.queue = None
        self._champion: dict | None = None

    def load(self) -> None:
        ctx = DataContext(case_store=CaseStore(self.case_dir / "case_store.sqlite")).warm_up()
        self._champion = champion_status(getattr(ctx.scorer, "model_sha", None))
        if self.require_champion and not self._champion["matches_served_model"]:
            raise RuntimeError("FINGUARD_REQUIRE_CHAMPION=1: " + self._champion["note"])
        self.ctx, self.tools = ctx, LockedTools(ctx)
        self.queue = self._queue(ctx)

    @staticmethod
    def _queue(ctx: DataContext) -> pd.DataFrame:
        from finguard import phase2
        from finguard.models import dataset
        from finguard.models.calibration import display_probability
        scorer = ctx.scorer
        name = scorer.model_path.parent.name                     # e.g. lgbm_numeric
        fs = dataset.load("main", name.split("_", 1)[1])
        ids = np.asarray(fs.transaction_id)[np.asarray(fs.masks["test"])]
        raw = np.asarray(phase2.load_preds(name)["test_raw"], dtype="float64")
        if len(raw) != len(ids):
            raise RuntimeError(f"{name}: {len(raw)} test predictions for {len(ids)} test transactions; rerun Phase 2")
        p = scorer.iso.predict(raw)
        t = ctx.table.iloc[ctx.position.reindex(ids).to_numpy(dtype=int)]
        q = pd.DataFrame({"transaction_id": ids.astype("int64"), "relative_day": t["day"].to_numpy(dtype="int64"),
                          "relative_hour": t["rel_hour"].to_numpy(dtype="int64") if "rel_hour" in t else
                          ((t[config.TIME].to_numpy() // 3600) % 24).astype("int64"),
                          "amount": t["TransactionAmt"].to_numpy(dtype="float64"),
                          "product_code": t["ProductCD"].astype("string").to_numpy(),
                          "card_network": t["card4"].astype("string").to_numpy(),
                          "card_type": t["card6"].astype("string").to_numpy(),
                          "has_identity_record": t["has_identity"].to_numpy() == 1 if "has_identity" in t else None,
                          "score": np.round(100 * np.clip(p, 0.01, 0.99), 1),
                          "probability_displayed": [display_probability(v) for v in p],
                          "band": scorer.bands.assign(raw), "_time": t[config.TIME].to_numpy(dtype="float64")})
        return q.reset_index(drop=True)

    def versions(self) -> dict:
        return self.ctx.versions()

    def champion(self) -> dict:
        return self._champion or {}

    def components(self) -> dict:
        ctx = self.ctx
        retriever, variant = ctx.retriever
        return {"transactions": int(len(ctx.table)), "model": ctx.scorer.model_version,
                "knowledge_base": f"{retriever.index.version} ({len(retriever.index.chunks)} passages, {variant})",
                "alert_queue": int(self.queue["band"].isin(["High", "Medium"]).sum()),
                "case_store": str(ctx.case_store.path)}

    def period(self) -> str:
        lo, hi = config.SPLIT_DAYS["test"]
        return f"test month, relative days {lo}-{hi} (scored once by the served model; labels never shown)"


class SyntheticWorld:
    name = "synthetic"
    synthetic = True

    def __init__(self, case_dir: Path | None = None):
        self.case_dir = Path(case_dir or CASE_DIR) / "synthetic"
        self.ctx = self.tools = self.queue = None

    def load(self) -> None:
        from finguard.api.synthetic import SYNTHETIC_FUNCTIONS, SyntheticContext
        self.ctx = SyntheticContext(self.case_dir)
        self.tools = LockedTools(self.ctx, functions=SYNTHETIC_FUNCTIONS)
        f = self.ctx.frame
        self.queue = f[QUEUE_COLUMNS].assign(_time=f["relative_day"] * 86400.0 + f["relative_hour"] * 3600.0)

    def versions(self) -> dict:
        return self.ctx.versions()

    def champion(self) -> dict:
        return {"alias": "champion", "model_id": None, "matches_served_model": False,
                "note": "synthetic world: no model is served"}

    def components(self) -> dict:
        retriever, variant = self.ctx.retriever
        return {"transactions": int(len(self.ctx.frame)), "model": "none (synthetic scores)",
                "knowledge_base": f"{retriever.index.version} (project documents only, {len(retriever.index.chunks)} passages)",
                "alert_queue": int(self.queue["band"].isin(["High", "Medium"]).sum()),
                "case_store": str(self.ctx.case_store.path)}

    def period(self) -> str:
        return "synthetic demo transactions (invented; not IEEE-CIS data)"


def make_world(name: str | None = None, case_dir: Path | None = None):
    name = (name or os.environ.get("FINGUARD_WORLD") or "real").lower()
    if name == "real":
        return RealWorld(case_dir)
    if name == "synthetic":
        return SyntheticWorld(case_dir)
    raise ValueError(f"unknown world '{name}': use real or synthetic")
