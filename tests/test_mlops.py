"""MLOps layer: data contracts (Pandera), drift monitoring (Evidently + PSI), tracking and the model
registry (MLflow), DVC metric files and the dvc.yaml checks. Synthetic data only; every test that
needs an optional library skips when it is not installed (CI's mlops job installs them all)."""
from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import _synthetic
from finguard import config
from finguard.data import schema as S
from finguard.mlops import contracts, drift, metrics, pipeline, tracking

DAY = 86_400


def need(module: str):
    return pytest.importorskip(module)


@pytest.fixture
def small_split(monkeypatch):
    """The synthetic frame covers 60 days: shrink the time split to fit it."""
    monkeypatch.setattr(config, "SPLIT_DAYS", {"train": (0, 39), "valid": (40, 49), "test": (50, 59)})


@pytest.fixture
def interim():
    return _synthetic.frame(n=3000, days=60, seed=1)


# --------------------------------------------------------------------------- contracts
def test_interim_contract_passes_on_a_correct_table(small_split, interim):
    need("pandera.pandas")
    res = contracts.validate(contracts.interim_schema("train", origin=DAY), interim)
    assert res["passed"], res["failures"]
    # the synthetic id_15 / id_28 / id_34 values are outside the documented domains: reported, not failed
    assert any("id_34" in w for w in res["warnings"]), res["warnings"]


@pytest.mark.parametrize("defect, check", [
    (lambda d: d.assign(TransactionID=np.r_[d.TransactionID.iloc[:1], d.TransactionID.iloc[:-1]]), "field_uniqueness"),
    (lambda d: d.assign(TransactionAmt=-d.TransactionAmt), "greater_than(0)"),
    (lambda d: d.assign(day=d["day"] + 1), "day = (TransactionDT"),
    (lambda d: d.assign(has_identity=np.int8(0)), "identity values only on rows"),
    (lambda d: d.assign(ProductCD="Z"), "isin"),
    (lambda d: d.drop(columns=["V1"]), "column_in_dataframe"),
    (lambda d: d.assign(extra=1.0), "column_in_schema"),
])
def test_interim_contract_catches_each_defect(small_split, interim, defect, check):
    need("pandera.pandas")
    res = contracts.validate(contracts.interim_schema("train", origin=DAY), defect(interim.copy()))
    assert not res["passed"]
    assert any(check in f["check"] or check in str(f["column"]) or check.split("(")[0] in f["check"] for f in res["failures"]), res["failures"]


def test_split_coverage_is_checked(interim, monkeypatch):
    need("pandera.pandas")
    monkeypatch.setattr(config, "SPLIT_DAYS", {"train": (0, 120), "valid": (121, 151), "test": (152, 181)})
    res = contracts.validate(contracts.interim_schema("train", origin=DAY), interim)
    assert not res["passed"] and any(f["check"].startswith("labelled days reach") for f in res["failures"])


def test_raw_contracts_need_the_exact_columns(interim):
    need("pandera.pandas")
    txn = interim[S.TRAIN_TXN_COLS]
    assert contracts.validate(contracts.raw_transaction_schema("train"), txn)["passed"]
    swapped = txn[[S.TRAIN_TXN_COLS[1], S.TRAIN_TXN_COLS[0]] + S.TRAIN_TXN_COLS[2:]]
    assert not contracts.validate(contracts.raw_transaction_schema("train"), swapped)["passed"]
    ident = interim.loc[interim.has_identity == 1, S.IDENTITY_COLS]
    assert contracts.validate(contracts.raw_identity_schema("train"), ident)["passed"]
    hyphen = ident.rename(columns={c: c.replace("_", "-") for c in S.ID_COLS})       # the test file's raw names
    assert not contracts.validate(contracts.raw_identity_schema("test"), hyphen)["passed"]


def test_a_new_card_network_is_a_warning_not_an_error(interim):
    need("pandera.pandas")
    txn = interim[S.TRAIN_TXN_COLS].copy()
    txn.loc[txn.index[:5], "card4"] = "unionpay"
    res = contracts.validate(contracts.raw_transaction_schema("train"), txn)
    assert res["passed"] and any("card4" in w for w in res["warnings"])


def _features(interim):
    from finguard.features.build import FeatureBuilder
    fb = FeatureBuilder(fit_days=(0, 39)).fit(interim)
    return fb.transform(interim), fb


def test_feature_contract(interim):
    need("pandera.pandas")
    feats, fb = _features(interim)
    sch = contracts.feature_schema(fb.columns_, fb.categorical_)
    assert contracts.validate(sch, feats)["passed"]
    bad = feats.copy()
    bad.loc[bad.index[0], "TransactionAmt"] = np.inf
    bad.loc[bad.index[1], "entity_txn_24h"] = -1
    bad.loc[bad.index[2], fb.categorical_[0]] = 0.5
    res = contracts.validate(sch, bad)
    assert not res["passed"] and {f["column"] for f in res["failures"]} >= {"TransactionAmt", "entity_txn_24h", fb.categorical_[0]}
    assert not contracts.validate(sch, feats[fb.columns_[::-1]])["passed"]                # order is part of the contract


def test_check_features_scans_the_whole_matrix_for_infinities(interim):
    need("pandera.pandas")
    feats, fb = _features(interim)
    X = feats.to_numpy(dtype="float32").copy()
    fs = SimpleNamespace(X=X, columns=fb.columns_, categorical=fb.categorical_, y=np.zeros(len(X)), scheme="main",
                         cols=lambda names: [fb.columns_.index(c) for c in names])
    assert contracts.check_features(fs, sample=500)["passed"]
    X[-1, 0] = np.inf                                                   # outside any small sample
    res = contracts.check_features(fs, sample=10)
    assert not res["passed"] and res["whole_matrix_infinities"] == 1
    with pytest.raises(contracts.ContractError):
        contracts.check_features(fs, sample=10, raise_on_fail=True)


def _scores(n=4000, seed=0):
    from finguard.models.calibration import fit_bands, fit_isotonic
    rs = np.random.RandomState(seed)
    y = (rs.rand(n) < 0.04).astype(int)
    raw = rs.normal(-4, 1.5, n) + 3 * y
    iso, bands = fit_isotonic(raw, y), fit_bands(raw)
    preds = {"valid_raw": raw, "valid_idx": np.arange(n)}
    frame = contracts.scores_frame(preds, "valid", np.arange(10_000, 10_000 + n), iso, bands.high, bands.medium)
    return frame, bands, iso


def test_score_contract():
    need("pandera.pandas")
    frame, bands, _ = _scores()
    sch = contracts.score_schema(bands.high, bands.medium, "valid")
    assert contracts.validate(sch, frame)["passed"]
    wrong = frame.copy()
    wrong.loc[wrong["raw_score"].idxmax(), "band"] = "Low"
    assert any(f["check"].startswith("band follows") for f in contracts.validate(sch, wrong)["failures"])
    flipped = frame.assign(probability=1 - frame["probability"])
    assert any(f["check"].startswith("calibrated probability") for f in contracts.validate(sch, flipped)["failures"])


def test_gate_check_reads_both_stage_files(tmp_path):
    out = {"data": tmp_path / "d.json", "model": tmp_path / "m.json"}
    assert not contracts.gate_check(out)[0]
    ok = {"run_at": "x", "contracts": [{"contract": "c", "passed": True, "warnings": []}]}
    out["data"].write_text(json.dumps(ok))
    assert not contracts.gate_check(out)[0]                              # model stage not run yet
    out["model"].write_text(json.dumps(ok))
    assert contracts.gate_check(out)[0]
    out["model"].write_text(json.dumps({"run_at": "x", "contracts": [{"contract": "scores_test", "passed": False, "warnings": []}]}))
    ok_, detail = contracts.gate_check(out)
    assert not ok_ and "scores_test" in detail


# --------------------------------------------------------------------------- drift
def _drift_frames(shift: float = 0.0, seed: int = 0, n: int = 3000):
    rs = np.random.RandomState(seed)
    def make(n, s, day0):
        d = pd.DataFrame({"TransactionAmt": rs.lognormal(3.5 + s, 1, n), "dist1": rs.exponential(50, n),
                          "C1": rs.poisson(2, n).astype(float), "C13": rs.poisson(5, n).astype(float),
                          "C14": rs.poisson(1, n).astype(float), "rel_hour": rs.randint(0, 24, n).astype(float)})
        p = [0.7, 0.1, 0.1, 0.05, 0.05] if not s else [0.3, 0.4, 0.1, 0.1, 0.1]
        d["ProductCD"] = rs.choice(list("WCRHS"), n, p=p)
        for c in ("card4", "card6", "P_emaildomain", "R_emaildomain", "DeviceType", "M4", "M6"):
            d[c] = rs.choice(["a", "b", None], n)
        d["has_identity"] = rs.randint(0, 2, n)
        d["day"] = day0 + rs.randint(0, 28, n)
        d[config.TARGET] = (rs.rand(n) < 0.04).astype(int)
        d["raw_score"] = rs.normal(-4 + s, 1.5, n) + 3 * d[config.TARGET]
        d[drift.SCORE] = 1 / (1 + np.exp(-d["raw_score"]))
        return d
    ref, test, later = make(n, 0.0, 121), make(n, shift, 152), make(n, shift, 212)
    high, medium = np.quantile(ref["raw_score"], [0.99, 0.95])          # bands frozen on the reference, as in Phase 2
    for d in (ref, test):
        d["band"] = np.where(d["raw_score"] >= high, "High", np.where(d["raw_score"] >= medium, "Medium", "Low"))
    return {"reference": ref, "test": test, "later": later.drop(columns=[config.TARGET, "raw_score", drift.SCORE])}


def test_psi_helpers():
    a = pd.Series(np.random.RandomState(0).choice(list("abc"), 5000))
    assert drift.psi_categorical(a, a) < 1e-9
    assert drift.psi_categorical(a, pd.Series(["a"] * 4000 + ["d"] * 1000)) > 0.25


def test_decision_policy():
    pol = drift.POLICY
    base = {"dataset": {"drifted_share": 0.0, "dataset_drift": False}, "score": {}}
    assert drift.decide(base, pol)[0] == "stable"
    assert drift.decide(base | {"score": {"psi": 0.12}}, pol)[0] == "review"
    assert drift.decide(base | {"score": {"psi": 0.30}}, pol)[0] == "retrain review"
    assert drift.decide(base | {"score": {"alert_share_current": 0.09}}, pol)[0] == "review"
    assert drift.decide(base | {"score": {"pr_auc_reference": 0.66, "pr_auc_current": 0.58}}, pol)[0] == "stable"
    assert drift.decide(base | {"score": {"pr_auc_reference": 0.66, "pr_auc_current": 0.50}}, pol)[0] == "review"
    assert drift.decide({"dataset": {"drifted_share": 0.8, "dataset_drift": True}, "score": {}}, pol)[0] == "retrain review"


def test_drift_run_writes_reports(tmp_path, monkeypatch):
    monkeypatch.setenv("FINGUARD_MLFLOW", "off")
    frames = _drift_frames(shift=0.8)
    res = drift.run(("test", "later"), html=True, frames=frames, out_dir=tmp_path, metrics_path=tmp_path / "drift.json",
                    log=lambda *a: None)
    test = res["windows"]["test"]
    drifted = {c["column"] for c in test["columns"] if c["drifted"]}
    assert {"TransactionAmt", "ProductCD"} <= drifted
    assert test["score"]["psi"] > 0.1 and test["decision"] in ("review", "retrain review")
    assert "pr_auc_current" in test["score"] and "psi" not in res["windows"]["later"]["score"]
    assert (tmp_path / "drift_test.summary.json").exists() and (tmp_path / "drift_trend.csv").exists()
    flat = json.loads((tmp_path / "drift.json").read_text())
    assert set(flat) == {"test", "later"} and "decision_level" in flat["test"]
    if drift.optional("evidently") is not None:
        assert (tmp_path / "drift_test.html").stat().st_size > 10_000 and test["evidently"]
    stable = drift.run(("test",), html=False, frames=_drift_frames(shift=0.0, seed=3), out_dir=tmp_path / "s",
                       metrics_path=tmp_path / "s.json", log=lambda *a: None)
    assert stable["windows"]["test"]["decision"] == "stable"


def test_evidently_snapshot_parsing():
    need("evidently")
    f = _drift_frames(shift=0.8)
    snap = drift.evidently_report(f["reference"], f["test"], 0.5, with_score=True)
    parsed = drift.parse_snapshot(snap.dict())
    cols = {c["column"]: c for c in parsed["columns"]}
    assert cols["TransactionAmt"]["drifted"] and not cols["C14"]["drifted"]
    assert drift.SCORE in cols and "psi" in str(cols[drift.SCORE]["method"]).lower()
    assert 0 < parsed["dataset"]["drifted_share"] < 1


# --------------------------------------------------------------------------- tracking helpers (no MLflow needed)
def test_metric_names_and_flattening():
    assert tracking.metric_name("test", "precision@1%") == "test.precision_at_1pct"
    flat = tracking.flatten({"valid": {"pr_auc": 0.6, "n": 10}, "ok": True, "note": "x",
                             "bands": [{"band": "High", "n": 5}, {"band": "Low", "n": 7}],
                             "curve": [0.1, 0.2], "nan": float("nan")})
    assert flat == {"valid.pr_auc": 0.6, "valid.n": 10.0, "ok": 1.0, "bands.High.n": 5.0, "bands.Low.n": 7.0}


# --------------------------------------------------------------------------- MLflow (skipped without it)
@pytest.fixture
def mlflow_store(tmp_path, monkeypatch):
    need("mlflow")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    monkeypatch.setenv("FINGUARD_MLFLOW", "auto")
    monkeypatch.setattr(tracking, "MODEL_NAME", "finguard-test-scorer")
    return tmp_path


def test_runs_are_logged_once_per_content(mlflow_store):
    note = mlflow_store / "note.txt"
    note.write_text("x")
    a = tracking.log_run("models", "m1", "test/m1", "hash-1", {"p": 1}, {"valid.pr_auc": 0.5}, steps={"curve": [0.1, 0.2, 0.3]},
                         artifacts=[note])
    b = tracking.log_run("models", "m1", "test/m1", "hash-1", {"p": 1}, {"valid.pr_auc": 0.5})
    assert a["status"] == "logged" and b == {"status": "exists", "run_id": a["run_id"], "experiment": tracking.EXPERIMENTS["models"]}
    c = tracking.client()
    run = c.get_run(a["run_id"])
    assert run.data.metrics["valid.pr_auc"] == 0.5 and run.data.params["p"] == "1"
    assert [m.value for m in c.get_metric_history(a["run_id"], "curve")] == [0.1, 0.2, 0.3]
    assert [x.path for x in c.list_artifacts(a["run_id"])] == ["note.txt"]
    assert list((mlflow_store / "mlruns").rglob("note.txt"))           # artifacts next to the SQLite store


def test_phase2_model_run_from_info_json(mlflow_store, monkeypatch):
    d = mlflow_store / "art" / "lgbm_numeric"
    d.mkdir(parents=True)
    (d / "info.json").write_text(json.dumps({"name": "lgbm_numeric", "scheme": "main", "best_iteration": 3, "params": {"num_leaves": 7},
                                             "valid": {"pr_auc": 0.6, "precision@1%": 0.9}, "test": {"pr_auc": 0.5},
                                             "valid_curve": [0.4, 0.5, 0.6]}))
    monkeypatch.setattr(tracking, "ART", mlflow_store / "art")
    r = tracking.log_phase2_model("lgbm_numeric")
    run = tracking.client().get_run(r["run_id"])
    assert run.data.metrics["valid.precision_at_1pct"] == 0.9 and run.data.params["lgbm.num_leaves"] == "7"
    assert tracking.log_phase2_model("lgbm_numeric")["status"] == "exists"


def test_registered_scorer_matches_the_serving_computation(mlflow_store):
    lgb = need("lightgbm")
    import mlflow
    from finguard.models.calibration import fit_bands, fit_isotonic
    rs = np.random.RandomState(0)
    cols = [f"f{i}" for i in range(6)]
    X = rs.rand(3000, 6).astype("float32")
    X[rs.rand(3000) < 0.1, 2] = np.nan
    y = ((X[:, 0] + rs.rand(3000) * 0.5) > 1.1).astype(int)
    booster = lgb.train({"objective": "binary", "verbose": -1, "num_threads": 1, "seed": 1}, lgb.Dataset(X, y, feature_name=cols), 30)
    model_file = mlflow_store / "model.txt"
    booster.save_model(str(model_file))
    raw = booster.predict(X, raw_score=True)
    iso, bands = fit_isotonic(raw, y), fit_bands(raw)
    bundle = {"model_file": "artifacts/phase2/lgbm_numeric/model.txt", "columns": {"columns": cols, "categorical": []},
              "isotonic": iso, "bands": bands, "codes_as": "numeric"}
    bundle_file = mlflow_store / "model_bundle.pkl"
    bundle_file.write_bytes(pickle.dumps(bundle))
    entry = {"model_id": "lgbm_numeric-v9", "name": "lgbm_numeric", "model_file": "model.txt", "gate": {"passed": True},
             "metrics": {"test": {"pr_auc": 0.5}}, "code": "test"}
    res = tracking.register_model(entry, model_file, bundle_file)
    assert res["status"] == "registered", res
    assert tracking.register_model(entry, model_file, bundle_file)["status"] == "exists"     # same model file: same version
    tracking.set_alias("champion", res["version"], "A. Reviewer", entry["model_id"])
    mlflow.set_tracking_uri(tracking.tracking_uri())
    loaded = mlflow.pyfunc.load_model(f"models:/{tracking.MODEL_NAME}@champion")
    out = loaded.predict(pd.DataFrame(X[:500].astype("float64"), columns=cols))
    np.testing.assert_allclose(out["raw_score"].to_numpy(), raw[:500], rtol=0, atol=1e-9)
    np.testing.assert_allclose(out["probability"].to_numpy(), iso.predict(raw[:500]), rtol=0, atol=1e-12)
    assert (out["band"].to_numpy() == bands.assign(raw[:500])).all()
    v = tracking.client().get_model_version_by_alias(tracking.MODEL_NAME, "champion")
    assert v.tags["champion_approved_by"] == "A. Reviewer" and v.tags["model_id"] == "lgbm_numeric-v9"


def test_tracking_is_a_no_op_when_disabled(monkeypatch):
    monkeypatch.setenv("FINGUARD_MLFLOW", "off")
    assert tracking.log_phase2_model("anything")["status"].startswith("skipped")
    assert tracking.sync(log=lambda *a: None) == {"status": "skipped"}


# --------------------------------------------------------------------------- DVC metrics and pipeline
def test_metric_files(tmp_path, monkeypatch):
    art, t = tmp_path / "art", tmp_path / "tables"
    (art / "lgbm_numeric").mkdir(parents=True)
    t.mkdir()
    (art / "selection.json").write_text(json.dumps({"codes_as": "numeric"}))
    (art / "lgbm_numeric" / "info.json").write_text(json.dumps({"valid": {"pr_auc": 0.66, "n": 1}, "test": {"pr_auc": 0.58, "recall@1%": 0.25}}))
    (art / "gate_report.json").write_text(json.dumps({"passed": True, "checks": [{"passed": True}]}))
    (t / "phase2_results.json").write_text(json.dumps({"block_bootstrap_ci95_lightgbm": {"pr_auc": [0.55, 0.61]},
                                                       "calibration": {"LightGBM, isotonic": {"brier": 0.0208, "ece": 0.004}}}))
    (t / "phase3_retrieval_results.json").write_text(json.dumps({"chosen_variant": "bm25", "summary": [{"variant": "bm25", "questions": 50, "hit_at_5": 0.8}],
                                                                 "by_split": [{"variant": "bm25", "split": "dev", "questions": 27, "hit_at_5": 0.81}]}))
    (t / "phase4_agent_summary_scripted.json").write_text(json.dumps({"llm": "scripted", "summary": {"orchestrated": {"cases": 20, "policy_compliance": 1.0}}}))
    monkeypatch.setattr(metrics, "ART", art)
    monkeypatch.setattr(metrics, "T", t)
    files = metrics.write(tmp_path / "metrics")
    m = json.loads(files["model"].read_text())
    assert m["test"] == {"pr_auc": 0.58, "recall_at_1pct": 0.25, "pr_auc_ci95_lo": 0.55, "pr_auc_ci95_hi": 0.61}
    assert m["gate"] == {"passed": True, "checks_failed": 0}
    assert json.loads(files["retrieval"].read_text())["dev"] == {"hit_at_5": 0.81}
    assert json.loads(files["agents"].read_text())["scripted"]["orchestrated"] == {"policy_compliance": 1.0}


def test_project_dvc_yaml_is_valid():
    problems = pipeline.validate()
    assert problems == [], problems
    stages = pipeline.load()["stages"]
    assert {"ingest", "contracts_data", "features", "lgbm", "gate", "knowledge_base", "monitoring"} <= set(stages)


def test_pipeline_validator_finds_mistakes(tmp_path):
    doc = {"stages": {
        "a": {"cmd": "python -m finguard.no_such_module", "deps": ["data/raw/x.csv"], "outs": ["out/a", {"out/a/b.json": {"cache": False}}]},
        "b": {"cmd": "python tools/missing.py", "deps": ["out/a", "src/nothing_here.py"], "outs": ["out/b"], "colour": "red"},
        "c": {"cmd": "python -m finguard.mlops", "deps": ["out/b"], "outs": [{"out/c": {"cache": False, "shape": 1}}, "out/b"]},
    }}
    found = "\n".join(pipeline.validate(doc, root=config.ROOT, tracked={"out/c/file.txt"}))
    for text in ("module finguard.no_such_module not found", "script tools/missing.py not found", "unknown key 'colour'",
                 "unknown key 'shape'", "out/b: output of both b and c", "overlapping outputs", "src/nothing_here.py"):
        assert text in found, text
    cyc = {"stages": {"x": {"cmd": "python -m finguard.mlops", "deps": ["o/y"], "outs": ["o/x"]},
                      "y": {"cmd": "python -m finguard.mlops", "deps": ["o/x"], "outs": ["o/y"]}}}
    assert any(p.startswith("cycle") for p in pipeline.validate(cyc, root=config.ROOT, tracked=set()))
    cached = {"stages": {"z": {"cmd": "python -m finguard.mlops", "outs": ["reports/tables/x.csv"]}}}
    assert any("tracked by Git" in p for p in pipeline.validate(cached, root=config.ROOT, tracked={"reports/tables/x.csv"}))


def test_mlops_cli_runs():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(Path(config.ROOT / "src"))] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    p = subprocess.run([sys.executable, "-m", "finguard.mlops", "pipeline"], cwd=config.ROOT, capture_output=True, text=True,
                       env=env, timeout=120)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "structure OK" in p.stdout
