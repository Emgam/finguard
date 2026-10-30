# %% [markdown]
# # 09 · Phase 2 additions: novelty flags, Isolation Forest, a stronger baseline, the scorer and the gate
#
# **FinGuard AI, Phase 2 (completed).** Project Report v3.0 listed five Phase 2 items that the
# first Phase 2 run left out. This notebook reads `python -m finguard.phase2b` outputs.
#
# 1. Explainable novelty flags, and whether they belong in the score.
# 2. Isolation Forest, alone and as an extra LightGBM input.
# 3. Why the baseline scored 0.419 against the design review's 0.445, and the strengthened baseline.
# 4. The calibrated 0-100 scorer the analyst tools will call (online = batch, checked).
# 5. The model gate and the registry entry.
#
# The test month (days 152-181) is only scored; every choice is made on the validation month.

# %%
import sys, pathlib, json
ROOT = pathlib.Path.cwd().resolve().parent if pathlib.Path.cwd().name == "notebooks" else pathlib.Path.cwd().resolve()
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from finguard import config, phase2
from finguard.evaluation import figures as F
from finguard.evaluation import metrics as M
from finguard.models import dataset

pd.set_option("display.max_columns", 30, "display.width", 200, "display.max_colwidth", 120)
T, A = config.TABLE_DIR, phase2.ART
load = lambda p: json.loads((A / p / "info.json").read_text())
fs = dataset.load("main", "numeric")
y_te = fs.y[fs.masks["test"]].astype(int)
main = phase2.load_preds("lgbm_numeric")
print("test transactions:", len(y_te), "| frauds:", int(y_te.sum()))

# %% [markdown]
# ## 1. Novelty flags
# Three rules on each proxy entity's earlier transactions (thresholds fixed in advance):
# a new device brand, a burst of at least 4 transactions within 24 h of first use, and an
# amount more than 3x the largest earlier amount. Rates below use transactions that have a
# proxy-entity key (rows without one are 13.7% fraud and are reported in notebook 08).

# %%
nov = load("novelty")
rates = pd.DataFrame(nov["rates"])
test_rates = rates[rates.part == "test"].reset_index(drop=True)
fig = F.fig25_novelty_flags(test_rates)
test_rates[["flag", "group", "n", "share_of_keyed_rows", "frauds", "fraud_rate", "lo", "hi"]]

# %%
print("fraud rate inside each band, test month, with and without any flag:")
display(pd.DataFrame(nov["within_test_bands"]))
c = nov["combiner"]
print("combiner inputs:", c["inputs"])
print("validation (two time-ordered halves) PR-AUC gain:", np.round(c["valid_time_cv_pr_auc_gain"], 4))
print("test PR-AUC gain (for the record):", round(c["paired_test_combined_minus_main"]["pr_auc"]["difference"], 4),
      np.round(c["paired_test_combined_minus_main"]["pr_auc"]["ci95"], 4))
print("decision:", c["decision"], "| rule:", c["rule"])

# %% [markdown]
# ## 2. Isolation Forest
# Alone: fitted on training-period legitimate transactions (as the report specifies). As a
# feature: refitted on all training-period transactions without labels, so that training frauds
# do not look unusual merely because they were left out of the fit.

# %%
iso = load("iforest")
iso_rows = []
for v, label in (("curated", "Isolation Forest, curated features"), ("all", "Isolation Forest, all features")):
    r = iso["variants"][v]
    iso_rows.append({"label": label, "pr_auc": r["test"]["pr_auc"], "lo": r["test_ci95"][0], "hi": r["test_ci95"][1],
                     "precision_at_1": r["test"]["precision@1%"], "valid_pr_auc": r["valid"]["pr_auc"],
                     "overlap_top1_with_lightgbm": r["test"]["overlap_top1pct_with_lightgbm"], "highlight": False})
lr2 = load("lr_v2")
res6 = json.loads((T / "phase2_results.json").read_text())
iso_rows.append({"label": "Logistic regression v2 (baseline)", "pr_auc": lr2["test"]["pr_auc"], "lo": lr2["test_ci95"]["pr_auc"][0],
                 "hi": lr2["test_ci95"]["pr_auc"][1], "precision_at_1": lr2["test"]["precision@1%"], "highlight": False})
iso_rows.append({"label": "LightGBM (main)", "pr_auc": res6["test_metrics"]["LightGBM"]["pr_auc"],
                 "lo": res6["bootstrap_ci95"]["LightGBM"]["pr_auc"][0], "hi": res6["bootstrap_ci95"]["LightGBM"]["pr_auc"][1],
                 "precision_at_1": res6["test_metrics"]["LightGBM"]["precision@1%"], "highlight": True})
unsup = pd.DataFrame(iso_rows)
fig = F.fig26_unsupervised(unsup, float(y_te.mean()))
unsup.drop(columns="highlight")

# %%
feat = load("iforest_feature")
d = feat["paired_test_with_minus_without"]["pr_auc"]
pd.DataFrame({"LightGBM": {"valid PR-AUC": load("lgbm_numeric")["valid"]["pr_auc"], "test PR-AUC": res6["test_metrics"]["LightGBM"]["pr_auc"]},
              "LightGBM + Isolation Forest score": {"valid PR-AUC": feat["valid"]["pr_auc"], "test PR-AUC": feat["test"]["pr_auc"]}}).T.assign(
    note=["", f"test difference {d['difference']:+.4f} (95% {d['ci95'][0]:+.4f} to {d['ci95'][1]:+.4f}); "
              f"importance rank {feat['iforest_gain_rank']} of {feat['n_features']}"])

# %% [markdown]
# ## 3. The baseline gap and the strengthened baseline
# The design review's baseline (0.445) used different preprocessing. Re-running it on the full
# training window, then applying the Phase 1 rules one at a time, isolates the cause.

# %%
lv = pd.DataFrame(load("lr_variants")["variants"])
lv["test_ci95"] = lv.test_ci95.map(lambda c: f"{c[0]:.3f}-{c[1]:.3f}")
lv["meaning"] = lv.variant.map(load("lr_variants")["meaning"])
lv.loc[len(lv)] = {"variant": "phase2_lr (notebook 05)", "inputs": 621, "C": 0.01, "valid_pr_auc": load("lr")["valid"]["pr_auc"],
                   "test_pr_auc": load("lr")["test"]["pr_auc"], "test_ci95": "0.400-0.438", "recall@5%": load("lr")["test"]["recall@5%"],
                   "meaning": "Phase 2 preprocessing: log only for heavy tails, duplicates dropped, frequencies for codes"}
lv[["variant", "inputs", "C", "valid_pr_auc", "test_pr_auc", "test_ci95", "meaning"]]

# %%
p = lr2["paired_lightgbm_minus_lr_v2"]
print(f"strengthened baseline (v2): test PR-AUC {lr2['test']['pr_auc']:.4f}  "
      f"[{lr2['test_ci95']['pr_auc'][0]:.3f}-{lr2['test_ci95']['pr_auc'][1]:.3f}], C = {lr2['C']}")
print(f"LightGBM - baseline v2: {p['pr_auc']['difference']:+.4f} (95% {p['pr_auc']['ci95'][0]:+.4f} to {p['pr_auc']['ci95'][1]:+.4f}); "
      f"recall@5% {p['recall@5%']['difference']:+.4f}")
pd.DataFrame({"Logistic regression v2": lr2["test"]}).T[["pr_auc", "roc_auc", "precision@1%", "recall@1%", "precision@5%", "recall@5%"]]

# %% [markdown]
# ## 4. The scorer
# `finguard.serving.Scorer` returns what the `fraud_score` and `explain_transaction` tools
# will: a 0-100 score (100 x calibrated probability), the band, the model version, the
# novelty flags and the top-5 reason families. Anonymized families name their signal group.
# The online path (one transaction + its proxy entity's earlier transactions from the
# `EntityStore`) must equal the batch features.

# %%
from finguard.data.ingest import load_table
from finguard.serving.entity_store import EntityStore
from finguard.serving.scorer import Scorer
sc = Scorer()
table = load_table("train").reset_index(drop=True)
store = EntityStore(table)
rs = np.random.RandomState(0)
check_rows = rs.choice(np.flatnonzero(fs.masks["test"]), 200, replace=False)
cached = sc.score_cached(table.loc[check_rows, "TransactionID"].tolist(), explain=False)
diff = max(abs(sc.score_transaction(table.iloc[[r]], store.history(table.iloc[[r]]), explain=False)["score"] - c["score"])
           for r, c in zip(check_rows, cached))
print(f"model version: {sc.model_version}")
print(f"max |online score - batch score| over 200 random test-month transactions: {diff}")

# %%
test_rows = np.flatnonzero(fs.masks["test"])
is_high = main["test_raw"] >= sc.bands.high
has_key = table["addr1"].notna().to_numpy()[test_rows] & table["D1"].notna().to_numpy()[test_rows]
example = table.iloc[[test_rows[np.flatnonzero(is_high & has_key)[0]]]]
out = sc.score_transaction(example, store.history(example))
print(json.dumps({k: v for k, v in out.items() if k != "reasons"}, indent=1))
pd.DataFrame(out["reasons"])[["label", "checkable", "contribution_log_odds", "signal_group", "text"]]

# %%
print(json.dumps(store.summary(example), indent=1))

# %% [markdown]
# ## 5. Model gate and registry

# %%
from finguard import gate, registry
report = gate.run()
entry = registry.register("lgbm_numeric", gate_report=report)
print("gate:", "PASSED" if report["passed"] else "FAILED", "| registered as", entry["model_id"], "| code", entry["code"])
print("promotion to champion is a human step:",
      f'python -m finguard.registry promote {entry["model_id"]} --alias champion --approved-by "<name>"')
pd.DataFrame(report["checks"])

# %%
summary = {"novelty": {"decision": c["decision"], "test_rates": test_rates.to_dict("records")},
           "isolation_forest": unsup.drop(columns="highlight").to_dict("records"),
           "iforest_feature": {"test_pr_auc": feat["test"]["pr_auc"], "paired_difference": d, "gain_rank": feat["iforest_gain_rank"]},
           "baseline_variants": lv.drop(columns="test_ci95").to_dict("records"),
           "baseline_v2": {"test": lr2["test"], "ci95": lr2["test_ci95"]["pr_auc"], "paired_lightgbm_minus_v2": p},
           "scorer": {"model_version": sc.model_version, "online_vs_batch_max_abs_diff": diff},
           "gate": {"passed": report["passed"], "model_id": entry["model_id"]}}
(T / "phase2b_summary.json").write_text(json.dumps(summary, indent=1, default=float))
print("saved reports/tables/phase2b_summary.json")
