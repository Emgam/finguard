# %% [markdown]
# # 04 · Feature inventory and evidence for the feature-engineering plan
#
# **FinGuard AI, Phase 1.** Run after notebooks 01-03 (it reads their tables).
#
# 1. A column-by-column inventory: group, type, missingness, signal, drift, redundancy,
#    interpretability, and a rule-based treatment recommendation.
# 2. Exploratory evidence (computed **causally**, from earlier transactions only) that the
#    planned proxy-entity features carry signal. Phase 2 implements them as tested code.
# 3. The engineered-feature specification and the Phase 2 entry checklist.
#
# **Tables:** feature_inventory, engineered_feature_plan, proxy_velocity_evidence.

# %%
import sys, pathlib, json
ROOT = pathlib.Path.cwd().resolve().parent if pathlib.Path.cwd().name == "notebooks" else pathlib.Path.cwd().resolve()
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from finguard import config
from finguard.data.ingest import load_table
from finguard.eda import style as S
from finguard.features.inventory import build_inventory
from finguard.features.proxy_entity import build_proxy_entity

pd.set_option("display.max_columns", 30, "display.width", 180, "display.max_colwidth", 80)
T = config.TABLE_DIR
train = load_table("train")
y = train[config.TARGET]
miss = pd.read_csv(T / "missing_by_column.csv", index_col=0)
sig = pd.read_csv(T / "univariate_signal.csv")
drift = pd.read_csv(T / "train_vs_later_drift.csv")
red = pd.read_csv(T / "redundancy.csv")
d_trend = pd.read_csv(T / "d_column_trend.csv", index_col=0).iloc[:, 0]
print("inputs loaded:", len(miss), "columns in the missingness table")

# %% [markdown]
# ## 1. Feature inventory
# One row per raw column. Treatments follow explicit rules in
# `finguard.features.inventory`, so the table can be regenerated when evidence changes.

# %%
inv = build_inventory(train, miss, sig, drift, red, d_trend)
inv.to_csv(T / "feature_inventory.csv")
display(pd.crosstab(inv.group, inv.use_in_model, margins=True))
pd.crosstab(inv.interpretability, inv.use_in_explanations)

# %%
print("Columns to drop or exclude:")
display(inv[inv.use_in_model.isin(["no", "no (derive only)"])][["group", "missing_train", "recommended_treatment", "note"]])
print("Columns to transform before use:")
inv[inv.use_in_model == "yes, transformed"][["group", "recommended_treatment", "note"]]

# %% [markdown]
# ### The most useful readable features
# The analyst can understand these directly; they carry the reason codes in Phase 2.

# %%
readable = inv[inv.interpretability == "readable"].sort_values("signal_strength", ascending=False)
readable[["group", "missing_train", "signal_strength", "drift_adversarial_auc", "recommended_treatment"]]

# %% [markdown]
# ## 2. Do proxy-entity features carry signal? (causal, exploratory)
#
# > Reminder: the proxy entity is **not a customer** (notebook 03, section 5).
#
# For every transaction we look only at **earlier** transactions of the same proxy entity
# (data sorted by time; `cumcount` and `shift` never look ahead). This is the same rule the
# Phase 2 feature code must follow. Labels are used here only to describe the association.

# %%
d = train[[config.TIME, "TransactionAmt", config.TARGET]].copy()
d["entity"] = build_proxy_entity(train)
d = d[d.entity.notna()].sort_values(config.TIME, kind="stable")
g = d.groupby("entity", sort=False)
d["prev_txn"] = g.cumcount()
d["secs_since_prev"] = d[config.TIME] - g[config.TIME].shift(1)
prev_sum = g["TransactionAmt"].cumsum() - d["TransactionAmt"]
d["amt_ratio"] = np.where(d.prev_txn > 0, d.TransactionAmt / (prev_sum / d.prev_txn.clip(lower=1)), np.nan)

def rates(col, bins, labels):
    b = pd.cut(d[col], bins=bins, labels=labels, right=False)
    t = S.rate_table(b.astype("string"), d[config.TARGET], missing_label="first transaction")
    t.insert(0, "feature", col)
    return t.reindex([*labels, "first transaction"]).dropna(subset=["n"])

ev = pd.concat([
    rates("prev_txn", [0, 1, 2, 5, 10, 1e9], ["0", "1", "2-4", "5-9", "10+"]),
    rates("secs_since_prev", [0, 60, 3600, 86400, 7 * 86400, 1e12], ["< 1 min", "1 min - 1 h", "1 h - 1 day", "1-7 days", "> 7 days"]),
    rates("amt_ratio", [0, 0.5, 0.9, 1.1, 2, 1e12], ["< 0.5x", "0.5-0.9x", "0.9-1.1x", "1.1-2x", "> 2x"]),
])
ev.to_csv(T / "proxy_velocity_evidence.csv")
ev.round(4)

# %% [markdown]
# Reading the table (rows that have a proxy entity):
# * **Time since the entity's previous transaction** is the strongest: fraud falls from about
#   6.6% when the previous transaction was under a minute earlier to about 0.9% after a week.
# * **Number of earlier transactions** rises modestly, from 2.0% (first) to 3.7% (10 or more).
# * **Amount ratio** is weak (about 1.9% to 3.2%): keep it only if it adds lift in Phase 2.
#
# These are associations within proxy entities, not descriptions of customer behaviour.

# %% [markdown]
# ## 3. Engineered-feature specification
# Every feature: exact computation, the Phase 1 evidence behind it, leakage risk, and
# whether it is available when a transaction is scored. Phase 2 implements these in
# `finguard.features` with a unit test for each causality rule.

# %%
plan = pd.DataFrame([
    # name, family, computation, evidence, leakage risk, available at scoring
    ["log_amt", "transaction", "log1p(TransactionAmt)", "skew 14.4 -> 0.49 (nb02)", "none", "yes"],
    ["amt_decile", "transaction", "decile bins fitted on the training period", "U-shaped fraud rate across deciles (fig07)", "bins fitted on all data", "yes"],
    ["amt_decimals", "transaction", "number of non-zero decimals (0-3)", "almost entirely a product-C proxy (fig08); keep only if it adds lift within product", "none", "yes"],
    ["rel_hour", "temporal", "floor(TransactionDT/3600) mod 24", "fraud 10.6% at relative hour 7 vs 2.3% at hour 13 (fig04)", "none; clock unknown, name it 'relative'", "yes"],
    ["rel_weekday", "temporal", "floor(TransactionDT/86400) mod 7", "weak (3.1-3.7%)", "none", "yes"],
    ["Dn for trending D-columns (D1, D4, D6, D8, D11-D15)", "temporal", "floor(TransactionDT/86400) - D", "|Spearman with day| >= 0.05, up to 0.12 (inventory)", "none", "yes"],
    ["has_identity", "identity", "identity record exists", "7.85% vs 2.09%; 12.3% vs 5.8% within product C (fig09)", "none", "yes, if identity arrives with the payment"],
    ["browser_family, browser_major", "device", "normalise id_31", "60.6% of raw strings new in later period; 0% after normalising", "none", "yes"],
    ["os_family", "device", "normalise id_30", "24.2% of raw strings new; 0% after normalising", "none", "yes"],
    ["device_brand", "device", "normalise DeviceInfo", "13.8% of raw strings new; 0% after normalising", "none", "yes"],
    ["screen_w, screen_h", "device", "split id_33 'WxH'", "readable device consistency signal", "none", "yes"],
    ["email_group_P, email_group_R, email_match", "identity", "provider groups; P == R", "R domain up to 16.5% vs 2.1% when empty (fig10)", "none", "yes"],
    ["freq_card1, freq_addr1, freq_email, freq_device", "rarity", "counts in the training period only", "card1 is the strongest single feature (AUC 0.73)", "leaks if counted over train + later data", "yes, from history"],
    ["entity_prev_txn", "proxy entity", "earlier transactions of the entity (time-ordered cumcount)", "2.0% (first) to 3.7% (10+ earlier), section 2", "must use earlier rows only", "yes, entity store"],
    ["entity_txn_24h, entity_txn_7d", "proxy entity", "earlier transactions in (t - window, t)", "bursts: short gaps are riskiest (section 2)", "must use earlier rows only", "yes, entity store"],
    ["entity_secs_since_prev", "proxy entity", "t minus previous entity transaction time", "6.6% (< 1 min) vs 0.9% (> 7 days), section 2", "must use earlier rows only", "yes, entity store"],
    ["entity_amt_ratio", "proxy entity", "amount / mean of earlier entity amounts", "weak, 1.9-3.2%: keep only if it adds lift", "must use earlier rows only", "yes, entity store"],
    ["entity_n_devices, entity_n_emails", "proxy entity", "expanding distinct count so far", "identity consistency", "must use earlier rows only", "yes, entity store"],
    ["entity_age_days", "proxy entity", "t minus first entity transaction", "newness", "must use earlier rows only", "yes, entity store"],
    ["n_missing, block_present_k (26), missing_pattern", "missingness", "row missing count; one flag per block; hashed block mask", "least-missing decile 10.0% fraud; blocks differ several-fold (fig06)", "none", "yes"],
    ["FORBIDDEN: entity or card target encoding", "-", "mean isFraud per entity/card", "labels propagate inside entities (fig15)", "severe", "no"],
    ["FORBIDDEN: whole-period entity aggregates", "-", "entity means over all rows", "uses future rows", "high", "no"],
    ["FORBIDDEN: raw TransactionDT, TransactionID", "-", "-", "adversarial AUC 1.0; Spearman 1.0 with time", "high", "no"],
], columns=["feature", "family", "computation", "phase-1 evidence", "leakage risk", "available at scoring"])
plan.to_csv(T / "engineered_feature_plan.csv", index=False)
plan

# %% [markdown]
# ## 4. Phase 2 entry checklist

# %%
manifest = json.loads((config.INTERIM_DIR / "manifest.json").read_text())
val = pd.read_csv(T / "validation_results.csv")
checks = pd.DataFrame([
    ["Raw files match the official release", all(v["matches_release"] for v in manifest["files"].values() if "matches_release" in v)],
    ["id-XX / id_XX defect fixed and tested", manifest["fixes"]["test_identity_renamed_columns"] == 38],
    ["No validation check FAILs", not (val.status == "FAIL").any()],
    ["Joined tables saved with manifest", (config.INTERIM_DIR / "manifest.json").exists()],
    ["16 Phase 1 figures produced", len(list(config.FIG_DIR.glob("fig*.png"))) == 16],
    ["Leakage register written", (T / "leakage_risks.csv").exists()],
    ["Feature inventory written", (T / "feature_inventory.csv").exists()],
    ["Engineered-feature plan written", (T / "engineered_feature_plan.csv").exists()],
    ["Time split fixed in config (days 0-120 / 121-151 / 152-181)", config.SPLIT_DAYS == {"train": (0, 120), "valid": (121, 151), "test": (152, 181)}],
], columns=["item", "done"])
checks

# %% [markdown]
# ## Findings
# * Every raw column has a documented treatment. Two columns are excluded on principle
#   (TransactionID, raw TransactionDT), nine are dropped for being over 99% missing, six text
#   fields are normalised, trending D-columns are re-expressed as day - D, and redundant V/C
#   columns are kept for trees only.
# * Readable features (amount, card network and type, email domains, device and browser
#   families) are the backbone of analyst-facing explanations; masked groups are explained
#   only at family level.
# * Causal proxy-entity features show real signal (time since the previous transaction above
#   all), which justifies building them in Phase 2 with tests that prove they never look ahead.
