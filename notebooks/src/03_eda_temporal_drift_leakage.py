# %% [markdown]
# # 03 · Time, drift, signal, redundancy, proxy entity and leakage
#
# **FinGuard AI, Phase 1.** Reads the joined tables saved by notebook 01.
#
# Questions answered here:
# 1. Is the fraud problem stable over time? What does the later (unlabelled) period look like?
# 2. Which features drift between training and the later period?
# 3. How much signal does each feature carry on its own, and how much is redundant?
# 4. What can a **proxy entity** tell us, and why it is **not** customer profiling?
# 5. Which leakage risks must Phase 2 guard against?
#
# **Figures:** fig02, fig03, fig04, fig13, fig14, fig15, fig16.
# **Tables:** train_vs_later_drift, univariate_signal, redundancy, d_column_trend,
# category_novelty, leakage_risks, proxy_entity_summary.

# %%
import sys, pathlib, json
ROOT = pathlib.Path.cwd().resolve().parent if pathlib.Path.cwd().name == "notebooks" else pathlib.Path.cwd().resolve()
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from finguard import config
from finguard.data.ingest import load_table
from finguard.eda import figures as F, missingness as M, signal as SG, drift as DR, style as S
from finguard.features import normalize as N, proxy_entity as P

pd.set_option("display.max_columns", 30, "display.width", 160)
train, test = load_table("train"), load_table("test")
y = train[config.TARGET]
FEATS = M.feature_columns(train)
STATS = {}
print(train.shape, test.shape, len(FEATS), "feature columns")

# %% [markdown]
# ## 1. Time
# TransactionDT is relative: day 0 is the first training transaction, and hours and weekdays
# are relative to an undisclosed clock.

# %%
fig = F.fig03_daily_volume(train, test)

# %%
fig = F.fig02_weekly_fraud_rate(train)
w = train.groupby(train.day // 7)[config.TARGET].mean()
STATS["weekly_rate_min"], STATS["weekly_rate_max"] = float(w.min()), float(w.max())
blocks30 = train.groupby((train.day // 30).clip(upper=5))[config.TARGET].agg(["size", "mean"]).rename_axis("30-day block")
STATS["rate_by_30d_block"] = blocks30["mean"].round(4).to_dict()
blocks30.round(4)

# %%
fig = F.fig04_hourly_profile(train)
h = train.groupby("rel_hour")[config.TARGET].agg(["size", "mean"])
STATS["hour_peak"] = {"hour": int(h["mean"].idxmax()), "rate": float(h["mean"].max())}
wd = train.groupby("rel_weekday")[config.TARGET].agg(["size", "mean"]).round(4)
print("relative weekday fraud rates range", f"{wd['mean'].min():.2%} - {wd['mean'].max():.2%} (weak)")

# %% [markdown]
# ## 2. Drift between training and the later period
# Single-feature adversarial AUC: how well one feature alone tells a training row from a
# later row. 0.5 means no drift. This is measurable without labels, so it is also the
# production monitoring signal.

# %%
drift = DR.adversarial_auc_per_feature(train, test, FEATS)
drift.to_csv(config.TABLE_DIR / "train_vs_later_drift.csv", index=False)
STATS["drift_over_0.6"] = int((drift[drift.column != config.TIME].adversarial_auc > 0.6).sum())
fig = F.fig14_adversarial_drift(drift)
drift.groupby("group").adversarial_auc.agg(["max", "median"]).round(3).sort_values("max", ascending=False)

# %% [markdown]
# ### Category novelty: why raw strings must be normalised
# Share of later-period values never seen in training, before and after normalisation.

# %%
nov = []
for col, fn in [("id_31", N.browser_family), ("id_30", N.os_family), ("DeviceInfo", N.device_brand),
                ("P_emaildomain", N.email_group), ("R_emaildomain", N.email_group)]:
    nov.append({"column": col, "raw: new in later period": DR.new_category_share(train[col], test[col]),
                "normalised: new in later period": DR.new_category_share(fn(train[col]), fn(test[col]))})
for col in ["card1", "addr1"]:
    nov.append({"column": col, "raw: new in later period": DR.new_category_share(train[col], test[col]),
                "normalised: new in later period": np.nan})
nov = pd.DataFrame(nov).set_index("column")
nov.to_csv(config.TABLE_DIR / "category_novelty.csv")
STATS["id_31_new_raw"] = float(nov.loc["id_31"].iloc[0])
nov.round(4)

# %% [markdown]
# ### D-columns trend with time
# The host describes D-columns only as "timedelta". Several rise with the calendar, which
# makes their raw values non-stationary. Normalising as `day - D` removes the trend.

# %%
dcols = [c for c in FEATS if c.startswith("D") and c[1:].isdigit()]
trend = pd.Series({c: train[[c, "day"]].dropna().corr("spearman").iloc[0, 1] for c in dcols}, name="spearman_with_day")
trend.to_csv(config.TABLE_DIR / "d_column_trend.csv")
STATS["d_trend"] = trend.round(3).to_dict()
d1n = np.floor(train[config.TIME] / config.SECONDS_PER_DAY) - train.D1
pd.DataFrame({"D1 median": train.groupby(train.day // 30).D1.median(),
              "day - D1 median": d1n.groupby(train.day // 30).median()}).rename_axis("30-day block")

# %% [markdown]
# ## 3. Single-feature signal
# Honest strength per feature: categorical columns are target-encoded on the first 60% of
# time and scored on the last 40%; numeric columns are ranked with NaN lowest.

# %%
sig = SG.univariate_strength(train, [c for c in FEATS if c != config.TIME])
sig.to_csv(config.TABLE_DIR / "univariate_signal.csv", index=False)
STATS["max_single_strength"] = float(sig.strength.max())
STATS["n_weak_lt_0.52"] = int((sig.strength < 0.52).sum())
fig = F.fig13_signal_by_group(sig)
sig.head(15).round(4)

# %%
print("Least stable numeric features (AUC first half vs second half):")
sig.dropna(subset=["half_shift"]).sort_values("half_shift", ascending=False).head(10)[
    ["column", "group", "auc_first_half", "auc_second_half", "half_shift"]].round(4)

# %% [markdown]
# ## 4. Redundancy
# Inside each missing-value block, a column is redundant if it correlates at |r| >= 0.95
# with an earlier kept column. C1-C14 have no missing values and are checked as one group.

# %%
blocks = M.nan_blocks(train)
c_block = pd.DataFrame([{"block": "C", "columns": [f"C{i}" for i in range(1, 15)]}])
red = pd.concat([SG.redundant_within_blocks(train, blocks), SG.redundant_within_blocks(train, c_block)], ignore_index=True)
red.to_csv(config.TABLE_DIR / "redundancy.csv", index=False)
from finguard.data.schema import feature_group
red["group"] = red.column.map(feature_group)
summary = red.groupby("group").redundant.agg(checked="size", redundant="sum")
STATS["redundant"] = summary.to_dict("index")
fig = F.fig16_vblock_correlation(train, blocks)
summary

# %% [markdown]
# ## 5. Proxy entity: what it is and what it is not
#
# > **IEEE-CIS has no customer, account or cardholder ID.** The proxy entity groups
# > transactions sharing card1, addr1 and the same first-seen day (day - D1). It is a
# > heuristic from the 2019 competition's top solutions. Analyses below are
# > **proxy-entity analysis**, never customer profiling.
#
# | Proxy-entity analysis can say | It cannot say |
# | --- | --- |
# | "This card/address/first-seen group made 6 transactions in 24 h" | "This customer made 6 transactions" |
# | "The amount is 8x the group's earlier average" | "The customer's normal spending is X" |
# | "This group's labels are uniform" | "This person is a fraudster" |
# | Velocity and consistency features for the model | Customer risk scores, customer segments, KYC conclusions |

# %%
entity = P.build_proxy_entity(train)
cov = P.coverage(entity)
purity = P.label_purity(entity, y)
with_fraud = purity[purity.frauds > 0]
cov.update({"entities_2plus_with_any_fraud": int(len(with_fraud)),
            "share_entirely_fraud": float((with_fraud.fraud_share == 1).mean()),
            "share_of_multi_entities_mixed": float(((purity.fraud_share > 0) & (purity.fraud_share < 1)).mean())})
c1 = P.label_purity(train.card1.astype("Int64").astype("string"), y)
cov["card1_groups_with_fraud_entirely_fraud"] = float((c1[c1.frauds > 0].fraud_share == 1).mean())
pd.Series(cov).to_frame("value")

# %% [markdown]
# ### Why random splits leak
# Labels are almost uniform inside an entity (the host's labelling propagates fraud to
# later linked transactions). A random split puts the same entity on both sides, so a model
# can recognise entities instead of learning fraud.

# %%
rng = np.random.default_rng(config.RANDOM_SEED)
r = rng.random(len(train))
rand_tr, rand_te = r < 0.83, r >= 0.83
time_tr, time_te = (train.day <= config.SPLIT_DAYS["valid"][1]).to_numpy(), (train.day >= config.SPLIT_DAYS["test"][0]).to_numpy()
ov_random = P.split_overlap(entity, y, rand_tr, rand_te)
ov_time = P.split_overlap(entity, y, time_tr, time_te)
fig = F.fig15_entity_label_purity(purity, ov_random, ov_time)
cov.update({"overlap_random": ov_random, "overlap_time": ov_time})
(config.TABLE_DIR / "proxy_entity_summary.json").write_text(json.dumps(cov, indent=2, default=float))
pd.DataFrame({"random split": ov_random, "time split": ov_time}).round(4)

# %% [markdown]
# ## 6. Leakage risk register
# Every risk below is backed by a number computed in notebooks 01-03 and maps to a rule
# Phase 2 must enforce (and test).

# %%
rho = train[[config.KEY, config.TIME]].corr("spearman").iloc[0, 1]
risks = pd.DataFrame([
    ["Random train/test split", f"{ov_random['test_frauds_entity_had_fraud_in_train']:.1%} of random-split test frauds belong to an entity already labelled fraud (time split: {ov_time['test_frauds_entity_had_fraud_in_train']:.1%})",
     "critical", "Time-ordered split only; report new vs known entities separately"],
    ["Entity label propagation", f"{cov['share_entirely_fraud']:.1%} of multi-transaction entities with fraud are entirely fraud; only {cov['share_of_multi_entities_mixed']:.1%} mix labels",
     "critical", "No target encoding of entities or card codes; label-delay gap in a sensitivity run"],
    ["TransactionID as a feature", f"Spearman with time = {rho:.4f}", "high", "Exclude"],
    ["Raw TransactionDT as a feature", "Adversarial AUC 1.0: later values never occur in training", "high", "Exclude; derive relative hour and weekday only"],
    ["Raw D-columns", f"Spearman with day up to {trend.abs().max():.2f}", "medium", "Normalise as day - D"],
    ["Statistics fitted on all data", "Frequencies, bins, medians, encoders computed over train + later rows", "high", "Fit every statistic on the training period only"],
    ["Whole-period entity aggregates", "Entity means over all rows use future transactions", "high", "Compute entity features from earlier rows only (time-ordered replay)"],
    ["Imputing informative missingness", "Present vs missing blocks differ up to several-fold in fraud rate (fig06)", "medium", "NaN for trees; block indicators for linear models"],
    ["Raw version and device strings", f"{STATS['id_31_new_raw']:.1%} of later browser strings unseen in training", "medium", "Normalise to families; monitor novelty"],
    ["Calibration and thresholds on the test month", "Would make the test month part of training", "high", "Fit calibrator and bands on the validation month; freeze before testing"],
], columns=["risk", "evidence", "severity", "rule for Phase 2"])
risks.to_csv(config.TABLE_DIR / "leakage_risks.csv", index=False)
risks

# %%
(config.TABLE_DIR / "phase1_stats_03.json").write_text(json.dumps(STATS, indent=2, default=float))
print(json.dumps({k: STATS[k] for k in ["weekly_rate_min", "weekly_rate_max", "hour_peak", "drift_over_0.6",
                                        "max_single_strength", "n_weak_lt_0.52", "redundant"]}, indent=1, default=float))

# %% [markdown]
# ## Findings
# * **The target drifts**: weekly fraud rate ranges roughly from 2% to 5%; the first month runs
#   lower than the next three. Time-ordered evaluation is mandatory.
# * **A real later period exists**: the unlabelled file starts 30 days after training and runs
#   183 days. Features drift (adversarial AUC above 0.6 for a group of V columns), browser
#   strings are mostly new, and normalising to families removes most of that novelty.
# * **Signal is spread thin**: no single feature exceeds AUC 0.73; many V columns are weak and
#   a large share are near-duplicates inside their blocks.
# * **Proxy entity**: covers about 89% of rows; labels are almost uniform inside entities, which
#   is consistent with the host's label propagation and makes random splits leak heavily.
# * The leakage register above lists the ten rules Phase 2 must implement and test.
