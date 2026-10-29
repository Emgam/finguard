# %% [markdown]
# # 02 · Missing values and fraud distributions
#
# **FinGuard AI, Phase 1.** Reads the joined tables saved by notebook 01.
#
# Questions answered here:
# 1. How much is missing, and does it come in blocks?
# 2. Is missingness random, structural (driven by product) or informative (linked to fraud)?
# 3. How does fraud vary with amount, product, identity availability, email domain, device and card?
#
# Rule used throughout: **compare within ProductCD before concluding anything.** Product
# drives both missingness and fraud rate, so many apparent signals are product effects.
#
# **Figures:** fig01, fig05, fig06, fig07, fig08, fig09, fig10, fig11, fig12.
# **Tables:** missing_by_column, nan_blocks, informative_missingness, segment_fraud_rates.

# %%
import sys, pathlib, json
ROOT = pathlib.Path.cwd().resolve().parent if pathlib.Path.cwd().name == "notebooks" else pathlib.Path.cwd().resolve()
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from finguard import config
from finguard.data.ingest import load_table
from finguard.eda import missingness as M, figures as F, style as S
from finguard.features.normalize import amount_decimals

pd.set_option("display.max_columns", 30, "display.width", 160)
train, test = load_table("train"), load_table("test")
y = train[config.TARGET]
BASE = float(y.mean())
STATS = {}
print(train.shape, test.shape, f"base fraud rate {BASE:.3%}")

# %% [markdown]
# ## 1. How much is missing
# Missing share per feature column, in training and in the later (unlabelled) period.

# %%
miss = M.missing_by_column(train, test)
miss.to_csv(config.TABLE_DIR / "missing_by_column.csv")
STATS["cell_missing_rate"] = float(train[M.feature_columns(train)].isna().to_numpy().mean())
buckets = M.bucket_counts(miss.missing_train)
STATS["missing_buckets"] = buckets.to_dict()
print(f"{STATS['cell_missing_rate']:.1%} of all feature cells are empty")
display(buckets.to_frame("columns"))
miss.groupby("group").missing_train.agg(["count", "mean", "min", "max"]).round(3).sort_values("mean", ascending=False)

# %% [markdown]
# ## 2. Missingness comes in blocks
# Columns whose missing masks are identical form a block: the same rows lack all of them.
# Each block is one piece of information ("this data source was absent"), not dozens.

# %%
blocks = M.nan_blocks(train)
STATS["n_blocks"], STATS["cols_in_blocks"] = len(blocks), int(blocks.n_cols.sum())
out = blocks.assign(columns=blocks["columns"].map(lambda c: ";".join(c)), groups=blocks["groups"].map(", ".join))
out.to_csv(config.TABLE_DIR / "nan_blocks.csv", index=False)
print(f"{len(blocks)} blocks cover {blocks.n_cols.sum()} columns")
blocks[["block", "first", "last", "n_cols", "groups", "missing_share", "fraud_rate_present", "fraud_rate_missing"]].round(4)

# %% [markdown]
# ## 3. Is it random? No: it is structural
# Missing share of each block by product. Whole blocks are 0% or 100% missing for some
# products, so missingness is not MCAR. It is at best MAR given ProductCD.

# %%
fig = F.fig05_block_missing_by_product(train, blocks)

# %% [markdown]
# ## 4. Is it informative? Yes
# Fraud rate when each block is present vs missing. If missingness carried no signal the
# two dots would overlap. They do not, so imputing values would destroy information.

# %%
fig = F.fig06_block_presence_fraud(blocks, BASE)

# %%
info = M.informative_missingness(train)
info.to_csv(config.TABLE_DIR / "informative_missingness.csv", index=False)
ident = train.groupby("has_identity")[config.TARGET].agg(["size", "mean"])
STATS["fraud_with_identity"], STATS["fraud_without_identity"] = float(ident.loc[1, "mean"]), float(ident.loc[0, "mean"])
STATS["share_of_fraud_with_identity"] = float(train.loc[y == 1, "has_identity"].mean())
print(f"identity record present: {STATS['fraud_with_identity']:.2%} fraud | absent: {STATS['fraud_without_identity']:.2%}")
print(f"{STATS['share_of_fraud_with_identity']:.1%} of all frauds carry identity data")
info.head(12).round(4)

# %% [markdown]
# ### Row-level view
# Rows with the fewest missing fields are the riskiest: completeness is itself a signal.

# %%
nmiss = train[M.feature_columns(train)].isna().sum(axis=1)
dec = pd.qcut(nmiss, 10, duplicates="drop")
row_view = pd.DataFrame({"d": dec, "y": y, "n": nmiss}).groupby("d", observed=True).agg(
    rows=("y", "size"), fraud_rate=("y", "mean"), mean_missing_fields=("n", "mean")).round(4)
STATS["fraud_rate_least_missing_decile"] = float(row_view.fraud_rate.iloc[0])
row_view

# %% [markdown]
# ### Missingness drifts
# Missing rates change between training and the later period: a model that learned
# "missing means X" will meet different proportions in production.

# %%
shift = miss.reindex(miss.missing_shift.abs().sort_values(ascending=False).index).head(10)
shift[["group", "missing_train", "missing_test", "missing_shift"]].round(3)

# %% [markdown]
# ## 5. Fraud by segment
# Fraud rate with 95% Wilson intervals for the main readable segments.

# %%
fig = F.fig01_fraud_rate_by_segment(train)

# %%
seg = []
for col in ["ProductCD", "card4", "card6", "DeviceType", "has_identity"]:
    t = S.rate_table(train[col], y).reset_index().rename(columns={"k": "value"})
    t.insert(0, "dimension", col)
    seg.append(t)
seg = pd.concat(seg).round(5)
seg.to_csv(config.TABLE_DIR / "segment_fraud_rates.csv", index=False)
prod = seg[seg.dimension == "ProductCD"].set_index("value")
STATS["product"] = {k: {"n": int(r.n), "rate": float(r.rate), "share_of_fraud": float(r.frauds / y.sum())} for k, r in prod.iterrows()}
prod

# %% [markdown]
# ## 6. Amount
# Amount alone carries almost no ranking signal; its relation to fraud is U-shaped.

# %%
fig = F.fig07_amount(train)
from sklearn.metrics import roc_auc_score
amt = train.TransactionAmt
STATS["amount"] = {"auc": float(roc_auc_score(y, amt)), "skew": float(amt.skew()), "skew_log": float(np.log1p(amt).skew()),
                   "median_fraud": float(amt[y == 1].median()), "median_legit": float(amt[y == 0].median()),
                   "p99": float(amt.quantile(0.99)), "max": float(amt.max())}
pd.Series(STATS["amount"]).round(3)

# %% [markdown]
# ### The decimals trap
# Amounts with three decimals look like a strong signal overall. Holding ProductCD fixed
# shows what is really going on.

# %%
fig = F.fig08_decimals_within_product(train)
d = amount_decimals(train.TransactionAmt)
ct = pd.crosstab(train.ProductCD, d, values=y, aggfunc="mean").round(4)
STATS["three_decimal_share_product_C"] = float(((d == 3) & (train.ProductCD == "C")).sum() / (d == 3).sum())
display(pd.crosstab(train.ProductCD, d).rename_axis(columns="decimals"))
ct.rename_axis(columns="decimals: fraud rate")

# %% [markdown]
# ## 7. Product and identity
# Identity data is absent by design for product W. Within a product, does identity
# presence still matter?

# %%
fig = F.fig09_product_identity(train)
pid = train.groupby(["ProductCD", "has_identity"])[config.TARGET].agg(["size", "mean"]).round(4)
STATS["product_C_identity"] = {"with": float(pid.loc[("C", 1), "mean"]), "without": float(pid.loc[("C", 0), "mean"])}
pid

# %% [markdown]
# ## 8. Email domains
# Readable, stable and directly explainable to an analyst.

# %%
fig = F.fig10_email_domains(train)

# %% [markdown]
# ## 9. Device: OS and browser families
# Raw version strings are normalised to families (`finguard.features.normalize`) because
# new versions appear every month.

# %%
fig = F.fig11_os_browser(train)

# %%
dt = train[train.has_identity == 1]
pd.crosstab(dt.ProductCD, dt.DeviceType.fillna("absent"), values=dt[config.TARGET], aggfunc="mean").round(4).rename_axis(
    columns="DeviceType: fraud rate (rows with identity)")

# %% [markdown]
# ## 10. Card network and type

# %%
fig = F.fig12_card_network_type(train)

# %%
(config.TABLE_DIR / "phase1_stats_02.json").write_text(json.dumps(STATS, indent=2, default=float))
print(json.dumps({k: STATS[k] for k in ["cell_missing_rate", "n_blocks", "cols_in_blocks", "fraud_with_identity",
                                        "fraud_without_identity", "three_decimal_share_product_C", "product_C_identity"]}, indent=1, default=float))

# %% [markdown]
# ## Findings
# * **45% of feature cells are empty**, concentrated in blocks: 371 columns fall into 26 blocks.
# * **Missingness is structural, not random.** Product W never has identity data; the
#   M-columns exist almost only for W; address codes are missing for 95% of product C.
# * **Missingness is informative.** Transactions with identity data are about 3.8x riskier;
#   the least-missing decile of rows has the highest fraud rate. Missing values must stay
#   visible to the model (NaN for trees, indicators for linear models), never be imputed away.
# * **Amount alone does not rank fraud** (AUC about 0.50); risk is U-shaped across deciles.
# * **The 3-decimal signal is product C in disguise**: almost all 3-decimal amounts are
#   product C, where fraud is 11-12% whatever the decimals.
# * **Product C** carries the highest fraud rate (11.7%) and 39% of all fraud; **credit**
#   cards about 2.7x debit; recipient email domains separate fraud more sharply than
#   purchaser domains; mobile and some browser families stand out once versions are normalised.
# * Device and identity effects must be read **within product**, because identity data
#   exists only for non-W products.
