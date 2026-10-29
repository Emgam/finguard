# %% [markdown]
# # 05 · Time split, point-in-time features and the logistic-regression baseline
#
# **FinGuard AI, Phase 2.** Run `python -m finguard.phase2` first (it trains every model and
# saves predictions in `artifacts/phase2/`); notebooks 05-08 read those outputs.
#
# This notebook checks that the inputs to every model follow the Phase 1 rules, then reports
# the baseline that every other model must beat.
#
# 1. Time split (days 0-120 / 121-151 / 152-181) and the 30-day label-delay variant.
# 2. Leakage guards on the real data: forbidden columns, point-in-time entity features,
#    training-window statistics.
# 3. The feature set (474 columns) by reason family.
# 4. Logistic regression: preprocessing, C chosen on validation, validation and test results.
#
# IEEE-CIS has **no customer identifier**. `entity_*` features describe the heuristic
# proxy entity (card1 + addr1 + first-seen day), never a customer.

# %%
import sys, pathlib, json
ROOT = pathlib.Path.cwd().resolve().parent if pathlib.Path.cwd().name == "notebooks" else pathlib.Path.cwd().resolve()
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from finguard import config, split, phase2
from finguard.data.ingest import load_table
from finguard.evaluation import metrics as M
from finguard.explain import reasons as R
from finguard.features.build import ENTITY_FEATURES, FORBIDDEN, entity_features_for
from finguard.models import dataset

pd.set_option("display.max_columns", 30, "display.width", 180, "display.max_colwidth", 90)
T = config.TABLE_DIR
fs = dataset.load("main", "numeric")
m = fs.masks
print("feature matrix:", fs.X.shape, "| categorical:", len(fs.categorical), "| scheme:", fs.scheme)

# %% [markdown]
# ## 1. Time split
# Validation is used for early stopping, model choice, calibration and band thresholds. The
# test month is scored once. The label-delay scheme drops days 91-120 from training, as if
# their chargeback labels had not arrived yet (sensitivity run in notebook 08).

# %%
rows = []
for scheme in split.SCHEMES:
    s = split.check(fs.day, scheme)
    mk = split.masks(fs.day, scheme)
    for p in split.PARTS:
        rows.append({"scheme": scheme, "part": p, "days": f"{s[p]['first_day']}-{s[p]['last_day']}",
                     "rows": s[p]["rows"], "frauds": int(fs.y[mk[p]].sum()), "fraud_rate": fs.y[mk[p]].mean()})
    rows.append({"scheme": scheme, "part": "unused", "days": "", "rows": s["unused_rows"], "frauds": None, "fraud_rate": None})
split_table = pd.DataFrame(rows)
split_table.to_csv(T / "phase2_split.csv", index=False)
split_table

# %% [markdown]
# ## 2. Leakage guards on the real data
# Unit tests (`tests/test_features_point_in_time.py`) prove these properties on synthetic
# data; here they are re-checked on IEEE-CIS itself.

# %%
checks = {}
checks["forbidden columns absent"] = not (FORBIDDEN & set(fs.columns))
checks["no 'customer' feature names"] = not any("customer" in c for c in fs.columns)
checks["statistics fitted on training-window rows only"] = fs.builder.fit_rows_ == int(m["train"].sum())

train_df = load_table("train")
early = train_df["day"].to_numpy() <= split.SCHEMES["main"]["train"][1]
recomputed = entity_features_for(train_df.loc[early])            # history truncated at day 120
cached = pd.DataFrame(np.asarray(fs.X[early][:, fs.cols(ENTITY_FEATURES)]), columns=ENTITY_FEATURES, index=recomputed.index)
diff = (recomputed - cached).abs().max().max()
checks["entity features of training rows unchanged when later rows are removed"] = bool(diff == 0)
checks["rows without an entity key get NaN entity features"] = bool(
    np.isnan(np.asarray(fs.X[:, fs.cols(['entity_prev_txn'])])[fs.entity == None]).all())  # noqa: E711
pd.Series(checks, name="passed").to_frame()

# %%
assert all(checks.values()), checks
print(f"max |difference| between causal recomputation and cached entity features: {diff}")
print("entity key available for", f"{np.mean(fs.entity != None):.1%}", "of rows")  # noqa: E711
del train_df, recomputed, cached

# %% [markdown]
# ## 3. The feature set
# Raw columns kept as they are, plus re-expressed ones (Dn = day - D instead of trending D
# columns; families instead of drifting browser / OS / device strings) and engineered ones
# (frequencies, proxy-entity features, missingness). `amt_decile` is used by the linear
# model only (trees are invariant to monotone binning). Counts are printed below.

# %%
fam = pd.Series({c: R.family(c) for c in fs.columns}, name="family")
engineered = set(ENTITY_FEATURES) | {c for c in fs.columns if c.startswith(("freq_", "block_present_", "screen_", "email_group", "email_match"))} \
    | {"n_missing", "missing_pattern", "amt_decimals", "amt_decile", "browser_family", "browser_major", "os_family", "os_major", "device_brand"} \
    | {f"{d}n" for d in ["D1", "D4", "D6", "D8", "D11", "D12", "D13", "D14", "D15"]}
fset = pd.DataFrame({"family": fam, "engineered": [c in engineered for c in fam.index],
                     "categorical": [c in fs.categorical for c in fam.index]})
summary = fset.groupby("family").agg(columns=("engineered", "size"), engineered=("engineered", "sum"),
                                     categorical=("categorical", "sum"))
summary["label"] = [R.FAMILY_LABEL[f] for f in summary.index]
summary["anonymized"] = [f in R.ANONYMIZED for f in summary.index]
summary = summary.sort_values("columns", ascending=False)
summary.to_csv(T / "phase2_feature_families.csv")
print(f"{len(fs.columns)} columns, {int(fset.engineered.sum())} engineered or re-expressed, {int(fset.categorical.sum())} categorical")
summary

# %% [markdown]
# ### Engineered proxy-entity features on the training window
# Fraud rate by feature value, computed on training rows only (days 0-120).

# %%
tr = m["train"]
def rate_by(col, edges, labels):
    x = np.asarray(fs.X[tr][:, fs.cols([col])]).ravel()
    b = pd.cut(pd.Series(x), edges, labels=labels, right=False).astype("string").fillna("no entity / no earlier txn")
    t = pd.DataFrame({"bucket": b, "y": fs.y[tr]}).groupby("bucket").y.agg(n="size", frauds="sum")
    t["fraud_rate"] = t.frauds / t.n
    return t.reindex(list(labels) + ["no entity / no earlier txn"])

display(rate_by("entity_secs_since_prev", [0, 60, 3600, 86400, 7 * 86400, np.inf],
                ["< 1 min", "1 min-1 h", "1 h-1 day", "1-7 days", "> 7 days"]))
rate_by("entity_txn_24h", [0, 1, 2, 5, 10, np.inf], ["0", "1", "2-4", "5-9", "10+"])

# %% [markdown]
# ## 4. Logistic-regression baseline
# Preprocessing (fitted on the training window): numeric codes replaced by their
# frequencies, duplicated columns inside NaN blocks dropped (|r| >= 0.95), signed log for
# heavy tails, median imputation, standardisation, missing indicators, one-hot for the 30
# most frequent levels. C is chosen on validation PR-AUC.

# %%
lr = phase2.load_info("lr")
prep = {k: lr[k] for k in ("n_inputs", "n_numeric", "n_dropped_redundant", "n_log", "n_indicators", "n_onehot_groups")}
print("preprocessing:", prep)
grid = pd.DataFrame(lr["grid"]).T.rename_axis("C")
print("chosen C =", lr["C"])
grid

# %%
res = pd.DataFrame({"validation": lr["valid"], "test": lr["test"]}).T
res[["n", "frauds", "prevalence", "pr_auc", "roc_auc", "precision@1%", "recall@1%", "precision@5%", "recall@5%"]]

# %% [markdown]
# ### What the baseline relies on
# Largest standardised coefficients (log-odds per standard deviation). Masked columns are
# listed as they are: their meaning is not disclosed.

# %%
coef = pd.read_csv(phase2.ART / "lr" / "coefficients.csv")
def readable_name(f):                                            # "card6==0" -> "card6==debit"
    if "==" in f and not f.endswith("==other"):
        c, k = f.split("==")
        levels = fs.builder.vocab_.get(c, [])
        return f"{c}=={levels[int(k)]}" if int(k) < len(levels) else f
    return f
coef["feature"] = coef.feature.map(readable_name)
coef["abs"] = coef.coef.abs()
coef["family"] = [R.family(f.split("__")[0].split("==")[0]) for f in coef.feature]
top = coef.sort_values("abs", ascending=False).head(15)[["feature", "coef", "family"]]
top

# %% [markdown]
# **Reading.** Large coefficients of opposite sign on correlated columns (for example C1 and
# C13) reflect collinearity, not opposite meanings, so linear coefficients are not used as
# explanations. The baseline's test PR-AUC (above) is the reference for notebook 06. Every
# check in section 2 passed, so all models see only point-in-time features with
# training-window statistics.
