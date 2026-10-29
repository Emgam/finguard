# %% [markdown]
# # 01 · Data validation: schema, joins, duplicates, imbalance
#
# **FinGuard AI, Phase 1: Data understanding.** Dataset: IEEE-CIS Fraud Detection (Kaggle, Vesta).
#
# This notebook checks the five raw files before any analysis: checksums, schema against the
# expected column lists, the `id-01` / `id_01` naming defect, key uniqueness, join integrity,
# duplicates, types, time coverage and class imbalance. It ends by saving the joined, fixed
# tables that notebooks 02-04 read.
#
# **Run order:** 01 → 02 → 03 → 04. Raw CSVs go in `data/raw/` (or set `FINGUARD_RAW_DIR`).
#
# **Outputs:** `data/interim/train.*`, `data/interim/test.*`, `data/interim/manifest.json`,
# `reports/tables/validation_results.csv`.

# %%
import sys, pathlib, json, platform
ROOT = pathlib.Path.cwd().resolve().parent if pathlib.Path.cwd().name == "notebooks" else pathlib.Path.cwd().resolve()
sys.path.insert(0, str(ROOT / "src"))   # not needed after `pip install -e .`

import numpy as np
import pandas as pd
from finguard import config
from finguard.data import ingest, schema, validate
from finguard.eda.style import wilson

pd.set_option("display.max_columns", 30, "display.width", 160)
config.TABLE_DIR.mkdir(parents=True, exist_ok=True)
print("python", platform.python_version(), "| pandas", pd.__version__, "| numpy", np.__version__)
print("raw data:", config.RAW_DIR)

# %% [markdown]
# ## 1. Raw files and checksums
# The MD5 checksums are compared with the official Kaggle release, so everyone on the
# project analyses byte-identical data.

# %%
rows = []
for name, fname in config.RAW_FILES.items():
    p = config.RAW_DIR / fname
    h = ingest.md5sum(p)
    rows.append({"file": fname, "MB": round(p.stat().st_size / 1e6, 1), "md5": h, "matches_release": h == config.RAW_MD5[name]})
files = pd.DataFrame(rows)
print(f"total size: {files.MB.sum() / 1000:.2f} GB")
files

# %% [markdown]
# ## 2. Load the raw tables with explicit dtypes
# Anonymised numerics are stored as float32 to halve memory. TransactionDT and
# TransactionAmt stay float64: TransactionDT exceeds 2^24 (float32 would round it) and
# amounts keep their decimals.

# %%
raw = {n: ingest.read_raw(n) for n in ["train_transaction", "train_identity", "test_transaction", "test_identity"]}
pd.DataFrame({n: {"rows": len(d), "columns": d.shape[1], "memory_MB": round(d.memory_usage(deep=True).sum() / 1e6)} for n, d in raw.items()}).T

# %% [markdown]
# ## 3. Schema verification
# Every file is compared with the expected column list in `finguard.data.schema`.
# The dataset-choice PDF claimed "871 columns": that is the sum of all five headers,
# including the unlabelled test files and the submission file, not the feature count.

# %%
schema_checks = pd.DataFrame(validate.check_raw_files(raw))
display(schema_checks)
n_headers = sum(d.shape[1] for d in raw.values()) + 2   # + sample_submission's 2 columns
print("sum of all five headers:", n_headers)

# %% [markdown]
# ### 3b. The `id-01` / `id_01` defect and why it matters
# `test_identity.csv` names its columns `id-01 ... id-38`; `train_identity.csv` uses
# `id_01 ... id_38`. A naive concatenation silently doubles those columns and leaves
# half of each one empty.

# %%
hyphen = [c for c in raw["test_identity"].columns if c.startswith("id-")]
print(f"{len(hyphen)} hyphenated columns in test_identity, e.g. {hyphen[:3]}")
naive = pd.concat([raw["train_identity"], raw["test_identity"]], ignore_index=True)
print("naive concat  ->", naive.shape[1], "columns;", f"id_01 empty for {naive['id_01'].isna().mean():.1%} of rows")
fixed_test_identity, renamed = ingest.normalize_identity_columns(raw["test_identity"])
fixed = pd.concat([raw["train_identity"], fixed_test_identity], ignore_index=True)
print("after the fix ->", fixed.shape[1], "columns;", f"id_01 empty for {fixed['id_01'].isna().mean():.1%} of rows")
assert list(fixed_test_identity.columns) == schema.IDENTITY_COLS
del naive, fixed

# %% [markdown]
# ## 4. Joins
# Identity rows are left-joined onto transactions by TransactionID. The join function fails
# on duplicate keys, orphan identity rows or any change in the number of transactions.

# %%
train = ingest.join_transaction_identity(raw["train_transaction"], raw["train_identity"])
test = ingest.join_transaction_identity(raw["test_transaction"], fixed_test_identity)
origin = float(raw["train_transaction"][config.TIME].min())
train, test = ingest.add_time_index(train, origin), ingest.add_time_index(test, origin)
pd.DataFrame({
    "transactions": [len(raw["train_transaction"]), len(raw["test_transaction"])],
    "after join": [len(train), len(test)],
    "with identity record": [int(train.has_identity.sum()), int(test.has_identity.sum())],
    "share with identity": [f"{train.has_identity.mean():.1%}", f"{test.has_identity.mean():.1%}"],
    "columns": [train.shape[1], test.shape[1]],
}, index=["train", "test (unlabelled)"])

# %% [markdown]
# ## 5. Keys, duplicates, label domain, time and leakage-prone columns
# All checks in one table. **WARN** marks a known issue that later steps handle;
# **FAIL** would stop the pipeline.

# %%
results = validate.run_all(raw, train, test)
results.to_csv(config.TABLE_DIR / "validation_results.csv", index=False)
print(results.status.value_counts().to_dict())
results

# %% [markdown]
# ## 6. Types and cardinality
# 31 text columns, the rest numeric. Six numeric columns are really category codes
# (card1, card2, card3, card5, addr1, addr2): arithmetic on them is meaningless.

# %%
from finguard.data.schema import feature_group
feat = [c for c in train.columns if c not in (config.KEY, config.TARGET, "has_identity", "day", "rel_hour", "rel_weekday")]
types = pd.DataFrame({
    "group": [feature_group(c) for c in feat],
    "kind": ["text" if c in schema.TEXT_COLS else ("numeric code" if c in schema.NUMERIC_CODE_COLS else "numeric") for c in feat],
    "n_unique": [train[c].nunique() for c in feat],
}, index=feat)
display(types.groupby(["group", "kind"]).size().unstack(fill_value=0))
print("usable feature columns:", len(feat), "(TransactionDT included; excluded later)")
types.sort_values("n_unique", ascending=False).query("kind != 'numeric'").head(12)

# %% [markdown]
# ## 7. Time coverage
# TransactionDT is a number of seconds from an undisclosed reference date. There is no
# calendar date and no time zone, so hour and weekday are **relative**.

# %%
span = lambda d: (d[config.TIME].max() - d[config.TIME].min()) / config.SECONDS_PER_DAY
pd.Series({
    "train first / last TransactionDT": f"{train[config.TIME].min():,.0f} / {train[config.TIME].max():,.0f}",
    "train span (days)": round(span(train), 1),
    "gap before the later file (days)": round((test[config.TIME].min() - train[config.TIME].max()) / config.SECONDS_PER_DAY, 1),
    "later file span (days)": round(span(test), 1),
    "train sorted by time": bool(train[config.TIME].is_monotonic_increasing),
    "Spearman(TransactionID, TransactionDT)": round(train[[config.KEY, config.TIME]].corr("spearman").iloc[0, 1], 6),
}).to_frame("value")

# %% [markdown]
# ## 8. Class imbalance
# 3.5% fraud. A model that predicts "legitimate" for everything is 96.5% accurate and
# useless, which is why accuracy is never a headline metric. The no-skill PR-AUC equals
# the prevalence.

# %%
y = train[config.TARGET]
k, n = int(y.sum()), len(y)
lo, hi = wilson(np.array([k]), np.array([n]))
pd.Series({
    "transactions": f"{n:,}", "frauds": f"{k:,}", "fraud rate": f"{k / n:.3%}",
    "95% interval": f"{lo[0]:.3%} - {hi[0]:.3%}", "legitimate per fraud": round((n - k) / k, 1),
    "accuracy of 'always legitimate'": f"{1 - k / n:.1%}", "no-skill PR-AUC": round(k / n, 4),
}).to_frame("value")

# %% [markdown]
# ## 9. There is no customer identifier
# No column identifies a customer, account or cardholder. `card1` is the closest
# candidate and it is not one: its most common value covers thousands of transactions
# spread over many billing-address codes. **Any grouping of transactions in this project
# is a heuristic proxy entity, never a customer** (notebook 03).

# %%
top = train.card1.value_counts().head(5)
pd.DataFrame({
    "transactions": top,
    "distinct addr1 codes": [train.loc[train.card1 == v, "addr1"].nunique() for v in top.index],
    "distinct P_emaildomain": [train.loc[train.card1 == v, "P_emaildomain"].nunique() for v in top.index],
    "frauds": [int(train.loc[train.card1 == v, config.TARGET].sum()) for v in top.index],
}).rename_axis("card1 value")

# %% [markdown]
# ## 10. Save the joined, fixed tables
# Parquet when pyarrow is installed (recommended), pickle otherwise. The manifest records
# shapes, checksums and the fixes applied, so every later result can be traced to this run.

# %%
p_train, p_test = ingest.save_table(train, "train"), ingest.save_table(test, "test")
manifest = {
    "files": {r.file: {"md5": r.md5, "matches_release": bool(r.matches_release)} for r in files.itertuples()},
    "fixes": {"test_identity_renamed_columns": len(renamed)},
    "time_origin_seconds": origin, "train_shape": list(train.shape), "test_shape": list(test.shape),
    "train_path": str(p_train), "test_path": str(p_test),
}
(config.INTERIM_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
print("saved", p_train.name, p_test.name, "and manifest.json")

# %% [markdown]
# ## Findings
# * All five files match the official release checksums; row counts match the published figures.
# * The only schema defect is the hyphenated `id-XX` naming in `test_identity`; it is fixed
#   in `normalize_identity_columns()` and covered by a unit test.
# * Keys are unique, every identity row has a transaction, the joins preserve row counts,
#   and train and test IDs do not overlap. Only 24.4% of training transactions have identity data.
# * Duplicates are negligible (3 exact rows; 487 rows identical except ID and time).
# * TransactionID is perfectly ordered by time and raw TransactionDT does not overlap between
#   periods: both are excluded from features.
# * 3.5% fraud: metrics must be PR-AUC and recall/precision at an alert budget, not accuracy.
# * No customer identifier exists; entity analysis in this project is proxy-entity analysis.
