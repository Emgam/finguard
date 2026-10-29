# FinGuard AI · Feature-Engineering Plan (input to Phase 2)

**Basis:** Phase 1 evidence in `reports/data_quality_report.md` and notebooks 02–04.
**Scope:** transaction-level scoring on IEEE-CIS only. No second dataset; no AML module.

## 1. Rules every feature must follow

1. **Point in time.** A feature for a transaction at time *t* uses only rows with time < *t*, plus
   statistics fitted on the training period. Each causal feature gets a unit test that shuffles future
   rows and checks that the feature does not change.
2. **Training-period statistics only.** Bins, frequencies, medians, encoders, calibrators and thresholds
   are fitted on the training window (relative days 0–120) and frozen.
3. **Honest names.** Anything built on the grouping key is called `entity_*` (proxy entity), never
   `customer_*`. Relative time is called `rel_hour` / `rel_weekday`.
4. **Forbidden:** raw TransactionID and TransactionDT; target encoding of entities or card codes;
   aggregates over the whole period; any statistic computed over the later (unlabelled) file.
5. **Missingness stays visible.** NaN for trees; explicit indicators for linear models; no blanket imputation.

## 2. What happens to the 432 raw columns

| Decision | Columns | Evidence |
| --- | --- | --- |
| Keep as is (NaN kept for trees) | 316 | Inventory rules |
| Normalise text: id_31 → browser family + major version, id_30 → OS family, DeviceInfo → brand, id_33 → width/height, emails → provider group (raw domain kept for trees) | 6 | 60.6% / 24.2% / 13.8% of raw values new in the later period; 0% after normalising |
| Re-express trending D-columns as day − D (D1, D4, D6, D8, D11–D15) | 9 | Spearman with day up to 0.12; D1 median 0 → 14 days |
| Trees only (dropped for logistic regression and explanations) | 91 (83 V, 8 C) | \|r\| ≥ 0.95 with a kept column in the same block |
| Treat numeric codes as categories (card1, card2, card3, card5, addr1, addr2) | 6 (within "keep") | Values have no order meaning |
| Exclude: raw TransactionDT (derive time features only) | 1 | Adversarial AUC 1.0 |
| Drop: > 99% missing (id_07, id_08, id_21–id_27) | 9 | Too little data |

## 3. Engineered features

| Feature | Exact computation | Phase 1 evidence | Leakage risk | At scoring time |
| --- | --- | --- | --- | --- |
| `log_amt` | log1p(TransactionAmt) | Skew 14.4 → 0.49 | none | yes |
| `amt_decile` | Deciles fitted on the training period | U-shaped: 5.6% / 5.1% at the ends (fig07) | bins on all data | yes |
| `amt_decimals` | Non-zero decimals (0–3) | Almost entirely a product-C proxy (fig08); keep only if it adds lift within product | none | yes |
| `rel_hour` | floor(TransactionDT / 3600) mod 24 | 10.6% at hour 7 vs 2.3% at hour 13 (fig04) | none | yes |
| `rel_weekday` | floor(TransactionDT / 86400) mod 7 | Weak (3.2%–3.7%) | none | yes |
| `Dn` (9 columns) | floor(TransactionDT / 86400) − D | Removes the calendar trend | none | yes |
| `has_identity` | Identity record exists | 7.85% vs 2.09%; 12.3% vs 5.8% within C | none | yes, if identity arrives with the payment |
| `browser_family`, `browser_major`, `os_family`, `device_brand`, `screen_w`, `screen_h` | Normalisers in `finguard.features.normalize` (unit-tested) | Opera 31.6%, Android WebView 20.4%, IE 1.8% (fig11) | none | yes |
| `email_group_P`, `email_group_R`, `email_match` | Provider groups; P == R | R domain up to 16.5% vs 2.1% when empty (fig10) | none | yes |
| `freq_card1`, `freq_addr1`, `freq_email`, `freq_device` | Counts in the training period | card1 is the strongest single feature (0.73) | **leaks if counted over later data** | yes, from history |
| `entity_prev_txn` | Earlier transactions of the proxy entity (time-ordered cumulative count) | 2.0% (first) → 3.7% (10+) | earlier rows only | yes, entity store |
| `entity_secs_since_prev` | t − previous entity transaction time | **6.6% (< 1 min) vs 0.9% (> 7 days)** | earlier rows only | yes, entity store |
| `entity_txn_24h`, `entity_txn_7d` | Earlier entity transactions in (t − window, t) | Short gaps are the riskiest | earlier rows only | yes, entity store |
| `entity_amt_ratio` | Amount ÷ mean of earlier entity amounts | Weak (1.9%–3.2%): keep only if it adds lift | earlier rows only | yes, entity store |
| `entity_n_devices`, `entity_n_emails` | Distinct device brands / email domains seen so far | Identity consistency (to be measured in Phase 2) | earlier rows only | yes, entity store |
| `entity_age_days` | t − first entity transaction | Newness | earlier rows only | yes, entity store |
| `n_missing`, `block_present_k` (26), `missing_pattern` | Row missing count; one flag per block; hashed block mask (rare → "other") | 10.0% fraud in the least-missing decile; blocks differ several-fold (fig06) | none | yes |

**Out of scope:** customer-level features, geographic change, "international" flags and merchant or
category behaviour. The data has no such fields.

## 4. Proxy-entity features: implementation contract

```
key          = card1 + "_" + addr1 + "_" + (floor(TransactionDT / 86400) - D1)   # NA if any part is missing
order        = stable sort by TransactionDT (ties keep file order)
prev_txn     = groupby(key).cumcount()
secs_prev    = TransactionDT - groupby(key).TransactionDT.shift(1)
txn_window   = count of earlier rows of key with TransactionDT in (t - window, t)
amt_ratio    = amount / (cumsum(amount) - amount) * prev_txn            # NaN when prev_txn == 0
```

Tests required in Phase 2:

* **Causality:** appending future rows never changes any earlier value.
* **Same values online and offline:** scoring one transaction against the entity store gives the batch value.
* **Missing key:** rows with a missing card1, addr1 or D1 get NaN entity features, never a guessed key.

## 5. Encoding by model family

| Model | Numeric | Categorical | Missing |
| --- | --- | --- | --- |
| Logistic regression (baseline) | log / signed-log, standardised with training statistics | One-hot for ≤ 30 levels, frequency for codes | Median (training period) + one indicator per block |
| LightGBM (main) | Raw (trees are scale-free) | Native categorical for text and codes (top levels from training, rest → "other") | NaN kept |
| Isolation Forest (experiment) | Curated amount, velocity and count features | Not used | Constant sentinel + indicators |

## 6. Phase 2 entry checklist (all met)

- [x] Raw files verified against the official checksums
- [x] `id-XX` / `id_XX` defect fixed and unit-tested
- [x] Joins and duplicates validated; no FAIL in `validation_results.csv`
- [x] 16 Phase 1 figures produced
- [x] Leakage register written (`leakage_risks.csv`)
- [x] Feature inventory written (`feature_inventory.csv`)
- [x] Engineered-feature specification written (`engineered_feature_plan.csv`)
- [x] Time split fixed in `finguard.config.SPLIT_DAYS` (days 0–120 / 121–151 / 152–181)
