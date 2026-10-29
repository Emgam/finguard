# FinGuard AI · Phase 1 Data-Quality Report

**Dataset:** IEEE-CIS Fraud Detection (Kaggle, Vesta), five CSV files, 1.35 GB
**Produced by:** notebooks `01`–`04` (executed outputs committed), package `finguard`, 14 unit tests
**Date:** 2026-10-12

## 1. Verdict

The data is **fit for Phase 2**, under ten rules listed in the leakage register (section 10).
No validation check fails. One schema defect (`id-01` vs `id_01`) is fixed and unit-tested.
The data has three structural limits that Phase 2 must respect:

1. **There is no customer, account or cardholder identifier.** Every grouping of transactions is a
   heuristic *proxy entity*, never a customer.
2. **Labels are propagated within linked transactions**, so random splits leak heavily.
   47.5% of random-split test frauds belong to an entity already labelled fraud, vs 18.5% with a time split.
3. **About 96% of the columns (414 of 432) have undisclosed meanings.** Readable explanations rest on
   18 readable or coded columns plus engineered features.

## 2. Provenance and reproducibility

| File | Rows × columns | Size | MD5 matches the official release |
| --- | --- | --- | --- |
| train_transaction.csv | 590,540 × 394 | 683.4 MB | yes |
| train_identity.csv | 144,233 × 41 | 26.5 MB | yes |
| test_transaction.csv (unlabelled) | 506,691 × 393 | 613.2 MB | yes |
| test_identity.csv (unlabelled) | 141,907 × 41 | 25.8 MB | yes |
| sample_submission.csv | 506,691 × 2 | 6.1 MB | yes |

Reproduce everything with `make data validate eda test`. Without Jupyter, use `make eda-fallback`.
The joined tables and a manifest (shapes, checksums, fixes) are written to `data/interim/`.

## 3. Schema and integrity

| Check | Result |
| --- | --- |
| Column names and order vs the expected schema | All match, except `test_identity`: 38 columns named `id-XX` instead of `id_XX` (**fixed**) |
| Effect of the defect if unfixed | A naive concatenation creates 79 columns, and `id_01` is empty for 49.6% of rows |
| TransactionID uniqueness | Unique in every file; train and later IDs never overlap |
| Join integrity | Every identity row matches a transaction; the left join preserves all 590,540 / 506,691 rows |
| Identity coverage | 24.4% of training transactions and 28.0% of later transactions have an identity record |
| Duplicates | 3 exact duplicate rows; 487 rows in 439 groups identical except ID and time, 8 groups with conflicting labels: negligible, kept |
| Label domain | `isFraud` ∈ {0, 1}, never missing |
| Amount | Strictly positive; skew 14.4 (0.49 after log) |
| Columns > 99% missing | 9: id_07, id_08, id_21–id_27 (dropped) |
| Constant columns | none |
| Usable feature columns | 432 after the join. The dataset-choice PDF's "871 columns" is the sum of all five file headers |

## 4. Class imbalance

20,663 frauds out of 590,540 transactions: **3.499%** (95% interval 3.452%–3.546%), or 27.6 legitimate
transactions per fraud. A model that always predicts "legitimate" is 96.5% accurate, so accuracy is
never a headline metric. The no-skill PR-AUC equals the prevalence, 0.035.

## 5. No customer identifier; proxy entity

No column identifies a customer, account or cardholder. `card1`, the closest candidate, is a shared
code. Its most frequent value covers 14,932 transactions across 57 address codes and 32 email domains.

FinGuard therefore defines a **proxy entity**: card1 + addr1 + (transaction day − D1). The construction
comes from top solutions of the 2019 competition. It is a heuristic resting on the host's group-level
description of D-columns as "timedelta".

| Measure | Value |
| --- | --- |
| Transactions with a proxy entity | 88.7% (card1, addr1 or D1 missing for the rest) |
| Proxy entities | 199,070: 115,513 single-transaction, 83,557 multi-transaction (median size 3) |
| Rows in multi-transaction entities | 69.1% of all rows; 77.9% of rows that have an entity |
| Multi-transaction entities with any fraud | 2,716, of which **54.6% are entirely fraud** |
| Multi-transaction entities that mix labels | 1.5% |
| Comparison: card1 groups with fraud that are entirely fraud | 3.6% |

| Proxy-entity analysis can say | It cannot say |
| --- | --- |
| "This card/address/first-seen group made 6 transactions in 24 h" | "This customer made 6 transactions" |
| "The amount is 8× the group's earlier average" | "The customer's normal spending is X" |
| "This group's labels are uniform" | "This person is a fraudster" |
| Velocity and consistency features for a model | Customer risk scores, customer segments, KYC conclusions |

The code, figures and reports use "entity" or "proxy entity", never "customer". The UI must do the same.

## 6. Missing values

* **Volume:** 45.3% of feature cells are empty. 18 columns are complete; 92 are ≤ 10% missing; 108 are 10–50%; 6 are 50–75%;
  196 are 75–90%; 3 are 90–99%; 9 are > 99%.
* **Blocks:** 371 columns fall into **26 blocks** that are missing together (largest: V217–V278, 46 columns,
  78% missing). Each block is one piece of information: "this data source was absent".
* **Not random, structural** (fig05): product W never has identity data, the M-columns exist almost only for W,
  and address codes are missing for 95% of product C. Missingness is at best MAR given ProductCD.
* **Informative** (fig06): transactions with an identity record show 7.85% fraud vs 2.09% without; 54.8% of all
  frauds carry identity data. Within product C the gap persists: 12.3% vs 5.8%. The least-missing decile of rows
  has the highest fraud rate (10.0%). Across 23 blocks, present data is riskier in 15 and missing data in 8.
* **Drifting:** the V35–V52 block falls from 28.6% to 15.2% missing in the later period; D13 from 89.5% to 75.6%.

**Treatment:** keep NaN for tree models; add one indicator per block, `has_identity` and a missing-field count;
impute (training-period median plus indicator) only for logistic regression and Isolation Forest; treat
missing text as its own category. No blanket median or mode imputation.

## 7. Fraud distributions

| Dimension | Finding | Figure |
| --- | --- | --- |
| Product | C: 11.7% fraud, 68,519 rows, 38.8% of all fraud. W: 2.0%, 439,670 rows, 43.4% of fraud. H 4.8%, S 5.9%, R 3.8% | fig01 |
| Amount | No ranking signal alone (AUC 0.50). U-shaped: 5.6% in the lowest decile, 5.1% in the highest | fig07 |
| Amount decimals | 61,931 of 61,933 three-decimal amounts are product C, where fraud is 11–12% whatever the decimals. Within W, round amounts are riskier (3.2% vs 0.7%) | fig08 |
| Identity × product | W never has identity data; within C, 12.3% with vs 5.8% without | fig09 |
| Email | Recipient domain is sharper: outlook.com 16.5%, icloud.com 12.9%, gmail.com 11.9% vs 2.1% when empty. Purchaser max: outlook.com 9.5% | fig10 |
| Device | Mobile 10.2%, desktop 6.5%, absent 2.1%. Browser families range from 1.8% (IE) to 31.6% (Opera, n = 449). Read within product | fig11 |
| Card | Credit 6.7% vs debit 2.4% (2.8×); Discover 7.7%; Visa and Mastercard near the base rate | fig12 |

## 8. Time and drift

* **Non-stationary target** (fig02): the weekly fraud rate ranges from 2.1% to 5.1%; the first 30 days run at 2.5%, the next 90 at about 4.0%.
* **Daily cycle** (fig04): fraud peaks at 10.6% in relative hour 7, among the quietest hours, and bottoms at 2.3% in hour 13.
  The real clock and time zone are undisclosed, so the hour is called "relative". The weekday effect is weak (3.2%–3.7%).
* **A real later period** (fig03): the unlabelled file starts 30 days after training ends and covers 183 days.
* **Feature drift** (fig14): 20 features exceed adversarial AUC 0.6, all V columns (max 0.614). Raw TransactionDT scores 1.0.
* **Category novelty:** 60.6% of later browser strings (id_31), 24.2% of OS strings and 13.8% of DeviceInfo values never occur
  in training. After normalising to families, 0% are new.
* **D-columns trend with time:** Spearman with day up to 0.12; the median D1 rises from 0 to 14 days. Normalise them as day − D.

## 9. Signal and redundancy

* No single feature exceeds a strength of 0.73 (card1). Recipient email domain reaches 0.70 and ProductCD 0.69.
  The 339 V columns have a median of 0.58, and 65 features are below 0.52 (fig13).
* Instability: V10 and V11 fall from AUC 0.40 to 0.31 between the two halves of the period.
* Redundancy inside blocks at |r| ≥ 0.95: **83 of 339 V columns** and **8 of 14 C columns** duplicate a kept column (fig16).

## 10. Leakage risk register

| Risk | Evidence | Severity | Rule for Phase 2 |
| --- | --- | --- | --- |
| Random train/test split | 47.5% of random-split test frauds are in an entity already labelled fraud (time split: 18.5%) | critical | Time-ordered split only; report new vs known entities separately |
| Entity label propagation | 54.6% of multi-transaction entities with fraud are entirely fraud; 1.5% mix labels | critical | No target encoding of entities or card codes; label-delay gap in a sensitivity run |
| TransactionID as a feature | Spearman with time 1.0000 | high | Exclude |
| Raw TransactionDT as a feature | Adversarial AUC 1.0 | high | Exclude; derive relative hour and weekday only |
| Raw D-columns | Spearman with day up to 0.12 | medium | Normalise as day − D |
| Statistics fitted on all data | Frequencies, bins, medians, encoders over train + later rows | high | Fit every statistic on the training period only |
| Whole-period entity aggregates | Entity means over all rows use future rows | high | Entity features from earlier rows only (time-ordered replay) |
| Imputing informative missingness | Fraud differs several-fold by block presence | medium | NaN for trees; block indicators for linear models |
| Raw version and device strings | 60.6% of later browser strings unseen | medium | Normalise to families; monitor novelty |
| Calibration and thresholds on the test month | Would make the test month part of training | high | Fit on the validation month; freeze before testing |

## 11. Usable features

`reports/tables/feature_inventory.csv` lists all 432 columns with a rule-based treatment:

| Use in model | Columns |
| --- | --- |
| Yes, as is (NaN kept) | 316 |
| Yes, after transformation (6 text fields normalised, 9 trending D-columns) | 15 |
| Trees only (near-duplicates; dropped for logistic regression and explanations) | 91 |
| Excluded (raw TransactionDT; derive time features only) | 1 |
| Dropped (> 99% missing) | 9 |

Interpretability: 11 readable columns (amount, card network and type, both email domains, DeviceType,
DeviceInfo, OS, browser, screen, TransactionDT), 7 undisclosed codes (ProductCD, card1/2/3/5, addr1/2),
and 414 masked columns. Masked columns can only be explained at family level ("anonymized signal group V217–V278").

## 12. Limitations and open points

* Meanings rest on the host's group-level descriptions. The label definition (chargeback-based, propagated to linked
  transactions, 120-day window) is as stated by the host on Kaggle and was not independently verifiable.
* No geography, merchant or customer attributes exist: geographic features, merchant analytics and fairness by customer group
  are out of scope.
* The proxy entity can merge or split real people. Its quality is supported indirectly (label uniformity, velocity signal),
  not verified.
* Open for Phase 2: whether `amt_decimals` adds lift within product, and whether the weak entity amount ratio earns its place.

## 13. Figure index

| # | File | Finding |
| --- | --- | --- |
| 1 | fig01_fraud_rate_by_segment.png | 4 of 18 segments run at ≥ 2× the base rate; ProductCD C leads at 11.7% |
| 2 | fig02_weekly_fraud_rate.png | Weekly fraud rate swings between 2.1% and 5.1% |
| 3 | fig03_daily_volume_train_vs_later.png | The unlabelled file starts 30 days after training ends |
| 4 | fig04_relative_hour_volume_and_rate.png | Fraud peaks at 10.6% in relative hour 7 |
| 5 | fig05_missing_blocks_by_product.png | Missingness is structural by product |
| 6 | fig06_block_presence_vs_fraud.png | Missingness carries signal both ways (15 vs 8 blocks) |
| 7 | fig07_amount_distribution_and_deciles.png | Amount alone barely separates fraud (AUC 0.50) |
| 8 | fig08_amount_decimals_within_product.png | The 3-decimal signal is product C |
| 9 | fig09_product_by_identity.png | Identity presence is mostly a product effect, yet 2.1× within C |
| 10 | fig10_email_domains.png | Recipient domains separate fraud more sharply |
| 11 | fig11_os_and_browser_family.png | Browser families range from 1.8% to 31.6% |
| 12 | fig12_card_network_and_type.png | Credit cards carry 2.8× the fraud rate of debit cards |
| 13 | fig13_single_feature_signal_by_group.png | No single feature exceeds 0.73 |
| 14 | fig14_train_vs_later_drift.png | 20 features drift past adversarial AUC 0.6 |
| 15 | fig15_proxy_entity_label_purity.png | Labels cluster inside proxy entities; random splits leak |
| 16 | fig16_v_block_correlation.png | Many V columns are near-duplicates |
