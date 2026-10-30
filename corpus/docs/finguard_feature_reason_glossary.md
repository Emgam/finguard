# FinGuard feature and reason-code glossary

Publisher: FinGuard project team. Generated from the FinGuard code and the Phase 2 evaluation outputs. Scope: the IEEE-CIS card-not-present e-commerce dataset (Vesta, 2019).

## 1. What the glossary covers

This glossary explains every input, reason family, novelty flag and band that a FinGuard alert can show. IEEE-CIS has no customer, account or cardholder identifier. Every feature named entity_ describes a proxy entity: card1 + addr1 + first-seen day (transaction day minus D1). A proxy entity is a heuristic grouping, never a customer, and it can merge or split real people. SHAP reasons explain what moved the model's score, not why fraud happened. Every alert is flagged for review, not a finding of fraud.

## 2. Risk score and bands

The risk score is 100 times the calibrated fraud probability of the main LightGBM model. Isotonic calibration is fitted on the validation month (relative days 121-151). A score of 40 means that about 40% of similar transactions in the validation month turned out to be fraud. The score is capped at 1-99 and the displayed probability at <1% and >99%, because the calibrator's plateaus at exactly 0 and 1 would claim certainty.

The bands follow analyst capacity. High is the top 1% of validation-month scores and Medium the next 4%; Low is the rest. The thresholds are frozen and applied to later transactions.

| Band | Rule | Calibrated probability at the threshold | Test-month fraud rate | Test-month share of transactions |
| --- | --- | --- | --- | --- |
| High | top 1% of validation scores | at least 87% | 90.1% | 0.93% |
| Medium | next 4% | at least 11% | 30.7% | 4.69% |
| Low | the rest | below 11% | 1.3% | 94.38% |

## 3. Reason families

A reason card lists up to five reason families that pushed the score up most. Each family sums the TreeSHAP values (log-odds) of its inputs. Readable families state the transaction's own values. Anonymized families have no disclosed meaning and are never given one; the card names the anonymized signal group (for example V279-V321) that contributed most.

| Reason family | Code | Meaning disclosed | Inputs |
| --- | --- | --- | --- |
| Proxy-entity velocity | entity_velocity | yes | entity_prev_txn, entity_secs_since_prev, entity_txn_24h, entity_txn_7d |
| Proxy-entity history | entity_history | yes | entity_amt_ratio, entity_n_devices, entity_n_emails, entity_age_days |
| Amount | amount | yes | TransactionAmt, amt_decimals |
| Product code | product | code only (the five product codes are undisclosed) | ProductCD |
| Card attributes | card | partly (network and type; other card fields are codes) | card1-card6, frequency of card1/card2/card3/card5 in the training window |
| Address and distance codes | address | no (undisclosed codes) | addr1, addr2, dist1, dist2, frequency of addr1/addr2 |
| Email domains | email | yes | P_emaildomain, R_emaildomain, provider groups, purchaser/recipient match, email frequency |
| Device and browser | device | yes | DeviceType, browser family and major version, OS family and major version, device brand, screen width and height, device frequency |
| Anonymized identity signals (id_XX) | identity_masked | no | id_01-id_38 (except those replaced by browser, OS, device and screen features) |
| Relative time of day | time | relative only | rel_hour, rel_weekday (relative to an undisclosed reference; real clock time unknown) |
| Anonymized counting features (C1-C14) | counting_C | no | C1-C14 |
| Anonymized time-delta features (D1-D15) | timedelta_D | no | D1-D15, with D1, D4, D6, D8, D11-D15 re-expressed as day minus D (D1n, D4n, ...) |
| Match flags (M1-M9) | match_M | no | M1-M9 |
| Vesta engineered features (V1-V339) | vesta_V | no | V1-V339 |
| Data completeness | completeness | yes | has_identity, n_missing, block_present_01-23, missing_pattern |

### 3.1 How each family is worded on a reason card

- **Proxy-entity velocity** (entity_velocity). Example: "Proxy-entity velocity: 6 earlier transaction(s) of the same proxy entity in the previous 24 h, 9 in 7 days; previous one 40 s earlier."
- **Proxy-entity history** (entity_history). Example: "Proxy-entity history: amount is 3.2x its earlier average; 2 device brand(s) and 1 purchaser email domain(s) seen so far; first seen 4 day(s) earlier in the data."
- **Amount** (amount). Example: "Amount: 117.00 (0 non-zero decimal place(s))."
- **Product code** (product). Example: "Product code: C (product meaning not disclosed by the data provider)."
- **Card attributes** (card). Example: "Card attributes: network visa, type credit; card1 code seen 76 time(s) in the training window (card codes are undisclosed)."
- **Address and distance codes** (address). Example: "Address and distance codes: billing-region code 299, country code 87; distance field not given (undisclosed codes)."
- **Email domains** (email). Example: "Email domains: purchaser gmail.com, recipient gmail.com."
- **Device and browser** (device). Example: "Device and browser: mobile, brand samsung, OS android, browser chrome 63."
- **Anonymized identity signals (id_XX)** (identity_masked). Example: "Anonymized identity signals (id_XX): meaning not disclosed by the data provider."
- **Relative time of day** (time). Example: "Relative time of day: relative hour 7 (the real clock time and time zone are not disclosed)."
- **Anonymized counting features (C1-C14)** (counting_C). Example: "Anonymized counting features (C1-C14): meaning not disclosed by the data provider."
- **Anonymized time-delta features (D1-D15)** (timedelta_D). Example: "Anonymized time-delta features (D1-D15): meaning not disclosed by the data provider."
- **Match flags (M1-M9)** (match_M). Example: "Match flags (M1-M9) (meaning not disclosed): M4=M0, M5 not given, M6=F."
- **Vesta engineered features (V1-V339)** (vesta_V). Example: "Vesta engineered features (V1-V339): meaning not disclosed by the data provider."
- **Data completeness** (completeness). Example: "Data completeness: 234 empty fields; identity record absent."

## 4. Engineered features

### 4.1 Proxy-entity features

Computed by replaying transactions in time order, from earlier transactions of the same proxy entity only. Transactions without a proxy-entity key (card1, addr1 or D1 missing) get empty values, never a guessed key.

| Feature | Definition |
| --- | --- |
| entity_prev_txn | Number of earlier transactions of the same proxy entity in the data, before this one. |
| entity_secs_since_prev | Seconds since the previous transaction of the same proxy entity; empty for its first transaction. |
| entity_txn_24h | Number of earlier transactions of the same proxy entity in the 24 hours before this one. |
| entity_txn_7d | Number of earlier transactions of the same proxy entity in the 7 days before this one. |
| entity_amt_ratio | This amount divided by the average amount of the proxy entity's earlier transactions. |
| entity_n_devices | Number of distinct device brands seen so far on the proxy entity, this transaction included. |
| entity_n_emails | Number of distinct purchaser email domains seen so far on the proxy entity, this transaction included. |
| entity_age_days | Days since the proxy entity's first transaction in the data. |

### 4.2 Other engineered features

| Feature | Definition |
| --- | --- |
| freq_card1 | Number of training-window transactions (relative days 0-120) with the same card1; 0 when the value was never seen. |
| freq_card2 | Number of training-window transactions (relative days 0-120) with the same card2; 0 when the value was never seen. |
| freq_card3 | Number of training-window transactions (relative days 0-120) with the same card3; 0 when the value was never seen. |
| freq_card5 | Number of training-window transactions (relative days 0-120) with the same card5; 0 when the value was never seen. |
| freq_addr1 | Number of training-window transactions (relative days 0-120) with the same addr1; 0 when the value was never seen. |
| freq_addr2 | Number of training-window transactions (relative days 0-120) with the same addr2; 0 when the value was never seen. |
| freq_email | Number of training-window transactions (relative days 0-120) with the same P_emaildomain; 0 when the value was never seen. |
| freq_device | Number of training-window transactions (relative days 0-120) with the same DeviceInfo; 0 when the value was never seen. |
| D1n, D4n, D6n, D8n, D11n-D15n | Transaction day minus the D column (D1, D4, D6, D8, D11, D12, D13, D14, D15). Removes the calendar trend of these time deltas. |
| amt_decimals | Number of non-zero decimal places in the amount (0-3). |
| browser_family, browser_major | Browser family (samsung, edge, chrome, firefox, opera, safari, android webview, internet explorer, other) and major version, from id_31. |
| os_family, os_major | Operating-system family (windows, ios, mac, android, linux, other) and major version, from id_30. |
| device_brand | Device brand from DeviceInfo (for example samsung, apple ios, apple mac, windows, motorola, lg, huawei, xiaomi, google, other). |
| screen_w, screen_h | Screen width and height in pixels, from id_33 (for example 2208x1242). |
| email_group_P, email_group_R | Email provider group of the purchaser and recipient domains (gmail, yahoo, microsoft, apple, aol, anonymous, protonmail, other). |
| email_match | 1 when purchaser and recipient email domains are equal; empty when either is missing. |
| has_identity | 1 when the transaction has an identity record (device and network data). |
| n_missing | Number of empty fields in the transaction. |
| block_present_01-23 | 1 when a group of columns that are always missing together is present. |
| missing_pattern | Code of the transaction's combination of present and missing column groups. |
| rel_hour, rel_weekday | Hour and day of week relative to the dataset's undisclosed reference time. |

## 5. Novelty flags

Novelty flags are explainable indicators computed from the proxy entity's earlier transactions. They are shown to the analyst next to the score. They are not part of the risk score: a combiner fitted on the validation month did not improve it.

| Flag | Rule |
| --- | --- |
| flag_new_device | New device brand for this proxy entity (another brand was used before). |
| flag_burst_first_day | Burst: at least 4 transactions within 24 h of the proxy entity's first transaction. |
| flag_amount_above_max | Amount more than 3x the proxy entity's largest earlier amount. |

Test-month fraud rates (transactions with a proxy-entity key):

| Flag | Flagged transactions | Fraud rate when flagged | Fraud rate when not flagged |
| --- | --- | --- | --- |
| flag_new_device | 277 | 4.3% | 2.3% |
| flag_burst_first_day | 2,783 | 3.4% | 2.2% |
| flag_amount_above_max | 457 | 2.2% | 2.3% |

## 6. Anomaly score

An Isolation Forest was tested as an unsupervised detector. Its most unusual transactions were almost never fraud, so it is not part of the risk score and is not shown on alerts.

## 7. Limits of these explanations

- About 85% of the model's inputs (403 of 474 columns) have no disclosed meaning.
- Proxy-entity history is a heuristic grouping, not a verified identity.
- Relative time is not clock time: the dataset's reference time and time zone are not disclosed.
- Fairness across customer groups cannot be measured: the data holds no customer attributes.
