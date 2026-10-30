# FinGuard data glossary: IEEE-CIS Fraud Detection

Publisher: FinGuard project team. Facts measured on the competition files during FinGuard Phase 1 (data-quality report). The only statement taken from the competition host is the label definition in section 3, marked as such. Dataset: IEEE-CIS Fraud Detection, card-not-present e-commerce transactions from Vesta, released for a 2019 Kaggle competition.

## 1. Files and size

| File | Rows | Columns | Notes |
| --- | --- | --- | --- |
| train_transaction | 590,540 | 394 | Labelled; one row per transaction |
| train_identity | 144,233 | 41 | Device and network data; joined on TransactionID |
| test_transaction | 506,691 | 393 | Unlabelled later period |
| test_identity | 141,907 | 41 | Names its columns id-01 to id-38 instead of id_01 to id_38 (fixed by FinGuard) |

24.4% of labelled transactions and 28.0% of later transactions have an identity record. After the join, 432 columns are usable as features.

## 2. Time

TransactionDT counts seconds from an undisclosed reference time; it is not a timestamp. The labelled file covers 182 days (relative days 0-181). The unlabelled later file starts 30 days after the labelled file ends and covers 183 days. The real clock time and time zone are unknown, so FinGuard speaks of relative hour and relative weekday. FinGuard splits the labelled file by time: training days 0-120, validation days 121-151, test days 152-181.

## 3. Label

isFraud is 1 for 20,663 of 590,540 labelled transactions, a fraud rate of 3.499% (95% interval 3.452%-3.546%), or 27.6 legitimate transactions per fraud.

As stated by the competition host, a reported chargeback on a card marks the transaction as fraud, and later transactions linked to it by user account, email address or billing address are labelled fraud too; transactions with no chargeback reported within 120 days are labelled legitimate. FinGuard could not verify this definition independently. Because labels are propagated inside linked transactions, random train/test splits leak: with a random split, 47.5% of test frauds belong to a proxy entity already labelled fraud in training, against 18.5% with a time split.

## 4. Column groups and what is disclosed

| Group | Columns | Disclosed meaning |
| --- | --- | --- |
| Transaction | TransactionID, TransactionDT, TransactionAmt | Identifier, relative time in seconds, amount |
| Product | ProductCD | A product code; the five values (W, C, R, H, S) are not explained |
| Card | card1-card6 | The values show that card4 is the network (visa, mastercard, american express, discover) and card6 the type (debit, credit, and a few 'debit or credit' and 'charge card'); card1, card2, card3 and card5 are undisclosed codes |
| Address | addr1, addr2 | Undisclosed numeric codes |
| Distance | dist1, dist2 | Undisclosed distance fields |
| Email | P_emaildomain, R_emaildomain | Purchaser and recipient email domains |
| Counting | C1-C14 | Masked; FinGuard calls them counting features after the dataset's documentation; individual meanings are not disclosed |
| Time deltas | D1-D15 | Masked; FinGuard calls them time-delta features after the dataset's documentation; individual meanings are not disclosed |
| Match | M1-M9 | Masked match flags (values T, F, or M0-M2 for M4); meanings not disclosed |
| Vesta features | V1-V339 | Features engineered by Vesta; meanings not disclosed |
| Identity | id_01-id_38, DeviceType, DeviceInfo | Device data: the values show that id_30 holds the operating system, id_31 the browser, id_33 the screen size, DeviceType mobile or desktop; the other id columns are masked |

About 96% of the original columns (414 of 432) have masked or undisclosed meanings. FinGuard never assigns a meaning to a masked column.

## 5. Missing values

45.3% of feature cells are empty. 371 columns fall into 26 blocks that are always missing together; the largest block is V217-V278 (46 columns, 78% missing). Missingness depends on the product: product W never has identity data, and address codes are missing for 95% of product C. Missingness predicts fraud: transactions with an identity record are 7.85% fraud against 2.09% without one. Nine columns are more than 99% missing (id_07, id_08, id_21-id_27) and are dropped.

## 6. Fraud by segment (labelled file)

| Segment | Fraud rate |
| --- | --- |
| Product C | 11.7% (38.8% of all fraud) |
| Product W | 2.0% |
| Credit cards | 6.7% |
| Debit cards | 2.4% |
| Mobile device type | 10.2% |
| Desktop device type | 6.5% |
| No device data | 2.1% |
| Relative hour 7 (highest) | 10.6% |
| Relative hour 13 (lowest) | 2.3% |

The weekly fraud rate moves between 2.1% and 5.1%: the target is not stationary.

## 7. The proxy entity

IEEE-CIS has no customer, account or cardholder identifier. FinGuard groups transactions by a proxy entity: card1 + addr1 + first-seen day (transaction day minus D1). It covers 88.7% of transactions and forms 199,070 groups (115,513 with a single transaction). Labels are nearly uniform inside a group: 54.6% of multi-transaction groups with any fraud are entirely fraud, and only 1.5% mix labels. The proxy entity is a heuristic, never a customer: it can merge or split real people.

## 8. Drift between the labelled and later periods

60.6% of browser strings, 24.2% of operating-system strings and 13.8% of device strings in the later file never occur in the labelled file. After normalising them into families (browser, OS, device brand), none are new. Twenty V columns separate the two periods with adversarial AUC above 0.6. The raw TransactionDT separates them perfectly and is never used as a feature; TransactionID rises with time and is excluded too.
