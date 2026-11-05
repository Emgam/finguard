# FinGuard case report: Transaction 3554656

**Flagged for review, not a finding of fraud.**

Case `eval-c01-orchestrated` · 2026-11-04 20:01:26 · request type **investigate** · orchestrated by scripted-test-model

> Investigate alert TX-3554656.

*Written by the scripted test model (deterministic, no LLM): a check of the pipeline, not of an LLM.*

## Summary

Transaction 3554656 is in the High band with a score of 96.2. [T1]

## Facts

- **Transaction 3554656** (test period, relative day 173, relative hour 2): amount 300, product code R, card visa credit, identity record yes. [T1]
- **Band High**, score 96.2 on a 0-100 scale (displayed probability 96%); model lgbm_numeric-v1. [T1]
- Novelty flags (shown, not part of the score): none on. [T1]
- Reason card (explains the score, not the cause) [T2]:
    1. Vesta engineered features (V1-V339): meaning not disclosed by the data provider; largest contribution from anonymized signal group V138-V166. *(anonymized, not checkable; +6.59 log-odds)*
    2. Anonymized counting features (C1-C14): meaning not disclosed by the data provider; largest contribution from anonymized signal group C1. *(anonymized, not checkable; +4.42 log-odds)*
    3. Amount: 300.00 (0 non-zero decimal place(s)). *(checkable; +1.53 log-odds)*
    4. Anonymized time-delta features (D1-D15): meaning not disclosed by the data provider; largest contribution from anonymized signal group D1n. *(anonymized, not checkable; +0.99 log-odds)*
    5. Email domains: purchaser gmail.com, recipient gmail.com. *(checkable; +0.87 log-odds)*
- Proxy entity (heuristic, not a customer): first transaction in the data, no earlier history. [T3]
- Earlier FinGuard cases for this transaction: none. [T8]

## Interpretation

- Amount: 300.00 (0 non-zero decimal place(s)). [T2]
- Email domains: purchaser gmail.com, recipient gmail.com. [T2]
- Anonymized signal groups also pushed the score up (Vesta engineered features (V1-V339); Anonymized counting features (C1-C14); Anonymized time-delta features (D1-D15)); their meaning is not disclosed, so they cannot be checked against the transaction. [T2]
- This is the proxy entity's first transaction in the data, so there is no earlier history to compare. [T3]
- In FinGuard's fictional playbook, these indicators are consistent with stolen card used for high-value goods, stolen card details used by a third party. [T4, S1]

## Guidance

> "Review the proxy-entity history: earlier transactions, bursts in the last 24 hours and 7 days, time since the previous transaction, amount against the earlier average, device brands and email domains seen so far."
>
> [S2] *FinGuard investigation playbook (fictional)*, FinGuard investigation playbook (fictional) > 4. Investigation steps (FinGuard project team, 2026-10-29; Project) *(fictional playbook)*

> "The payment service provider should report statistical information on: a. total payment transactions in line with the different breakdowns in Annex 2 and in accordance with Guideline 1; and b. total fraudulent payment transactions in line with the different breakdowns in Annex 2 and as defined in Guideline 1.6(a)."
>
> [S7] *Guidelines on fraud reporting under PSD2 (EBA/GL/2018/05), consolidated version*, 3.1 Guidelines on fraud data reporting applicable to Payment Service Providers > Guideline 2: General data requirements, pages 17-18 (European Banking Authority, 2018-07-18; EU)

## Suggested checks

*Suggestions for the analyst, not required steps.*

- Compare the amount with the proxy entity's earlier amounts and with the usual amounts for the product code. [T4]
- Check the purchaser and recipient email domains against the domains the proxy entity used before. [T4]

## Limitations

- The proxy entity (card1 + addr1 + first-seen day) is a heuristic grouping: it can merge or split real people and is never a customer or a verified identity.
- Reasons explain the model's score, not the cause of fraud. Anonymized signal groups (V, C, D, M, id) have no disclosed meaning and cannot be checked against the transaction.
- The band leads; the calibrated probability is secondary and capped at "<1%" and ">99%". In the alert range it overstated the test-month fraud rate by 4-5 points.
- Guidance comes from 11 registered documents; a passage is context for the analyst's decision, not legal advice. Passages marked fictional come from FinGuard's fictional playbook.

## Sources

- [T1] `fraud_score(transaction_id=3554656)`: ok
- [T2] `explain_transaction(transaction_id=3554656)`: ok
- [T3] `entity_history(transaction_id=3554656, window_days=7)`: ok
- [T4] `indicator_map(reason_families=['vesta_V', 'counting_C', 'amount', 'timedelta_D', 'email'], novelty_flags=[], entity_available=True, request_type=investigate)`: ok
- [T5] `policy_search(query=amount far above the earlier amounts of the same card account takeover, k=3)`: ok
- [T6] `policy_search(query=purchaser and recipient email domains differ anonymous email domain, k=3)`: ok
- [T7] `policy_search(query=which fraudulent payment transactions must a payment service provider report, k=3, tier=1)`: ok
- [T8] `case_store(action=read, transaction_id=3554656, entity_key=11019_432_174)`: ok
- [S1] *FinGuard investigation playbook (fictional)*, FinGuard project team, 2026-10-29; Project; tier 3; section: FinGuard investigation playbook (fictional) > 5. Indicators and what to check next; corpus/docs/finguard_investigation_playbook.md **FICTIONAL**
- [S2] *FinGuard investigation playbook (fictional)*, FinGuard project team, 2026-10-29; Project; tier 3; section: FinGuard investigation playbook (fictional) > 4. Investigation steps (also: 5. Indicators and what to check next); corpus/docs/finguard_investigation_playbook.md **FICTIONAL**
- [S3] *FinGuard feature and reason-code glossary*, FinGuard project team, 2026-10-29; Project; tier 1; section: FinGuard feature and reason-code glossary > 4. Engineered features > 4.1 Proxy-entity features; corpus/docs/finguard_feature_reason_glossary.md
- [S4] *FinGuard investigation playbook (fictional)*, FinGuard project team, 2026-10-29; Project; tier 3; section: FinGuard investigation playbook (fictional) > 5. Indicators and what to check next; corpus/docs/finguard_investigation_playbook.md **FICTIONAL**
- [S5] *FinGuard feature and reason-code glossary*, FinGuard project team, 2026-10-29; Project; tier 1; section: FinGuard feature and reason-code glossary > 4. Engineered features > 4.2 Other engineered features; corpus/docs/finguard_feature_reason_glossary.md
- [S6] *FinGuard feature and reason-code glossary*, FinGuard project team, 2026-10-29; Project; tier 1; section: FinGuard feature and reason-code glossary > 3. Reason families > 3.1 How each family is worded on a reason card; corpus/docs/finguard_feature_reason_glossary.md
- [S7] *Guidelines on fraud reporting under PSD2 (EBA/GL/2018/05), consolidated version*, European Banking Authority, 2018-07-18; EU; tier 1; section: 3.1 Guidelines on fraud data reporting applicable to Payment Service Providers > Guideline 2: General data requirements; pages 17-18; https://www.eba.europa.eu/sites/default/files/document_library/Final%20Report%20on%20EBA%20Guidelines%20on%20fraud%20reporting%20-%20Consolidated%20version.pdf
- [S8] *Guidelines on fraud reporting under PSD2 (EBA/GL/2018/05), consolidated version*, European Banking Authority, 2018-07-18; EU; tier 1; section: 3.1 Guidelines on fraud data reporting applicable to Payment Service Providers > Guideline 2: General data requirements; pages 18-19; https://www.eba.europa.eu/sites/default/files/document_library/Final%20Report%20on%20EBA%20Guidelines%20on%20fraud%20reporting%20-%20Consolidated%20version.pdf
- [S9] *Guidelines on fraud reporting under PSD2 (EBA/GL/2018/05), consolidated version*, European Banking Authority, 2018-07-18; EU; tier 1; section: 3.1 Guidelines on fraud data reporting applicable to Payment Service Providers > Guideline 1: Payment transactions and fraudulent payment transactions; pages 16; https://www.eba.europa.eu/sites/default/files/document_library/Final%20Report%20on%20EBA%20Guidelines%20on%20fraud%20reporting%20-%20Consolidated%20version.pdf

## Run record

- Model lgbm_numeric-v1; index 4e38771857537a9d (lsa-tfidf-svd256, re-ranker lexical-v1, variant bm25); indicator map 1.0.0; prompts orchestrator@1.0.0, investigation@1.0.0, report@1.0.0; LLM scripted-test-model.
- Evidence policy: passed; claims cited 10/10, numbers traceable 4/4; first agent draft: 10/10 cited, 4/4 numbers traceable, 0 violation(s).
- 8 tool calls, 3 LLM calls, 0.48 s.
