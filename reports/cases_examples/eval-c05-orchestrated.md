# FinGuard case report: Transaction 3515334

**Flagged for review, not a finding of fraud.**

Case `eval-c05-orchestrated` · 2026-11-04 20:01:28 · request type **why_high_risk** · orchestrated by scripted-test-model

> Why was TX-3515334 scored this way?

*Written by the scripted test model (deterministic, no LLM): a check of the pipeline, not of an LLM.*

## Summary

Transaction 3515334 is in the Medium band with a score of 82.8. [T1]

## Facts

- **Transaction 3515334** (test period, relative day 159, relative hour 20): amount 20, product code S, card visa debit, identity record yes. [T1]
- **Band Medium**, score 82.8 on a 0-100 scale (displayed probability 83%); model lgbm_numeric-v1. [T1]
- Novelty flags (shown, not part of the score): none on. [T1]
- Reason card (explains the score, not the cause) [T2]:
    1. Card attributes: network visa, type debit; card1 code seen 751 time(s) in the training window (card codes are undisclosed). *(checkable; +2.26 log-odds)*
    2. Anonymized time-delta features (D1-D15): meaning not disclosed by the data provider; largest contribution from anonymized signal group D1n. *(anonymized, not checkable; +1.89 log-odds)*
    3. Anonymized identity signals (id_XX): meaning not disclosed by the data provider; largest contribution from anonymized signal group id_20. *(anonymized, not checkable; +1.47 log-odds)*
    4. Address and distance codes: billing-region code 272, country code 87; distance field not given (undisclosed codes). *(checkable; +1.02 log-odds)*
    5. Email domains: purchaser not given, recipient yahoo.com. *(checkable; +0.99 log-odds)*
- Proxy entity (heuristic, not a customer): 75 earlier transactions, 0 in the previous 24 h, 1 in the previous 7 days; amount 20 vs earlier mean 16.93 and max 25; device brands seen: windows; email domains seen: none. [T3]

## Interpretation

- Card attributes: network visa, type debit; card1 code seen 751 time(s) in the training window (card codes are undisclosed). [T2]
- Address and distance codes: billing-region code 272, country code 87; distance field not given (undisclosed codes). [T2]
- Email domains: purchaser not given, recipient yahoo.com. [T2]
- Anonymized signal groups also pushed the score up (Anonymized time-delta features (D1-D15); Anonymized identity signals (id_XX)); their meaning is not disclosed, so they cannot be checked against the transaction. [T2]
- The proxy entity had 75 earlier transactions, 0 of them in the 24 hours before this one. The amount is 1.18 times its earlier average. [T3]

## Suggested checks

*Suggestions for the analyst, not required steps.*

- Verify each checkable reason above against the transaction record. [T2]

## Limitations

- The proxy entity (card1 + addr1 + first-seen day) is a heuristic grouping: it can merge or split real people and is never a customer or a verified identity.
- Reasons explain the model's score, not the cause of fraud. Anonymized signal groups (V, C, D, M, id) have no disclosed meaning and cannot be checked against the transaction.
- The band leads; the calibrated probability is secondary and capped at "<1%" and ">99%". In the alert range it overstated the test-month fraud rate by 4-5 points.
- Guidance comes from 11 registered documents; a passage is context for the analyst's decision, not legal advice. Passages marked fictional come from FinGuard's fictional playbook.

## Sources

- [T1] `fraud_score(transaction_id=3515334)`: ok
- [T2] `explain_transaction(transaction_id=3515334)`: ok
- [T3] `entity_history(transaction_id=3515334, window_days=7)`: ok

## Run record

- Model lgbm_numeric-v1; index 4e38771857537a9d (lsa-tfidf-svd256, re-ranker lexical-v1, variant bm25); indicator map 1.0.0; prompts orchestrator@1.0.0, investigation@1.0.0, report@1.0.0; LLM scripted-test-model.
- Evidence policy: passed; claims cited 7/7, numbers traceable 9/9; first agent draft: 7/7 cited, 9/9 numbers traceable, 0 violation(s).
- 3 tool calls, 3 LLM calls, 0.29 s.
