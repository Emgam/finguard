# FinGuard case report: Knowledge-base question

**Flagged for review, not a finding of fraud.**

Case `eval-c16-orchestrated` · 2026-11-04 20:01:33 · request type **typology** · orchestrated by scripted-test-model

> What is the Mastercard reason code for card-not-present fraud?

*Written by the scripted test model (deterministic, no LLM): a check of the pipeline, not of an LLM.*

## Summary

Not covered by the sources. [T1]

## Limitations

- Answers come only from the 11 registered documents; anything else is "not covered by the sources".
- A passage is context for the analyst's decision, not legal advice. Passages marked fictional come from FinGuard's fictional playbook.

## Sources

- [T1] `policy_search(query=What is the Mastercard reason code for card-not-present fraud?, k=5)`: not_covered

## Run record

- Model lgbm_numeric-v1; index 4e38771857537a9d (lsa-tfidf-svd256, re-ranker lexical-v1, variant bm25); indicator map 1.0.0; prompts orchestrator@1.0.0, investigation@1.0.0, report@1.0.0; LLM scripted-test-model.
- Evidence policy: passed; claims cited 1/1, numbers traceable 0/0; first agent draft: 1/1 cited, 0/0 numbers traceable, 0 violation(s).
- 1 tool calls, 2 LLM calls, 0.05 s.
