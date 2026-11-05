# FinGuard case report: Knowledge-base question

**Flagged for review, not a finding of fraud.**

Case `eval-c13-orchestrated` · 2026-11-04 20:01:32 · request type **typology** · orchestrated by scripted-test-model

> What does Visa dispute condition 10.4 cover?

*Written by the scripted test model (deterministic, no LLM): a check of the pipeline, not of an LLM.*

## Summary

The quoted passages below address the question; read them with their source and date. [S1]

## Guidance

> "Compelling Evidence 3.0 (CE3.0) is a framework available for Dispute Condition 10.4 Other Fraud - Card‐Absent Environment."
>
> [S1] *Visa Dispute Management Guide for Merchants: Best Practices, Tools and Resources*, Visa Dispute Management Guide for Merchants > Visa Dispute Categories and Conditions > What's Covered, pages 84-85 (Visa, 2026-10; Card network (international))

> "This dispute applies when a cardholder claims they did not authorize or participate in a transaction completed in a card‐absent environment, such as online, phone, or mail order."
>
> [S2] *Visa Dispute Management Guide for Merchants: Best Practices, Tools and Resources*, Visa Dispute Management Guide for Merchants > Visa Dispute Categories and Conditions > What you can do to help prevent this dispute, pages 28-29 (Visa, 2026-10; Card network (international))

> "Step 5: Consider CE3.0 (if fraud - 10.4) − Do you have at least two prior undisputed transactions? − Are the timing and data requirements met?"
>
> [S3] *Visa Dispute Management Guide for Merchants: Best Practices, Tools and Resources*, Visa Dispute Management Guide for Merchants > Conclusion and Next Steps > Questions to Ask Your Acquirer, pages 109-110 (Visa, 2026-10; Card network (international))

## Limitations

- Answers come only from the 11 registered documents; anything else is "not covered by the sources".
- A passage is context for the analyst's decision, not legal advice. Passages marked fictional come from FinGuard's fictional playbook.

## Sources

- [T1] `policy_search(query=What does Visa dispute condition 10.4 cover?, k=5)`: ok
- [S1] *Visa Dispute Management Guide for Merchants: Best Practices, Tools and Resources*, Visa, 2026-10; Card network (international); tier 1; section: Visa Dispute Management Guide for Merchants > Visa Dispute Categories and Conditions > What's Covered (also: Important to Remember); pages 84-85; https://by.visa.com/dam/VCOM/global/support-legal/documents/merchants-dispute-management-guidelines.pdf
- [S2] *Visa Dispute Management Guide for Merchants: Best Practices, Tools and Resources*, Visa, 2026-10; Card network (international); tier 1; section: Visa Dispute Management Guide for Merchants > Visa Dispute Categories and Conditions > What you can do to help prevent this dispute (also: Condition 10.4 - Other Fraud - Card-Absent Environment What this dispute is about; Terminology Note; When this dispute typically occurs); pages 28-29; https://by.visa.com/dam/VCOM/global/support-legal/documents/merchants-dispute-management-guidelines.pdf
- [S3] *Visa Dispute Management Guide for Merchants: Best Practices, Tools and Resources*, Visa, 2026-10; Card network (international); tier 1; section: Visa Dispute Management Guide for Merchants > Conclusion and Next Steps > Questions to Ask Your Acquirer (also: Appendix: Dispute Handling Checklist; Step 1: Understand the dispute; Step 2: Identify the transaction type; Step 3: Assess response eligibility; Step 4: Evaluate your evidence); pages 109-110; https://by.visa.com/dam/VCOM/global/support-legal/documents/merchants-dispute-management-guidelines.pdf
- [S4] *Visa Dispute Management Guide for Merchants: Best Practices, Tools and Resources*, Visa, 2026-10; Card network (international); tier 1; section: Visa Dispute Management Guide for Merchants > Visa Dispute Categories and Conditions > How to Assess Whether CE3.0 May Apply (also: How CE3.0 Evaluates Prior Cardholder Activity); pages 86-87; https://by.visa.com/dam/VCOM/global/support-legal/documents/merchants-dispute-management-guidelines.pdf
- [S5] *Visa Dispute Management Guide for Merchants: Best Practices, Tools and Resources*, Visa, 2026-10; Card network (international); tier 1; section: Visa Dispute Management Guide for Merchants > Visa Dispute Categories and Conditions > How CE3.0 May Be Used During the Dispute Process (also: Before a Dispute Is Filed; After a Dispute Has Been Initiated); pages 88-89; https://by.visa.com/dam/VCOM/global/support-legal/documents/merchants-dispute-management-guidelines.pdf

## Run record

- Model lgbm_numeric-v1; index 4e38771857537a9d (lsa-tfidf-svd256, re-ranker lexical-v1, variant bm25); indicator map 1.0.0; prompts orchestrator@1.0.0, investigation@1.0.0, report@1.0.0; LLM scripted-test-model.
- Evidence policy: passed; claims cited 4/4, numbers traceable 0/0; first agent draft: 4/4 cited, 0/0 numbers traceable, 0 violation(s).
- 1 tool calls, 2 LLM calls, 0.07 s.
