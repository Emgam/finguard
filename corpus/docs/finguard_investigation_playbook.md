# FinGuard investigation playbook (fictional)

Publisher: FinGuard project team. Status: FICTIONAL. This playbook was written for the FinGuard prototype. It is not the procedure of any real institution. Roles, response times and amount limits are illustrative and must be replaced by the host institution's own rules.

## 1. Purpose and scope

The playbook tells a fraud analyst how to work a FinGuard alert on a card-not-present (CNP) e-commerce payment, from triage to a documented decision. FinGuard recommends and the analyst decides. FinGuard never blocks a payment, contacts a customer, labels a customer as fraudulent or files a report. Every FinGuard alert reads "Flagged for review, not a finding of fraud".

## 2. Roles

| Role | Responsibility |
| --- | --- |
| Level-1 fraud analyst | Works High and Medium alerts, records the decision and the evidence |
| Level-2 investigator | Takes escalated cases, linked cases and disputed decisions |
| Compliance officer (AML) | Decides whether a suspicion of money laundering must be reported to the financial intelligence unit |
| Team lead | Sets daily alert capacity, reviews quality samples, approves threshold changes |

## 3. Triage by band

| Band | Queue | Fictional response time | Who works it |
| --- | --- | --- | --- |
| High | Priority queue | Start within 2 business hours | Level-1 analyst; escalate when section 7 applies |
| Medium | Standard queue | Start within 1 business day | Level-1 analyst |
| Low | Not queued | None | A 1% random sample goes to weekly quality review |

When the daily alert volume exceeds capacity, the team lead works High alerts first and records the backlog. The band shares are a drift alarm: if more than 6% of a day's transactions land in High or Medium (the plan is 5%), the team lead informs the model owner.

## 4. Investigation steps

1. Confirm the transaction: TransactionID, amount, product code, card network and type, time relative to the data's reference.
2. Read the risk score and band. The calibrated probability is secondary to the band; it is capped at ">99%" and "<1%".
3. Read the reason card. Separate checkable reasons (amount, card, address codes, email, device, proxy-entity velocity and history, data completeness) from anonymized reasons (V, C, D, M and id groups). Only checkable reasons can be verified against the transaction.
4. Review the proxy-entity history: earlier transactions, bursts in the last 24 hours and 7 days, time since the previous transaction, amount against the earlier average, device brands and email domains seen so far. Remember that the proxy entity is a heuristic grouping, not a customer.
5. Review the novelty flags: new device brand for the proxy entity, burst within 24 hours of first use, amount far above the proxy entity's earlier maximum.
6. Retrieve guidance with the knowledge base: the fraud typology the indicators are consistent with, the card-network dispute rules, and the reporting obligations if fraud is confirmed. Cite every passage used.
7. Decide one outcome from section 6 and document it (section 8).

## 5. Indicators and what to check next

| Indicator on the alert | What to check next | Typology to look up |
| --- | --- | --- |
| Several transactions on the proxy entity within minutes or hours | Amounts, products and devices of the burst; whether earlier transactions in the burst were declined or disputed | Card testing, account takeover |
| New device brand for the proxy entity | Whether the device change coincides with a new email domain or a change of shipping details | Account takeover |
| Amount far above the proxy entity's earlier amounts | Earlier amounts and products; whether the purchase type changed | Account takeover, stolen card used for high-value goods |
| Purchaser and recipient email domains differ, or an anonymous email domain | Domain history on the proxy entity | Stolen card details used by a third party |
| Identity record absent and many empty fields | Whether this is normal for the product code | Not an indicator on its own |
| Transaction cannot be linked to a proxy entity | Which of card1, addr1 or D1 is missing | Not an indicator on its own |

## 6. Outcomes

| Outcome | When | What the analyst does |
| --- | --- | --- |
| Confirmed fraud | The cardholder or issuer confirms the transaction was not authorised, or a fraud chargeback is received | Record the outcome and the evidence; follow the card-network dispute process; consider escalation under section 7 |
| False positive | The evidence supports a legitimate transaction | Close the case with the reason; record which reasons misled |
| Inconclusive | Evidence is missing | Keep the case open for up to 5 business days (fictional), then close as inconclusive |

Outcomes are recorded with their date and the model version, because confirmed labels arrive late and only for reviewed alerts.

## 7. Escalation

Escalate to a Level-2 investigator when any of these hold (fictional limits):

- the alert is linked to three or more other alerts in the last 30 days through the same card, address code, device or email domain;
- the amount exceeds 5,000 in the transaction currency;
- the analyst disagrees with a High band after completing section 4.

Escalate to the compliance officer when the case suggests money laundering or organised fraud: funds moved on through mule accounts, repeated payments structured to stay under limits, or links to other confirmed fraud cases. The compliance officer, not the analyst, decides whether a suspicious transaction report is filed with the financial intelligence unit under the applicable law. The analyst must never tell the customer that a report is being considered.

## 8. Documentation

Each case record holds: the TransactionID; the band, score and model version; the reason card as shown; the proxy-entity history consulted; the knowledge-base passages cited, with document, section and page; the outcome and its reason; the analyst and the date. FinGuard case reports carry five sections: Facts, Interpretation, Guidance, Suggested checks and Limitations.

## 9. What analysts must not do

- Treat the proxy entity as a customer or as a verified identity.
- Give a meaning to an anonymized feature group.
- Read a SHAP reason as the cause of the fraud; it explains the model's score only.
- Block a card, refuse a payment or contact a customer on the score alone.
- Quote guidance without its source, date and jurisdiction.

## 10. Quality review

Each week the team lead reviews a sample of closed cases: 10 High, 10 Medium and the 1% Low sample. The review checks the evidence, the citations and the outcome. Disagreements are recorded and fed back to the model owner with the TransactionIDs.
