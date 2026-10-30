"""Generate the FinGuard feature and reason-code glossary (a project document in the corpus).

    python -m finguard.rag.glossary     # writes corpus/docs/finguard_feature_reason_glossary.md

The glossary is generated from the code and the Phase 2 outputs, so it cannot drift from what
the model and the reason cards actually do. It is the "reason-code catalogue" deliverable of
Phase 2 and tier-1 "system" document of the knowledge base.
"""
from __future__ import annotations

import json
from pathlib import Path

from finguard import config
from finguard.explain import reasons as R
from finguard.features import novelty as NV
from finguard.features.build import ENTITY_FEATURES, FREQ_SOURCES, TRENDING_D

ART = Path(config.ROOT / "artifacts" / "phase2")
OUT = Path(config.ROOT / "corpus" / "docs" / "finguard_feature_reason_glossary.md")

FEATURE_DEFINITIONS = {
    "entity_prev_txn": "Number of earlier transactions of the same proxy entity in the data, before this one.",
    "entity_secs_since_prev": "Seconds since the previous transaction of the same proxy entity; empty for its first transaction.",
    "entity_txn_24h": "Number of earlier transactions of the same proxy entity in the 24 hours before this one.",
    "entity_txn_7d": "Number of earlier transactions of the same proxy entity in the 7 days before this one.",
    "entity_amt_ratio": "This amount divided by the average amount of the proxy entity's earlier transactions.",
    "entity_n_devices": "Number of distinct device brands seen so far on the proxy entity, this transaction included.",
    "entity_n_emails": "Number of distinct purchaser email domains seen so far on the proxy entity, this transaction included.",
    "entity_age_days": "Days since the proxy entity's first transaction in the data.",
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def render() -> str:
    res = _load(config.TABLE_DIR / "phase2_results.json")
    nov = _load(ART / "novelty" / "info.json")
    bands = {b["band"]: b for b in res.get("bands", {}).get("test", [])}
    thr = res.get("bands", {}).get("thresholds_calibrated", {})
    L = []
    a = L.append
    a("# FinGuard feature and reason-code glossary")
    a("")
    a("Publisher: FinGuard project team. Generated from the FinGuard code and the Phase 2 evaluation outputs. "
      "Scope: the IEEE-CIS card-not-present e-commerce dataset (Vesta, 2019).")
    a("")
    a("## 1. What the glossary covers")
    a("")
    a("This glossary explains every input, reason family, novelty flag and band that a FinGuard alert can show. "
      "IEEE-CIS has no customer, account or cardholder identifier. Every feature named entity_ describes a proxy entity: "
      "card1 + addr1 + first-seen day (transaction day minus D1). A proxy entity is a heuristic grouping, never a customer, "
      "and it can merge or split real people. SHAP reasons explain what moved the model's score, not why fraud happened. "
      "Every alert is flagged for review, not a finding of fraud.")
    a("")
    a("## 2. Risk score and bands")
    a("")
    a("The risk score is 100 times the calibrated fraud probability of the main LightGBM model. Isotonic calibration is fitted "
      "on the validation month (relative days 121-151). A score of 40 means that about 40% of similar transactions in the "
      "validation month turned out to be fraud. The score is capped at 1-99 and the displayed probability at <1% and >99%, "
      "because the calibrator's plateaus at exactly 0 and 1 would claim certainty.")
    a("")
    a("The bands follow analyst capacity. High is the top 1% of validation-month scores and Medium the next 4%; Low is the rest. "
      "The thresholds are frozen and applied to later transactions.")
    a("")
    if bands:
        a("| Band | Rule | Calibrated probability at the threshold | Test-month fraud rate | Test-month share of transactions |")
        a("| --- | --- | --- | --- | --- |")
        a(f"| High | top 1% of validation scores | at least {thr.get('high', 0):.0%} | {bands['High']['fraud_rate']:.1%} | {bands['High']['share_of_rows']:.2%} |")
        a(f"| Medium | next 4% | at least {thr.get('medium', 0):.0%} | {bands['Medium']['fraud_rate']:.1%} | {bands['Medium']['share_of_rows']:.2%} |")
        a(f"| Low | the rest | below {thr.get('medium', 0):.0%} | {bands['Low']['fraud_rate']:.1%} | {bands['Low']['share_of_rows']:.2%} |")
        a("")
    a("## 3. Reason families")
    a("")
    a("A reason card lists up to five reason families that pushed the score up most. Each family sums the TreeSHAP values "
      "(log-odds) of its inputs. Readable families state the transaction's own values. Anonymized families have no disclosed "
      "meaning and are never given one; the card names the anonymized signal group (for example V279-V321) that contributed most.")
    a("")
    a("| Reason family | Code | Meaning disclosed | Inputs |")
    a("| --- | --- | --- | --- |")
    disclosed = {"product": "code only (the five product codes are undisclosed)", "card": "partly (network and type; other card fields are codes)",
                 "address": "no (undisclosed codes)", "time": "relative only"}
    inputs = {
        "entity_velocity": "entity_prev_txn, entity_secs_since_prev, entity_txn_24h, entity_txn_7d",
        "entity_history": "entity_amt_ratio, entity_n_devices, entity_n_emails, entity_age_days",
        "amount": "TransactionAmt, amt_decimals",
        "product": "ProductCD",
        "card": "card1-card6, frequency of card1/card2/card3/card5 in the training window",
        "address": "addr1, addr2, dist1, dist2, frequency of addr1/addr2",
        "email": "P_emaildomain, R_emaildomain, provider groups, purchaser/recipient match, email frequency",
        "device": "DeviceType, browser family and major version, OS family and major version, device brand, screen width and height, device frequency",
        "identity_masked": "id_01-id_38 (except those replaced by browser, OS, device and screen features)",
        "time": "rel_hour, rel_weekday (relative to an undisclosed reference; real clock time unknown)",
        "counting_C": "C1-C14",
        "timedelta_D": "D1-D15, with D1, D4, D6, D8, D11-D15 re-expressed as day minus D (D1n, D4n, ...)",
        "match_M": "M1-M9",
        "vesta_V": "V1-V339",
        "completeness": "has_identity, n_missing, block_present_01-23, missing_pattern",
    }
    for fam, label in R.FAMILY_LABEL.items():
        a(f"| {label} | {fam} | {disclosed.get(fam, 'no' if fam in R.ANONYMIZED else 'yes')} | {inputs[fam]} |")
    a("")
    a("### 3.1 How each family is worded on a reason card")
    a("")
    examples = {
        "entity_velocity": "Proxy-entity velocity: 6 earlier transaction(s) of the same proxy entity in the previous 24 h, 9 in 7 days; previous one 40 s earlier.",
        "entity_history": "Proxy-entity history: amount is 3.2x its earlier average; 2 device brand(s) and 1 purchaser email domain(s) seen so far; first seen 4 day(s) earlier in the data.",
        "amount": "Amount: 117.00 (0 non-zero decimal place(s)).",
        "product": "Product code: C (product meaning not disclosed by the data provider).",
        "card": "Card attributes: network visa, type credit; card1 code seen 76 time(s) in the training window (card codes are undisclosed).",
        "address": "Address and distance codes: billing-region code 299, country code 87; distance field not given (undisclosed codes).",
        "email": "Email domains: purchaser gmail.com, recipient gmail.com.",
        "device": "Device and browser: mobile, brand samsung, OS android, browser chrome 63.",
        "time": "Relative time of day: relative hour 7 (the real clock time and time zone are not disclosed).",
        "completeness": "Data completeness: 234 empty fields; identity record absent.",
        "match_M": "Match flags (M1-M9) (meaning not disclosed): M4=M0, M5 not given, M6=F.",
    }
    for fam, label in R.FAMILY_LABEL.items():
        text = examples.get(fam, f"{label}: meaning not disclosed by the data provider.")
        a(f"- **{label}** ({fam}). Example: \"{text}\"")
    a("")
    a("## 4. Engineered features")
    a("")
    a("### 4.1 Proxy-entity features")
    a("")
    a("Computed by replaying transactions in time order, from earlier transactions of the same proxy entity only. "
      "Transactions without a proxy-entity key (card1, addr1 or D1 missing) get empty values, never a guessed key.")
    a("")
    a("| Feature | Definition |")
    a("| --- | --- |")
    for f in ENTITY_FEATURES:
        a(f"| {f} | {FEATURE_DEFINITIONS[f]} |")
    a("")
    a("### 4.2 Other engineered features")
    a("")
    a("| Feature | Definition |")
    a("| --- | --- |")
    for name, src in FREQ_SOURCES.items():
        a(f"| {name} | Number of training-window transactions (relative days 0-120) with the same {src}; 0 when the value was never seen. |")
    a(f"| D1n, D4n, D6n, D8n, D11n-D15n | Transaction day minus the D column ({', '.join(TRENDING_D)}). Removes the calendar trend of these time deltas. |")
    a("| amt_decimals | Number of non-zero decimal places in the amount (0-3). |")
    a("| browser_family, browser_major | Browser family (samsung, edge, chrome, firefox, opera, safari, android webview, internet explorer, other) and major version, from id_31. |")
    a("| os_family, os_major | Operating-system family (windows, ios, mac, android, linux, other) and major version, from id_30. |")
    a("| device_brand | Device brand from DeviceInfo (for example samsung, apple ios, apple mac, windows, motorola, lg, huawei, xiaomi, google, other). |")
    a("| screen_w, screen_h | Screen width and height in pixels, from id_33 (for example 2208x1242). |")
    a("| email_group_P, email_group_R | Email provider group of the purchaser and recipient domains (gmail, yahoo, microsoft, apple, aol, anonymous, protonmail, other). |")
    a("| email_match | 1 when purchaser and recipient email domains are equal; empty when either is missing. |")
    a("| has_identity | 1 when the transaction has an identity record (device and network data). |")
    a("| n_missing | Number of empty fields in the transaction. |")
    a("| block_present_01-23 | 1 when a group of columns that are always missing together is present. |")
    a("| missing_pattern | Code of the transaction's combination of present and missing column groups. |")
    a("| rel_hour, rel_weekday | Hour and day of week relative to the dataset's undisclosed reference time. |")
    a("")
    a("## 5. Novelty flags")
    a("")
    a("Novelty flags are explainable indicators computed from the proxy entity's earlier transactions. They are shown to the "
      "analyst next to the score. They are not part of the risk score: a combiner fitted on the validation month did not "
      "improve it.")
    a("")
    a("| Flag | Rule |")
    a("| --- | --- |")
    for f in NV.FLAGS:
        a(f"| {f} | {NV.FLAG_TEXT[f]}. |")
    a("")
    rates = [r for r in nov.get("rates", []) if r["part"] == "test"]
    if rates:
        a("Test-month fraud rates (transactions with a proxy-entity key):")
        a("")
        a("| Flag | Flagged transactions | Fraud rate when flagged | Fraud rate when not flagged |")
        a("| --- | --- | --- | --- |")
        for f in NV.FLAGS:
            on = next(r for r in rates if r["flag"] == f and r["group"] == "flag on")
            off = next(r for r in rates if r["flag"] == f and r["group"] == "flag off")
            a(f"| {f} | {on['n']:,} | {on['fraud_rate']:.1%} | {off['fraud_rate']:.1%} |")
        a("")
    a("## 6. Anomaly score")
    a("")
    a("An Isolation Forest was tested as an unsupervised detector. Its most unusual transactions were almost never fraud, so it "
      "is not part of the risk score and is not shown on alerts.")
    a("")
    a("## 7. Limits of these explanations")
    a("")
    a("- About 85% of the model's inputs (403 of 474 columns) have no disclosed meaning.")
    a("- Proxy-entity history is a heuristic grouping, not a verified identity.")
    a("- Relative time is not clock time: the dataset's reference time and time zone are not disclosed.")
    a("- Fairness across customer groups cannot be measured: the data holds no customer attributes.")
    return "\n".join(L) + "\n"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render(), encoding="utf-8")
    from finguard.rag.corpus import record_hash
    digest = record_hash("finguard_feature_reason_glossary")
    print(f"wrote {OUT} (sha256 {digest[:16]}... recorded in corpus/registry.json)")


if __name__ == "__main__":
    main()
