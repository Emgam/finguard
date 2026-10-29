"""Phase 2 figures (fig17-fig24), same visual system as Phase 1 (finguard.eda.style).

Each function takes small, already-computed tables so the numbers in a figure are the
numbers in the report tables. Titles state the finding; subtitles state the method.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from finguard import config
from finguard.eda import style as S

SOURCE2 = ("Source: IEEE-CIS labelled file, time split days 0-120 / 121-151 / 152-181; test month = days 152-181 "
           "(85,430 transactions), FinGuard Phase 2")


def _labels(ax, y, vals, texts, xmax):
    for yp, v, t in zip(y, vals, texts):
        ax.text(v + xmax * 0.012, yp, t, va="center", fontsize=8.5, color=S.INK2)


# ----------------------------------------------------------------------------- 17
def fig17_model_comparison(t: pd.DataFrame, prevalence: float):
    """t: rows = models (label, pr_auc, lo, hi, highlight bool), top to bottom in display order."""
    fig, ax = S.new(10.5, 4.6)
    ypos = np.arange(len(t))[::-1]
    colors = [S.ACCENT if h else S.GREY for h in t.highlight]
    ax.barh(ypos, t.pr_auc, height=0.6, color=colors, zorder=2)
    ax.errorbar(t.pr_auc, ypos, xerr=[t.pr_auc - t.lo, t.hi - t.pr_auc], fmt="none", ecolor=S.INK2, elinewidth=0.9, capsize=3, zorder=3)
    xmax = float(t.hi.max())
    _labels(ax, ypos, t.hi, [f"{v:.3f}  [{lo:.3f}-{hi:.3f}]" for v, lo, hi in zip(t.pr_auc, t.lo, t.hi)], xmax)
    ax.set_yticks(ypos, t.label)
    ax.set_xlim(0, xmax * 1.33)
    ax.set_xlabel("PR-AUC (average precision) on the test month")
    S.grid(ax, "x")
    S.ref_line(ax, prevalence, f"random {prevalence:.3f}")
    main, lr = t[t.highlight].iloc[0], t[t.label.str.startswith("Logistic")].iloc[0]
    S.headline(fig, f"LightGBM reaches PR-AUC {main.pr_auc:.3f} on the unseen test month, "
                    f"{main.pr_auc / lr.pr_auc:.2f}x the logistic-regression baseline ({lr.pr_auc:.3f})",
               "Average precision, 95% bootstrap intervals (1,000 resamples); trained on days 0-120 "
               "(0-90 for label delay), early-stopped on 121-151",
               source=SOURCE2)
    fig.subplots_adjust(left=0.33, bottom=0.17)
    S.save(fig, "fig17_model_comparison")
    return fig


# ----------------------------------------------------------------------------- 18
def fig18_pr_curves(y: np.ndarray, scores: dict[str, np.ndarray], main: str, ks=(0.01, 0.05)):
    fig, ax = S.new(9.5, 5.6)
    for name, s in scores.items():
        p, r, _ = precision_recall_curve(y, s)
        is_main = name == main
        ax.plot(r, p, color=S.ACCENT if is_main else S.GREY_DARK, linewidth=2.2 if is_main else 1.6,
                zorder=3 if is_main else 2, label=name)
        order = np.argsort(-s, kind="stable")
        ys = y[order]
        for k in ks:
            n = int(round(k * len(y)))
            prec, rec = ys[:n].mean(), ys[:n].sum() / y.sum()
            ax.plot(rec, prec, "o", markersize=8, color=S.ACCENT if is_main else S.GREY_DARK,
                    markeredgecolor=S.SURFACE, markeredgewidth=2, zorder=4)
            if is_main:
                ax.annotate(f"top {k:.0%}: precision {prec:.0%}, recall {rec:.0%}", (rec, prec), xytext=(10, 4),
                            textcoords="offset points", fontsize=8.5, color=S.INK)
    ax.axhline(y.mean(), color=S.GREY_DARK, linestyle=(0, (4, 3)), linewidth=1)
    ax.text(0.01, y.mean() + 0.012, f"random = prevalence {y.mean():.1%}", ha="left", fontsize=8, color=S.INK2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall (share of test frauds caught)")
    ax.set_ylabel("Precision (share of alerts that are fraud)")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
    ax.legend(loc="upper right")
    S.grid(ax, "both")
    ys = y[np.argsort(-scores[main], kind="stable")]
    n1, n5 = int(round(ks[0] * len(y))), int(round(ks[1] * len(y)))
    p1, r5 = ys[:n1].mean(), ys[:n5].sum() / y.sum()
    S.headline(fig, f"Reviewing the top {ks[0]:.0%}: {p1:.0%} of alerts are fraud; reviewing the top {ks[1]:.0%} catches "
                    f"{r5:.0%} of all test fraud",
               "Precision-recall curves on the test month; dots mark the top 1% and top 5% of transactions by score",
               source=SOURCE2)
    S.save(fig, "fig18_precision_recall_curves")
    return fig


# ----------------------------------------------------------------------------- 19
def fig19_reliability(rel_raw: pd.DataFrame, rel_cal: pd.DataFrame, ece_raw: float, ece_cal: float):
    """rel_*: finguard.evaluation.metrics.reliability() tables on the test month."""
    fig, ax = S.new(8.6, 6.0)
    both = pd.concat([rel_raw, rel_cal])
    lo_x = 10 ** np.floor(np.log10(max(float(both.mean_predicted.min()), 1e-7)))
    lo_y = 10 ** np.floor(np.log10(max(float(both.lo.min()), 1e-7)))
    d0 = max(lo_x, lo_y)
    ax.plot([d0, 1], [d0, 1], color=S.GREY_DARK, linestyle=(0, (4, 3)), linewidth=1, zorder=1)
    for t, color, label in [(rel_raw, S.GREY_DARK, f"uncalibrated (ECE {ece_raw:.4f})"),
                            (rel_cal, S.ACCENT, f"isotonic, fitted on validation (ECE {ece_cal:.4f})")]:
        x = t.mean_predicted.clip(lower=lo_x)
        yv = t.observed_rate.clip(lower=lo_y)
        ax.errorbar(x, yv, yerr=[yv - t.lo.clip(lower=lo_y), t.hi.clip(lower=lo_y) - yv], fmt="o-", color=color, markersize=6,
                    elinewidth=0.8, capsize=2, markeredgecolor=S.SURFACE, markeredgewidth=1.5, label=label, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo_x, 1)
    ax.set_ylim(lo_y, 1)
    ax.set_xlabel("Mean predicted fraud probability (20 equal-size bins, log scale)")
    ax.set_ylabel("Observed fraud rate (log scale)")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v * 100:g}%")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v * 100:g}%")
    ax.legend(loc="upper left")
    S.grid(ax, "both")
    verb = "cuts" if ece_cal < ece_raw else "does not cut"
    S.headline(fig, f"Isotonic calibration {verb} test-month ECE from {ece_raw:.4f} to {ece_cal:.4f}",
               "Fitted on the validation month; test month shown; dashed line = perfect calibration.\nEach dot = 1/20 of "
               "transactions, 95% Wilson intervals; bins on one isotonic plateau share a predicted value",
               source=SOURCE2, top=0.82)
    S.save(fig, "fig19_reliability")
    return fig


# ----------------------------------------------------------------------------- 20
def fig20_bands(valid_t: pd.DataFrame, test_t: pd.DataFrame):
    """band tables from finguard.models.calibration.band_table (validation and test)."""
    fig, (a1, a2) = S.new(11, 4.9, ncols=2, gridspec_kw={"width_ratios": [1.25, 1]})
    bands = list(test_t.band)
    x = np.arange(len(bands))
    w = 0.36
    for off, t, color, lab in [(-w / 2, valid_t, S.GREY, "validation (thresholds set here)"), (w / 2, test_t, S.ACCENT, "test month")]:
        a1.bar(x + off, t.fraud_rate, width=w, color=color, zorder=2, label=lab)
        a1.errorbar(x + off, t.fraud_rate, yerr=[t.fraud_rate - t.lo, t.hi - t.fraud_rate], fmt="none", ecolor=S.INK2,
                    elinewidth=0.8, capsize=2, zorder=3)
        for xi, v in zip(x + off, t.fraud_rate):
            a1.text(xi, v + 0.025, f"{v:.1%}", ha="center", fontsize=8.5, color=S.INK, fontweight="bold" if color == S.ACCENT else None)
    a1.set_xticks(x, [f"{b}\n({p:.0%} of traffic planned)" for b, p in zip(bands, test_t.planned_share)])
    a1.set_ylim(0, 1.08)
    a1.yaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
    a1.set_title("Fraud rate inside each band", fontsize=10)
    a1.legend(loc="upper right")
    S.grid(a1, "y")
    a2.barh([1, 0], [test_t.share_of_rows.iloc[0] + test_t.share_of_rows.iloc[1], test_t.share_of_frauds.iloc[0] + test_t.share_of_frauds.iloc[1]],
            color=S.GREY, height=0.5, zorder=2)
    a2.barh([1, 0], [test_t.share_of_rows.iloc[0], test_t.share_of_frauds.iloc[0]], color=S.ACCENT, height=0.5, zorder=3)
    a2.text(test_t.share_of_rows.iloc[:2].sum() + 0.01, 1, f"{test_t.share_of_rows.iloc[:2].sum():.1%} of test traffic", va="center", fontsize=8.5)
    a2.text(test_t.share_of_frauds.iloc[:2].sum() + 0.01, 0, f"{test_t.share_of_frauds.iloc[:2].sum():.1%} of test frauds", va="center", fontsize=8.5)
    a2.set_yticks([1, 0], ["High + Medium\nshare of traffic", "High + Medium\nshare of frauds"])
    a2.set_xlim(0, 1)
    a2.xaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
    a2.set_title("Alert load vs fraud captured (blue = High only)", fontsize=10)
    S.grid(a2, "x")
    S.headline(fig, f"On the test month {test_t.fraud_rate.iloc[0]:.0%} of High-band and {test_t.fraud_rate.iloc[1]:.0%} of "
                    f"Medium-band transactions are fraud, vs {test_t.fraud_rate.iloc[2]:.1%} in Low",
               "Bands = top 1% / next 4% / rest of validation-month scores; thresholds frozen and applied to the test month; "
               "95% Wilson intervals", source=SOURCE2, top=0.82)
    fig.subplots_adjust(wspace=0.32, bottom=0.2, left=0.06)
    S.save(fig, "fig20_risk_bands")
    return fig


# ----------------------------------------------------------------------------- 21
def fig21_weekly(wk: pd.DataFrame):
    """wk: columns first_day, pr_auc, prevalence, part ('valid'/'test')."""
    fig, ax = S.new(11.5, 5.0)
    for part, (lab, col) in {"valid": ("Validation (early stopping, calibration)", "#e9eef6"), "test": ("Test", "#e3e2dc")}.items():
        a, b = config.SPLIT_DAYS[part]
        ax.axvspan(a, b + 1, color=col, zorder=0)
        ax.text((a + b + 1) / 2, 0.97, lab, ha="center", va="top", fontsize=8.5, color=S.INK2, transform=ax.get_xaxis_transform())
    x = wk.first_day + 3.5
    ax.plot(x, wk.pr_auc, color=S.ACCENT, marker="o", markersize=6, markeredgecolor=S.SURFACE, markeredgewidth=1.5, zorder=3)
    for xi, v in zip(x, wk.pr_auc):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=8, color=S.INK2)
    ax.set_xlim(config.SPLIT_DAYS["valid"][0], config.SPLIT_DAYS["test"][1] + 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Relative day (7-day windows from the start of validation)")
    ax.set_ylabel("PR-AUC within the week")
    S.grid(ax, "y")
    first, third = wk.pr_auc.iloc[0], wk.pr_auc.iloc[2]
    later = wk.pr_auc.iloc[2:]
    S.headline(fig, f"Weekly PR-AUC drops from {first:.2f} to {third:.2f} in the three weeks after training, "
                    f"then holds at {later.min():.2f}-{later.max():.2f}",
               "Main LightGBM, fixed after training on days 0-120; validation weeks also chose the stopping round; "
               f"weekly fraud prevalence varies ({wk.prevalence.min():.1%}-{wk.prevalence.max():.1%})",
               source=SOURCE2)
    fig.subplots_adjust(bottom=0.17)
    S.save(fig, "fig21_weekly_performance")
    return fig


# ----------------------------------------------------------------------------- 22
def fig22_entity_groups(t: pd.DataFrame):
    """t: rows = entity status groups (label, n, share_rows, prevalence, pr_auc, lo, hi)."""
    fig, (a1, a2) = S.new(11, 4.7, ncols=2, gridspec_kw={"width_ratios": [1.4, 1]})
    ypos = np.arange(len(t))[::-1]
    hi_mask = t.label.str.contains("never seen")
    a1.barh(ypos, t.pr_auc, height=0.6, color=[S.ACCENT if h else S.GREY for h in hi_mask], zorder=2)
    a1.errorbar(t.pr_auc, ypos, xerr=[t.pr_auc - t.lo, t.hi - t.pr_auc], fmt="none", ecolor=S.INK2, elinewidth=0.8, capsize=2, zorder=3)
    a1.scatter(t.prevalence, ypos, marker="|", s=220, color=S.INK, zorder=4)
    _labels(a1, ypos, t.hi, [f"{v:.3f}" for v in t.pr_auc], 1.0)
    a1.set_yticks(ypos, t.label)
    a1.set_xlim(0, 1.1)
    a1.set_xlabel("PR-AUC within the group (test month); black tick = prevalence (random)")
    S.grid(a1, "x")
    a2.barh(ypos, t.share_rows, height=0.6, color=S.GREY, zorder=2)
    _labels(a2, ypos, t.share_rows, [f"{s:.1%} of rows, fraud {p:.1%}" for s, p in zip(t.share_rows, t.prevalence)], 1.0)
    a2.set_yticks(ypos, [""] * len(t))
    a2.set_xlim(0, 1.0)
    a2.xaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
    a2.set_title("Size and fraud rate of each group", fontsize=10)
    S.grid(a2, "x")
    new = t[hi_mask].iloc[0]
    known = t[t.label.str.contains("had a fraud label")].iloc[0]
    S.headline(fig, f"Entities never seen in training are the honest test: PR-AUC {new.pr_auc:.3f}, "
                    f"vs {known.pr_auc:.3f} for entities that already had a fraud label",
               "Proxy entity = card1 + addr1 + (day - D1), a heuristic grouping, not a customer; groups by what "
               "days 0-120 contained; 95% bootstrap intervals",
               source=SOURCE2, top=0.82)
    fig.subplots_adjust(left=0.27, wspace=0.08, bottom=0.17)
    S.save(fig, "fig22_new_vs_known_entities")
    return fig


# ----------------------------------------------------------------------------- 23
def fig23_family_importance(t: pd.DataFrame):
    """t: rows = families (label, share, anonymized bool), sorted by share descending."""
    fig, ax = S.new(10.5, 5.8)
    ypos = np.arange(len(t))[::-1]
    ax.barh(ypos, t.share, height=0.62, color=[S.GREY if a else S.ACCENT for a in t.anonymized], zorder=2)
    _labels(ax, ypos, t.share, [f"{v:.1%}" for v in t.share], float(t.share.max()))
    ax.set_yticks(ypos, t.label)
    ax.set_xlim(0, float(t.share.max()) * 1.2)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
    ax.set_xlabel("Share of total mean |SHAP| (5,000 random test transactions)")
    S.grid(ax, "x")
    anon = float(t[t.anonymized].share.sum())
    S.headline(fig, f"{anon:.0%} of the model's attribution sits in anonymized families; readable families carry {1 - anon:.0%}",
               "TreeSHAP values from LightGBM summed per reason family; blue = readable or engineered families an analyst can check",
               source=SOURCE2)
    fig.subplots_adjust(left=0.3)
    S.save(fig, "fig23_reason_family_importance")
    return fig


# ----------------------------------------------------------------------------- 24
def fig24_reason_stability(overlap: np.ndarray, freq: pd.DataFrame, target: float = 0.6):
    """overlap: per alerted transaction mean pairwise top-5 overlap; freq: (label, share_in_top5, anonymized)."""
    fig, (a1, a2) = S.new(12.5, 5.2, ncols=2, gridspec_kw={"width_ratios": [1, 1.15]})
    bins = np.linspace(0, 1, 11)
    a1.hist(overlap, bins=bins, color=S.GREY, edgecolor=S.SURFACE, zorder=2)
    mean = float(np.mean(overlap))
    a1.axvline(mean, color=S.ACCENT, linewidth=2, zorder=3)
    a1.text(mean - 0.01, a1.get_ylim()[1] * 0.95, f"mean {mean:.2f}", ha="right", va="top", fontsize=9, fontweight="bold", color=S.INK)
    a1.axvline(target, color=S.GREY_DARK, linestyle=(0, (4, 3)), linewidth=1, zorder=3)
    a1.text(target - 0.01, a1.get_ylim()[1] * 0.85, f"target {target:.1f}", ha="right", va="top", fontsize=8.5, color=S.INK2)
    a1.set_xlabel("Mean pairwise overlap of top-5 families (5 retrains)")
    a1.set_ylabel("Alerted test transactions")
    S.grid(a1, "y")
    ypos = np.arange(len(freq))[::-1]
    a2.barh(ypos, freq.share_in_top5, height=0.62, color=[S.GREY if a else S.ACCENT for a in freq.anonymized], zorder=2)
    _labels(a2, ypos, freq.share_in_top5, [f"{v:.0%}" for v in freq.share_in_top5], 1.0)
    a2.set_yticks(ypos, freq.label)
    a2.set_xlim(0, 1.15)
    a2.xaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
    a2.set_title("How often each family is among the top 5 reasons (main model)", fontsize=10)
    S.grid(a2, "x")
    verdict = "stable" if mean >= target else "less stable than targeted"
    S.headline(fig, f"Reason families are {verdict} across retrains: top-5 overlap {mean:.2f} on average (target at least {target:.1f})",
               "High- and Medium-band test transactions; 5 LightGBM retrains with different random seeds, same data and settings",
               source=SOURCE2, top=0.84)
    fig.subplots_adjust(wspace=0.72, left=0.07, bottom=0.15)
    S.save(fig, "fig24_reason_stability")
    return fig
