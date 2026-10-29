"""Evaluation metrics for an alert-budget fraud model.

Ranking metrics use the raw model score (ties are rare); calibration metrics use the
calibrated probability. "Top k%" means the k% highest-scored transactions of the period
being evaluated: an analyst team that can review k% of traffic.

All intervals are percentile bootstrap intervals over transactions (1,000 resamples by
default). Transactions of one proxy entity are correlated, so these intervals are, if
anything, slightly too narrow; the entity-block bootstrap in `bootstrap(..., groups=...)`
is reported alongside for the headline metric.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from finguard.eda.style import wilson

KS = (0.01, 0.05)


def _topk(y_sorted: np.ndarray, k: float, total_pos: int) -> tuple[float, float]:
    n = max(int(round(k * len(y_sorted))), 1)
    hits = y_sorted[:n].sum()
    return hits / n, hits / max(total_pos, 1)


def ranking(y: np.ndarray, s: np.ndarray, ks: tuple[float, ...] = KS) -> dict:
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype="float64")
    ys = y[np.argsort(-s, kind="stable")]
    out = {"n": int(len(y)), "frauds": int(y.sum()), "prevalence": float(y.mean()),
           "pr_auc": float(average_precision_score(y, s)), "roc_auc": float(roc_auc_score(y, s))}
    for k in ks:
        p, r = _topk(ys, k, int(y.sum()))
        out[f"precision@{k:.0%}"] = float(p)
        out[f"recall@{k:.0%}"] = float(r)
    return out


def _weighted_topk(y_sorted: np.ndarray, w_sorted: np.ndarray, k: float) -> tuple[float, float]:
    cw = np.cumsum(w_sorted)
    n = max(round(k * cw[-1]), 1)
    take = cw <= n
    hits = float((y_sorted * w_sorted)[take].sum())
    return hits / max(float(w_sorted[take].sum()), 1.0), hits / max(float((y_sorted * w_sorted).sum()), 1.0)


def bootstrap(y: np.ndarray, s: np.ndarray, B: int = 1000, seed: int = 0, ks: tuple[float, ...] = KS,
              groups: np.ndarray | None = None) -> dict[str, tuple[float, float]]:
    """95% percentile intervals.

    Each resample is expressed as integer weights (how often each row was drawn). With
    `groups` (e.g. the proxy entity), whole groups are drawn: a block bootstrap that respects
    the correlation between transactions of one entity. Rows without a group are their own group.
    """
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype="float64")
    order = np.argsort(-s, kind="stable")
    y_sorted = y[order]
    rs = np.random.RandomState(seed)
    if groups is None:
        gid, n_groups = np.arange(len(y)), len(y)
    else:
        g = pd.Series(groups, dtype=object)
        key = g.where(g.notna(), pd.Series([f"__row{i}" for i in range(len(g))], dtype=object))
        gid = pd.factorize(key)[0]
        n_groups = int(gid.max()) + 1
    vals: dict[str, list[float]] = {"pr_auc": [], "roc_auc": []} | {f"{m}@{k:.0%}": [] for k in ks for m in ("precision", "recall")}
    for _ in range(B):
        w = np.bincount(rs.randint(0, n_groups, n_groups), minlength=n_groups)[gid].astype("float64")
        if (w * y).sum() == 0:
            continue
        vals["pr_auc"].append(average_precision_score(y, s, sample_weight=w))
        vals["roc_auc"].append(roc_auc_score(y, s, sample_weight=w))
        for k in ks:
            p, r = _weighted_topk(y_sorted, w[order], k)
            vals[f"precision@{k:.0%}"].append(p)
            vals[f"recall@{k:.0%}"].append(r)
    return {m: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for m, v in vals.items()}


def paired_difference(y: np.ndarray, s_a: np.ndarray, s_b: np.ndarray, B: int = 1000, seed: int = 0,
                      ks: tuple[float, ...] = KS) -> dict:
    """Metric(a) - metric(b) on the same resampled transactions: estimate, 95% CI, share of resamples <= 0."""
    y = np.asarray(y).astype(int)
    rs = np.random.RandomState(seed)
    names = ["pr_auc"] + [f"recall@{k:.0%}" for k in ks]
    diffs: dict[str, list[float]] = {n: [] for n in names}

    def metrics(yi, si):
        ys = yi[np.argsort(-si, kind="stable")]
        return [average_precision_score(yi, si)] + [_topk(ys, k, int(yi.sum()))[1] for k in ks]

    point = np.subtract(metrics(y, s_a), metrics(y, s_b))
    for _ in range(B):
        i = rs.randint(0, len(y), len(y))
        d = np.subtract(metrics(y[i], s_a[i]), metrics(y[i], s_b[i]))
        for n, v in zip(names, d):
            diffs[n].append(v)
    return {n: {"difference": float(p), "ci95": (float(np.percentile(diffs[n], 2.5)), float(np.percentile(diffs[n], 97.5))),
                "share_resamples_not_better": float(np.mean(np.asarray(diffs[n]) <= 0))} for n, p in zip(names, point)}


# --------------------------------------------------------------------------- calibration
def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error with equal-count bins (ranked, so plateaus do not merge bins)."""
    t = reliability(y, p, bins)
    return float((t["n"] / t["n"].sum() * (t["observed_rate"] - t["mean_predicted"]).abs()).sum())


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    d = pd.DataFrame({"y": np.asarray(y).astype(int), "p": np.asarray(p, dtype="float64")})
    d["bin"] = pd.qcut(d["p"].rank(method="first"), bins, labels=False)
    t = d.groupby("bin").agg(n=("y", "size"), frauds=("y", "sum"), mean_predicted=("p", "mean"), observed_rate=("y", "mean"))
    t["lo"], t["hi"] = wilson(t["frauds"].to_numpy(), t["n"].to_numpy())
    return t.reset_index()


def calibration(y: np.ndarray, p: np.ndarray, bins: int = 10) -> dict:
    y = np.asarray(y).astype(int)
    return {"brier": float(brier_score_loss(y, p)), "ece": ece(y, p, bins), "mean_predicted": float(np.mean(p)),
            "observed_rate": float(y.mean()),
            "brier_constant_rate": float(brier_score_loss(y, np.full(len(y), y.mean())))}


# --------------------------------------------------------------------------- over time and by group
def by_group(y: np.ndarray, s: np.ndarray, group: np.ndarray, min_frauds: int = 30, ks: tuple[float, ...] = KS) -> pd.DataFrame:
    """Ranking metrics per group. Top-k uses each group's own k% (group-local alert budget)."""
    rows = []
    for g in pd.unique(group):
        m = group == g
        if np.asarray(y)[m].sum() >= min_frauds and (np.asarray(y)[m] == 0).any():
            rows.append({"group": g} | ranking(np.asarray(y)[m], np.asarray(s)[m], ks))
    return pd.DataFrame(rows)


def weekly(y: np.ndarray, s: np.ndarray, day: np.ndarray, min_frauds: int = 30) -> pd.DataFrame:
    week = np.asarray(day) // 7
    t = by_group(y, s, week, min_frauds=min_frauds).rename(columns={"group": "week"}).sort_values("week")
    t["first_day"] = t["week"] * 7
    return t.reset_index(drop=True)
