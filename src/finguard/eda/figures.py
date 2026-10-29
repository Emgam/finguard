"""The 16 prioritised Phase 1 figures (critical review, section 5).

Every function takes prepared data, draws one figure, saves it to reports/figures and
returns the matplotlib Figure. Titles are computed from the data so they cannot drift
from the numbers they describe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from finguard import config
from finguard.eda import style as S
from finguard.features.normalize import amount_decimals, browser_family, os_family

Y = config.TARGET


def _base(df: pd.DataFrame) -> float:
    return float(df[Y].mean())


def _hbars(ax, t: pd.DataFrame, labels: list[str], base: float, highlight: np.ndarray | None = None,
           show_n: bool = True, xmax: float | None = None) -> None:
    """Horizontal fraud-rate bars with Wilson whiskers and value labels."""
    ypos = np.arange(len(t))[::-1]
    colors = [S.ACCENT if (highlight is not None and h) else S.GREY for h in (highlight if highlight is not None else [False] * len(t))]
    ax.barh(ypos, t.rate, height=0.62, color=colors, zorder=2)
    ax.errorbar(t.rate, ypos, xerr=[t.rate - t.lo, t.hi - t.rate], fmt="none", ecolor=S.INK2, elinewidth=0.8, capsize=2, zorder=3)
    for yp, (_, r) in zip(ypos, t.iterrows()):
        txt = S.pct(r.rate) + (f"   n={int(r.n):,}" if show_n else "")
        ax.text(r.hi + (xmax or t.hi.max()) * 0.012, yp, txt, va="center", fontsize=8, color=S.INK2)
    ax.set_yticks(ypos, labels)
    ax.set_xlim(0, (xmax or t.hi.max()) * 1.32)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
    S.grid(ax, "x")
    S.ref_line(ax, base, f"base {S.pct(base)}")


# ----------------------------------------------------------------------------- 1
def fig01_fraud_rate_by_segment(df: pd.DataFrame):
    base = _base(df)
    dims = {"ProductCD": df.ProductCD, "card type (card6)": df.card6, "network (card4)": df.card4,
            "DeviceType": df.DeviceType, "identity record": df.has_identity.map({1: "present", 0: "absent"})}
    parts = []
    for name, s in dims.items():
        t = S.rate_table(s, df[Y], min_n=1000)
        t.index = [f"{name} = {k}" for k in t.index]
        parts.append(t)
    t = pd.concat(parts).sort_values("rate", ascending=False)
    hi = (t.rate >= 2 * base).to_numpy()
    fig, ax = S.new(10.5, 6.6)
    _hbars(ax, t, list(t.index), base, hi)
    top = t.index[0]
    S.headline(fig, f"{hi.sum()} of {len(t)} segments run at twice the base rate or more; {top} leads at {S.pct(t.rate.iloc[0])}",
               "Fraud rate per segment with 95% Wilson intervals; segments with at least 1,000 transactions; blue = at least 2x the base rate",
               top=0.86)
    fig.subplots_adjust(left=0.27)
    S.save(fig, "fig01_fraud_rate_by_segment")
    return fig


# ----------------------------------------------------------------------------- 2
def fig02_weekly_fraud_rate(df: pd.DataFrame):
    w = df.groupby(df.day // 7)[Y].agg(n="size", frauds="sum")
    w["rate"] = w.frauds / w.n
    base = _base(df)
    fig, ax = S.new(10.5, 4.8)
    x = (w.index.to_numpy() * 7 + 3.5)
    shades = {"valid": ("Validation", "#e9eef6"), "test": ("Test", "#e3e2dc")}
    for k, (lab, col) in shades.items():
        a, b = config.SPLIT_DAYS[k]
        ax.axvspan(a, b + 1, color=col, zorder=0)
        ax.text((a + b + 1) / 2, 0.064, lab, ha="center", fontsize=8.5, color=S.INK2)
    a, b = config.SPLIT_DAYS["train"]
    ax.text((a + b + 1) / 2, 0.064, "Train", ha="center", fontsize=8.5, color=S.INK2)
    ax.plot(x, w.rate, color=S.ACCENT, marker="o", markersize=4.5, zorder=3)
    for idx in (w.rate.idxmax(), w.rate.idxmin()):
        ax.annotate(S.pct(w.rate[idx]), (idx * 7 + 3.5, w.rate[idx]), xytext=(0, 9 if idx == w.rate.idxmax() else -15),
                    textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold", color=S.INK)
    S.ref_line(ax, base, f"overall {S.pct(base)}", orient="h")
    ax.set_ylim(0, 0.07)
    ax.set_xlim(0, 182)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
    ax.set_xlabel("Relative day of the training file (calendar dates undisclosed)")
    S.grid(ax, "y")
    S.headline(fig, f"Weekly fraud rate swings between {S.pct(w.rate.min())} and {S.pct(w.rate.max())}: the target is not stationary",
               "Share of transactions labelled fraud per week; shaded windows are the time split of Project Report v3.0 (6.4)")
    fig.subplots_adjust(bottom=0.16)
    S.save(fig, "fig02_weekly_fraud_rate")
    return fig


# ----------------------------------------------------------------------------- 3
def fig03_daily_volume(train: pd.DataFrame, test: pd.DataFrame):
    dtr, dte = train.day.value_counts().sort_index(), test.day.value_counts().sort_index()
    gap = int(dte.index.min() - dtr.index.max() - 1)
    fig, ax = S.new(10.5, 4.6)
    ax.fill_between(dtr.index, dtr.values, color=S.GREY, alpha=0.55, linewidth=0)
    ax.plot(dtr.index, dtr.values, color=S.GREY_DARK, linewidth=1.2)
    ax.fill_between(dte.index, dte.values, color=S.ACCENT, alpha=0.18, linewidth=0)
    ax.plot(dte.index, dte.values, color=S.ACCENT, linewidth=1.2)
    ymax = max(dtr.max(), dte.max())
    ax.text(float(np.mean(dtr.index.to_numpy())), ymax * 1.05, f"Labelled training file\n{len(train):,} transactions, {dtr.index.max() + 1} days",
            ha="center", fontsize=8.5, color=S.INK2)
    ax.text(float(np.mean(dte.index.to_numpy())), ymax * 1.05, f"Unlabelled later file\n{len(test):,} transactions, {dte.index.max() - dte.index.min() + 1} days",
            ha="center", fontsize=8.5, color=S.INK2)
    ax.annotate("", xy=(dte.index.min(), ymax * 0.5), xytext=(dtr.index.max(), ymax * 0.5),
                arrowprops=dict(arrowstyle="<->", color=S.INK2, lw=0.9))
    ax.text((dte.index.min() + dtr.index.max()) / 2, ymax * 0.53, f"{gap}-day gap", ha="center", fontsize=8.5, color=S.INK)
    ax.set_ylim(0, ymax * 1.25)
    ax.set_xlabel("Relative day (day 0 = first training transaction)")
    ax.set_ylabel("Transactions per day")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k")
    S.grid(ax, "y")
    S.headline(fig, f"The unlabelled file starts {gap} days after training ends: a real later period for drift monitoring",
               "Daily transaction counts of both files on one relative clock; the later file has no labels and is never used for metrics")
    fig.subplots_adjust(bottom=0.17)
    S.save(fig, "fig03_daily_volume_train_vs_later")
    return fig


# ----------------------------------------------------------------------------- 4
def fig04_hourly_profile(df: pd.DataFrame):
    h = df.groupby("rel_hour")[Y].agg(n="size", frauds="sum")
    h["rate"] = h.frauds / h.n
    base = _base(df)
    fig, (a1, a2) = S.new(10.5, 6.0, nrows=2, sharex=True, gridspec_kw={"height_ratios": [1, 1.25]})
    a1.bar(h.index, h.n, color=S.GREY, width=0.72, zorder=2)
    a1.set_ylabel("Transactions")
    a1.yaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k")
    S.grid(a1, "y")
    k_lo, k_hi = S.wilson(h.frauds.to_numpy(), h.n.to_numpy())
    a2.fill_between(h.index, k_lo, k_hi, color=S.ACCENT, alpha=0.15, linewidth=0)
    a2.plot(h.index, h.rate, color=S.ACCENT, marker="o", markersize=4)
    top = h.rate.idxmax()
    a2.annotate(S.pct(h.rate[top]), (top, h.rate[top]), xytext=(8, 4), textcoords="offset points", fontsize=8.5, fontweight="bold")
    S.ref_line(a2, base, f"base {S.pct(base)}", orient="h")
    a2.set_ylabel("Fraud rate")
    a2.yaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
    a2.set_xticks(range(0, 24, 2))
    a2.set_xlabel("Relative hour = floor(TransactionDT / 3600) mod 24 (true clock and time zone undisclosed)")
    S.grid(a2, "y")
    quiet = h.n.nsmallest(4).index.tolist()
    S.headline(fig, f"Fraud peaks at {S.pct(h.rate.max())} in relative hour {top}, among the quietest hours of the day",
               f"Top: volume per relative hour. Bottom: fraud rate with 95% band. Quietest hours: {', '.join(map(str, sorted(quiet)))}",
               top=0.86)
    S.save(fig, "fig04_relative_hour_volume_and_rate")
    return fig


# ----------------------------------------------------------------------------- 5
def fig05_block_missing_by_product(df: pd.DataFrame, blocks: pd.DataFrame):
    from finguard.eda.missingness import block_missing_by
    m = block_missing_by(df, blocks, "ProductCD")
    m["all"] = [df[f].isna().mean() for f in blocks["first"]]
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", ["#f4f8fd"] + S.SEQ)
    fig, ax = S.new(9.5, 9.0)
    im = ax.imshow(m.to_numpy(), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.iat[i, j]
            ax.text(j, i, f"{v * 100:.0f}", ha="center", va="center", fontsize=7.5, color="white" if v > 0.6 else S.INK)
    ax.set_xticks(range(m.shape[1]), [f"Product {c}" if c != "all" else "All" for c in m.columns])
    ax.set_yticks(range(m.shape[0]), m.index, fontsize=7.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Missing share", color=S.INK2)
    cb.ax.yaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
    cb.outline.set_visible(False)
    full = int(((m.drop(columns="all") > 0.99) | (m.drop(columns="all") < 0.01)).all(axis=1).sum())
    S.headline(fig, "Missingness is structural: whole column blocks vanish for specific products",
               f"Missing share (%) of each block of columns that go missing together, by ProductCD. {len(m)} blocks cover {int(blocks.n_cols.sum())} columns",
               top=0.9)
    fig.subplots_adjust(left=0.24, right=0.95, bottom=0.07)
    S.save(fig, "fig05_missing_blocks_by_product")
    return fig


# ----------------------------------------------------------------------------- 6
def fig06_block_presence_fraud(blocks: pd.DataFrame, base: float, lo: float = 0.01, hi: float = 0.995):
    keep = blocks.missing_share.between(lo, hi)
    dropped = blocks.loc[~keep, "block"].tolist()
    b = blocks[keep].sort_values("fraud_rate_present").reset_index(drop=True)
    fig, ax = S.new(10.5, 8.6)
    yp = np.arange(len(b))
    for i, r in b.iterrows():
        ax.plot([r.fraud_rate_missing, r.fraud_rate_present], [i, i], color=S.GRID, linewidth=2.2, zorder=1)
    ax.scatter(b.fraud_rate_missing, yp, s=34, color=S.GREY_DARK, zorder=3, label="block missing")
    ax.scatter(b.fraud_rate_present, yp, s=34, color=S.ACCENT, zorder=3, label="block present")
    ax.set_yticks(yp, [f"{r.label}  ·  {r.missing_share * 100:.0f}% missing" for _, r in b.iterrows()], fontsize=7.5)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
    S.grid(ax, "x")
    S.ref_line(ax, base, f"base {S.pct(base)}")
    ax.legend(loc="lower right")
    n_more = int((b.fraud_rate_present > b.fraud_rate_missing).sum())
    S.headline(fig, f"Missingness carries signal both ways: present data is riskier in {n_more} of {len(b)} blocks, missing data in {len(b) - n_more}",
               f"Fraud rate when a block is present (blue) vs missing (grey); blocks missing in under {lo:.0%} or over {hi:.1%} of rows omitted ({', '.join(dropped)})",
               top=0.89)
    fig.subplots_adjust(left=0.36, bottom=0.08)
    S.save(fig, "fig06_block_presence_vs_fraud")
    return fig


# ----------------------------------------------------------------------------- 7
def fig07_amount(df: pd.DataFrame):
    base = _base(df)
    amt = df.TransactionAmt
    auc = roc_auc_score(df[Y], amt)
    fig, (a1, a2) = S.new(11, 4.9, ncols=2, gridspec_kw={"width_ratios": [1, 1.15]})
    bins = np.linspace(np.log10(amt.min()), np.log10(amt.max()), 80)
    for lab, mask, col in [("legitimate", df[Y] == 0, S.GREY_DARK), ("fraud", df[Y] == 1, S.ACCENT)]:
        h, e = np.histogram(np.log10(amt[mask]), bins=bins, density=True)
        a1.plot((e[:-1] + e[1:]) / 2, h, color=col, linewidth=1.8, label=lab)
    a1.set_xticks([0, 1, 2, 3, 4], ["$1", "$10", "$100", "$1k", "$10k"])
    a1.set_xlabel("TransactionAmt (log scale; currency undisclosed)")
    a1.set_ylabel("Density")
    a1.legend(loc="upper right")
    a1.set_title("Distributions overlap almost entirely", fontsize=10)
    S.grid(a1, "y")
    dec = pd.qcut(amt, 10, duplicates="drop")
    t = pd.DataFrame({"d": dec, "y": df[Y]}).groupby("d", observed=True).y.agg(n="size", frauds="sum")
    t["rate"] = t.frauds / t.n
    t["lo"], t["hi"] = S.wilson(t.frauds.to_numpy(), t.n.to_numpy())
    labels = [f"{iv.left:,.0f}-{iv.right:,.0f}" for iv in t.index]
    x = np.arange(len(t))
    hi = np.zeros(len(t), bool); hi[[0, -1]] = True
    a2.bar(x, t.rate, color=[S.ACCENT if h else S.GREY for h in hi], width=0.7, zorder=2)
    a2.errorbar(x, t.rate, yerr=[t.rate - t.lo, t.hi - t.rate], fmt="none", ecolor=S.INK2, elinewidth=0.8, capsize=2, zorder=3)
    a2.set_xticks(x, labels, rotation=35, ha="right", fontsize=7.5)
    a2.yaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
    S.ref_line(a2, base, f"base {S.pct(base)}", orient="h")
    a2.set_title("Fraud rate by amount decile", fontsize=10)
    S.grid(a2, "y")
    S.headline(fig, f"Amount alone barely separates fraud (AUC {auc:.2f}); risk is highest in the lowest and highest deciles",
               "Left: density of log10(amount) by class. Right: fraud rate per amount decile with 95% intervals",
               top=0.83)
    fig.subplots_adjust(bottom=0.24, wspace=0.25)
    S.save(fig, "fig07_amount_distribution_and_deciles")
    return fig


# ----------------------------------------------------------------------------- 8
def fig08_decimals_within_product(df: pd.DataFrame, min_n: int = 400):
    base = _base(df)
    dec = amount_decimals(df.TransactionAmt)
    series = {"all products": np.ones(len(df), bool), "product C": (df.ProductCD == "C").to_numpy(),
              "product W": (df.ProductCD == "W").to_numpy()}
    colors = {"all products": S.GREY, "product C": S.ACCENT, "product W": S.SECOND}
    fig, ax = S.new(10.5, 5.0)
    width = 0.26
    n3, n3c = int((dec == 3).sum()), int(((dec == 3) & (df.ProductCD == "C")).sum())
    w = df.ProductCD == "W"
    w0, w2 = df[Y][w & (dec == 0)].mean(), df[Y][w & (dec == 2)].mean()
    for i, (name, mask) in enumerate(series.items()):
        t = S.rate_table(dec[mask], df[Y][mask], min_n=min_n)
        xs = t.index.astype(int).to_numpy() + (i - 1) * width
        ax.bar(xs, t.rate, width=width * 0.92, color=colors[name], label=name, zorder=2)
        ax.errorbar(xs, t.rate, yerr=[t.rate - t.lo, t.hi - t.rate], fmt="none", ecolor=S.INK2, elinewidth=0.8, capsize=2, zorder=3)
        for x_, r in zip(xs, t.itertuples()):
            ax.text(x_, r.hi + 0.003, f"{r.rate * 100:.1f}%\nn={r.n:,}", ha="center", fontsize=6.8, color=S.INK2)
    ax.set_xticks([0, 1, 2, 3], ["0 decimals", "1 decimal", "2 decimals", "3 decimals"])
    ax.yaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
    S.ref_line(ax, base, f"base {S.pct(base)}", orient="h")
    ax.legend(loc="upper left")
    ax.set_ylim(0, 0.16)
    S.grid(ax, "y")
    S.headline(fig, f"The 3-decimal 'signal' is product C: {n3c:,} of {n3:,} such amounts belong to it",
               f"Within C, decimals change little. Within W, round amounts are riskier ({S.pct(w0)} vs {S.pct(w2)} at 2 decimals): a weaker, separate effect. Bars shown when n >= {min_n}")
    S.save(fig, "fig08_amount_decimals_within_product")
    return fig


# ----------------------------------------------------------------------------- 9
def fig09_product_identity(df: pd.DataFrame, min_n: int = 100):
    base = _base(df)
    prods = df.ProductCD.value_counts().index.tolist()
    cnt = pd.crosstab(df.ProductCD, df.has_identity).reindex(prods)
    fig, (a1, a2) = S.new(11, 4.8, ncols=2)
    yp = np.arange(len(prods))[::-1]
    with_id = cnt.get(1, pd.Series(0, index=prods)).to_numpy()
    without = cnt.get(0, pd.Series(0, index=prods)).to_numpy()
    a1.barh(yp, without, color=S.GREY, height=0.6, label="no identity record")
    a1.barh(yp, with_id, left=without, color=S.ACCENT, height=0.6, label="identity record")
    a1.set_yticks(yp, [f"Product {p}" for p in prods])
    a1.xaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k")
    a1.legend(loc="lower right")
    a1.set_title("Transactions by product and identity availability", fontsize=10)
    S.grid(a1, "x")
    w = 0.36
    for k, (flag, col, lab) in enumerate([(0, S.GREY, "no identity record"), (1, S.ACCENT, "identity record")]):
        sub = df[df.has_identity == flag]
        t = S.rate_table(sub.ProductCD, sub[Y], min_n=min_n).reindex(prods)
        ys = yp + (0.5 - k) * w
        ok = t.n.notna().to_numpy()
        a2.barh(ys[ok], t.rate[ok], height=w * 0.92, color=col, label=lab, zorder=2)
        a2.errorbar(t.rate[ok], ys[ok], xerr=[(t.rate - t.lo)[ok], (t.hi - t.rate)[ok]], fmt="none", ecolor=S.INK2, elinewidth=0.8, capsize=2, zorder=3)
        for y_, r in zip(ys[ok], t[ok].itertuples()):
            a2.text(r.hi + 0.002, y_, f"{r.rate * 100:.1f}%", va="center", fontsize=7.5, color=S.INK2)
    a2.set_yticks(yp, [f"Product {p}" for p in prods])
    a2.xaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
    S.ref_line(a2, base, f"base {S.pct(base)}")
    a2.set_title(f"Fraud rate (groups with n >= {min_n})", fontsize=10, pad=14)
    S.grid(a2, "x")
    c = df[df.ProductCD == "C"]
    rc1, rc0 = c[c.has_identity == 1][Y].mean(), c[c.has_identity == 0][Y].mean()
    w_share = df[df.ProductCD == "W"].has_identity.mean()
    S.headline(fig, f"Product W never has identity data ({w_share:.0%}); within product C, identity rows still run at {rc1 / rc0:.1f}x the fraud rate",
               f"Identity presence is mostly a product effect: compare within a product. Product C: {S.pct(rc1)} with vs {S.pct(rc0)} without",
               top=0.82)
    fig.subplots_adjust(wspace=0.3)
    S.save(fig, "fig09_product_by_identity")
    return fig


# ----------------------------------------------------------------------------- 10
def fig10_email_domains(df: pd.DataFrame, top: int = 10, min_n: int = 1000):
    base = _base(df)
    fig, axes = S.new(11.5, 5.6, ncols=2)
    best = {}
    tabs = {c: S.rate_table(df[c], df[Y], min_n=min_n).sort_values("n", ascending=False).head(top) for c in ["P_emaildomain", "R_emaildomain"]}
    xmax = max(t.hi.max() for t in tabs.values())
    for ax, col, name in zip(axes, ["P_emaildomain", "R_emaildomain"], ["Purchaser (P_emaildomain)", "Recipient (R_emaildomain)"]):
        t = tabs[col]
        hi = (t.rate >= 2 * base).to_numpy()
        _hbars(ax, t, [f"{k}" for k in t.index], base, hi, xmax=xmax)
        ax.set_title(name, fontsize=10, pad=14)
        best[col] = t.drop(index="absent", errors="ignore").rate.max()
    S.headline(fig, f"Recipient email domains separate fraud more sharply: up to {S.pct(best['R_emaildomain'])} vs {S.pct(best['P_emaildomain'])} for purchaser domains",
               f"Top {top} domains by volume (n >= {min_n:,}); blue = at least 2x the base rate; 'absent' = field empty",
               top=0.83)
    fig.subplots_adjust(wspace=0.55, left=0.12)
    S.save(fig, "fig10_email_domains")
    return fig


# ----------------------------------------------------------------------------- 11
def fig11_os_browser(df: pd.DataFrame, min_n: int = 300):
    base = _base(df)
    fig, axes = S.new(11.5, 5.0, ncols=2)
    fams = {"Operating system (id_30 family)": os_family(df.id_30), "Browser (id_31 family)": browser_family(df.id_31)}
    tabs = {k: S.rate_table(v, df[Y], min_n=min_n).sort_values("rate", ascending=False) for k, v in fams.items()}
    xmax = max(t.hi.max() for t in tabs.values())
    for ax, (name, t) in zip(axes, tabs.items()):
        hi = (t.rate >= 2 * base).to_numpy()
        _hbars(ax, t, list(t.index), base, hi, xmax=xmax)
        ax.set_title(name, fontsize=10, pad=14)
    b = tabs["Browser (id_31 family)"].drop(index="absent", errors="ignore")
    absent = float(fams["Browser (id_31 family)"].isna().mean())
    S.headline(fig, f"Browser families range from {S.pct(b.rate.min())} ({b.rate.idxmin()}) to {S.pct(b.rate.max())} ({b.rate.idxmax()}); {absent:.0%} of rows have no browser data",
               f"Fraud rate per family (n >= {min_n}); version strings collapsed to families because they change monthly; 'absent' = no identity data",
               top=0.83)
    fig.subplots_adjust(wspace=0.5, left=0.12)
    S.save(fig, "fig11_os_and_browser_family")
    return fig


# ----------------------------------------------------------------------------- 12
def fig12_card_network_type(df: pd.DataFrame, min_n: int = 1000):
    base = _base(df)
    fig, axes = S.new(11, 3.8, ncols=2)
    out = {c: S.rate_table(df[c], df[Y], min_n=min_n).sort_values("rate", ascending=False) for c in ["card4", "card6"]}
    xmax = max(t.hi.max() for t in out.values())
    for ax, col, name in zip(axes, ["card4", "card6"], ["Card network (card4)", "Card type (card6)"]):
        t = out[col]
        hi = (t.rate >= 2 * base).to_numpy()
        _hbars(ax, t, list(t.index), base, hi, xmax=xmax)
        ax.set_title(name, fontsize=10, pad=14)
    ratio = out["card6"].loc["credit", "rate"] / out["card6"].loc["debit", "rate"]
    S.headline(fig, f"Credit cards carry {ratio:.1f}x the fraud rate of debit cards; Discover leads the networks",
               f"Fraud rate per category (n >= {min_n:,}) with 95% intervals", top=0.8)
    fig.subplots_adjust(wspace=0.5, left=0.12, bottom=0.16)
    S.save(fig, "fig12_card_network_and_type")
    return fig


# ----------------------------------------------------------------------------- 13
def fig13_signal_by_group(sig: pd.DataFrame):
    order = sig.groupby("group").strength.max().sort_values().index.tolist()
    fig, ax = S.new(10.5, 6.0)
    rng = np.random.default_rng(0)
    for i, g in enumerate(order):
        s = sig[sig.group == g]
        ax.scatter(s.strength, i + rng.uniform(-0.22, 0.22, len(s)), s=12, color=S.GREY, alpha=0.8, zorder=2, linewidths=0)
        top = s.iloc[s.strength.argmax()]
        ax.scatter([top.strength], [i], s=34, color=S.ACCENT, zorder=3)
        ax.text(top.strength + 0.004, i, f"{top.column} {top.strength:.2f}", va="center", fontsize=7.5, color=S.INK2)
    counts = sig.group.value_counts()
    ax.set_yticks(range(len(order)), [f"{g} ({counts[g]})" for g in order])
    ax.set_xlim(0.49, 0.78)
    S.grid(ax, "x")
    S.ref_line(ax, 0.5, "no signal")
    v = sig[sig.group == "V1-339"].strength
    S.headline(fig, f"No single feature exceeds AUC {sig.strength.max():.2f}; the {len(v)} V columns cluster around {v.median():.2f}",
               "Single-feature strength = max(AUC, 1 - AUC); categorical columns fitted on the first 60% of time, scored on the last 40%; blue = best per group",
               top=0.86)
    fig.subplots_adjust(left=0.2)
    S.save(fig, "fig13_single_feature_signal_by_group")
    return fig


# ----------------------------------------------------------------------------- 14
def fig14_adversarial_drift(drift: pd.DataFrame, top: int = 25, threshold: float = 0.6):
    d = drift[drift.column != config.TIME].head(top).iloc[::-1]
    n_over = int((drift[drift.column != config.TIME].adversarial_auc > threshold).sum())
    groups = drift[(drift.column != config.TIME) & (drift.adversarial_auc > threshold)].group.value_counts()
    fig, ax = S.new(10, 6.4)
    colors = [S.ACCENT if v > threshold else S.GREY for v in d.adversarial_auc]
    ax.barh(range(len(d)), d.adversarial_auc - 0.5, left=0.5, color=colors, height=0.62, zorder=2)
    for i, v in enumerate(d.adversarial_auc):
        ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=7.5, color=S.INK2)
    ax.set_yticks(range(len(d)), d.column)
    ax.set_xlim(0.5, max(0.66, d.adversarial_auc.max() + 0.03))
    S.grid(ax, "x")
    S.ref_line(ax, threshold, f"{threshold}")
    grp = (f"all of them {groups.index[0]} columns" if len(groups) == 1
           else ", ".join(f"{g} ({n})" for g, n in groups.items()))
    S.headline(fig, f"{n_over} features drift past adversarial AUC {threshold} between the two periods",
               f"Single-feature AUC for 'is this row from the later file?' ({grp}). Raw TransactionDT scores 1.0 and is excluded from chart and features",
               top=0.88)
    fig.subplots_adjust(left=0.14)
    S.save(fig, "fig14_train_vs_later_drift")
    return fig


# ----------------------------------------------------------------------------- 15
def fig15_entity_label_purity(purity: pd.DataFrame, overlap_random: dict, overlap_time: dict):
    with_fraud = purity[purity.frauds > 0]
    all_fraud = float((with_fraud.fraud_share == 1).mean())
    mixed = float(((purity.fraud_share > 0) & (purity.fraud_share < 1)).mean())
    fig, (a1, a2) = S.new(11, 4.8, ncols=2, gridspec_kw={"width_ratios": [1.35, 1]})
    a1.hist(with_fraud.fraud_share, bins=np.linspace(0, 1, 21), color=S.GREY, edgecolor=S.SURFACE, zorder=2)
    counts, _ = np.histogram(with_fraud.fraud_share, bins=np.linspace(0, 1, 21))
    a1.bar(0.975, counts[-1], width=0.05, color=S.ACCENT, edgecolor=S.SURFACE, zorder=3)
    a1.text(0.95, counts[-1], f"{all_fraud:.0%} entirely fraud ", ha="right", va="top", fontsize=8.5, fontweight="bold")
    a1.set_xlabel("Share of the entity's transactions labelled fraud")
    a1.set_ylabel("Proxy entities")
    a1.xaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
    a1.set_title(f"{len(with_fraud):,} proxy entities with 2+ transactions and any fraud", fontsize=10)
    S.grid(a1, "y")
    vals = [overlap_random["test_frauds_entity_had_fraud_in_train"], overlap_time["test_frauds_entity_had_fraud_in_train"]]
    a2.bar([0, 1], vals, color=[S.ACCENT, S.GREY], width=0.55, zorder=2)
    for x_, v in zip([0, 1], vals):
        a2.text(x_, v + 0.01, f"{v:.1%}", ha="center", fontsize=9, fontweight="bold")
    a2.set_xticks([0, 1], ["random split\n(66 / 17 / 17%)", "time split\n(days 0-151 vs 152-181)"])
    a2.yaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
    a2.set_ylim(0, max(vals) * 1.25)
    a2.set_title("Test frauds whose entity already had a fraud label", fontsize=10)
    S.grid(a2, "y")
    S.headline(fig, f"Labels cluster inside proxy entities, so random splits leak ({vals[0]:.0%} vs {vals[1]:.0%} 'known' test frauds)",
               f"Proxy entity = card1 + addr1 + (day - D1), a heuristic grouping, NOT a customer. Only {mixed:.1%} of multi-transaction entities mix labels",
               top=0.82)
    fig.subplots_adjust(wspace=0.28, bottom=0.18)
    S.save(fig, "fig15_proxy_entity_label_purity")
    return fig


# ----------------------------------------------------------------------------- 16
def fig16_vblock_correlation(df: pd.DataFrame, blocks: pd.DataFrame, block_id: str | None = None,
                             threshold: float = 0.95, sample: int = 150_000):
    b = blocks.iloc[0] if block_id is None else blocks.set_index("block").loc[block_id].rename(block_id)
    bid = b.block if "block" in b else b.name
    cols = [c for c in b["columns"] if c.startswith("V")]
    sub = df[cols].dropna().sample(min(sample, df[cols].dropna().shape[0]), random_state=config.RANDOM_SEED)
    r = np.abs(np.corrcoef(sub.to_numpy(dtype="float64").T))
    r = np.nan_to_num(r)
    n_pairs = int((np.triu(r, 1) >= threshold).sum())
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", ["#f4f8fd"] + S.SEQ)
    fig, ax = S.new(9.5, 9.2)
    im = ax.imshow(r, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=90, fontsize=6)
    ax.set_yticks(range(len(cols)), cols, fontsize=6)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("|Pearson r|", color=S.INK2)
    cb.outline.set_visible(False)
    S.headline(fig, f"Inside block {bid} ({len(cols)} V columns), {n_pairs} column pairs correlate at |r| >= {threshold}",
               f"Absolute correlation on {len(sub):,} rows where the block is present: many V columns are near-duplicates",
               top=0.9)
    fig.subplots_adjust(left=0.08, bottom=0.08)
    S.save(fig, "fig16_v_block_correlation")
    return fig
