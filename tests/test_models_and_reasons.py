import numpy as np
import pandas as pd

import _synthetic
from finguard.evaluation import metrics as M
from finguard.explain import reasons as R
from finguard.features.build import FeatureBuilder
from finguard.models import calibration as C
from finguard.models.linear import LinearPreprocessor


def test_topk_precision_and_recall_on_a_toy_ranking():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1] * 10)
    s = np.r_[np.linspace(1, 0.9, 10), np.linspace(0.5, 0, 90)]
    y = np.r_[np.ones(5), np.zeros(5), y[10:]]                       # top 10 rows: 5 frauds
    r = M.ranking(y, s, ks=(0.05, 0.10))
    assert r["precision@5%"] == 1.0 and r["precision@10%"] == 0.5
    assert abs(r["recall@10%"] - 5 / y.sum()) < 1e-12


def test_weighted_topk_equals_plain_topk_with_unit_weights():
    rs = np.random.RandomState(0)
    y = (rs.rand(5000) < 0.05).astype(int)
    s = rs.rand(5000) + y
    order = np.argsort(-s, kind="stable")
    p, r = M._weighted_topk(y[order], np.ones(5000), 0.01)
    ref = M.ranking(y, s, ks=(0.01,))
    assert abs(p - ref["precision@1%"]) < 1e-12 and abs(r - ref["recall@1%"]) < 1e-12


def test_bootstrap_interval_contains_point_estimate():
    rs = np.random.RandomState(1)
    y = (rs.rand(4000) < 0.05).astype(int)
    s = rs.rand(4000) + 0.8 * y
    point = M.ranking(y, s)["pr_auc"]
    lo, hi = M.bootstrap(y, s, B=100)["pr_auc"]
    lo_g, hi_g = M.bootstrap(y, s, B=100, groups=np.repeat(np.arange(1000), 4))["pr_auc"]
    assert lo < point < hi and lo_g < point < hi_g


def test_bands_hit_their_budget_on_the_fitting_month():
    rs = np.random.RandomState(2)
    raw = rs.randn(100_000)
    b = C.fit_bands(raw)
    shares = pd.Series(b.assign(raw)).value_counts(normalize=True)
    assert abs(shares["High"] - 0.01) < 1e-3 and abs(shares["Medium"] - 0.04) < 1e-3
    t = C.band_table((raw > 2).astype(int), raw, b)
    assert list(t.band) == ["High", "Medium", "Low"] and abs(t.share_of_rows.sum() - 1) < 1e-12


def test_isotonic_keeps_order_and_ece_is_small_when_calibrated():
    rs = np.random.RandomState(3)
    raw = rs.randn(50_000)
    p_true = 1 / (1 + np.exp(-(raw * 1.5 - 3)))
    y = (rs.rand(50_000) < p_true).astype(int)
    iso = C.fit_isotonic(raw, y)
    p = iso.predict(raw)
    assert np.all(np.diff(p[np.argsort(raw)]) >= -1e-12)
    assert M.ece(y, p) < 0.01


def test_linear_preprocessor_outputs_finite_values_and_drops_duplicates():
    df = _synthetic.frame(4000, seed=8)              # >= 1000 complete rows per block
    fb = FeatureBuilder(fit_days=(0, 39))
    X = fb.fit_transform(df)
    X["V2"] = X["V1"] * 2.0                                          # perfect duplicate inside the V block
    prep = LinearPreprocessor(fb).fit(X)
    Z = prep.transform(X)
    assert np.isfinite(Z).all()
    assert "V2" in prep.dropped_redundant_ and "card1" not in prep.numeric_
    assert Z.shape[1] == len(prep.feature_names_)


def test_every_model_column_has_a_reason_family_and_renders():
    df = _synthetic.frame(1500, seed=9)
    fb = FeatureBuilder(fit_days=(0, 39))
    X = fb.fit_transform(df)
    fams = {R.family(c) for c in X.columns}
    assert fams <= set(R.FAMILY_LABEL)
    row = pd.Series({c: np.nan for c in X.columns})                   # worst case: everything missing
    for f in R.FAMILY_LABEL:
        text = R.render(f, row)
        assert text and "customer" not in text.lower()


def test_family_sum_and_top_families():
    cols = ["V1", "V2", "C1", "entity_txn_24h", "TransactionAmt"]
    contrib = np.array([[0.5, 0.25, -0.1, 1.0, 0.0, -2.0]])            # last column = bias
    fam = R.family_matrix(contrib, cols)
    assert abs(fam.loc[0, "vesta_V"] - 0.75) < 1e-12
    assert R.top_families(fam, k=5) == [["entity_velocity", "vesta_V"]]   # only positive families
    st = R.stability([[["a", "b"]], [["a", "c"]], [["a", "b"]]], k=2)
    assert abs(st.mean_pairwise_overlap[0] - (0.5 + 1.0 + 0.5) / 3) < 1e-12


def test_displayed_probability_never_claims_certainty():
    assert C.display_probability(1.0) == ">99%" and C.display_probability(0.0) == "<1%"
    assert C.display_probability(0.871) == "87%"
