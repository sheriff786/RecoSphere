"""Phase 8: the A/B statistics must be calibrated before we trust any experiment."""
import numpy as np

from ab import stats as S


def test_srm_detects_broken_split():
    assert not S.srm_test({"c": 5010, "t": 4990}, {"c": .5, "t": .5})["srm"]
    assert S.srm_test({"c": 5300, "t": 4700}, {"c": .5, "t": .5})["srm"]


def test_compare_means_finds_real_effect_and_ci_covers():
    rng = np.random.default_rng(0)
    c = rng.binomial(1, 0.10, 20000); t = rng.binomial(1, 0.11, 20000)
    r = S.compare_means(t, c)
    assert r["p_value"] < 0.05 and r["ci"][0] < 0.01 < r["ci"][1] and r["rel_ci"][0] < 0.1 < r["rel_ci"][1]


def test_aa_is_calibrated():
    rng = np.random.default_rng(1)
    y = rng.poisson(0.3, 6000)
    r = S.aa_false_positive_rate([f"u{i}" for i in range(6000)], y, n_sims=400)
    assert 0.02 < r["false_positive_rate"] < 0.08


def test_peeking_inflates_false_positives():
    rng = np.random.default_rng(2)
    r = S.peeking_false_positive(rng.poisson(0.3, 8000), n_looks=20, n_sims=300)
    assert r["fpr_fixed_horizon"] < 0.09 and r["fpr_peeking"] > 2 * r["fpr_fixed_horizon"]


def test_cuped_keeps_effect_and_reduces_variance():
    rng = np.random.default_rng(3)
    n = 20000
    x = rng.gamma(2, 1, n)                                   # pre-period activity
    g = rng.random(n) < 0.5
    y = 0.5 * x + rng.normal(0, 1, n) + 0.05 * g             # true effect 0.05
    adj, theta, red = S.cuped(y, x)
    raw, cup = S.compare_means(y[g], y[~g]), S.compare_means(adj[g], adj[~g])
    assert red > 0.2 and cup["se"] < raw["se"] and abs(cup["diff"] - 0.05) < 3 * cup["se"]


def test_power_math_roundtrip():
    n = S.sample_size(sd=1.0, abs_effect=0.05)
    assert abs(S.mde(1.0, n) - 0.05) < 0.001


def test_team_draft_interleaving_is_fair():
    rng = np.random.default_rng(4)
    a, b = list(range(0, 20)), list(range(100, 120))
    out, team = S.team_draft_interleave(a, b, 10, rng)
    assert len(out) == 10 and len(set(out)) == 10 and abs(team.count("A") - team.count("B")) <= 1
    same, t2 = S.team_draft_interleave(a, a, 10, rng)        # identical rankers -> identical list
    assert same == a[:10]
