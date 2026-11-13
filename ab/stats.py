"""
A/B statistics (Phase 8). Pure numpy/scipy functions, unit-tested in tests/test_ab_stats.py.

  srm_test          sample ratio mismatch: did the split come out as configured? (if not: STOP, data is broken)
  compare_means     difference in means, Welch CI, relative lift with delta-method CI, p-value
  cuped             variance reduction with a PRE-experiment covariate (Deng et al. 2013)
  mde / sample_size power analysis: smallest effect we can detect / users needed for a target effect
  aa_false_positive_rate   A/A calibration: split one arm in two many times -> should reject ~alpha
  peeking_false_positive   what checking the p-value every day does to the false-positive rate
  team_draft_interleave    Netflix/Bing-style interleaving: both rankers in ONE list, clicks credited per team
"""
import hashlib

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------------- data quality
def srm_test(counts: dict, expected_share: dict) -> dict:
    """Chi-square goodness of fit of user counts per arm vs the configured traffic split.
    p < 0.001 = sample ratio mismatch -> assignment/logging bug, results are NOT trustworthy."""
    arms = list(counts)
    obs = np.array([counts[a] for a in arms], float)
    exp = np.array([expected_share[a] for a in arms], float) * obs.sum()
    chi2, p = stats.chisquare(obs, exp)
    return {"counts": counts, "expected": dict(zip(arms, exp.round(1).tolist())), "chi2": float(chi2),
            "p_value": float(p), "srm": bool(p < 0.001)}


# ---------------------------------------------------------------------------------- effect estimates
def compare_means(y_t: np.ndarray, y_c: np.ndarray, alpha: float = 0.05) -> dict:
    """Treatment vs control on a per-USER metric. Users are the randomization unit, so they are the
    independent observations (not requests / impressions -> those are correlated within a user)."""
    y_t, y_c = np.asarray(y_t, float), np.asarray(y_c, float)
    m_t, m_c = y_t.mean(), y_c.mean()
    v_t, v_c = y_t.var(ddof=1) / len(y_t), y_c.var(ddof=1) / len(y_c)
    diff, se = m_t - m_c, np.sqrt(v_t + v_c)
    z = stats.norm.ppf(1 - alpha / 2)
    p = 2 * stats.norm.sf(abs(diff / se)) if se > 0 else 1.0
    # relative lift m_t/m_c - 1, delta method: Var(m_t/m_c) ~ v_t/m_c^2 + m_t^2 v_c/m_c^4
    rel = m_t / m_c - 1 if m_c else np.nan
    rel_se = np.sqrt(v_t / m_c ** 2 + m_t ** 2 * v_c / m_c ** 4) if m_c else np.nan
    return {"n_t": len(y_t), "n_c": len(y_c), "mean_t": m_t, "mean_c": m_c, "diff": diff, "se": se,
            "ci": (diff - z * se, diff + z * se), "p_value": float(p),
            "rel_lift": rel, "rel_ci": (rel - z * rel_se, rel + z * rel_se)}


def cuped(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Y_cuped = Y - theta (X - mean X), theta = cov(X,Y)/var(X), X measured BEFORE the experiment.
    Same expected treatment effect (X is independent of the arm), lower variance by corr(X,Y)^2.
    theta is estimated on the pooled data of both arms (standard practice).
    Returns (adjusted y, theta, variance reduction)."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    if x.var() == 0:
        return y, 0.0, 0.0
    theta = np.cov(x, y)[0, 1] / x.var(ddof=1)
    adj = y - theta * (x - x.mean())
    return adj, float(theta), float(1 - adj.var() / y.var())


# ---------------------------------------------------------------------------------- power
def mde(sd: float, n_per_arm: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest absolute difference detectable with `power` (two-sided, equal arms)."""
    return float((stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) * sd * np.sqrt(2 / n_per_arm))


def sample_size(sd: float, abs_effect: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Users PER ARM needed to detect `abs_effect`."""
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return int(np.ceil(2 * (z * sd / abs_effect) ** 2))


# ---------------------------------------------------------------------------------- calibration
def _hash_split(ids, salt):
    return np.array([int(hashlib.sha256(f"{salt}:{u}".encode()).hexdigest()[:15], 16) % 2 for u in ids], bool)


def aa_false_positive_rate(ids, y, n_sims: int = 500, alpha: float = 0.05, seed: int = 0) -> dict:
    """Split the SAME population (one arm, one model) into two random halves many times with new salts.
    There is no real effect, so a correct pipeline rejects ~alpha of the time. Much more -> the stats
    are broken (wrong unit, dependence, bug); much less -> test is conservative."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float)
    ps = []
    for s in range(n_sims):
        g = _hash_split(ids, f"aa-{seed}-{s}") if len(ids) < 5000 else rng.random(len(y)) < 0.5
        ps.append(compare_means(y[g], y[~g], alpha)["p_value"])
    ps = np.array(ps)
    return {"n_sims": n_sims, "false_positive_rate": float((ps < alpha).mean()),
            "p_uniform_ks_p": float(stats.kstest(ps, "uniform").pvalue)}


def peeking_false_positive(y, n_looks: int = 20, n_sims: int = 500, alpha: float = 0.05, seed: int = 0) -> dict:
    """A/A test, users arrive over time; the analyst checks the p-value after every 1/n_looks of traffic
    and stops at the first p < alpha. Compares that 'peeking' error rate with a single look at the end."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float)
    n = len(y)
    looks = np.linspace(n / n_looks, n, n_looks).astype(int)
    peek_hits = end_hits = 0
    for _ in range(n_sims):
        order = rng.permutation(n)
        g = rng.random(n) < 0.5
        yy, gg = y[order], g[order]
        cs_t, cs_c = np.cumsum(yy * gg), np.cumsum(yy * ~gg)
        cq_t, cq_c = np.cumsum(yy ** 2 * gg), np.cumsum(yy ** 2 * ~gg)
        cn_t, cn_c = np.cumsum(gg), np.cumsum(~gg)
        stopped = False
        for L in looks:
            nt, nc = cn_t[L - 1], cn_c[L - 1]
            if nt < 2 or nc < 2:
                continue
            mt, mc = cs_t[L - 1] / nt, cs_c[L - 1] / nc
            vt = (cq_t[L - 1] / nt - mt ** 2) * nt / (nt - 1) / nt
            vc = (cq_c[L - 1] / nc - mc ** 2) * nc / (nc - 1) / nc
            p = 2 * stats.norm.sf(abs(mt - mc) / np.sqrt(vt + vc)) if vt + vc > 0 else 1
            if p < alpha and not stopped:
                peek_hits += 1; stopped = True
            if L == looks[-1] and p < alpha:
                end_hits += 1
    return {"n_looks": n_looks, "n_sims": n_sims, "fpr_peeking": peek_hits / n_sims, "fpr_fixed_horizon": end_hits / n_sims}


# ---------------------------------------------------------------------------------- interleaving
def team_draft_interleave(a: list, b: list, k: int, rng) -> tuple[list, list]:
    """Team-draft interleaving (Radlinski et al. 2008). Teams take turns (coin flip when tied) picking their
    highest-ranked item not yet in the list. Returns (list, team per position: 'A'/'B')."""
    out, team, seen = [], [], set()
    ia = ib = n_a = n_b = 0
    while len(out) < k and (ia < len(a) or ib < len(b)):
        pick_a = n_a < n_b or (n_a == n_b and rng.random() < 0.5)
        if pick_a and ia >= len(a):
            pick_a = False
        if not pick_a and ib >= len(b):
            pick_a = True
        src, idx = (a, ia) if pick_a else (b, ib)
        while idx < len(src) and src[idx] in seen:
            idx += 1
        if pick_a:
            ia = idx + 1
        else:
            ib = idx + 1
        if idx < len(src):
            out.append(src[idx]); seen.add(src[idx]); team.append("A" if pick_a else "B")
            n_a, n_b = n_a + pick_a, n_b + (not pick_a)
    return out, team


def interleaving_test(wins_b: np.ndarray, wins_a: np.ndarray) -> dict:
    """Per user: did team B get more clicks than A (win), fewer (loss), or equal (tie)?
    Delta = (wins_B - wins_A) / (wins + losses); sign test on non-tied users."""
    wb, wa = int(np.sum(wins_b)), int(np.sum(wins_a))
    n = wb + wa
    p = float(stats.binomtest(wb, n, 0.5).pvalue) if n else 1.0
    return {"users_b_won": wb, "users_a_won": wa, "delta": (wb - wa) / n if n else 0.0, "p_value": p}
