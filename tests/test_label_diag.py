"""Label diagnostics under simulated nulls and planted effects."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pmgraph.label_diag import (assign_halves, blind_score, boot_edge, half_table, persistence, split_half,  # noqa: E402
                                variance_function)


def _bets(n_wallets, weeks, per_week, seed, tau=0.0, week_dep=0.0):
    """Long bets at calibrated prices. tau: sd of a wallet's true edge. week_dep: share of a wallet's
    bets in one close week that resolve on one common draw (dependence inside a week only)."""
    rng = np.random.default_rng(seed)
    n = weeks * per_week
    wallet = np.repeat(np.arange(n_wallets), n)
    week = np.tile(np.repeat(np.arange(weeks), per_week), n_wallets)
    pi = rng.uniform(0.2, 0.8, len(wallet))
    theta = np.repeat(rng.normal(0, tau, n_wallets), n)
    shared = rng.random((n_wallets, weeks))[wallet, week]
    U = np.where(rng.random(len(wallet)) < week_dep, shared, rng.random(len(wallet)))
    W = (U < pi + theta).astype(int)
    # 2025-01-02 is a Thursday, like 1970-01-01, so simulated weeks line up with assign_halves' weeks
    close = pd.Timestamp("2025-01-02", tz="UTC") + pd.to_timedelta(week * 7 + rng.integers(0, 7, len(week)), "D")
    return pd.DataFrame({"wallet": wallet, "market_id": np.arange(len(wallet)), "event_id": None, "w": W,
                         "A": 1.0, "C": pi, "pibar": pi, "B": pi, "profit": W - pi, "Q": 1.0,
                         "first_ts": 0, "close_time": close})


def test_variance_function_recovers_planted_parameters():
    rng = np.random.default_rng(0)
    s = rng.uniform(0.01, 0.08, 20000)
    x = rng.normal(-0.015, np.sqrt(0.03 ** 2 + 1.5 * s ** 2))
    f = variance_function(x, s)
    assert abs(f["mu"] + 0.015) < 0.002 and abs(f["tau2"] - 9e-4) < 2e-4 and abs(f["phi"] - 1.5) < 0.15, f
    null = variance_function(rng.normal(0, s), s)
    assert null["tau2"] < 5e-5 and abs(null["phi"] - 1) < 0.05, null
    assert abs(variance_function(x, s, tau2=9e-4)["phi"] - 1.5) < 0.1       # one parameter held fixed


def test_halves_do_not_depend_on_row_order():
    b = _bets(200, 6, 5, 8).assign(event_id=lambda d: (d.market_id // 3).astype(str))
    b["close_time"] = b["close_time"].dt.floor("7D")                    # many tied close times
    sh = b.sample(frac=1, random_state=0)
    for scheme in ("market", "event", "week", "time"):
        a, c = pd.Series(assign_halves(b, scheme), b.index), pd.Series(assign_halves(sh, scheme), sh.index)
        assert (a == c.reindex(b.index)).all(), scheme
        assert 0.3 < a.mean() < 0.7


def test_split_half_null_has_no_shared_variance():
    b = _bets(1500, 12, 6, 1)
    for scheme in ("market", "week", "time"):
        r = split_half(half_table(b, scheme))
        assert r["cov_lo"] < 0 < r["cov_hi"], (scheme, r)
        assert 0.9 < r["phi"] < 1.1, (scheme, r)


def test_split_half_recovers_persistent_edge():
    b = _bets(3000, 12, 6, 2, tau=0.05)
    for scheme in ("market", "time"):
        r = split_half(half_table(b, scheme))
        assert 0.0019 < r["cov"] < 0.0031 and r["cov_lo"] > 0, (scheme, r)   # tau^2 = 0.0025


def test_week_dependence_shows_in_market_split_only():
    b = _bets(2000, 12, 6, 3, week_dep=0.6)
    mk, wk, tm = (split_half(half_table(b, s)) for s in ("market", "week", "time"))
    assert mk["cov_lo"] > 0 and mk["cov"] > 5 * abs(wk["cov"]), (mk, wk)
    assert wk["cov_lo"] < 0 < wk["cov_hi"] and tm["cov_lo"] < 0 < tm["cov_hi"], (wk, tm)
    assert wk["phi"] > 1.3                                    # the dependence stays inside each half


def test_persistence_flat_under_null_rising_with_planted_edge():
    null = persistence(half_table(_bets(3000, 12, 6, 6), "time"), q=5)
    assert (null["xB_c"].abs() < 3 * null["xB_c_se"] + 0.1).all(), null     # past z says nothing
    alt = persistence(half_table(_bets(3000, 12, 6, 7, tau=0.05), "time"), q=5)
    assert alt["xB_c"].iloc[-1] - alt["xB_c"].iloc[0] > 3 and alt["xB_c"].is_monotonic_increasing, alt


def test_blind_score_needs_unequal_noise_scales():
    rng = np.random.default_rng(4)
    n = 40000
    s = rng.uniform(0.02, 0.08, n)
    ap = lambda z: average_precision_score(z > 2, blind_score(s, z > 2))  # noqa: E731
    z0 = rng.normal(size=n)                                   # centred null: z does not depend on s
    assert ap(z0) < 1.15 * (z0 > 2).mean()
    z1 = (rng.normal(-0.015, 0.03, n) + s * rng.normal(size=n)) / s      # negative mean + real spread
    assert ap(z1) > 1.3 * (z1 > 2).mean()                     # the noise scale alone now "predicts skill"


def test_edge_metrics_random_and_informative_scores():
    rng = np.random.default_rng(5)
    n = 4000
    s = rng.uniform(0.02, 0.08, n)
    theta = rng.normal(0, 0.03, n)
    x = theta + s * rng.normal(size=n)
    out = boot_edge({"random": rng.random(n), "informed": theta + 0.03 * rng.normal(size=n)},
                    x, 1 / (9e-4 + s ** 2), x / s, B=200)
    est, lo, hi = out["random"]
    assert (lo[:4] < 0).all() and (hi[:4] > 0).all(), (lo, hi)        # CIs cover 0 for a random score
    est, lo, hi = out["informed"]
    assert (lo[:4] > 0).all() and est[2] > 0.02, (est, lo)            # top decile beats the bottom one
