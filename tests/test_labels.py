"""Sanity checks for the skill test on simulated wallets."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pmgraph.labels import assign_types, bh, skill_test  # noqa: E402


def _sim(n_wallets, n_bets, edge, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_wallets):
        pi = rng.uniform(0.05, 0.95, n_bets)                 # calibrated price = true prob (null)
        W = (rng.random(n_bets) < np.clip(pi + edge, 0, 1)).astype(int)
        A = rng.choice([-1.0, 1.0], n_bets)                  # long or short the reference token
        W_eff = np.where(A > 0, W, (rng.random(n_bets) < np.clip(pi - edge, 0, 1)).astype(int))
        rows.append(pd.DataFrame({"wallet": f"w{i}", "market_id": np.arange(n_bets), "w": W_eff,
                                  "A": A, "C": A * pi, "pibar": pi, "B": A * pi,
                                  "profit": A * (W_eff - pi), "Q": 1.0, "first_ts": 0}))
    return pd.concat(rows)


def test_null_is_calibrated():
    t = skill_test(_sim(2000, 60, 0.0, 0))
    rate = (t["p"] < 0.05).mean()
    assert 0.03 < rate < 0.07, rate


def test_informed_wallets_detected():
    t = skill_test(_sim(200, 300, 0.08, 1))
    assert (t["p"] < 0.05).mean() > 0.9


def test_bh_and_types():
    p = np.r_[np.full(10, 1e-6), np.random.default_rng(0).uniform(size=990)]
    assert bh(p, 0.1)[:10].all() and bh(p, 0.1).sum() < 30
    t = assign_types(skill_test(_sim(100, 40, 0.0, 2)), 30, 0.1)
    assert set(t["type"]) <= {"skilled_winner", "lucky_winner", "unlucky_skilled", "unskilled_loser"}


def _sim_events(n_wallets, n_events, per_event, seed):
    """Null wallets whose sibling markets in one event resolve together (one uniform per event)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_wallets):
        ev = np.repeat(np.arange(n_events), per_event)
        pi = rng.uniform(0.05, 0.95, len(ev))
        U = rng.random(n_events)[ev]
        W = (U < pi).astype(int)
        A = np.repeat(rng.choice([-1.0, 1.0], n_events), per_event)   # same side within an event
        rows.append(pd.DataFrame({"wallet": f"w{i}", "market_id": np.arange(len(ev)), "event_id": ev.astype(str),
                                  "w": W, "A": A, "C": A * pi, "pibar": pi, "B": A * pi,
                                  "profit": A * (W - pi), "Q": 1.0, "first_ts": 0}))
    return pd.concat(rows)


def test_event_unit_fixes_sibling_overdispersion():
    b = _sim_events(1500, 30, 4, 3)
    mk, ev = skill_test(b, unit="market"), skill_test(b, unit="event")
    assert mk["z"].std() > 1.3                       # market-level treats 120 correlated bets as independent
    assert 0.9 < ev["z"].std() < 1.1, ev["z"].std()
    assert 0.03 < (ev["p"] < 0.05).mean() < 0.07
    assert (ev["n_bets"] == 30).all() and (ev["n_markets"] == 120).all()


def test_event_unit_equals_market_unit_for_singleton_events():
    b = _sim(50, 40, 0.02, 4)
    b["event_id"] = b["market_id"].astype(str)
    a, e = skill_test(b, unit="market"), skill_test(b, unit="event")
    np.testing.assert_allclose(a["z"], e["z"])


def test_event_mc_pvalues_match_analytic():
    b = _sim_events(150, 40, 3, 5)
    t = skill_test(b, n_sim=4000, unit="event")
    assert np.corrcoef(t["p"], t["p_mc"])[0, 1] > 0.95
