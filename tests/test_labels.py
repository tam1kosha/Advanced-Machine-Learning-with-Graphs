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
