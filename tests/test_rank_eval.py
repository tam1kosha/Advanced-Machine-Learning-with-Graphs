"""Pooled ranking metrics and the paired bootstrap under a simulated null."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pmgraph.rank_eval import (bet_bucket, bootstrap, fold_ranks, metrics, precision_at,  # noqa: E402
                               spearman, spearman_strat, summarise)


def _data(n, seed, signal=0.0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    y = (z > 2).astype(int)
    s = signal * z + rng.normal(size=n)
    return y, z, s


def test_precision_at_ties():
    y = np.array([1, 0, 1, 0, 0])
    assert precision_at(y, np.array([5, 4, 3, 2, 1.0]), 2) == 0.5
    assert precision_at(y, np.zeros(5), 2) == 0.4          # all tied -> base rate


def test_null_metrics_at_chance():
    M = np.array([metrics(*_data(5000, i)) for i in range(30)])
    assert abs(M[:, 2].mean()) < 0.01                      # spearman ~ 0
    assert np.isnan(M[:, 3]).all()                         # no groups given
    assert abs(M[:, 1].mean() - 0.0228) < 0.01             # precision@100 ~ base rate P(z > 2)


def test_paired_bootstrap_null_coverage():
    """Two uninformative scorers: the 95% CI of their difference covers 0 about 95% of the time."""
    cover = []
    for i in range(40):
        y, z, s1 = _data(1500, i)
        s2 = np.random.default_rng(1000 + i).normal(size=len(y))
        g = bet_bucket(np.random.default_rng(2000 + i).choice([35, 70, 300], len(y)))
        b = bootstrap(y, z, {"a": s1, "b": s2}, B=150, seed=i, g=g)
        d = summarise(b["a"][0] - b["b"][0], b["a"][1] - b["b"][1])
        cover.append([d[m]["lo"] <= 0 <= d[m]["hi"] for m in d])
    cov = np.mean(cover, axis=0)
    assert (cov >= 0.85).all(), cov


def test_signal_detected_and_fold_ranks():
    y, z, s = _data(3000, 7, signal=1.0)
    fold = np.arange(len(y)) % 5
    s_mixed = np.where(fold == 0, 100 * s, s)              # one fold on another scale
    r = fold_ranks(s_mixed, fold)
    assert 0 < r.min() and r.max() == 1
    g = bet_bucket(np.random.default_rng(0).choice([35, 70, 300], len(y)))
    b = bootstrap(y, z, {"m": r, "null": np.zeros(len(y))}, B=100, g=g)
    d = summarise(b["m"][0] - b["null"][0], b["m"][1] - b["null"][1])
    assert all(d[m]["lo"] > 0 for m in d), d


def test_strat_spearman_removes_activity():
    """z is higher among very active wallets: a score that only knows activity correlates with z
    overall, but not within bet-count buckets; a real within-bucket signal survives stratification."""
    rng = np.random.default_rng(3)
    n = 20000
    n_bets = rng.choice([35, 70, 300], n)
    z = rng.normal(size=n) + 0.5 * (n_bets == 300)            # higher z among very active wallets
    s = np.log(n_bets) + 0.01 * rng.normal(size=n)            # knows activity, nothing else
    g = bet_bucket(n_bets)
    assert spearman(z, s) > 0.1
    assert abs(spearman_strat(z, s, g)) < 0.03
    s2 = z + rng.normal(size=n)                               # real within-bucket signal survives
    assert spearman_strat(z, s2, g) > 0.5
