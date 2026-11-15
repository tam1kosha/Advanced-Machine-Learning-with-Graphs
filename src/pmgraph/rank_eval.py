"""Rare-class evaluation with more power (NOTES iteration 2, next step 1; labels unchanged).

One 20% test fold holds ~70 skilled new wallets, so per-seed AUPRC is noisy. With K-fold CV every
wallet is scored exactly once: pooling the out-of-fold scores uses all positives (~5x more).
Scores are turned into within-fold percentile ranks before pooling, so folds whose models output
different score scales stay comparable. Uncertainty: wallet bootstrap of the pooled metric; methods
are compared by a paired bootstrap (same resampled wallets for every method).
Metrics for the skill target (label z > 2, z = continuous label z-score):
    auprc       average precision for z > 2
    p_at_100    share of z > 2 among the 100 top-scored wallets (expected value under ties)
    spearman_z  rank correlation of the score with the label z (uses every wallet, not only positives)
    spearman_z_strat  the same within label bet-count buckets (30-49, 50-99, >= 100), size-weighted.
                Under real skill z grows with sqrt(bets), so spearman_z also rewards predicting activity;
                the stratified version removes that between-bucket part.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

METRICS = ("auprc", "p_at_100", "spearman_z", "spearman_z_strat")
BET_BUCKETS = (30, 50, 100)


def bet_bucket(n_bets: np.ndarray) -> np.ndarray:
    return np.searchsorted(BET_BUCKETS, n_bets, side="right")


def precision_at(y: np.ndarray, s: np.ndarray, K: int) -> float:
    """Expected precision of the top-K scores when ties at the cut-off are broken at random."""
    K = min(K, len(s))
    thr = np.partition(s, -K)[-K]
    above, tie = s > thr, s == thr
    return float((y[above].sum() + (K - above.sum()) * y[tie].mean()) / K)


def spearman(z: np.ndarray, s: np.ndarray) -> float:
    a, b = rankdata(z), rankdata(s)
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else 0.0


def spearman_strat(z: np.ndarray, s: np.ndarray, g: np.ndarray) -> float:
    """Size-weighted mean of the within-group Spearman correlations (groups with < 20 wallets skipped)."""
    r, w = [], []
    for v in np.unique(g):
        m = g == v
        if m.sum() >= 20:
            r.append(spearman(z[m], s[m]))
            w.append(m.sum())
    return float(np.average(r, weights=w)) if w else np.nan


def metrics(y: np.ndarray, z: np.ndarray, s: np.ndarray, K: int = 100, g: np.ndarray | None = None) -> np.ndarray:
    return np.array([average_precision_score(y, s) if 0 < y.sum() < len(y) else np.nan,
                     precision_at(y, s, K), spearman(z, s),
                     spearman_strat(z, s, g) if g is not None else np.nan])


def fold_ranks(s: np.ndarray, fold: np.ndarray) -> np.ndarray:
    """Percentile rank of each score within its CV fold (ties share the average rank)."""
    r = np.empty(len(s))
    for f in np.unique(fold):
        m = fold == f
        r[m] = rankdata(s[m]) / m.sum()
    return r


def bootstrap(y: np.ndarray, z: np.ndarray, scores: dict, B: int = 1000, seed: int = 0, K: int = 100,
              g: np.ndarray | None = None):
    """Point estimates and B paired bootstrap replicates per method: {m: (est[M], reps[B, M])}.
    g: group per wallet (e.g. bet_bucket(n_bets)) for spearman_z_strat; nan without it."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), (B, len(y)))
    out = {}
    for m, s in scores.items():
        reps = np.array([metrics(y[i], z[i], s[i], K, None if g is None else g[i]) for i in idx])
        out[m] = (metrics(y, z, s, K, g), reps)
    return out


def summarise(est: np.ndarray, reps: np.ndarray, level: float = 0.95) -> dict:
    """Estimate, percentile CI and the bootstrap share of replicates <= 0 (one-sided p for a difference)."""
    a = (1 - level) / 2
    lo, hi = np.nanquantile(reps, [a, 1 - a], axis=0)
    return {m: dict(est=est[j], lo=lo[j], hi=hi[j], p_le0=float(np.mean(reps[:, j] <= 0)))
            for j, m in enumerate(METRICS)}
