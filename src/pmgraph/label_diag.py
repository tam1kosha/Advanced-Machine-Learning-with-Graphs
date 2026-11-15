"""Label diagnostics (iteration 4): what does the skill label measure?

Wallet i has excess edge per bet x_i = T_i - mu0_i and null sd s_i (`sd0` of labels.skill_test), so
its label is z_i = x_i / s_i. Random-effects model across wallets:
    x_i = theta_i + e_i,   theta_i ~ N(mu, tau2),   e_i ~ N(0, phi * s_i^2)
  mu   != 0  the null is not centred (e.g. the average wallet pays the spread)
  tau2 >  0  wallets really differ in edge (the thing a model should predict)
  phi  >  1  a wallet's bets are dependent, so the null sd is too small (noise, not skill)
Then z_i has mean mu / s_i and variance phi + tau2 / s_i^2: any of the three makes P(z > 2) depend
on the noise scale s_i, which a classifier can learn without knowing anything about theta_i.

tau2 and phi cannot be told apart from one cross-section if both scale alike, so they are separated
by split halves: a wallet's label bets are split in two, and the covariance across wallets of the
two half edges estimates the variance that is shared by the halves. Schemes (`assign_halves`):
  market  random half per market           shares persistent edge + any dependence across markets
  event   random half per event            ... minus dependence between sibling markets
  week    alternating market-close weeks   ... minus dependence inside a week
  time    earlier vs later closes          persistent edge only (test-retest over time)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score

from .labels import skill_test
from .rank_eval import spearman

SCHEMES = ("market", "event", "week", "time")
EDGE_METRICS = ("spearman_edge", "wcorr_edge", "top_minus_bottom", "spearman_z", "auprc_hi", "auprc_lo")


# ---------------------------------------------------------------- variance function
def variance_function(x, s, tau2: float | None = None, phi: float | None = None) -> dict:
    """ML fit of x_i ~ N(mu, tau2 + phi * s_i^2). Pass tau2 or phi to hold it fixed."""
    x, s2 = np.asarray(x, float), np.asarray(s, float) ** 2

    def unpack(p):
        p = list(p)
        t = tau2 if tau2 is not None else np.exp(p.pop(0))
        f = phi if phi is not None else np.exp(p.pop(0))
        return t, f

    def fit(p):
        t, f = unpack(p)
        v = t + f * s2
        mu = np.sum(x / v) / np.sum(1 / v)
        return mu, 0.5 * np.sum(np.log(v) + (x - mu) ** 2 / v)

    p0 = [np.log(max(np.var(x) / 2, 1e-10))] * (tau2 is None) + [0.0] * (phi is None)
    p = minimize(lambda q: fit(q)[1], p0, method="Nelder-Mead",
                 options=dict(xatol=1e-4, fatol=1e-6, maxiter=2000)).x if p0 else []
    t, f = unpack(p)
    mu, nll = fit(p)
    return dict(mu=float(mu), tau2=float(t), phi=float(f), nll=float(nll), n=len(x))


# ---------------------------------------------------------------- split halves
def assign_halves(b: pd.DataFrame, scheme: str, seed: int = 0) -> np.ndarray:
    """0/1 half of every bet (rows of a labels.bets_from_sides frame; week/time need `close_time`).
    Does not depend on the row order of b (DuckDB returns bets in no fixed order)."""
    if scheme in ("market", "event"):
        key = b["market_id"] if scheme == "market" else b["event_id"].fillna("m" + b["market_id"].astype(str))
        uniq, codes = np.unique(key.to_numpy(), return_inverse=True)
        return np.random.default_rng(seed).integers(0, 2, len(uniq))[codes]
    if scheme == "week":
        t0 = pd.Timestamp("1970-01-01", tz=b["close_time"].dt.tz)
        return (((b["close_time"] - t0) // pd.Timedelta(days=7)) % 2).to_numpy().astype(int)
    if scheme == "time":                     # ties in close_time are broken by market_id
        o = b[["wallet", "close_time", "market_id"]].sort_values(["wallet", "close_time", "market_id"])
        g = o.groupby("wallet", sort=False)
        return (g.cumcount() >= g["market_id"].transform("size") / 2).astype(int).reindex(b.index).to_numpy()
    raise ValueError(scheme)


def half_table(b: pd.DataFrame, scheme: str, seed: int = 0, unit: str = "market") -> pd.DataFrame:
    """Per wallet: excess edge x, null sd s and bet count n of each half (x0, s0, n0, x1, s1, n1)."""
    h = assign_halves(b, scheme, seed)
    parts = []
    for v in (0, 1):
        t = skill_test(b[h == v], unit=unit)
        parts.append(pd.DataFrame({f"x{v}": t["T"] - t["mu0"], f"s{v}": t["sd0"], f"n{v}": t["n_bets"]}))
    return parts[0].join(parts[1], how="inner")


def split_half(H: pd.DataFrame, min_half: int = 10, B: int = 500, seed: int = 0) -> dict:
    """Covariance of the two half edges across wallets (= variance shared by the halves), wallet
    bootstrap CI, and what is left of a half's variance in units of its null variance (phi)."""
    H = H[(H.n0 >= min_half) & (H.n1 >= min_half)]
    x0, x1 = H["x0"].to_numpy(), H["x1"].to_numpy()
    n = len(H)
    if n < 20:
        return dict(n=n)
    cov = lambda i: np.cov(x0[i], x1[i])[0, 1]  # noqa: E731
    reps = [cov(i) for i in np.random.default_rng(seed).integers(0, n, (B, n))]
    c, var_half = float(np.cov(x0, x1)[0, 1]), (x0.var(ddof=1) + x1.var(ddof=1)) / 2
    null_half = float((H["s0"] ** 2 + H["s1"] ** 2).mean() / 2)
    return dict(n=n, cov=c, cov_lo=float(np.quantile(reps, .025)), cov_hi=float(np.quantile(reps, .975)),
                corr=float(np.corrcoef(x0, x1)[0, 1]), spearman=spearman(x0, x1),
                var_half=float(var_half), null_var_half=null_half, phi=float((var_half - c) / null_half))


# ---------------------------------------------------------------- persistence and reference scores
def persistence(H: pd.DataFrame, q: int = 10, min_half: int = 10) -> pd.DataFrame:
    """Wallets sorted into q groups by the z of half 0; what half 1 then shows per group (edges in
    cents). With the `time` scheme this is the classic sort on past performance."""
    H = H[(H.n0 >= min_half) & (H.n1 >= min_half)]
    d = pd.DataFrame({"zA": H.x0 / H.s0, "xA_c": 100 * H.x0, "xB_c": 100 * H.x1, "zB": H.x1 / H.s1})
    g = d.groupby(pd.qcut(d["zA"], q, labels=False))
    out = g.agg(n=("zA", "size"), zA=("zA", "mean"), xA_c=("xA_c", "mean"), xB_c=("xB_c", "mean"),
                xB_c_median=("xB_c", "median"), xB_c_se=("xB_c", "sem"), zB=("zB", "mean"),
                p_hi=("zB", lambda z: (z > 2).mean()), p_lo=("zB", lambda z: (z < -2).mean()))
    return out.rename_axis("group").reset_index()


def blind_score(s, y, bins: int = 20, seed: int = 0) -> np.ndarray:
    """Skill-blind score for a binary label y: the label rate of the wallet's noise-scale bin (quantile
    bins of s), cross-fitted on two random folds so that a wallet's own label never enters its score.
    It knows how noisy the wallet's label is and nothing else."""
    s, y = np.asarray(s, float), np.asarray(y, float)
    b = pd.qcut(s, bins, labels=False, duplicates="drop")
    f = np.random.default_rng(seed).integers(0, 2, len(s))
    out = np.zeros(len(s))
    for v in (0, 1):
        rate = pd.Series(y[f != v]).groupby(b[f != v]).mean()
        out[f == v] = pd.Series(b[f == v]).map(rate).fillna(y[f != v].mean()).to_numpy()
    return out


# ---------------------------------------------------------------- scores against realised edge
def edge_metrics(r, x, w, z, thr: float = 2.0) -> np.ndarray:
    """A score r against a realised excess edge x (inverse-variance weights w, z = x / s). EDGE_METRICS:
    Spearman with x; weighted Pearson of r with x; weighted mean x of the top minus the bottom score
    decile; Spearman with z; AUPRC for z > thr; AUPRC of the negated score for z < -thr (a low score
    should mean a losing wallet)."""
    r, x, w, z = (np.asarray(a, float) for a in (r, x, w, z))
    rc, xc = r - np.average(r, weights=w), x - np.average(x, weights=w)
    den = np.sqrt((w * rc * rc).sum() * (w * xc * xc).sum())
    lo, hi = np.quantile(r, [.1, .9])
    top, bot = r >= hi, r <= lo
    tmb = (np.average(x[top], weights=w[top]) - np.average(x[bot], weights=w[bot])
           if top.mean() < .5 and bot.mean() < .5 else np.nan)       # nan for (near-)constant scores
    ap = lambda y, s: average_precision_score(y, s) if 0 < y.sum() < len(y) else np.nan  # noqa: E731
    return np.array([spearman(x, r), (w * rc * xc).sum() / den if den > 0 else 0.0, tmb, spearman(z, r),
                     ap(z > thr, r), ap(z < -thr, -r)])


def boot_edge(scores: dict, x, w, z, B: int = 300, seed: int = 0, reps: bool = False) -> dict:
    """{method: (est, lo, hi)} over EDGE_METRICS; the same resampled wallets for every method.
    reps=True appends the B bootstrap replicates (for paired differences between methods)."""
    x, w, z = (np.asarray(a, float) for a in (x, w, z))
    idx = np.random.default_rng(seed).integers(0, len(x), (B, len(x)))
    out = {}
    for name, r in scores.items():
        r = np.asarray(r, float)
        R = np.array([edge_metrics(r[i], x[i], w[i], z[i]) for i in idx])
        ci = np.nanquantile(R, [.025, .975], axis=0)
        out[name] = (edge_metrics(r, x, w, z), ci[0], ci[1]) + ((R,) if reps else ())
    return out
