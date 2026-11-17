"""Four-type labels from market resolution (proposal §2).

Per bet m (= a wallet's position in one market inside the label window), with fills i:
    s_i = +1 long / -1 short the reference token, q_i shares, p_i price, pi_i = calib(p_i)
    Q = sum q_i,  A = sum s q / Q  (net direction, in [-1, 1]),  C = sum s q pi / Q
    d_m      = A * W - C          signed outcome minus calibrated price, per share
    profit_m = sum s q (W - p)    dollars at resolution (no fees: see README)
Skill statistic T = mean_m d_m (every bet counted equally). Null: W_m ~ Bernoulli(pibar_m)
independently, pibar_m = sum q pi / Q. One-sided p-value (analytic normal, optional Monte
Carlo), Benjamini-Hochberg across wallets.

Event-level bets (unit="event"). Sibling markets of one event (election sub-markets, nested
"by date X" markets, spreads/totals of one game) are not independent. With unit="event" the
wallet's market bets inside one event are averaged into one event bet, d_e = mean_{m in e} d_m,
and T = mean_e d_e; events are independent. Within an event the null couples the sibling
outcomes so that the wallet's payoffs are comonotone: one U_e ~ U(0,1) per event,
W_m = 1[U_e < pibar_m] if long, 1[U_e > 1 - pibar_m] if short (all its bets win for small U_e).
A comonotone sum has the largest variance of any dependence with these marginals, so the test
stays valid (conservative) whatever the true dependence, e.g. mutually exclusive candidates.
Exact null variance of d_e, with a_m = |A_m| / n_e and q_m = P(bet m pays):
    Var = sum_{m,m'} a_m a_m' (min(q_m, q_m') - q_m q_m').
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.isotonic import IsotonicRegression

TYPES = ["skilled_winner", "lucky_winner", "unlucky_skilled", "unskilled_loser"]
# Net direction is a dimensionless ratio in [-1, 1]. Treat machine-roundoff around an inclusive
# cutoff as equality; a mathematically 0.1 position must not disappear when aggregate order changes.
DIRECTION_ATOL = 1e-12


def fit_calibration(con, fills: Path, cutoff: str, bins: int = 100) -> pd.DataFrame:
    """Favourite-longshot calibration price -> P(win), market-equal weights, monotone.

    Uses only markets that closed before `cutoff` (no look-ahead) and symmetrises
    each fill (p, w) with (1 - p, 1 - w) so the curve is consistent for both tokens.
    """
    df = con.execute(f"""
      WITH f AS (
        SELECT price p, w, 1.0 / count(*) OVER (PARTITION BY market_id) wt
        FROM '{fills}'
        WHERE w IS NOT NULL AND close_time < TIMESTAMPTZ '{cutoff}'),
      x AS (SELECT p, w, wt FROM f UNION ALL SELECT 1 - p, 1 - w, wt FROM f)
      SELECT least(floor(p * {bins}), {bins - 1})::INT b, sum(w * wt) / sum(wt) rate, sum(wt) wsum
      FROM x GROUP BY 1 ORDER BY 1""").fetchdf()
    grid = pd.DataFrame({"b": np.arange(bins)})
    df = grid.merge(df, on="b", how="left")
    mid = (df["b"] + 0.5) / bins
    ok = df["rate"].notna()
    iso = IsotonicRegression(y_min=1e-4, y_max=1 - 1e-4, out_of_bounds="clip")
    iso.fit(mid[ok], df.loc[ok, "rate"], sample_weight=df.loc[ok, "wsum"])
    df["pi"] = iso.predict(mid)
    df["mid"] = mid
    return df[["b", "mid", "rate", "wsum", "pi"]]


def bets(con, fills: Path, calib: pd.DataFrame, min_abs_dir: float = 0.1,
         start: str | None = None, end: str | None = None,
         closed_before: str | None = None) -> pd.DataFrame:
    """(wallet, market) bets on resolved markets with fills in [start, end).

    closed_before: keep only markets that closed before this time (for leak-free history).
    """
    where = "w IS NOT NULL"
    if closed_before:
        where += f" AND close_time < TIMESTAMPTZ '{closed_before}'"
    if start:
        where += f" AND ts >= TIMESTAMPTZ '{start}'"
    if end:
        where += f" AND ts < TIMESTAMPTZ '{end}'"
    sides = f"""
        WITH f AS (SELECT * FROM '{fills}' WHERE {where})
        SELECT taker wallet, market_id, event_id, ts, CASE WHEN taker_long THEN 1 ELSE -1 END sgn,
               qty, price, w FROM f
        UNION ALL
        SELECT maker,        market_id, event_id, ts, CASE WHEN taker_long THEN -1 ELSE 1 END,
               qty, price, w FROM f"""
    return bets_from_sides(con, sides, calib, min_abs_dir)


def bets_from_sides(con, sides: str, calib: pd.DataFrame, min_abs_dir: float = 0.1) -> pd.DataFrame:
    """Bets from a relation with columns wallet, market_id, event_id, ts, sgn (+1 long the reference
    token), qty, price, w (resolved only). Filtering (windows, leak rules) is the caller's job.
    Compensated sums reduce order-dependent roundoff. The inclusive net-direction cutoff allows
    DIRECTION_ATOL on the ratio; label consistency checks retain their original strict tolerance.
    """
    bins = len(calib)
    con.register("calib", calib[["b", "pi"]])
    return con.execute(f"""
      WITH s AS (SELECT *, least(floor(price * {bins}), {bins - 1})::INT b FROM ({sides})),
      j AS (SELECT s.*, c.pi FROM s JOIN calib c USING (b))
      SELECT wallet, market_id,
             any_value(event_id)                AS event_id,
             any_value(w)                       AS w,
             min(ts)                            AS first_ts,
             count(*)                           AS n_fills,
             fsum(qty)                           AS Q,
             fsum(sgn * qty) / fsum(qty)          AS A,
             fsum(sgn * qty * pi) / fsum(qty)     AS C,
             fsum(qty * pi) / fsum(qty)           AS pibar,
             fsum(sgn * qty * price) / fsum(qty)  AS B,
             fsum(sgn * qty * (w - price))       AS profit
      FROM j GROUP BY 1, 2
      HAVING abs(fsum(sgn * qty)) / fsum(qty) >= {min_abs_dir - DIRECTION_ATOL}
    """).fetchdf()


UNITS = ("market", "event")


def labels_file(processed: Path, split: str, unit: str = "market") -> Path:
    """market-level labels keep the iteration-1 name; other units get a suffix."""
    return processed / (f"{split}_labels.parquet" if unit == "market" else f"{split}_labels_{unit}.parquet")


def skill_test(b: pd.DataFrame, n_sim: int = 0, seed: int = 0, unit: str = "market") -> pd.DataFrame:
    """Per-wallet skill statistic, analytic z-test and (optionally) Monte Carlo p-value.

    unit="market": every (wallet, market) bet is one independent observation (iteration-1 default).
    unit="event":  bets are averaged within (wallet, event_id); see module docstring.
    """
    if unit not in UNITS:
        raise ValueError(unit)
    b = b.copy()
    b["d"] = b["A"] * b["w"] - b["C"]
    b["d0"] = b["A"] * b["pibar"] - b["C"]
    b["raw_edge"] = b["A"] * b["w"] - b["B"]          # outcome minus price paid, uncalibrated
    b["grp"] = b["market_id"] if unit == "market" else b["event_id"].fillna("m" + b["market_id"].astype(str))
    # weight of each market bet in T: 1 / (#markets of its group * #groups of the wallet)
    n_in = b.groupby(["wallet", "grp"])["d"].transform("size")
    n_grp = b.groupby("wallet")["grp"].transform("nunique")
    b["wt"] = 1.0 / (n_in * n_grp)
    var_g = _group_null_var(b, n_in)
    g = b.groupby("wallet")
    out = pd.DataFrame({
        "n_bets": g["grp"].nunique(),
        "n_markets": g.size(),
        "T": (b["d"] * b["wt"]).groupby(b["wallet"]).sum(),
        "mu0": (b["d0"] * b["wt"]).groupby(b["wallet"]).sum(),
        "raw_edge": (b["raw_edge"] * b["wt"]).groupby(b["wallet"]).sum(),
        "profit": g["profit"].sum(),
        "volume_sh": g["Q"].sum(),
        "first_label_ts": g["first_ts"].min(),
    })
    out["sd0"] = np.sqrt(var_g.groupby(level="wallet").sum()) / out["n_bets"]
    out["z"] = (out["T"] - out["mu0"]) / out["sd0"].clip(lower=1e-9)
    out["p"] = stats.norm.sf(out["z"])
    if n_sim:
        out["p_mc"] = _mc_pvalues(b, out, n_sim, seed)
    return out


def _group_null_var(b: pd.DataFrame, n_in: pd.Series) -> pd.Series:
    """Null variance of each (wallet, group) mean bet under comonotone payoffs (module docstring)."""
    q = np.where(b["A"] > 0, b["pibar"], 1 - b["pibar"])
    s = pd.DataFrame({"wallet": b["wallet"].to_numpy(), "grp": b["grp"].to_numpy(), "q": q,
                      "a": (np.abs(b["A"]) / n_in).to_numpy()}).sort_values(["wallet", "grp", "q"])
    key = [s["wallet"], s["grp"]]
    # sorted by q: sum_{m,m'} a a' min(q, q') = sum_m a_m q_m (a_m + 2 * sum_{m' after m} a_m')
    after = s["a"].groupby(key).transform("sum") - s["a"].groupby(key).cumsum()
    s["S"] = s["a"] * s["q"] * (s["a"] + 2 * after)
    s["aq"] = s["a"] * s["q"]
    g = s.groupby(["wallet", "grp"])
    return (g["S"].sum() - g["aq"].sum() ** 2).clip(lower=0)


def _mc_pvalues(b, out, n_sim, seed):
    """One uniform per group and draw; within a group the wallet's payoffs are comonotone."""
    rng = np.random.default_rng(seed)
    p = pd.Series(np.nan, index=out.index)
    for wlt, grp in b.groupby("wallet"):
        gi = pd.factorize(grp["grp"])[0]
        U = rng.random((n_sim, gi.max() + 1))[:, gi]
        pi, long = grp["pibar"].to_numpy(), grp["A"].to_numpy() > 0
        W = np.where(long, U < pi, U > 1 - pi)
        Tsim = (W * grp["A"].to_numpy() - grp["C"].to_numpy()) @ grp["wt"].to_numpy()
        p[wlt] = (1 + (Tsim >= out.at[wlt, "T"] - 1e-12).sum()) / (n_sim + 1)
    return p


def bh(p: np.ndarray, alpha: float) -> np.ndarray:
    """Benjamini-Hochberg rejections."""
    p = np.asarray(p)
    n = len(p)
    order = np.argsort(p)
    thresh = alpha * np.arange(1, n + 1) / n
    passed = p[order] <= thresh
    k = passed.nonzero()[0].max() + 1 if passed.any() else 0
    rej = np.zeros(n, bool)
    rej[order[:k]] = True
    return rej


def assign_types(w: pd.DataFrame, min_bets: int, alpha: float, pcol: str = "p") -> pd.DataFrame:
    w = w[w["n_bets"] >= min_bets].copy()
    w["skilled"] = bh(w[pcol].to_numpy(), alpha)
    w["profitable"] = w["profit"] > 0
    w["type"] = np.select(
        [w.skilled & w.profitable, ~w.skilled & w.profitable, w.skilled & ~w.profitable],
        TYPES[:3], TYPES[3])
    w["y"] = w["type"].map({t: i for i, t in enumerate(TYPES)})
    return w
