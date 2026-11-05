"""Four-type labels from market resolution (proposal §2).

Per bet m (= a wallet's position in one market inside the label window), with fills i:
    s_i = +1 long / -1 short the reference token, q_i shares, p_i price, pi_i = calib(p_i)
    Q = sum q_i,  A = sum s q / Q  (net direction, in [-1, 1]),  C = sum s q pi / Q
    d_m      = A * W - C          signed outcome minus calibrated price, per share
    profit_m = sum s q (W - p)    dollars at resolution (no fees: see README)
Skill statistic T = mean_m d_m (every bet counted equally). Null: W_m ~ Bernoulli(pibar_m)
independently, pibar_m = sum q pi / Q. One-sided p-value (analytic normal, optional Monte
Carlo), Benjamini-Hochberg across wallets.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.isotonic import IsotonicRegression

TYPES = ["skilled_winner", "lucky_winner", "unlucky_skilled", "unskilled_loser"]


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
    bins = len(calib)
    con.register("calib", calib[["b", "pi"]])
    where = "w IS NOT NULL"
    if closed_before:
        where += f" AND close_time < TIMESTAMPTZ '{closed_before}'"
    if start:
        where += f" AND ts >= TIMESTAMPTZ '{start}'"
    if end:
        where += f" AND ts < TIMESTAMPTZ '{end}'"
    return con.execute(f"""
      WITH f AS (SELECT *, least(floor(price * {bins}), {bins - 1})::INT b FROM '{fills}' WHERE {where}),
      s AS (
        SELECT taker wallet, market_id, ts, CASE WHEN taker_long THEN 1 ELSE -1 END sgn, qty, price, w, b FROM f
        UNION ALL
        SELECT maker,        market_id, ts, CASE WHEN taker_long THEN -1 ELSE 1 END,     qty, price, w, b FROM f),
      j AS (SELECT s.*, c.pi FROM s JOIN calib c USING (b))
      SELECT wallet, market_id,
             any_value(w)                       AS w,
             min(ts)                            AS first_ts,
             count(*)                           AS n_fills,
             sum(qty)                           AS Q,
             sum(sgn * qty) / sum(qty)          AS A,
             sum(sgn * qty * pi) / sum(qty)     AS C,
             sum(qty * pi) / sum(qty)           AS pibar,
             sum(sgn * qty * price) / sum(qty)  AS B,
             sum(sgn * qty * (w - price))       AS profit
      FROM j GROUP BY 1, 2
      HAVING abs(sum(sgn * qty)) / sum(qty) >= {min_abs_dir}
    """).fetchdf()


def skill_test(b: pd.DataFrame, n_sim: int = 0, seed: int = 0) -> pd.DataFrame:
    """Per-wallet skill statistic, analytic z-test and (optionally) Monte Carlo p-value."""
    b = b.copy()
    b["d"] = b["A"] * b["w"] - b["C"]
    b["d0"] = b["A"] * b["pibar"] - b["C"]
    b["v0"] = b["A"] ** 2 * b["pibar"] * (1 - b["pibar"])
    b["raw_edge"] = b["A"] * b["w"] - b["B"]          # outcome minus price paid, uncalibrated
    g = b.groupby("wallet")
    out = pd.DataFrame({
        "n_bets": g.size(),
        "T": g["d"].mean(),
        "mu0": g["d0"].mean(),
        "sd0": np.sqrt(g["v0"].sum()) / g.size(),
        "raw_edge": g["raw_edge"].mean(),
        "profit": g["profit"].sum(),
        "volume_sh": g["Q"].sum(),
        "first_label_ts": g["first_ts"].min(),
    })
    out["z"] = (out["T"] - out["mu0"]) / out["sd0"].clip(lower=1e-9)
    out["p"] = stats.norm.sf(out["z"])
    if n_sim:
        out["p_mc"] = _mc_pvalues(b, out, n_sim, seed)
    return out


def _mc_pvalues(b, out, n_sim, seed):
    rng = np.random.default_rng(seed)
    p = pd.Series(np.nan, index=out.index)
    for wlt, grp in b.groupby("wallet"):
        W = rng.random((n_sim, len(grp))) < grp["pibar"].to_numpy()
        Tsim = (W * grp["A"].to_numpy() - grp["C"].to_numpy()).mean(1)
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
