"""Wallet features and the counterparty graph, both computed strictly before the cutoff."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .labels import bets, skill_test

CATS = ["Politics", "Sports", "Crypto", "Finance", "Culture", "Tech", "Weather"]


def wallet_features(con, fills: Path, cutoff: str, calib: pd.DataFrame) -> pd.DataFrame:
    """Behavioural features from graph-window fills (all of which are < cutoff)."""
    cat_cols = ",\n".join(f"avg((category = '{c}')::INT) AS cat_{c.lower()}" for c in CATS)
    f = con.execute(f"""
      WITH s AS (
        SELECT taker wallet, maker cp, 0 is_maker, CASE WHEN taker_long THEN price ELSE 1 - price END bp,
               usd, ts, market_id, category, hours_to_close FROM '{fills}'
        UNION ALL
        SELECT maker, taker, 1, CASE WHEN taker_long THEN 1 - price ELSE price END,
               usd, ts, market_id, category, hours_to_close FROM '{fills}')
      SELECT wallet,
             count(*)                                         AS n_fills,
             ln(1 + sum(usd))                                 AS log_usd,
             ln(1 + median(usd))                              AS log_med_usd,
             avg(is_maker)                                    AS maker_share,
             avg(bp)                                          AS avg_buy_price,
             avg((bp < 0.10)::INT)                            AS frac_longshot,
             avg((bp > 0.90)::INT)                            AS frac_sureshot,
             avg(ln(1 + greatest(hours_to_close, 0)))         AS avg_log_h_to_close,
             count(DISTINCT market_id)                        AS n_markets,
             count(DISTINCT cp)                               AS n_counterparties,
             count(DISTINCT date_trunc('day', ts))            AS active_days,
             date_diff('hour', min(ts), TIMESTAMPTZ '{cutoff}') / 24.0 AS age_days,
             date_diff('hour', max(ts), TIMESTAMPTZ '{cutoff}') / 24.0 AS days_since_last,
             {cat_cols}
      FROM s GROUP BY 1""").fetchdf().set_index("wallet")
    f["log_n_fills"] = np.log1p(f.pop("n_fills"))
    for c in ["n_markets", "n_counterparties", "active_days"]:
        f[c] = np.log1p(f[c])

    # history-only skill: same test, restricted to markets closed before the cutoff
    hb = bets(con, fills, calib, closed_before=cutoff)
    h = skill_test(hb)[["n_bets", "T", "z", "raw_edge", "profit"]]
    h.columns = ["hist_n_bets", "hist_T", "hist_z", "hist_raw_edge", "hist_profit"]
    f = f.join(h)
    f["hist_n_bets"] = np.log1p(f["hist_n_bets"].fillna(0))
    f["hist_profit_slog"] = np.sign(f["hist_profit"]) * np.log1p(f["hist_profit"].abs())
    f = f.drop(columns="hist_profit")
    return f


def counterparty_graph(con, fills: Path, index: pd.Index) -> sp.csr_matrix:
    """Undirected weighted adjacency, weight = log(1 + #trades between the pair)."""
    ids = pd.DataFrame({"wallet": index.to_numpy(), "nid": np.arange(len(index), dtype=np.int32)})
    con.register("wallet_ids", ids)
    e = con.execute(f"""
      SELECT a.nid r, b.nid c, count(*) n FROM '{fills}' f
      JOIN wallet_ids a ON a.wallet = f.taker JOIN wallet_ids b ON b.wallet = f.maker
      GROUP BY 1, 2""").fetchnumpy()
    r, c = e["r"].astype(np.int32), e["c"].astype(np.int32)
    w = np.log1p(e["n"]).astype(np.float32)
    del e
    A = sp.coo_matrix((w, (r, c)), shape=(len(index),) * 2).tocsr()
    A = (A + A.T).tocsr()
    A.sum_duplicates()
    return A


def sym_norm(A: sp.csr_matrix, self_loops: bool = True) -> sp.csr_matrix:
    if self_loops:
        A = A + sp.eye(A.shape[0], dtype=A.dtype, format="csr")
    d = np.asarray(A.sum(1)).ravel()
    dinv = sp.diags(1 / np.sqrt(np.maximum(d, 1e-12)))
    return (dinv @ A @ dinv).tocsr()


def neighbour_features(A: sp.csr_matrix, X: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Weighted mean of neighbours' features (one hop) + degree stats: a cheap graph baseline."""
    d = np.asarray(A.sum(1)).ravel()
    P = sp.diags(1 / np.maximum(d, 1e-12)) @ A
    V = X[cols].fillna(0).to_numpy()
    out = pd.DataFrame(P @ V, index=X.index, columns=[f"nbr_{c}" for c in cols])
    out["deg_w"] = np.log1p(d)
    out["deg"] = np.log1p(np.diff(A.indptr))
    return out
