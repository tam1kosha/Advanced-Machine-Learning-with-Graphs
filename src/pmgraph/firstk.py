"""Inductive "first-k" snapshots: observe a wallet's first k post-cutoff fills, label what comes after.

Everything over fills runs in DuckDB on integer wallet ids (`nid`). Graph wallets (a node in the
pre-cutoff graph) keep their row index in `{split}_features.parquet`; other wallets get ids >= N.

For wallet w with post-cutoff fills ordered by time, r = 1, 2, ...:
  observed(w, k) = its first k post-cutoff fills (+ all its pre-cutoff fills if it is an existing wallet)
  label(w, k)    = bets on fills with ts > t_j(w) in markets not in observed(w, j), resolved markets
                   only, kept if >= min_resolved_bets bets; j = k (label_after="k", the default) or
                   j = K = max k (label_after="kmax": one population and label for every k, much smaller).
Observed and labelled fills never overlap (asserted per k).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .features import add_history, side_features, sides_sql
from .labels import bets_from_sides, skill_test

SIDE_COLS = "cp, is_maker, long_, bp, usd, ts, market_id, event_id, category, hours_to_close, price, qty, w, close_time"


def register_ids(con, graph_index: pd.Index, label_fills: Path) -> int:
    """TEMP TABLE ids(wallet, nid): graph wallets 0..N-1 (feature-row order), then label-window-only wallets."""
    N = len(graph_index)
    con.register("gidx", pd.DataFrame({"wallet": graph_index.to_numpy(), "nid": np.arange(N, dtype=np.int64)}))
    con.execute(f"""
      CREATE OR REPLACE TEMP TABLE ids AS
      WITH a AS (SELECT DISTINCT w FROM (SELECT taker w FROM '{label_fills}' UNION ALL
                                         SELECT maker FROM '{label_fills}')),
           n AS (SELECT w FROM a ANTI JOIN gidx g ON g.wallet = a.w)
      SELECT wallet, nid FROM gidx
      UNION ALL SELECT w, {N} + row_number() OVER (ORDER BY w) - 1 FROM n""")
    con.unregister("gidx")
    return N


def int_sides(con, fills: Path, name: str, keep: str | None = None) -> None:
    """TEMP TABLE `name`: sides of `fills` with integer wallet / counterparty ids.
    keep: optional relation with a `nid` column to restrict the wallet side to."""
    semi = f"SEMI JOIN {keep} k ON k.nid = i.nid" if keep else ""
    con.execute(f"""
      CREATE OR REPLACE TEMP TABLE {name} AS
      WITH s AS ({sides_sql(fills, "event_id, price, qty, w, close_time")})
      SELECT i.nid, c.nid cp, s.is_maker, s.long_, s.bp, s.usd, s.ts, s.market_id, s.event_id, s.category,
             s.hours_to_close, s.price, s.qty, s.w, s.close_time
      FROM s JOIN ids i ON i.wallet = s.wallet JOIN ids c ON c.wallet = s.cp {semi}""")


def prepare(con, N: int, K: int, min_bets: int, graph_fills: Path) -> None:
    """Needs TEMP TABLE ls (int_sides of the label window). Creates TEMP TABLEs
    cand (wallets that could get a label), obs (their first K post-cutoff fills, rank rn),
    gs (pre-cutoff sides of existing candidates) and gm (their pre-cutoff markets)."""
    con.execute(f"""
      CREATE OR REPLACE TEMP TABLE cand AS
      SELECT nid FROM ls GROUP BY nid
      HAVING count(*) > 1 AND count(DISTINCT CASE WHEN w IS NOT NULL THEN market_id END) >= {min_bets}""")
    con.execute(f"""
      CREATE OR REPLACE TEMP TABLE obs AS
      SELECT * FROM (SELECT ls.*, row_number() OVER (PARTITION BY nid ORDER BY ts, market_id, cp, is_maker) rn
                     FROM ls SEMI JOIN cand USING (nid))
      WHERE rn <= {K}""")
    int_sides(con, graph_fills, "gs", keep=f"(SELECT nid FROM cand WHERE nid < {N})")
    con.execute("CREATE OR REPLACE TEMP TABLE gm AS SELECT DISTINCT nid, market_id FROM gs")


def population(con, N: int, j: int, min_bets: int, unit: str, calib: pd.DataFrame) -> pd.DataFrame:
    """Label after the j-th post-cutoff fill (see module docstring) + leakage asserts.
    Returns one row per labelled wallet: label columns, t_j (time of the j-th fill), new."""
    con.execute(f"""
      CREATE OR REPLACE TEMP TABLE om AS
      SELECT DISTINCT nid, market_id FROM (SELECT nid, market_id FROM obs WHERE rn <= {j}
                                           UNION ALL SELECT nid, market_id FROM gm)""")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE tj AS
                    SELECT nid, max(ts) t_j FROM obs WHERE rn <= {j} GROUP BY 1 HAVING count(*) = {j}""")
    b = bets_from_sides(con, """
      SELECT ls.nid wallet, ls.market_id, ls.event_id, ls.ts, CASE WHEN ls.long_ THEN 1 ELSE -1 END sgn,
             ls.qty, ls.price, ls.w
      FROM ls JOIN tj USING (nid)
      ANTI JOIN om ON om.nid = ls.nid AND om.market_id = ls.market_id
      WHERE ls.ts > tj.t_j AND ls.w IS NOT NULL""", calib)
    lab = skill_test(b, unit=unit)
    lab = lab[lab["n_bets"] >= min_bets].copy()
    lab.index.name = "nid"
    pop = lab.join(con.execute("SELECT * FROM tj").fetchdf().set_index("nid"), how="inner")
    pop["new"] = pop.index >= N
    pop["skill"] = (pop["z"] > 2).astype(int)
    pop["profitable"] = (pop["profit"] > 0).astype(int)

    # ---- leakage checks ----
    con.register("pop_j", pd.DataFrame({"nid": pop.index.to_numpy()}))
    bb = b[b["wallet"].isin(pop.index)]
    first_lab = bb.groupby("wallet")["first_ts"].min().reindex(pop.index)
    last_obs = con.execute(f"""
      SELECT nid, max(ts) m FROM (SELECT nid, ts FROM obs WHERE rn <= {j} UNION ALL SELECT nid, ts FROM gs)
      SEMI JOIN pop_j USING (nid) GROUP BY 1""").fetchdf().set_index("nid")["m"].reindex(pop.index)
    assert (last_obs < first_lab).all(), f"j={j}: max observed ts >= min labelled ts"
    con.register("lab_mk", bb[["wallet", "market_id"]].rename(columns={"wallet": "nid"}))
    clash = con.execute("SELECT count(*) FROM lab_mk JOIN om USING (nid, market_id)").fetchall()[0][0]
    assert clash == 0, f"j={j}: {clash} labelled (wallet, market) pairs are in the observed set"
    con.unregister("lab_mk")
    con.unregister("pop_j")
    return pop


def first_k_features(con, k: int, calib: pd.DataFrame, wallets: np.ndarray) -> pd.DataFrame:
    """Wallet features on observed(w, k), as of the k-th fill. Needs TEMP TABLEs obs, gs."""
    con.register("fk_w", pd.DataFrame({"nid": np.asarray(wallets)}))
    con.execute(f"""
      CREATE OR REPLACE TEMP TABLE fk_asof AS
      SELECT nid, max(ts) t_asof FROM obs SEMI JOIN fk_w USING (nid) WHERE rn <= {k} GROUP BY 1""")
    con.execute(f"""
      CREATE OR REPLACE TEMP VIEW fk_sides AS
      SELECT nid wallet, {SIDE_COLS}, t_asof FROM obs JOIN fk_asof USING (nid) WHERE rn <= {k}
      UNION ALL SELECT nid, {SIDE_COLS}, t_asof FROM gs JOIN fk_asof USING (nid)""")
    f = side_features(con, "SELECT * FROM fk_sides", "t_asof")
    hb = bets_from_sides(con, """
      SELECT wallet, market_id, event_id, ts, CASE WHEN long_ THEN 1 ELSE -1 END sgn, qty, price, w
      FROM fk_sides WHERE w IS NOT NULL AND close_time < t_asof""", calib)
    f = add_history(f, hb)
    f.index.name = "nid"
    con.unregister("fk_w")
    return f
