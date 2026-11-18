"""Gate C: as-of nodes for markets that have no node in the frozen (pre-cutoff) graph.

61-78% of new wallets' first-k fills are in markets with no pre-cutoff trade, so in iterations 2-4
the wallet-market channel never reached them. Here every (query wallet q, k, such market m) gets a
private market node built only from fills with ts < t_k(q), the time of q's k-th fill:
  features  same columns as the frozen market nodes (hetero.market_features), as of t_k:
            category, log volume / fills / distinct takers, log age, last price, closed before t_k
            and the outcome if so, log hours to scheduled expiry
  in-edges  (wallet, held, market) from up to `max_traders` frozen wallets (pre-cutoff nodes) that
            entered m most recently before t_k, attrs = HELD_ATTRS of their fills in m before t_k
Like the query wallet itself, the node only receives messages, so nothing after t_k(q) and nothing
from another query wallet's activity reaches q. Every fill used is checked to be < t_k (assert).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import CATS


def build(con, inst: pd.DataFrame, sides: str, N: int, max_traders: int, markets_parquet) -> tuple[pd.DataFrame, pd.DataFrame]:
    """inst: iid, nid (query wallet), market_id, t_k. sides: TEMP TABLE of label-window int sides
    (firstk.int_sides columns). Returns (features per iid, in-edges: iid, src nid, HELD_ATTRS)."""
    con.register("pm_inst_df", inst[["iid", "nid", "market_id", "t_k"]])
    con.execute("CREATE OR REPLACE TEMP TABLE pm_inst AS SELECT * FROM pm_inst_df")
    con.unregister("pm_inst_df")
    con.execute("CREATE OR REPLACE TEMP TABLE pm_mk AS SELECT DISTINCT market_id FROM pm_inst")
    # every fill once (its taker side): market activity as of each time (RANGE frame: ties counted together)
    con.execute(f"""
      CREATE OR REPLACE TEMP TABLE pm_cum AS
      SELECT market_id, ts, price, sum(usd) OVER w cum_usd, count(*) OVER w cum_n,
             min(ts) OVER (PARTITION BY market_id) first_ts
      FROM {sides} SEMI JOIN pm_mk USING (market_id) WHERE is_maker = 0
      WINDOW w AS (PARTITION BY market_id ORDER BY ts)""")
    con.execute(f"""
      CREATE OR REPLACE TEMP TABLE pm_tk AS
      SELECT market_id, ets ts, count(*) OVER (PARTITION BY market_id ORDER BY ets) cum_takers
      FROM (SELECT market_id, nid, min(ts) ets FROM {sides} SEMI JOIN pm_mk USING (market_id)
            WHERE is_maker = 0 GROUP BY 1, 2)""")
    con.execute(f"""
      CREATE OR REPLACE TEMP TABLE pm_meta AS
      SELECT s.market_id, any_value(s.category) category, any_value(s.close_time) close_time, any_value(s.w) w,
             any_value(mk.expiration_time)::TIMESTAMPTZ expiration_time
      FROM {sides} s SEMI JOIN pm_mk USING (market_id)
      LEFT JOIN read_parquet('{markets_parquet}') mk ON mk.market_id = s.market_id GROUP BY 1""")
    cats = ", ".join(f"(m.category = '{c}')::INT AS cat_{c.lower()}" for c in CATS)
    F = con.execute(f"""
      SELECT i.iid, {cats},
             ln(1 + coalesce(c.cum_usd, 0)) log_volume, ln(1 + coalesce(c.cum_n, 0)) log_fills,
             ln(1 + coalesce(t.cum_takers, 0)) log_takers,
             ln(1 + greatest(coalesce(date_diff('hour', c.first_ts, i.t_k), 0), 0)) log_age_h,
             coalesce(c.price, 0.5) last_price,
             coalesce(m.close_time < i.t_k, false)::INT closed_before_cutoff,
             CASE WHEN m.close_time < i.t_k THEN coalesce(m.w, 0.5) - 0.5 ELSE 0 END outcome_if_closed,
             ln(1 + greatest(coalesce(date_diff('second', i.t_k, m.expiration_time) / 3600.0, 0), 0)) log_h_to_expiry,
             c.ts::VARCHAR used_ts, t.ts::VARCHAR used_ts2, i.t_k::VARCHAR t_k_s
      FROM pm_inst i
      ASOF LEFT JOIN pm_cum c ON c.market_id = i.market_id AND i.t_k > c.ts
      ASOF LEFT JOIN pm_tk t ON t.market_id = i.market_id AND i.t_k > t.ts
      JOIN pm_meta m ON m.market_id = i.market_id
      ORDER BY i.iid""").fetchdf().set_index("iid")
    tk = pd.to_datetime(F.pop("t_k_s"))
    for c in ["used_ts", "used_ts2"]:
        u = pd.to_datetime(F.pop(c))
        assert (u.isna() | (u < tk)).all(), f"as-of market features use a fill at/after t_k ({c})"
    # traders: frozen wallets, entry rank per market; the max_traders latest entrants before t_k
    con.execute(f"""
      CREATE OR REPLACE TEMP TABLE pm_s AS
      SELECT market_id, nid, ts, long_, qty, usd, price, hours_to_close, is_maker
      FROM {sides} SEMI JOIN pm_mk USING (market_id) WHERE nid < {N}""")
    con.execute("""
      CREATE OR REPLACE TEMP TABLE pm_e AS
      SELECT market_id, nid, ets, row_number() OVER (PARTITION BY market_id ORDER BY ets, nid) r
      FROM (SELECT market_id, nid, min(ts) ets FROM pm_s GROUP BY 1, 2)""")
    con.execute(f"""
      CREATE OR REPLACE TEMP TABLE pm_cand AS
      WITH p AS (SELECT i.iid, i.nid qn, i.market_id, i.t_k, e.r rmax
                 FROM pm_inst i ASOF JOIN pm_e e ON e.market_id = i.market_id AND i.t_k > e.ets)
      SELECT p.iid, p.market_id, p.t_k, e.nid FROM p
      JOIN pm_e e ON e.market_id = p.market_id AND e.r BETWEEN p.rmax - {max_traders} AND p.rmax
      WHERE e.nid <> p.qn
      QUALIFY row_number() OVER (PARTITION BY p.iid ORDER BY e.r DESC) <= {max_traders}""")
    con.execute("""
      CREATE OR REPLACE TEMP TABLE pm_c AS
      SELECT market_id, nid, ts,
             sum(CASE WHEN long_ THEN qty ELSE -qty END) OVER w csq, sum(qty) OVER w cq, sum(usd) OVER w cu,
             sum(price * qty) OVER w cpq, sum(ln(1 + greatest(hours_to_close, 0))) OVER w clh,
             count(*) OVER w cn, sum(is_maker) OVER w cm
      FROM pm_s SEMI JOIN (SELECT DISTINCT market_id, nid FROM pm_cand) USING (market_id, nid)
      WINDOW w AS (PARTITION BY market_id, nid ORDER BY ts)""")
    E = con.execute("""
      SELECT c.iid, c.nid src, x.csq / x.cq net_dir, ln(1 + x.cu) log_usd, x.cpq / x.cq vwap_price,
             x.clh / x.cn log_h_to_close, x.cm / x.cn maker_share,
             (epoch(x.ts) < epoch(c.t_k)) before_tk
      FROM pm_cand c ASOF JOIN pm_c x ON x.market_id = c.market_id AND x.nid = c.nid AND c.t_k > x.ts""").fetchdf()
    assert E["before_tk"].all(), "as-of market edge uses a fill at/after t_k"
    return F.astype(np.float32), E.drop(columns="before_tk")
