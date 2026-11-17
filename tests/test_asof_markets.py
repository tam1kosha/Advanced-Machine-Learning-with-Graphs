"""Gate C as-of market nodes: only fills before the query wallet's k-th fill are used."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pmgraph.asof_markets import build  # noqa: E402
from pmgraph.prep import connect  # noqa: E402

T0 = pd.Timestamp("2025-02-01", tz="UTC")
H = pd.Timedelta(hours=1)


def _sides():
    # market 7: frozen wallets 1, 2, 3 (N = 10), query wallet 50; t_k(50) = T0 + 5h
    r = []
    def fill(nid, h, long_, qty=10.0, price=0.4, maker=0):
        r.append(dict(nid=nid, cp=99, is_maker=maker, long_=long_, bp=price, usd=qty * price, ts=T0 + h * H,
                      market_id=7, event_id="e", category="Sports", hours_to_close=24.0, price=price, qty=qty,
                      w=1, close_time=T0 + 48 * H))
    fill(1, 1, True)               # trader 1: long before t_k ...
    fill(1, 6, False, qty=100)     # ... and a large short after t_k (must not count)
    fill(2, 2, False)              # trader 2: short before t_k
    fill(3, 7, True)               # trader 3 enters after t_k (must not appear)
    fill(50, 3, True)              # the query wallet itself (not its own neighbour)
    fill(11, 4, True)              # wallet without a frozen node (nid >= N): counts in activity, not as edge
    fill(2, 5, True, maker=1)      # exactly at t_k: excluded (strictly before)
    return pd.DataFrame(r)


def test_asof_market_uses_only_fills_before_tk(tmp_path):
    con = connect()
    con.register("s_df", _sides())
    con.execute("CREATE TEMP TABLE s AS SELECT * FROM s_df")
    mk = tmp_path / "markets.parquet"
    pd.DataFrame({"market_id": [7], "expiration_time": [T0 + 48 * H]}).to_parquet(mk)
    inst = pd.DataFrame({"iid": [0], "nid": [50], "market_id": [7], "t_k": [T0 + 5 * H]})
    F, E = build(con, inst, "s", N=10, max_traders=30, markets_parquet=mk)
    import numpy as np
    assert np.isclose(F.loc[0, "log_fills"], np.log1p(4))           # taker fills at 1, 2, 3, 4 h
    assert F.loc[0, "cat_sports"] == 1 and F.loc[0, "closed_before_cutoff"] == 0
    assert np.isclose(F.loc[0, "log_age_h"], np.log1p(4))           # first fill 1 h, t_k 5 h
    assert np.isclose(F.loc[0, "log_h_to_expiry"], np.log1p(43))
    e = E.set_index("src")
    assert sorted(e.index) == [1, 2]                                # not 3 (later), 50 (self), 11 (no node)
    assert e.loc[1, "net_dir"] == 1.0 and e.loc[2, "net_dir"] == -1.0   # trader 1's later short ignored
    assert e.loc[2, "maker_share"] == 0.0                           # trader 2's fill at t_k ignored


def test_asof_market_keeps_latest_entrants():
    con = connect()
    r = [dict(nid=n, cp=99, is_maker=0, long_=True, bp=.5, usd=5., ts=T0 + n * H, market_id=7, event_id="e",
              category="Crypto", hours_to_close=1., price=.5, qty=10., w=0, close_time=T0 + 2 * H) for n in range(1, 9)]
    con.register("s_df", pd.DataFrame(r))
    con.execute("CREATE TEMP TABLE s AS SELECT * FROM s_df")
    inst = pd.DataFrame({"iid": [0], "nid": [99], "market_id": [7], "t_k": [T0 + 6.5 * H]})
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        mk = Path(d) / "m.parquet"
        pd.DataFrame({"market_id": [7], "expiration_time": pd.Series([pd.NaT], dtype="datetime64[ns, UTC]")}).to_parquet(mk)
        F, E = build(con, inst, "s", N=100, max_traders=3, markets_parquet=mk)
    assert sorted(E.src) == [4, 5, 6]                               # 3 latest entrants before 6.5 h
    assert F.loc[0, "closed_before_cutoff"] == 1 and F.loc[0, "outcome_if_closed"] == -0.5
