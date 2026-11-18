"""Query-node sampler: correct neighbours, and query wallets never send messages."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pmgraph.hetero import CSR, Graph, sample_batch, train_hetero  # noqa: E402


def _toy(seed=0):
    rng = np.random.default_rng(seed)
    nw, nm, nq = 50, 8, 6
    cp = rng.integers(0, nw, (2, 300))
    held = np.stack([rng.integers(0, nw, 120), rng.integers(0, nm, 120)])
    frozen = {"cp": CSR.build(cp[1], cp[0], np.ones(300), nw),
              "held": CSR.build(held[1], held[0], np.ones((120, 5)), nm),
              "rev_held": CSR.build(held[0], held[1], np.ones((120, 5)), nw)}
    qcp = np.stack([rng.integers(0, nw, 20), rng.integers(0, nq, 20)])
    qh = np.stack([rng.integers(0, nm, 10), rng.integers(0, nq, 10)])
    query = {"cp": CSR.build(qcp[1], qcp[0], np.ones(20), nq),
             "rev_held": CSR.build(qh[1], qh[0], np.ones((10, 5)), nq)}
    g = Graph(x={"wallet": rng.random((nw, 3), dtype=np.float32), "market": rng.random((nm, 2), dtype=np.float32)},
              frozen=frozen, xq=rng.random((nq, 3), dtype=np.float32), query=query)
    return g, cp, nq


def test_csr_rows_match_edges():
    _, cp, _ = _toy()
    c = CSR.build(cp[1], cp[0], np.arange(300), 50)
    v = np.array([3, 17, 42])
    pos, vi = c.rows(v)
    for i, node in enumerate(v):
        assert sorted(c.src[pos[vi == i]]) == sorted(cp[0][cp[1] == node])


def test_query_nodes_only_receive():
    g, _, nq = _toy()
    b = sample_batch(g, np.arange(nq), ["cp", "rev_held", "held"], (4, 3), np.random.default_rng(1))
    n_s = b["n_seeds"]
    for (s_t, _, d_t), ei in b["ei"].items():
        if s_t == "wallet":
            assert (ei[0] >= n_s).all(), "a query seed is used as a message source"
        if d_t == "wallet" and s_t == "wallet":
            assert ei.shape[1] > 0
    assert b["x"]["wallet"].shape[0] > n_s and "market" in b["x"]


def _reg_graph(seed, signal):
    """Query wallets whose target is the mean first feature of their counterparties (+ noise), or pure noise."""
    rng = np.random.default_rng(seed)
    nw, nq, deg = 300, 600, 4
    xw = rng.normal(size=(nw, 3)).astype(np.float32)
    src = rng.integers(0, nw, nq * deg)
    dst = np.repeat(np.arange(nq), deg)
    f = rng.integers(0, nw, (2, 1500))
    g = Graph(x={"wallet": xw}, frozen={"cp": CSR.build(f[1], f[0], np.ones(1500), nw)},
              xq=rng.normal(size=(nq, 3)).astype(np.float32), query={"cp": CSR.build(dst, src, np.ones(nq * deg), nq)})
    y = signal * xw[src, 0].reshape(nq, deg).mean(1) + 0.3 * rng.normal(size=nq)
    y[rng.random(nq) < 0.1] = np.nan                   # wallets without a target are skipped, not fatal
    return g, y[:, None]


def test_regression_head_learns_neighbour_signal_and_nothing_under_null():
    hp = dict(hidden=16, fanout=[4, 2], batch=128, epochs=25, patience=25, lr=0.01, eval_reps=3, threads=1)
    idx = {"train": np.arange(0, 360), "val": np.arange(360, 480)}
    te = np.arange(480, 600)
    r = {}
    for name, signal in [("signal", 1.0), ("null", 0.0)]:
        g, y = _reg_graph(5, signal)
        s = train_hetero(g, ["cp"], y, idx, 0, hp, log=lambda *a: None, objectives=["reg"])
        ok = np.isfinite(y[te, 0])
        r[name] = np.corrcoef(s[te, 0][ok], y[te, 0][ok])[0, 1]
    assert r["signal"] > 0.5 and abs(r["null"]) < 0.3, r


def test_held_edges_pd_matches_sql_with_missing_close_time():
    """Query (pandas) and frozen (DuckDB) position attrs agree, incl. NULL / negative hours_to_close."""
    import duckdb
    import pandas as pd

    from pmgraph.hetero import HELD_ATTRS, held_edges, held_edges_pd

    o = pd.DataFrame({"nid": [0, 0, 0, 1, 1, 2], "market_id": [5, 5, 6, 5, 6, 6],
                      "long_": [True, False, True, True, False, True],
                      "qty": [1.0, 0.5, 2.0, 3.0, 1.0, 4.0], "price": [0.4, 0.6, 0.2, 0.5, 0.7, 0.3],
                      "usd": [0.4, 0.3, 0.4, 1.5, 0.7, 1.2], "is_maker": [0, 1, 0, 0, 1, 1],
                      "hours_to_close": [10.0, np.nan, np.nan, -3.0, 2.0, np.nan]})
    con = duckdb.connect()
    con.register("o_view", o)
    sql = held_edges(con, "o_view").sort_values(["nid", "market_id"]).reset_index(drop=True)
    pdf = held_edges_pd(o).sort_values(["nid", "market_id"]).reset_index(drop=True)
    assert np.isfinite(pdf[HELD_ATTRS].to_numpy()).all()
    np.testing.assert_allclose(pdf[HELD_ATTRS].to_numpy(float), sql[HELD_ATTRS].to_numpy(float), rtol=1e-9)
