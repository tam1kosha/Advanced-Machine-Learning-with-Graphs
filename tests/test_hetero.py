"""Query-node sampler: correct neighbours, and query wallets never send messages."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pmgraph.hetero import CSR, Graph, sample_batch  # noqa: E402


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
