"""Wallet-market heterogeneous graph (frozen at the cutoff), query-node insertion and hetero GraphSAGE.

Frozen graph (only fills with ts < cutoff), node types wallet / market:
  (wallet, held, market)  one edge per (wallet, market) position, attrs HELD_ATTRS
  (wallet, cp, wallet)    counterparty pairs (iteration-1 adjacency, symmetric), attr log weight
  (wallet, mf, wallet)    moved-first u -> v: v's first taker fill on a (market, side) comes within
                          `moved_first_window_min` after u's, among the `moved_first_max_leaders`
                          entries just before it; kept if the pair co-moves >= `moved_first_min_count`
                          times; attr log1p(count)
Message passing also uses the reverses (market, rev_held, wallet) and (wallet, rev_mf, wallet).

Query nodes. A wallet being scored is a separate node that only *receives* messages from the
frozen graph (its own pre-cutoff edges if it existed, plus the edges of its first k post-cutoff
fills). Nothing from one query wallet's post-cutoff activity reaches another query wallet.

Training samples a fixed fan-out of in-neighbours per relation and hop (with replacement, like
GraphSAGE) in numpy; no pyg-lib / torch-sparse needed. Memory: frozen CSR arrays (~1-2 GB).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv
from torch_geometric.nn.conv import MessagePassing

from .features import CATS

HELD_ATTRS = ["net_dir", "log_usd", "vwap_price", "log_h_to_close", "maker_share"]
HELD_SQL = """
      sum(CASE WHEN long_ THEN qty ELSE -qty END) / sum(qty)       AS net_dir,
      ln(1 + sum(usd))                                              AS log_usd,
      sum(price * qty) / sum(qty)                                   AS vwap_price,
      avg(ln(1 + greatest(hours_to_close, 0)))                      AS log_h_to_close,
      avg(is_maker)                                                 AS maker_share"""
REL_DIMS = {"held": len(HELD_ATTRS), "rev_held": len(HELD_ATTRS), "cp": 1, "mf": 1, "rev_mf": 1}
RELS = {  # relation -> (src type, dst type)
    "held": ("wallet", "market"), "rev_held": ("market", "wallet"),
    "cp": ("wallet", "wallet"), "mf": ("wallet", "wallet"), "rev_mf": ("wallet", "wallet")}


# ------------------------------------------------------------------ frozen graph construction (DuckDB)
def market_features(con, fills, markets_parquet, cutoff: str) -> pd.DataFrame:
    """Per market traded before the cutoff: category, activity, time features, resolved outcome if
    the market closed before the cutoff (known at the cutoff), last pre-cutoff price."""
    cats = ", ".join(f"(any_value(category) = '{c}')::INT AS cat_{c.lower()}" for c in CATS)
    m = con.execute(f"""
      WITH f AS (SELECT * FROM '{fills}'),
      a AS (SELECT market_id, {cats},
                   ln(1 + sum(usd)) log_volume, ln(1 + count(*)) log_fills,
                   ln(1 + count(DISTINCT taker)) log_takers,
                   ln(1 + date_diff('hour', min(ts), TIMESTAMPTZ '{cutoff}')) log_age_h,
                   arg_max(price, ts) last_price,
                   any_value(close_time) close_time, any_value(w) w
            FROM f GROUP BY 1)
      SELECT a.*, mk.expiration_time FROM a LEFT JOIN read_parquet('{markets_parquet}') mk USING (market_id)
      ORDER BY market_id""").fetchdf().set_index("market_id")
    cut = pd.Timestamp(cutoff, tz="UTC")
    closed = (m.pop("close_time") < cut).fillna(False).to_numpy()
    w = m.pop("w").to_numpy(dtype=float)
    m["closed_before_cutoff"] = closed.astype(float)
    m["outcome_if_closed"] = np.where(closed, np.nan_to_num(w, nan=0.5) - 0.5, 0.0)
    exp = pd.to_datetime(m.pop("expiration_time"), utc=True)
    h = ((exp - cut).dt.total_seconds() / 3600).fillna(0).clip(lower=0)
    m["log_h_to_expiry"] = np.log1p(h)
    return m.astype(np.float32)


def held_edges(con, sides_table: str, where: str = "") -> pd.DataFrame:
    """Aggregate a sides table (int ids, columns as firstk.int_sides) into (nid, market_id) positions."""
    return con.execute(f"""SELECT nid, market_id, {HELD_SQL} FROM {sides_table} {where}
                           GROUP BY 1, 2""").fetchdf()


def held_edges_pd(o: pd.DataFrame) -> pd.DataFrame:
    """Same aggregation as HELD_SQL, in pandas, for the (small) query-fill table."""
    o = o.assign(sq=np.where(o.long_, o.qty, -o.qty), pq=o.price * o.qty,
                 lh=np.log1p(o.hours_to_close.clip(lower=0)))
    g = o.groupby(["nid", "market_id"])
    e = pd.DataFrame({"net_dir": g.sq.sum() / g.qty.sum(), "log_usd": np.log1p(g.usd.sum()),
                      "vwap_price": g.pq.sum() / g.qty.sum(), "log_h_to_close": g.lh.mean(),
                      "maker_share": g.is_maker.mean()})
    return e.reset_index()


def entries_sql(sides_table: str, where: str = "") -> str:
    """First taker fill per (wallet, market, side): the moment a wallet *makes* a bet."""
    return f"""SELECT nid, market_id, long_, min(ts) ts FROM {sides_table}
               WHERE is_maker = 0 {('AND ' + where) if where else ''} GROUP BY 1, 2, 3"""


def moved_first_frozen(con, entries: str, window_min: int, max_leaders: int, min_count: int) -> pd.DataFrame:
    """Leader -> follower pairs among entries (a TEMP TABLE name), aggregated over markets."""
    return con.execute(f"""
      WITH e AS (SELECT *, row_number() OVER (PARTITION BY market_id, long_ ORDER BY ts, nid) r FROM {entries})
      SELECT a.nid src, b.nid dst, count(*) n
      FROM e b JOIN e a ON a.market_id = b.market_id AND a.long_ = b.long_
                        AND a.r BETWEEN b.r - {max_leaders} AND b.r - 1
      WHERE a.nid <> b.nid AND b.ts - a.ts <= INTERVAL {window_min} MINUTE AND b.ts > a.ts
      GROUP BY 1, 2 HAVING count(*) >= {min_count}""").fetchdf()


def moved_first_query(con, q_entries: str, others: str, window_min: int, max_leaders: int) -> pd.DataFrame:
    """Moved-first edges between query entries (nid, market_id, long_, ts, rn) and frozen-wallet
    entries `others` (nid, market_id, long_, ts) at any time. Returns per (query nid, query rn):
      dir='lead'   other entered <= window before the query wallet (other -> query)
      dir='follow' other entered <= window after it; usable at k only if its ts <= t_k (other -> query, rev)
    Up to `max_leaders` nearest entries per direction."""
    return con.execute(f"""
      WITH j AS (
        SELECT q.nid, q.rn, o.nid other, o.ts t_other, (o.ts < q.ts) is_lead,
               abs(epoch(o.ts) - epoch(q.ts)) dt
        FROM {q_entries} q JOIN {others} o
          ON o.market_id = q.market_id AND o.long_ = q.long_
         AND o.ts BETWEEN q.ts - INTERVAL {window_min} MINUTE AND q.ts + INTERVAL {window_min} MINUTE
        WHERE o.nid <> q.nid AND o.ts <> q.ts)
      SELECT nid, rn, other, t_other, CASE WHEN is_lead THEN 'lead' ELSE 'follow' END dir FROM j
      QUALIFY row_number() OVER (PARTITION BY nid, rn, is_lead ORDER BY dt) <= {max_leaders}""").fetchdf()


# ------------------------------------------------------------------ CSR structures
@dataclass
class CSR:
    """In-edges grouped by destination: src[indptr[v]:indptr[v+1]] are v's in-neighbours."""
    indptr: np.ndarray
    src: np.ndarray
    attr: np.ndarray

    @staticmethod
    def build(dst: np.ndarray, src: np.ndarray, attr: np.ndarray, n_dst: int) -> "CSR":
        o = np.argsort(dst, kind="stable")
        counts = np.bincount(dst, minlength=n_dst)
        indptr = np.zeros(n_dst + 1, np.int64)
        np.cumsum(counts, out=indptr[1:])
        a = np.asarray(attr, np.float32)
        # reshape(0, -1) cannot infer the attribute dimension for an empty relation.
        a = a[:, None] if a.ndim == 1 else a
        return CSR(indptr, src[o].astype(np.int64), a[o])

    def rows(self, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Edge positions of all in-edges of nodes v, and the index into v of each edge."""
        st, en = self.indptr[v], self.indptr[v + 1]
        cnt = en - st
        pos = np.repeat(st - np.r_[0, np.cumsum(cnt)[:-1]], cnt) + np.arange(cnt.sum())
        return pos, np.repeat(np.arange(len(v)), cnt)


def frozen_csrs(data, scalers: dict | None = None) -> tuple[dict[str, CSR], dict]:
    """In-edge CSR per relation (incl. reverses) from a frozen HeteroData; edge attrs standardised
    with per-relation (mean, sd) fitted here unless `scalers` is given."""
    n = {"wallet": data["wallet"].num_nodes, "market": data["market"].num_nodes}
    raw = {"held": data["wallet", "held", "market"], "cp": data["wallet", "cp", "wallet"],
           "mf": data["wallet", "mf", "wallet"]}
    scalers = dict(scalers or {})
    out = {}
    for r, st in raw.items():
        ei, ea = st.edge_index.numpy(), st.edge_attr.numpy()
        if r not in scalers:
            scalers[r] = ((ea.mean(0), ea.std(0) + 1e-6) if len(ea) else
                          (np.zeros(REL_DIMS[r]), np.ones(REL_DIMS[r])))
        ea = (ea - scalers[r][0]) / scalers[r][1]
        s_t, d_t = RELS[r]
        out[r] = CSR.build(ei[1], ei[0], ea, n[d_t])
        rev = "rev_" + r if r != "cp" else None      # cp is stored symmetric already
        if rev:
            out[rev] = CSR.build(ei[0], ei[1], ea, n[s_t])
    scalers["rev_held"], scalers["rev_mf"] = scalers["held"], scalers["mf"]
    return out, scalers


@dataclass
class Graph:
    """Frozen graph (x per node type, CSR per relation) + query nodes (features, CSR per relation)."""
    x: dict[str, np.ndarray]
    frozen: dict[str, CSR]
    xq: np.ndarray = None
    query: dict[str, CSR] = field(default_factory=dict)


def sample_in(csr: CSR, v: np.ndarray, fanout: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """Fixed fan-out in-neighbour sample with replacement. Returns (edge positions, index into v)."""
    st = csr.indptr[v]
    deg = csr.indptr[v + 1] - st
    has = np.flatnonzero(deg > 0)
    r = (rng.random((len(has), fanout)) * deg[has, None]).astype(np.int64)
    return (st[has, None] + r).ravel(), np.repeat(has, fanout)


def sample_batch(g: Graph, seeds: np.ndarray, rels: list[str], fanout: tuple[int, int], rng) -> dict:
    """Two-hop computation graph for query seeds, as local-index tensors for HeteroConv.

    Local wallet ids: [0, n_seeds) query seeds, then frozen wallets. Local market ids: frozen markets.
    Hop 1 = query CSR (in-edges of the seeds), hop 2 = frozen CSR (in-edges of hop-1 nodes).
    """
    f1, f2 = fanout
    edges = {r: [] for r in rels}       # r -> list of (src frozen id, dst local id, attr)
    hop1 = {"wallet": [], "market": []}
    for r in rels:
        s_t, d_t = RELS[r]
        if d_t != "wallet":            # seeds are wallets: hop 1 uses relations into wallets only
            continue
        pos, di = sample_in(g.query[r], seeds, f1, rng)
        src = g.query[r].src[pos]
        edges[r].append(("q", src, di, g.query[r].attr[pos]))
        hop1[s_t].append(src)
    h1 = {t: np.unique(np.concatenate(v)) if v else np.zeros(0, np.int64) for t, v in hop1.items()}
    hop2 = {"wallet": [], "market": []}
    for r in rels:
        s_t, d_t = RELS[r]
        if len(h1[d_t]) == 0:
            continue
        pos, di = sample_in(g.frozen[r], h1[d_t], f2, rng)
        src = g.frozen[r].src[pos]
        edges[r].append(("f", src, h1[d_t][di], g.frozen[r].attr[pos]))
        hop2[s_t].append(src)
    nodes = {t: np.unique(np.concatenate([h1[t]] + hop2[t])) for t in h1}
    n_s = len(seeds)
    x = {"wallet": torch.from_numpy(np.concatenate([g.xq[seeds], g.x["wallet"][nodes["wallet"]]]))}
    # HeteroConv still receives empty relations in isolated batches; keep an empty market tensor
    # so its source projection/message function has a valid input rather than None.
    if "market" in g.x:
        x["market"] = torch.from_numpy(g.x["market"][nodes["market"]])
    loc = {t: (lambda ids, t=t, off=(n_s if t == "wallet" else 0): off + np.searchsorted(nodes[t], ids))
           for t in nodes}
    ei, ea = {}, {}
    for r in rels:
        s_t, d_t = RELS[r]
        parts = edges[r]
        if not parts:
            continue
        srcs, dsts, attrs = [], [], []
        for kind, src, dst, a in parts:
            srcs.append(loc[s_t](src))
            dsts.append(dst if kind == "q" else loc[d_t](dst))
            attrs.append(a)
        key = (s_t, r, d_t)
        ei[key] = torch.from_numpy(np.stack([np.concatenate(srcs), np.concatenate(dsts)]).astype(np.int64))
        ea[key] = torch.from_numpy(np.concatenate(attrs))
    return {"x": x, "ei": ei, "ea": ea, "n_seeds": n_s}


# ------------------------------------------------------------------ model
class EdgeSAGEConv(MessagePassing):
    """h_v' = W_root h_v + mean_{(u, e) -> v} relu(W_msg [h_u, e])  (GraphSAGE-mean with edge features)."""

    def __init__(self, d_in: int, d_out: int, d_edge: int):
        super().__init__(aggr="mean")
        self.msg = nn.Linear(d_in + d_edge, d_out)
        self.root = nn.Linear(d_in, d_out, bias=False)

    def forward(self, x, edge_index, edge_attr):
        xs, xd = x if isinstance(x, tuple) else (x, x)
        out = self.propagate(edge_index, x=(xs, xd), edge_attr=edge_attr, size=(xs.size(0), xd.size(0)))
        return out + self.root(xd)

    def message(self, x_j, edge_attr):
        return F.relu(self.msg(torch.cat([x_j, edge_attr], -1)))


class HeteroSAGE(nn.Module):
    """Type-specific input projections, L x HeteroConv(EdgeSAGEConv per relation, sum), multitask head.
    With rels=["cp"] this is a homogeneous GraphSAGE on the counterparty graph."""

    def __init__(self, in_dims: dict[str, int], rels: list[str], d_h: int = 64, n_layers: int = 2,
                 n_out: int = 2, dropout: float = 0.2):
        super().__init__()
        self.types = sorted({t for r in rels for t in RELS[r]})
        self.inp = nn.ModuleDict({t: nn.Linear(in_dims[t], d_h) for t in self.types})
        self.convs = nn.ModuleList(
            HeteroConv({(RELS[r][0], r, RELS[r][1]): EdgeSAGEConv(d_h, d_h, REL_DIMS[r]) for r in rels}, aggr="sum")
            for _ in range(n_layers))
        self.norms = nn.ModuleList(nn.ModuleDict({t: nn.LayerNorm(d_h) for t in self.types})
                                   for _ in range(n_layers))
        self.head = nn.Linear(d_h, n_out)
        self.dropout = dropout

    def forward(self, b: dict) -> torch.Tensor:
        h = {t: F.relu(self.inp[t](x)) for t, x in b["x"].items() if t in self.inp}
        for conv, norms in zip(self.convs, self.norms):
            new = conv(h, b["ei"], edge_attr_dict=b["ea"])
            h = {t: F.dropout(F.relu(norms[t](new[t] if t in new else h[t])), self.dropout, self.training)
                 for t in h}
        return self.head(h["wallet"][: b["n_seeds"]])


def train_hetero(g: Graph, rels: list[str], Y: np.ndarray, idx: dict, seed: int, hp: dict, log=print,
                 objectives: list[str] | None = None, checkpoint_path: str | Path | None = None,
                 metadata: dict | None = None) -> np.ndarray:
    """Multitask BCE on query seeds; early stopping on mean val AUPRC. Returns scores for all query nodes.
    objectives (iteration 4): "binary" (default) or "reg" per column of Y. A "reg" column is standardised
    on the training rows and fit with a Huber loss, rows with a NaN target are skipped, and its
    validation score is the Pearson correlation instead of the AUPRC.
    Device: hp["device"], else CUDA when available, else CPU. Neighbour sampling stays on the CPU
    (numpy), so on a GPU raise hp["batch"] / hp["hidden"] rather than expect a speed-up at the defaults."""
    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(hp.get("threads", 2))
    dims = {t: v.shape[1] for t, v in g.x.items()}
    dev = torch.device(hp.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    if os.environ.get("PMGRAPH_REQUIRE_CUDA") == "1" and dev.type != "cuda":
        raise RuntimeError("CUDA was required but is unavailable (or device was set to CPU). Check the torch build/driver.")
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
        torch.cuda.reset_peak_memory_stats(dev)
    started = time.perf_counter()
    model = HeteroSAGE(dims, rels, hp.get("hidden", 64), 2, Y.shape[1], hp.get("dropout", 0.2)).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"    device={dev} parameters={n_params:,} seed={seed} settings={hp}")
    opt = torch.optim.AdamW(model.parameters(), lr=hp.get("lr", 3e-3), weight_decay=1e-4)
    reg = [o == "reg" for o in (objectives or ["binary"] * Y.shape[1])]
    Y = Y.astype(np.float32)
    target_mean, target_std = np.zeros(Y.shape[1]), np.ones(Y.shape[1])
    for j in np.flatnonzero(reg):
        values = Y[idx["train"], j]
        values = values[np.isfinite(values)]
        if len(values) < 2 or values.std() < 1e-8:
            raise ValueError(f"target {j}: regression needs at least two finite, nonconstant training values")
        target_mean[j], target_std[j] = values.mean(), values.std()
        Y[:, j] = (Y[:, j] - target_mean[j]) / target_std[j]
    Yt = torch.from_numpy(Y)
    pos = Yt[idx["train"]].mean(0).clamp(min=1e-3).to(dev)

    def batch(seeds):
        b = sample_batch(g, seeds, rels, fan, rng)
        return {"x": {t: v.to(dev) for t, v in b["x"].items()}, "ei": {r: v.to(dev) for r, v in b["ei"].items()},
                "ea": {r: v.to(dev) for r, v in b["ea"].items()}, "n_seeds": b["n_seeds"]}

    def loss_fn(out, y):
        if not any(reg):
            return F.binary_cross_entropy_with_logits(out, y, pos_weight=(1 - pos) / pos)
        tot = out.sum() * 0
        for j, r in enumerate(reg):
            ok = ~torch.isnan(y[:, j])
            if ok.any():
                tot = tot + (F.smooth_l1_loss(out[ok, j], y[ok, j]) if r else
                             F.binary_cross_entropy_with_logits(out[ok, j], y[ok, j], pos_weight=(1 - pos[j]) / pos[j]))
        return tot / len(reg)

    def val_score(y, s):
        ok = np.isfinite(y)
        if ok.sum() < 2:
            raise ValueError("regression validation fold needs at least two finite targets")
        return float(np.corrcoef(y[ok], s[ok])[0, 1]) if y[ok].std() > 0 and s[ok].std() > 0 else 0.0
    fan, bs = tuple(hp.get("fanout", [10, 5])), hp.get("batch", 1024)
    eval_bs = hp.get("eval_batch", bs)
    if any(int(v) <= 0 for v in [bs, eval_bs, *fan, hp.get("epochs", 40), hp.get("eval_reps", 3)]):
        raise ValueError("batch sizes, fanout, epochs and eval_reps must be positive")

    def predict(nodes, reps):
        model.eval()
        out = np.zeros((len(nodes), Y.shape[1]), np.float32)
        with torch.no_grad():
            for _ in range(reps):
                for i in range(0, len(nodes), eval_bs):
                    s = nodes[i:i + eval_bs]
                    out[i:i + len(s)] += model(batch(s)).cpu().numpy()
        return out / reps

    best, best_state, bad, best_epoch = -np.inf, None, 0, 0
    for ep in range(hp.get("epochs", 40)):
        model.train()
        perm = rng.permutation(idx["train"])
        for i in range(0, len(perm), bs):
            s = perm[i:i + bs]
            if not np.isfinite(Y[s]).any():
                continue
            opt.zero_grad()
            loss = loss_fn(model(batch(s)), Yt[s].to(dev))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss at epoch {ep + 1}, seed {seed}")
            loss.backward()
            opt.step()
        sv = predict(idx["val"], 1)
        score = np.mean([val_score(Y[idx["val"], j], sv[:, j]) if reg[j] else
                         average_precision_score(Y[idx["val"], j], sv[:, j]) for j in range(Y.shape[1])])
        if score > best:
            best, bad = score, 0
            best_epoch = ep + 1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if ep % 10 == 9:
            log(f"    {'+'.join(rels)} seed{seed} ep{ep+1} loss {loss.item():.4f} val-AP {score:.4f}")
        if bad >= hp.get("patience", 8):
            break
    if best_state is None:
        raise RuntimeError("no finite validation score was produced; inspect targets and graph inputs")
    model.load_state_dict(best_state)
    scores = predict(np.arange(len(g.xq)), hp.get("eval_reps", 3))
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    stats = dict(seed=seed, device=str(dev), parameters=n_params, epochs=ep + 1, best_epoch=best_epoch,
                 best_val_score=float(best), elapsed_seconds=time.perf_counter() - started,
                 process_peak_rss_bytes=int(rss if sys.platform == "darwin" else rss * 1024),
                 peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(dev) if dev.type == "cuda" else 0,
                 peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(dev) if dev.type == "cuda" else 0,
                 device_name=torch.cuda.get_device_name(dev) if dev.type == "cuda" else "CPU",
                 torch_version=str(torch.__version__), settings=hp, metadata=metadata or {})
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        bundle = dict(state_dict={k: v.detach().cpu() for k, v in best_state.items()}, in_dims=dims,
                      rels=rels, hidden=hp.get("hidden", 64), n_layers=2, n_out=Y.shape[1],
                      dropout=hp.get("dropout", 0.2), objectives=objectives or ["binary"] * Y.shape[1],
                      target_mean=target_mean.tolist(), target_std=target_std.tolist(), training=stats)
        tmp = path.with_suffix(".pt.tmp")
        torch.save(bundle, tmp)
        tmp.replace(path)
        path.with_suffix(".json").write_text(json.dumps(stats, indent=2) + "\n")
    log(f"    completed: {json.dumps(stats)}")
    return scores
