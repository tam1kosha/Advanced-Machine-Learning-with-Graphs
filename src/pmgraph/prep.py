"""Cleaning: raw trades -> canonical fills for one time window.

Canonical form (one row per trade):
  ts, market_id, event_id, category, price   price of the market's reference token (outcome_idx 0)
  taker_long  taker is long the reference token after this fill
  w           1 if the reference token won, 0 if it lost, NULL if unresolved by sample end
  qty, usd    shares and dollar notional of the fill
  maker, taker, hours_to_close
Selling token 0 at p is the same exposure as buying token 1 at 1-p, so every fill
maps to one signed position on the reference token.

Filters: binary markets only, self-trades dropped, degenerate prices dropped.
"""
from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path

import duckdb

from . import ROOT


def trade_files(raw_dir: Path, start: str, end: str) -> list[str]:
    """Day-partition files with start <= day < end."""
    d, e = date.fromisoformat(start), date.fromisoformat(end)
    out, missing = [], []
    while d < e:
        f = raw_dir / f"trades/year={d.year}/month={d.month:02d}/day={d.day:02d}/data.parquet"
        if f.exists():
            out.append(str(f))
        else:
            missing.append(d.isoformat())
        d += timedelta(days=1)
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} daily trade partitions in [{start}, {end}): "
                                f"{', '.join(missing[:5])}; finish scripts/download.py before preparing data")
    return out


def connect(mem: str = "5GB", threads: int | None = None) -> duckdb.DuckDBPyConnection:
    """DuckDB with a memory cap. threads defaults to 2 (env PMGRAPH_THREADS): the per-wallet
    DISTINCT/median aggregates spill per thread, and at 12 threads x 5 GB the wallet-feature
    query filled 70 GB of temp space, while 2 threads finish it in 14 s."""
    threads = threads or int(os.environ.get("PMGRAPH_THREADS", 2))
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{mem}'")
    con.execute(f"SET temp_directory='{ROOT / 'data' / 'tmp'}'")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET TimeZone='UTC'")  # cutoff literals are UTC
    if threads:
        con.execute(f"SET threads={threads}")
    return con


def build_fills(cfg: dict, start: str, end: str, out: Path, con=None) -> dict:
    raw = ROOT / cfg["data"]["raw_dir"]
    files = trade_files(raw, start, end)
    if not files:
        raise FileNotFoundError(f"no trade files in [{start}, {end})")
    c = cfg["cleaning"]
    con = con or connect()
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW preds AS
        SELECT prediction_id, market_id AS mid, outcome_idx, n_outcomes
        FROM read_parquet('{raw / "predictions.parquet"}');
        CREATE OR REPLACE TEMP VIEW mkts AS
        SELECT market_id AS mid, close_time FROM read_parquet('{raw / "markets.parquet"}');
    """)
    flist = "[" + ",".join(f"'{f}'" for f in files) + "]"
    out.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"""
        COPY (
          SELECT
            t.timestamp::TIMESTAMPTZ                      AS ts,
            p.mid                                         AS market_id,
            t.event_id                                    AS event_id,
            t.category                                    AS category,
            CASE WHEN p.outcome_idx = 0 THEN t.price ELSE 1 - t.price END AS price,
            (t.taker_bought = (p.outcome_idx = 0))        AS taker_long,
            CASE WHEN t.winner IS NULL THEN NULL
                 WHEN (t.winner = (p.outcome_idx = 0)) THEN 1 ELSE 0 END::TINYINT AS w,
            t.quantity                                    AS qty,
            t.quantity * t.price                          AS usd,
            t.maker_address                               AS maker,
            t.taker_address                               AS taker,
            m.close_time::TIMESTAMPTZ                     AS close_time,
            date_diff('second', t.timestamp, m.close_time) / 3600.0 AS hours_to_close
          FROM read_parquet({flist}, hive_partitioning=false) t
          JOIN preds p USING (prediction_id)
          LEFT JOIN mkts m ON m.mid = p.mid
          WHERE p.n_outcomes = 2
            AND t.price BETWEEN {c['min_price']} AND {c['max_price']}
            AND t.quantity > 0
            {"AND t.maker_address <> t.taker_address" if c['drop_self_trades'] else ""}
        ) TO '{out}' (FORMAT parquet, COMPRESSION zstd)
    """)
    raw_n = con.execute(f"SELECT count(*) FROM read_parquet({flist}, hive_partitioning=false)").fetchone()[0]
    kept = con.execute(f"SELECT count(*), count(w), sum(usd) FROM read_parquet('{out}')").fetchone()
    return {"window": [start, end], "raw_trades": raw_n, "kept": kept[0],
            "kept_resolved": kept[1], "usd": kept[2]}
