"""Download the vgregoire/polymarket-users tables needed for the project.

Examples
--------
# metadata only (markets, events, predictions; ~150 MB)
python scripts/download.py --meta

# day-partitioned trades for a date range (inclusive months)
python scripts/download.py --trades 2024-07 2025-06

# lifetime user tables (87 features, terminal PnL; ~1.2 GB) — for cross-checks only,
# they are full-sample aggregates and must NOT be used as model features (leakage).
python scripts/download.py --user-tables
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = "vgregoire/polymarket-users"
META = ["markets.parquet", "events.parquet", "predictions.parquet", "README.md"]
USER = ["user_features.parquet", "user_pnl_summary.parquet"]


def months(start: str, end: str) -> list[str]:
    y, m = map(int, start.split("-"))
    ye, me = map(int, end.split("-"))
    out = []
    while (y, m) <= (ye, me):
        out.append(f"trades/year={y}/month={m:02d}/*")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--meta", action="store_true")
    ap.add_argument("--user-tables", action="store_true")
    ap.add_argument("--trades", nargs=2, metavar=("YYYY-MM", "YYYY-MM"))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    patterns: list[str] = []
    if a.meta:
        patterns += META
    if a.user_tables:
        patterns += USER
    if a.trades:
        patterns += months(*a.trades)
    if not patterns:
        ap.error("nothing selected")

    Path(a.out).mkdir(parents=True, exist_ok=True)
    snapshot_download(
        REPO,
        repo_type="dataset",
        local_dir=a.out,
        allow_patterns=patterns,
        max_workers=a.workers,
    )
    print("done:", patterns[:3], "..." if len(patterns) > 3 else "")


if __name__ == "__main__":
    main()
