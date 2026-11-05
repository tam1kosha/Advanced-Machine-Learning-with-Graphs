"""Step 1: clean raw trades into canonical fills for each split's graph and label windows."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pmgraph import ROOT, load_config  # noqa: E402
from pmgraph.prep import build_fills, connect  # noqa: E402

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "mvp")
splits = sys.argv[2:] or ["train"]
con = connect()
stats = {}
for s in splits:
    sp = cfg["splits"][s]
    for part, (a, b) in {"graph": (sp["graph_start"], sp["cutoff"]),
                         "label": (sp["cutoff"], sp["label_end"])}.items():
        out = ROOT / cfg["data"]["interim_dir"] / f"{s}_{part}_fills.parquet"
        stats[f"{s}/{part}"] = st = build_fills(cfg, a, b, out, con)
        print(s, part, st, flush=True)
(ROOT / "outputs/tables").mkdir(parents=True, exist_ok=True)
with open(ROOT / "outputs/tables/prepare_stats.json", "w") as f:
    json.dump(stats, f, indent=2, default=str)
