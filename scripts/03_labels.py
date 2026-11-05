"""Step 3: calibrate prices, build label-window bets, run the skill test, assign four types.

usage: python scripts/03_labels.py [split] [unit ...]
  unit: market | event | all   (default: configs labels.bet_unit). Outputs per unit:
  data/processed/{split}_labels[_event].parquet, outputs/tables/labels_{split}[_event].json,
  outputs/figures/labels_{split}[_event].png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pmgraph import ROOT, load_config  # noqa: E402
from pmgraph.labels import UNITS, assign_types, bets, fit_calibration, labels_file, skill_test  # noqa: E402
from pmgraph.prep import connect  # noqa: E402

cfg = load_config()
split = sys.argv[1] if len(sys.argv) > 1 else "train"
units = sys.argv[2:] or [cfg["labels"].get("bet_unit", "market")]
units = list(UNITS) if units == ["all"] else units
sp, lc = cfg["splits"][split], cfg["labels"]
I, P = ROOT / cfg["data"]["interim_dir"], ROOT / cfg["data"]["processed_dir"]
P.mkdir(parents=True, exist_ok=True)
con = connect()

calib = fit_calibration(con, I / f"{split}_graph_fills.parquet", sp["cutoff"], lc["calib_bins"])
calib.to_parquet(P / f"{split}_calibration.parquet")

b = bets(con, I / f"{split}_label_fills.parquet", calib)
print("bets", len(b), "wallets", b["wallet"].nunique(), "events", b["event_id"].nunique(), flush=True)
BUCKETS = [30, 50, 100, 250, np.inf]

for unit in units:
    sfx = "" if unit == "market" else f"_{unit}"
    w = skill_test(b, unit=unit)
    lab = assign_types(w, lc["min_resolved_bets"], lc["fdr_alpha"])

    # Monte Carlo check of the analytic p-values on a random subsample
    sub = lab.sample(min(400, len(lab)), random_state=0).index
    mc = skill_test(b[b.wallet.isin(sub)], n_sim=lc["n_sim"], unit=unit)
    agree = float(np.corrcoef(mc["p"], mc["p_mc"])[0, 1])

    lab.to_parquet(labels_file(P, split, unit))
    bk = pd.cut(lab["n_bets"], BUCKETS, right=False,
                labels=[f"{a}-{c - 1}" if np.isfinite(c) else f">={a}" for a, c in zip(BUCKETS, BUCKETS[1:])])
    by_b = lab.groupby(bk, observed=True).agg(n=("p", "size"), p_lt_05=("p", lambda p: (p < 0.05).mean()),
                                              z_sd=("z", "std"), skill_z2=("z", lambda z: (z > 2).mean()))
    summary = {
        "bet_unit": unit,
        "labelled_wallets": len(lab),
        "type_counts": lab["type"].value_counts().to_dict(),
        "type_share": lab["type"].value_counts(normalize=True).round(4).to_dict(),
        "share_profitable": float(lab.profitable.mean()),
        "share_skilled": float(lab.skilled.mean()),
        "share_z_gt_2": float((lab.z > 2).mean()),
        "median_bets": float(lab.n_bets.median()),
        "median_markets_per_bet": float((lab.n_markets / lab.n_bets).median()),
        "z_mean": float(lab.z.mean()), "z_sd": float(lab.z.std()),
        "share_p_lt_05": float((lab.p < 0.05).mean()),
        "by_n_bets": by_b.round(4).reset_index().astype({"n_bets": str}).to_dict("records"),
        "p_analytic_vs_mc_corr": agree,
    }
    print(json.dumps(summary, indent=2), flush=True)
    with open(ROOT / f"outputs/tables/labels_{split}{sfx}.json", "w") as f:
        json.dump(summary, f, indent=2)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(calib["mid"], calib["rate"], ".", color="#bbb", label="binned (market-weighted)")
    ax[0].plot(calib["mid"], calib["pi"], color="#2a6fdb", label="isotonic fit")
    ax[0].plot([0, 1], [0, 1], ls="--", lw=1, color="#999")
    ax[0].set(xlabel="price", ylabel="P(win)", title="Favourite–longshot calibration")
    ax[0].legend(frameon=False)
    colors = dict(zip(["skilled_winner", "lucky_winner", "unlucky_skilled", "unskilled_loser"],
                      ["#2a6fdb", "#f0ad4e", "#5cb85c", "#bbbbbb"]))
    for t, g in lab.groupby("type"):
        ax[1].scatter(g["z"].clip(-10, 15), np.sign(g.profit) * np.log10(1 + g.profit.abs()),
                      s=3, alpha=.4, color=colors[t], label=f"{t} ({len(g)})")
    ax[1].set(xlabel=f"skill z-score ({unit}-level bets)", ylabel="sign·log10(1+|profit $|)",
              title="Four trader types")
    ax[1].legend(frameon=False, fontsize=7, markerscale=3)
    fig.tight_layout()
    fig.savefig(ROOT / f"outputs/figures/labels_{split}{sfx}.png", dpi=150)
    plt.close(fig)
