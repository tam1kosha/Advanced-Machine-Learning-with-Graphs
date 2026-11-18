"""Fixed wallet holdout and evaluation for the GNN-only tuning interface (experiment 6)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .label_diag import EDGE_METRICS, boot_edge, edge_metrics, variance_function
from .rank_eval import fold_ranks


def wallet_holdout(nids: np.ndarray, strat: np.ndarray, seed: int) -> tuple:
    """One 60/20/20 split, identical for every initialization/model/setting and row-order invariant."""
    order = np.argsort(nids)
    train, test = train_test_split(order, test_size=.2, random_state=seed, stratify=strat[order])
    train, val = train_test_split(train, test_size=.25, random_state=seed, stratify=strat[train])
    return train, val, test


def centered_edge_target(H: pd.DataFrame, fit_idx=None) -> np.ndarray:
    """Half-A target; estimate its variance function using training wallets only when supplied."""
    ok = (H["n0"] >= 10).to_numpy()
    x, s = H["x0"].to_numpy(), H["s0"].to_numpy()
    if not (np.isfinite(x[ok]).all() and np.isfinite(s[ok]).all() and (s[ok] > 0).all()):
        raise ValueError("half-A labels contain invalid edge/noise values")
    fit = ok.copy()
    if fit_idx is not None:
        fit[:] = False
        fit[fit_idx] = ok[fit_idx]
    if fit.sum() < 2:
        raise ValueError("not enough training-half labels for target normalization")
    v = variance_function(x[fit], s[fit])
    return np.where(ok, (x - v["mu"]) / np.sqrt(v["tau2"] + v["phi"] * s ** 2), np.nan)[:, None]


def validate_hparams(hp: dict) -> None:
    counts = ["hidden", "batch", "epochs", "patience", "eval_reps", "threads"]
    if any(not isinstance(hp[k], int) or isinstance(hp[k], bool) or hp[k] < 1 for k in counts):
        raise ValueError(f"positive integer values required for {counts}")
    fan = hp["fanout"]
    if not isinstance(fan, (list, tuple)) or len(fan) != 2 or any(type(v) is not int or v < 1 for v in fan):
        raise ValueError("fanout needs two positive integers; this sampler supports two layers")
    if type(hp.get("eval_batch", hp["batch"])) is not int or hp.get("eval_batch", hp["batch"]) < 1:
        raise ValueError("eval_batch must be a positive integer")
    for name in ("lr", "weight_decay", "grad_clip", "dropout"):
        if not isinstance(hp[name], (float, int)) or isinstance(hp[name], bool) or not np.isfinite(hp[name]):
            raise ValueError(f"{name} must be finite")
    if hp["lr"] <= 0 or hp["weight_decay"] < 0 or hp["grad_clip"] < 0 or not 0 <= hp["dropout"] < 1:
        raise ValueError("lr must be positive; weight_decay/grad_clip nonnegative; dropout in [0, 1)")


def report_holdout(processed: Path, out: Path, plan: dict, phase: str, B: int = 200,
                   tables: Path | None = None) -> Path:
    """Evaluate only val or test wallets, averaging within-seed ranks on the same fixed holdout.

    Metrics use half-B realised edge, not the half-A target used for fitting/early stopping.
    Bootstrap uncertainty is over wallets for the seed ensemble. sd_seeds measures initialization
    variation separately. This does not choose hyperparameters from test scores.
    """
    if phase not in ("val", "test") or B < 0:
        raise ValueError("phase must be val/test and bootstrap count nonnegative")
    preds = processed / f"exp6_preds_{plan['tag']}"
    rows, paired = [], []
    for split in plan["splits"]:
        pop = pd.read_parquet(processed / f"{split}_fk_pop.parquet")
        halves = pd.read_parquet(processed / f"{split}_fk_halves.parquet").query("scheme == 'week'")
        for k in plan["ks"]:
            pk = pop[pop.k == k].reset_index(drop=True)
            strat = pk.skill.to_numpy() * 4 + pk.profitable.to_numpy() * 2 + pk.new.to_numpy()
            tr, va, te = wallet_holdout(pk.nid.to_numpy(), strat, plan["split_seed"])
            chosen = va if phase == "val" else te
            expected = pd.Index(np.sort(pk.nid.to_numpy()[chosen]), name="nid")
            prediction_sets, per_seed = {}, {}
            for model in plan["models"]:
                sfx = "_val" if phase == "val" else ""
                path = preds / f"{split}_k{k}_{model}{sfx}.parquet"
                d = pd.read_parquet(path)
                if (set(d.seed) != set(plan["seeds"]) or d.duplicated(["nid", "seed"]).any()
                        or not np.isfinite(d.s_edge).all()):
                    raise ValueError(f"invalid/incomplete holdout predictions: {path}")
                for _, part in d.groupby("seed"):
                    if set(part.nid) != set(expected):
                        raise ValueError(f"wrong {phase} wallet set: {path}")
                matrix = d.pivot(index="nid", columns="seed", values="s_edge").reindex(expected)
                ranked = np.stack([fold_ranks(matrix[sd].to_numpy(), np.zeros(len(expected))) for sd in plan["seeds"]])
                per_seed[model] = ranked
                prediction_sets[model] = ranked.mean(0)
            H = halves[halves.k == k].set_index("nid").reindex(expected)
            ok = (H.n1 >= 10).to_numpy()
            x, s = H.x1.to_numpy(), H.s1.to_numpy()
            ok = ok & np.isfinite(x) & np.isfinite(s) & (s > 0)
            if ok.sum() < 20:
                raise ValueError(f"{split} k={k}: fewer than 20 {phase} wallets with usable half-B labels")
            v = variance_function(x[ok], s[ok])
            weights = 1 / (v["tau2"] + v["phi"] * s ** 2)
            new = pk.set_index("nid").new.reindex(expected).to_numpy()
            for population, mask in [("all", ok), ("new", ok & new), ("existing", ok & ~new)]:
                if mask.sum() < 20:
                    continue
                scores = {name: score[mask] for name, score in prediction_sets.items()}
                z = x[mask] / s[mask]
                if B:
                    results = boot_edge(scores, x[mask], weights[mask], z, B=B, seed=k, reps=True)
                else:
                    results = {name: (edge_metrics(score, x[mask], weights[mask], z),
                                      np.full(len(EDGE_METRICS), np.nan), np.full(len(EDGE_METRICS), np.nan))
                               for name, score in scores.items()}
                for model, values in results.items():
                    estimate, lo, hi = values[:3]
                    seed_metrics = np.stack([edge_metrics(score[mask], x[mask], weights[mask], z)
                                             for score in per_seed[model]])
                    sd = seed_metrics.std(axis=0, ddof=1) if len(seed_metrics) > 1 else np.zeros(len(EDGE_METRICS))
                    for j, metric in enumerate(EDGE_METRICS):
                        rows.append(dict(split=split, k=k, phase=phase, pop=population, model=model,
                                         metric=metric, n=int(mask.sum()), seeds=len(plan["seeds"]),
                                         est=estimate[j], lo=lo[j], hi=hi[j], sd_seeds=sd[j],
                                         unit="edge_per_share" if metric == "top_minus_bottom" else
                                              ("average_precision" if metric.startswith("auprc") else "correlation")))
                if {"sage", "hgnn"} <= set(results):
                    delta = results["hgnn"][0] - results["sage"][0]
                    lo, hi = (np.nanquantile(results["hgnn"][3] - results["sage"][3], [.025, .975], axis=0)
                              if B else (np.full(len(EDGE_METRICS), np.nan), np.full(len(EDGE_METRICS), np.nan)))
                    for j, metric in enumerate(EDGE_METRICS):
                        paired.append(dict(split=split, k=k, phase=phase, pop=population, metric=metric,
                                           comparison="hgnn-minus-sage", diff=delta[j], lo=lo[j], hi=hi[j]))
    out.mkdir(parents=True, exist_ok=True)
    tables = tables or out
    tables.mkdir(parents=True, exist_ok=True)
    path = tables / f"exp6_{plan['tag']}_{phase}_metrics.csv"
    result = pd.DataFrame(rows)
    result.to_csv(path, index=False)
    paired_frame = pd.DataFrame(paired, columns=["split", "k", "phase", "pop", "metric", "comparison", "diff", "lo", "hi"])
    paired_frame.to_csv(tables / f"exp6_{plan['tag']}_{phase}_paired.csv", index=False)
    sections = [f"Experiment 6: {plan['tag']}, {phase} wallets, half-B evaluation (fixed within-period holdout)"]
    for metric in ("wcorr_edge", "top_minus_bottom", "auprc_hi"):
        for population in ("new", "existing"):
            table = result[(result.metric == metric) & (result["pop"] == population)].copy()
            if metric == "top_minus_bottom":
                table[["est", "lo", "hi", "sd_seeds"]] *= 100
            label = metric + (" (cents per share)" if metric == "top_minus_bottom" else "")
            sections.append(f"{label}, {population} wallets\n" +
                            table[["split", "k", "model", "n", "est", "lo", "hi", "sd_seeds"]].to_string(index=False))
    sections.append("Paired weighted edge correlation, hgnn minus sage\n" +
                    paired_frame[paired_frame.metric == "wcorr_edge"].to_string(index=False))
    summary = "\n\n".join(sections) + "\n"
    (out / f"{phase}_report.log").write_text(summary)
    print(summary, flush=True)
    print(f"Saved {phase} evaluation: {path}", flush=True)
    return path
