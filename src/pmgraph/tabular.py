"""Torch-free XGBoost worker.

On macOS, xgboost (Homebrew libomp) and torch (bundled libomp) segfault when both are used in one
process, in either import order. Scripts that also train GNNs call this module in a subprocess:
    python -m pmgraph.tabular in.npz out.npz
in.npz: X (n, d), Y (n, t), seeds (s,), and fit_<seed> / te_<seed> index arrays; out.npz: S_<seed> (n, t).
Optional (iteration 4): obj (t,) "binary" | "reg" per target column and W (n, t) sample weights;
fit rows whose target is NaN are dropped, so targets may be defined on different wallets.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def xgb_scores(Xtr, ytr, Xte, seed: int, objective: str = "binary", wtr=None) -> np.ndarray:
    if objective == "reg":             # same capacity as the classifier
        from xgboost import XGBRegressor
        m = XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, tree_method="hist", n_jobs=2, random_state=seed, verbosity=0)
        m.fit(Xtr, ytr, sample_weight=wtr)
        return m.predict(Xte)
    from xgboost import XGBClassifier
    pos = max(ytr.mean(), 1e-3)
    m = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
                      colsample_bytree=0.8, scale_pos_weight=(1 - pos) / pos,
                      tree_method="hist", n_jobs=2, random_state=seed, verbosity=0)
    m.fit(Xtr, ytr, sample_weight=wtr)
    return m.predict_proba(Xte)[:, 1]


def run_in_subprocess(X: np.ndarray, Y: np.ndarray, splits: dict[int, tuple[np.ndarray, np.ndarray]],
                      objectives: list[str] | None = None, W: np.ndarray | None = None) -> dict:
    """splits: seed -> (fit idx, test idx). Returns seed -> scores (n, t), filled on the test rows.
    objectives: "binary" (default) or "reg" per column of Y; W: sample weights, same shape as Y."""
    with tempfile.TemporaryDirectory() as d:
        i, o = Path(d) / "in.npz", Path(d) / "out.npz"
        arrs = {"X": X, "Y": Y, "seeds": np.array(list(splits))}
        if objectives is not None:
            arrs["obj"] = np.array(objectives)
        if W is not None:
            arrs["W"] = W
        for s, (fit, te) in splits.items():
            arrs[f"fit_{s}"], arrs[f"te_{s}"] = fit, te
        np.savez(i, **arrs)
        src = str(Path(__file__).resolve().parents[1])
        subprocess.run([sys.executable, "-c", f"import sys; sys.path.insert(0, {src!r}); "
                        f"from pmgraph.tabular import _main; _main({str(i)!r}, {str(o)!r})"], check=True)
        r = np.load(o)
        return {int(s): r[f"S_{s}"] for s in splits}


def _main(inp: str, out: str) -> None:
    a = np.load(inp)
    X, Y = a["X"], a["Y"]
    obj = a["obj"] if "obj" in a else ["binary"] * Y.shape[1]
    W = a["W"] if "W" in a else None
    res = {}
    for s in a["seeds"]:
        fit, te = a[f"fit_{s}"], a[f"te_{s}"]
        S = np.zeros(Y.shape, np.float32)
        for j in range(Y.shape[1]):
            f = fit[np.isfinite(Y[fit, j])]
            S[te, j] = xgb_scores(X[f], Y[f, j], X[te], int(s), str(obj[j]), None if W is None else W[f, j])
        res[f"S_{s}"] = S
    np.savez(out, **res)
