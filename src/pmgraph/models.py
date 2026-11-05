"""Baselines and graph models for the first (transductive, single-cutoff) experiments."""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

OWN_EXCLUDE = ("nbr_", "deg", "in_label")


def own_cols(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if not c.startswith(OWN_EXCLUDE)]


def matrix(X: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Standardised features; NaN history -> 0 plus a missing indicator."""
    M = X[cols].copy()
    M["hist_missing"] = M["hist_z"].isna().astype(float)
    M = M.fillna(0.0).to_numpy(np.float32)
    return StandardScaler().fit_transform(M).astype(np.float32)


# ---------------------------------------------------------------- tabular
def xgb_scores(Xtr, ytr, Xte, seed: int) -> np.ndarray:
    pos = max(ytr.mean(), 1e-3)
    m = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
                      colsample_bytree=0.8, scale_pos_weight=(1 - pos) / pos,
                      tree_method="hist", n_jobs=2, random_state=seed, verbosity=0)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def lr_scores(Xtr, ytr, Xte) -> np.ndarray:
    m = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


# ---------------------------------------------------------------- graph, training-free
def spectral_embedding(S: sp.csr_matrix, dim: int = 32, seed: int = 0) -> np.ndarray:
    """Top eigenvectors of the normalised adjacency (cheap stand-in for node2vec on CPU)."""
    from scipy.sparse.linalg import eigsh
    vals, vecs = eigsh(S.astype(np.float64), k=dim, which="LA", tol=1e-4,
                       v0=np.random.default_rng(seed).random(S.shape[0]))
    return (vecs * vals).astype(np.float32)


def label_propagation(S: sp.csr_matrix, idx_tr, y_tr, alpha: float = 0.9, iters: int = 20):
    """Zhou et al. propagation of centred train labels; returns a score per node."""
    Y = np.zeros(S.shape[0], np.float32)
    Y[idx_tr] = y_tr - y_tr.mean()
    Fv = Y.copy()
    for _ in range(iters):
        Fv = alpha * (S @ Fv) + (1 - alpha) * Y
    return Fv


# ---------------------------------------------------------------- GraphSAGE (sparse, full batch)
def to_torch_sparse(P: sp.csr_matrix) -> torch.Tensor:
    P = P.tocsr().astype(np.float32)
    return torch.sparse_csr_tensor(torch.from_numpy(P.indptr).long(), torch.from_numpy(P.indices).long(),
                                   torch.from_numpy(P.data), size=P.shape)


class SAGE(nn.Module):
    """h' = W_self h + W_nbr mean_{u in N(v)} h_u  (mean aggregator as a sparse matmul)."""

    def __init__(self, d_in: int, d_h: int = 64, n_layers: int = 2, n_out: int = 2, dropout: float = 0.2):
        super().__init__()
        dims = [d_in] + [d_h] * n_layers
        self.self_l = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.nbr_l = nn.ModuleList(nn.Linear(a, b, bias=False) for a, b in zip(dims[:-1], dims[1:]))
        self.norms = nn.ModuleList(nn.LayerNorm(b) for b in dims[1:])
        self.head = nn.Linear(d_h, n_out)
        self.dropout = dropout

    def forward(self, x, P):
        for ls, ln_, nrm in zip(self.self_l, self.nbr_l, self.norms):
            x = F.relu(nrm(ls(x) + ln_(torch.sparse.mm(P, x))))
            x = F.dropout(x, self.dropout, self.training)
        return self.head(x)


def train_sage(Xn: np.ndarray, P: sp.csr_matrix, idx: dict, Y: np.ndarray, seed: int,
               epochs: int = 150, lr: float = 5e-3, patience: int = 25, log=print) -> np.ndarray:
    """Multi-task (one logit per target column of Y). Early stopping on mean val AUPRC."""
    from sklearn.metrics import average_precision_score
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    x, Pt = torch.from_numpy(Xn), to_torch_sparse(P)
    Yt = torch.from_numpy(Y.astype(np.float32))
    tr, va = torch.from_numpy(idx["train"]), idx["val"]
    pos = Yt[tr].mean(0).clamp(min=1e-3)
    model = SAGE(Xn.shape[1], n_out=Y.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(x, Pt)
        loss = F.binary_cross_entropy_with_logits(out[tr], Yt[tr], pos_weight=(1 - pos) / pos)
        loss.backward()
        opt.step()
        if ep % 5 == 4:
            model.eval()
            with torch.no_grad():
                s = model(x, Pt)[va].numpy()
            score = np.mean([average_precision_score(Y[va, j], s[:, j]) for j in range(Y.shape[1])])
            if score > best:
                best, bad = score, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad += 5
            if ep % 25 == 24:
                log(f"    sage seed{seed} ep{ep+1} loss {loss.item():.4f} val-AP {score:.4f}")
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model(x, Pt).numpy()
