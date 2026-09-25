from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import average_precision, hits_at_k, mrr, roc_auc


def device_of() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_adj(n: int, edges: np.ndarray, device: torch.device) -> torch.Tensor:
    """Â = D^{-1/2} (A+I) D^{-1/2} como matriz densa |V|x|V| (Cora cabe)."""
    a = np.eye(n, dtype=np.float32)
    for u, v in edges:
        u, v = int(u), int(v)
        a[u, v] = 1.0
        a[v, u] = 1.0
    deg = a.sum(axis=1)
    dinv = np.power(np.maximum(deg, 1e-12), -0.5)
    a = (dinv[:, None] * a) * dinv[None, :]
    return torch.from_numpy(a).to(device)


class GCN(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, out_dim: int = 64, dropout: float = 0.5):
        super().__init__()
        self.w1 = nn.Linear(in_dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, out_dim, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        h = a_hat @ x
        h = F.relu(self.w1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = a_hat @ h
        return self.w2(h)


def pair_scores(z: torch.Tensor, pairs: np.ndarray) -> torch.Tensor:
    src = torch.as_tensor(pairs[:, 0], device=z.device, dtype=torch.long)
    dst = torch.as_tensor(pairs[:, 1], device=z.device, dtype=torch.long)
    return (z[src] * z[dst]).sum(dim=-1)


def train_gcn(
    x_np: np.ndarray,
    mp_edges: np.ndarray,
    pos_train: np.ndarray,
    n: int,
    *,
    seed: int,
    hidden: int = 64,
    epochs: int = 120,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    neg_per_pos: int = 1,
    dropout: float = 0.5,
) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = device_of()
    x = torch.from_numpy(x_np).to(dev)
    a_hat = normalize_adj(n, mp_edges, dev)
    model = GCN(x.shape[1], hidden, hidden, dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(seed + 17)
    forbidden = {(int(u), int(v)) for u, v in mp_edges}
    forbidden |= {(int(u), int(v)) for u, v in pos_train}

    model.train()
    for _ in range(epochs):
        neg = []
        need = len(pos_train) * neg_per_pos
        while len(neg) < need:
            u = int(rng.integers(0, n))
            v = int(rng.integers(0, n))
            if u == v:
                continue
            key = (u, v) if u < v else (v, u)
            if key in forbidden:
                continue
            neg.append(key)
        neg_arr = np.asarray(neg[:need], dtype=np.int64)
        z = model(x, a_hat)
        pos_s = pair_scores(z, pos_train)
        neg_s = pair_scores(z, neg_arr)
        scores = torch.cat([pos_s, neg_s])
        labels = torch.cat(
            [
                torch.ones_like(pos_s),
                torch.zeros_like(neg_s),
            ]
        )
        loss = F.binary_cross_entropy_with_logits(scores, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        z = model(x, a_hat).cpu().numpy()
    return z


def score_numpy(z: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    return (z[pairs[:, 0]] * z[pairs[:, 1]]).sum(axis=1)


def metrics_from_scores(pos: np.ndarray, neg: np.ndarray) -> dict[str, float]:
    y = [1] * len(pos) + [0] * len(neg)
    s = np.concatenate([pos, neg]).tolist()
    return {
        "auc": float(roc_auc(y, s)),
        "ap": float(average_precision(y, s)),
        "hits3": float(hits_at_k(y, s, 3)),
        "hits10": float(hits_at_k(y, s, 10)),
        "mrr": float(mrr(y, s)),
    }
