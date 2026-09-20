from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .metrics import average_precision, hits_at_k, mrr, roc_auc

def device_of():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normalize_adj(n, edges, device):
    a = np.eye(n, dtype=np.float32)
    for u, v in edges:
        u, v = int(u), int(v)
        a[u, v] = a[v, u] = 1.0
    dinv = np.power(np.maximum(a.sum(1), 1e-12), -0.5)
    a = (dinv[:, None] * a) * dinv[None, :]
    return torch.from_numpy(a).to(device)

class GCN(nn.Module):
    def __init__(self, in_dim, hidden=32, out_dim=32, dropout=0.5):
        super().__init__()
        self.w1 = nn.Linear(in_dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, out_dim, bias=False)
        self.dropout = dropout
    def forward(self, x, a_hat):
        h = F.relu(self.w1(a_hat @ x))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.w2(a_hat @ h)

def pair_scores(z, pairs):
    src = torch.as_tensor(pairs[:, 0], device=z.device, dtype=torch.long)
    dst = torch.as_tensor(pairs[:, 1], device=z.device, dtype=torch.long)
    return (z[src] * z[dst]).sum(-1)

def train_gcn(x_np, mp_edges, pos_train, n, *, seed, hidden=32, epochs=50, lr=0.01, weight_decay=5e-4):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = device_of()
    x = torch.from_numpy(np.ascontiguousarray(x_np)).to(dev)
    a_hat = normalize_adj(n, mp_edges, dev)
    model = GCN(x.shape[1], hidden, hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(seed + 17)
    forbidden = {(int(u), int(v)) if u < v else (int(v), int(u)) for u, v in mp_edges}
    model.train()
    for _ in range(epochs):
        neg = []
        while len(neg) < len(pos_train):
            u, v = int(rng.integers(0, n)), int(rng.integers(0, n))
            if u == v: continue
            key = (u, v) if u < v else (v, u)
            if key in forbidden: continue
            neg.append(key)
        z = model(x, a_hat)
        scores = torch.cat([pair_scores(z, pos_train), pair_scores(z, np.asarray(neg, np.int64))])
        labels = torch.cat([torch.ones(len(pos_train), device=dev), torch.zeros(len(neg), device=dev)])
        loss = F.binary_cross_entropy_with_logits(scores, labels)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        return model(x, a_hat).cpu().numpy()

def score_numpy(z, pairs):
    return (z[pairs[:, 0]] * z[pairs[:, 1]]).sum(1)

def metrics_from_scores(pos, neg):
    y = [1]*len(pos) + [0]*len(neg)
    s = np.concatenate([pos, neg]).tolist()
    return {"auc": float(roc_auc(y, s)), "ap": float(average_precision(y, s)),
            "hits3": float(hits_at_k(y, s, 3)), "mrr": float(mrr(y, s))}
