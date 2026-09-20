from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def normalize_adj(n, edges, device):
    a = np.eye(n, dtype=np.float32)
    for u, v in edges:
        u, v = int(u), int(v)
        a[u, v] = a[v, u] = 1.0
    dinv = np.power(np.maximum(a.sum(1), 1e-12), -0.5)
    a = (dinv[:, None] * a) * dinv[None, :]
    return torch.from_numpy(a).to(device)

class GCN(nn.Module):
    def __init__(self, in_dim, hidden=32, dropout=0.5):
        super().__init__()
        self.w1 = nn.Linear(in_dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, hidden, bias=False)
        self.dropout = dropout
    def forward(self, x, a_hat):
        h = F.relu(self.w1(a_hat @ x))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.w2(a_hat @ h)
