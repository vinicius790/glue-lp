from __future__ import annotations
import math
import numpy as np
from .graph import Node, Edge

KINDS = ["resource", "material", "tool", "station", "goal"]

def embed_gcn(nodes, mp, hidden=8, seed=7):
    n = len(nodes)
    adj = [set() for _ in range(n)]
    for e in mp:
        adj[e.source].add(e.target); adj[e.target].add(e.source)
    X = []
    max_u = max(nd.unlock for nd in nodes) or 1
    for nd in nodes:
        oh = [1.0 if nd.kind == k else 0.0 for k in KINDS]
        X.append(oh + [nd.unlock / max_u, len(adj[nd.id]) / max(n-1,1), 1.0])
    X = np.asarray(X)
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, math.sqrt(2/(hidden+X.shape[1])), (hidden, X.shape[1]))
    def agg(X):
        out = np.zeros_like(X)
        for v in range(n):
            acc = X[v].copy()
            for u in adj[v]: acc += X[u]
            out[v] = acc / (len(adj[v]) + 1)
        return out
    H = np.maximum(agg(X) @ W1.T, 0)
    return H

def score_pairs(Z, pairs):
    return [float((Z[u] * Z[v]).sum()) for u, v in pairs]
