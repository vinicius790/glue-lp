from __future__ import annotations
import math
import numpy as np
from .protocol import undirected_adj
from .rng import mulberry32

KINDS = ["resource", "material", "tool", "station", "goal"]

def embed_gcn(nodes, mp, hidden=8, seed=7):
    rnd = mulberry32(seed)
    adj = undirected_adj(len(nodes), mp)
    max_u = max(nd.unlock for nd in nodes) or 1
    X = []
    for nd in nodes:
        oh = [1.0 if nd.kind == k else 0.0 for k in KINDS]
        X.append(oh + [nd.unlock/max_u, len(adj[nd.id])/max(len(nodes)-1,1), 1.0])
    X = np.asarray(X)
    in_dim = X.shape[1]
    scale = math.sqrt(2/(hidden+in_dim))
    W1 = np.array([[(rnd()*2-1)*scale for _ in range(in_dim)] for _ in range(hidden)])
    def agg(X):
        out = np.zeros_like(X)
        for v in range(len(nodes)):
            acc = X[v].copy()
            for u in adj[v]: acc += X[u]
            out[v] = acc/(len(adj[v])+1)
        return out
    return np.maximum(agg(X) @ W1.T, 0)

def score_pairs(Z, pairs):
    return [float(Z[u] @ Z[v]) for u, v in pairs]

def adamic_adar(n, mp, pairs):
    adj = undirected_adj(n, mp)
    deg = [len(s) for s in adj]
    out = []
    for u, v in pairs:
        a, b = adj[u], adj[v]
        small, large = (a, b) if len(a)<=len(b) else (b, a)
        s = 0.0
        for x in small:
            if x in large:
                s += 1.0/math.log(max(deg[x], 2))
        out.append(s)
    return out
