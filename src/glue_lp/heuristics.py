from __future__ import annotations
import math
import numpy as np

def scores_cn(adj, pairs):
    return np.asarray([float(len(adj[int(u)] & adj[int(v)])) for u, v in pairs])

def scores_aa(adj, deg, pairs):
    out = []
    for u, v in pairs:
        inter = adj[int(u)] & adj[int(v)]
        out.append(sum(1.0 / math.log(max(deg[x], 2)) for x in inter))
    return np.asarray(out)

def scores_pa(deg, pairs):
    return np.asarray([float(deg[int(u)] * deg[int(v)]) for u, v in pairs])
