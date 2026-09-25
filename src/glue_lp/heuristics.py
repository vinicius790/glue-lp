from __future__ import annotations

import math

import numpy as np

from .splits import build_adj


def _score_pairs(n: int, mp: np.ndarray, pairs: np.ndarray, kind: str) -> np.ndarray:
    adj = build_adj(n, mp)
    deg = [len(s) for s in adj]
    out = np.zeros(len(pairs), dtype=np.float64)
    for i, (u, v) in enumerate(pairs):
        u, v = int(u), int(v)
        a, b = adj[u], adj[v]
        small, large = (a, b) if len(a) <= len(b) else (b, a)
        inter = [x for x in small if x in large]
        if kind == "cn":
            out[i] = float(len(inter))
        elif kind == "aa":
            s = 0.0
            for x in inter:
                s += 1.0 / math.log(max(deg[x], 2))
            out[i] = s
        elif kind == "pa":
            out[i] = float(deg[u] * deg[v])
        else:
            raise ValueError(kind)
    return out


def scores_cn(n: int, mp: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    return _score_pairs(n, mp, pairs, "cn")


def scores_aa(n: int, mp: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    return _score_pairs(n, mp, pairs, "aa")


def scores_pa(n: int, mp: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    return _score_pairs(n, mp, pairs, "pa")
