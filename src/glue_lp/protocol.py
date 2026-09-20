from __future__ import annotations
from dataclasses import dataclass
from .graph import Edge, Node, edge_key
from .rng import mulberry32, shuffle

@dataclass
class Split:
    mp_edges: list
    positives: list
    negatives: list
    cutoff: int

def undirected_adj(n, edges):
    adj = [set() for _ in range(n)]
    for e in edges:
        adj[e.source].add(e.target)
        adj[e.target].add(e.source)
    return adj

def leakage_exists(mp, positives):
    keys = {edge_key(e.source, e.target) for e in mp}
    return any(edge_key(e.source, e.target) in keys for e in positives)

def assert_no_leakage(mp, positives):
    if leakage_exists(mp, positives):
        raise AssertionError("invariante violada: aresta-alvo em E_mp")

def _common(adj, u, v):
    a, b = adj[u], adj[v]
    if len(a) > len(b):
        a, b = b, a
    return sum(1 for x in a if x in b)

def split_graph(nodes, edges, *, split="temporal", negatives="hard", mode="valid", seed=7, holdout=0.25, neg_per_pos=2):
    rnd = mulberry32(seed)
    n = len(nodes)
    ordered = sorted(edges, key=lambda e: (e.time, e.source, e.target))
    if split == "temporal":
        times = sorted({e.time for e in ordered})
        cutoff = times[max(0, len(times) - 3)] if times else 0
        train = [e for e in ordered if e.time < cutoff]
        positives = [e for e in ordered if e.time >= cutoff]
        if not positives:
            k = max(1, int(len(ordered) * holdout))
            train, positives = ordered[:-k], ordered[-k:]
    else:
        cutoff = 0
        shuffled = list(ordered)
        shuffle(shuffled, rnd)
        k = max(1, int(len(shuffled) * holdout))
        positives = shuffled[:k]
        pkeys = {edge_key(e.source, e.target) for e in positives}
        train = [e for e in shuffled if edge_key(e.source, e.target) not in pkeys]
    mp = list(train) + (list(positives) if mode == "leaky" else [])
    if mode == "valid":
        assert_no_leakage(mp, positives)
    present = {edge_key(e.source, e.target) for e in train + positives}
    candidates = [(i, j) for i in range(n) for j in range(i + 1, n) if edge_key(i, j) not in present]
    adj = undirected_adj(n, mp)
    ranked = [(s, t, _common(adj, s, t)) for s, t in candidates]
    if negatives == "hard":
        ranked.sort(key=lambda x: (-x[2], x[0], x[1]))
    else:
        shuffle(ranked, rnd)
    want = max(len(positives) * neg_per_pos, len(positives))
    negs = [(s, t) for s, t, _ in ranked[:want]]
    return Split(mp_edges=mp, positives=positives, negatives=negs, cutoff=cutoff)
