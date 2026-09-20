from __future__ import annotations
import numpy as np

def _key(u, v):
    return (int(u), int(v)) if u < v else (int(v), int(u))

def edge_set(edges):
    return {_key(u, v) for u, v in edges}

def random_edge_split(edges, *, seed, test_frac=0.10, val_frac=0.05):
    rng = np.random.default_rng(seed)
    e = edges[rng.permutation(len(edges))]
    n_test = max(1, int(len(e)*test_frac))
    n_val = max(1, int(len(e)*val_frac))
    return {"test": e[:n_test], "val": e[n_test:n_test+n_val], "train": e[n_test+n_val:]}

def inductive_node_split(n, edges, *, seed, holdout_frac=0.15):
    rng = np.random.default_rng(seed)
    hold = set(rng.choice(n, size=max(1, int(n*holdout_frac)), replace=False).tolist())
    seen = set(range(n)) - hold
    train, test = [], []
    for u, v in edges:
        if int(u) in seen and int(v) in seen:
            train.append((int(u), int(v)))
        else:
            test.append((int(u), int(v)))
    return {"train": np.asarray(train, np.int64), "test": np.asarray(test, np.int64),
            "seen_nodes": np.asarray(sorted(seen), np.int64),
            "inductive_nodes": np.asarray(sorted(hold), np.int64)}

def sample_uniform_negatives(n, forbidden, k, rng):
    out, guard = [], 0
    while len(out) < k and guard < k*40:
        guard += 1
        u, v = int(rng.integers(0, n)), int(rng.integers(0, n))
        if u == v: continue
        key = _key(u, v)
        if key in forbidden: continue
        forbidden.add(key); out.append(key)
    return np.asarray(out, np.int64)

def build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[int(u)].add(int(v)); adj[int(v)].add(int(u))
    return adj

def common_neighbors_count(adj, u, v):
    a, b = adj[u], adj[v]
    if len(a) > len(b): a, b = b, a
    return sum(1 for x in a if x in b)

def sample_hard_negatives(n, adj, forbidden, positives, per_pos, rng, pool=80):
    out, used = [], set(forbidden)
    for u, v in positives:
        cands, tries = [], 0
        while len(cands) < pool and tries < pool*20:
            tries += 1
            a = int(u if rng.random() < 0.5 else v)
            b = int(rng.integers(0, n))
            if a == b: continue
            key = _key(a, b)
            if key in used: continue
            cands.append((key[0], key[1], common_neighbors_count(adj, key[0], key[1])))
        cands.sort(key=lambda t: -t[2])
        for s, t, _ in cands[:per_pos]:
            used.add((s, t)); out.append((s, t))
    return np.asarray(out[:len(positives)*per_pos] if out else out, np.int64)

def sample_degree_matched(n, adj, forbidden, positives, per_pos, rng, tol=2):
    deg = [len(s) for s in adj]
    buckets = {}
    for i, d in enumerate(deg):
        buckets.setdefault(d, []).append(i)
    out, used = [], set(forbidden)
    for u, v in positives:
        u, v = int(u), int(v)
        pool = []
        for d in range(max(0, deg[v]-tol), deg[v]+tol+1):
            pool.extend(buckets.get(d, []))
        rng.shuffle(pool)
        got = 0
        for w in pool:
            if got >= per_pos: break
            if w == u: continue
            key = _key(u, w)
            if key in used: continue
            used.add(key); out.append(key); got += 1
        while got < per_pos:
            extra = sample_uniform_negatives(n, used, 1, rng)
            if len(extra)==0: break
            out.append(tuple(extra[0])); got += 1
    return np.asarray(out, np.int64)
