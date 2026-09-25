from __future__ import annotations

import numpy as np


def _key(u: int, v: int) -> tuple[int, int]:
    return (int(u), int(v)) if u < v else (int(v), int(u))


def edge_set(edges: np.ndarray) -> set[tuple[int, int]]:
    return {_key(u, v) for u, v in edges}


def random_edge_split(
    edges: np.ndarray,
    *,
    seed: int,
    test_frac: float = 0.10,
    val_frac: float = 0.05,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(edges))
    e = edges[perm]
    n_test = max(1, int(len(e) * test_frac))
    n_val = max(1, int(len(e) * val_frac))
    test = e[:n_test]
    val = e[n_test : n_test + n_val]
    train = e[n_test + n_val :]
    return {"train": train, "val": val, "test": test}


def inductive_node_split(
    n: int,
    edges: np.ndarray,
    *,
    seed: int,
    holdout_frac: float = 0.15,
) -> dict:
    """Nós novos no teste. Treino usa só o subgrafo induzido pelos nós vistos."""
    rng = np.random.default_rng(seed)
    hold = max(1, int(n * holdout_frac))
    inductive = set(rng.choice(n, size=hold, replace=False).tolist())
    seen = [i for i in range(n) if i not in inductive]
    seen_set = set(seen)
    train, test_ind = [], []
    for u, v in edges:
        u_seen, v_seen = int(u) in seen_set, int(v) in seen_set
        if u_seen and v_seen:
            train.append((int(u), int(v)))
        else:
            # at least one endpoint is inductive → holdout edge
            test_ind.append((int(u), int(v)))
    return {
        "seen_nodes": np.asarray(seen, dtype=np.int64),
        "inductive_nodes": np.asarray(sorted(inductive), dtype=np.int64),
        "train": np.asarray(train, dtype=np.int64),
        "test": np.asarray(test_ind, dtype=np.int64),
    }


def sample_uniform_negatives(
    n: int,
    forbidden: set[tuple[int, int]],
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    out = []
    guard = 0
    while len(out) < k and guard < k * 40:
        guard += 1
        u = int(rng.integers(0, n))
        v = int(rng.integers(0, n))
        if u == v:
            continue
        key = _key(u, v)
        if key in forbidden:
            continue
        forbidden.add(key)
        out.append(key)
    return np.asarray(out, dtype=np.int64)


def common_neighbors_count(adj: list[set[int]], u: int, v: int) -> int:
    a, b = adj[u], adj[v]
    if len(a) > len(b):
        a, b = b, a
    return sum(1 for x in a if x in b)


def sample_hard_negatives(
    n: int,
    adj: list[set[int]],
    forbidden: set[tuple[int, int]],
    positives: np.ndarray,
    per_pos: int,
    rng: np.random.Generator,
    pool: int = 80,
) -> np.ndarray:
    """Para cada positivo, sorteia `pool` não-arestas e fica com as de maior CN."""
    out: list[tuple[int, int]] = []
    used = set(forbidden)
    for u, v in positives:
        cands: list[tuple[int, int, int]] = []
        tries = 0
        while len(cands) < pool and tries < pool * 20:
            tries += 1
            a = int(u if rng.random() < 0.5 else v)
            b = int(rng.integers(0, n))
            if a == b:
                continue
            key = _key(a, b)
            if key in used:
                continue
            cands.append((key[0], key[1], common_neighbors_count(adj, key[0], key[1])))
        cands.sort(key=lambda t: -t[2])
        for s, t, _ in cands[:per_pos]:
            used.add((s, t))
            out.append((s, t))
    if len(out) < len(positives) * per_pos:
        extra = sample_uniform_negatives(
            n, used, len(positives) * per_pos - len(out), rng
        )
        out.extend(map(tuple, extra.tolist()))
    return np.asarray(out[: len(positives) * per_pos], dtype=np.int64)


def build_adj(n: int, edges: np.ndarray) -> list[set[int]]:
    adj: list[set[int]] = [set() for _ in range(n)]
    for u, v in edges:
        adj[int(u)].add(int(v))
        adj[int(v)].add(int(u))
    return adj


def sample_degree_matched(
    n: int,
    adj: list[set[int]],
    forbidden: set[tuple[int, int]],
    positives: np.ndarray,
    per_pos: int,
    rng: np.random.Generator,
    tol: int = 2,
) -> np.ndarray:
    """Negativo (u,w) com |deg(w)-deg(v)| <= tol, v o outro extremo do positivo."""
    deg = [len(s) for s in adj]
    buckets: dict[int, list[int]] = {}
    for i, d in enumerate(deg):
        buckets.setdefault(d, []).append(i)
    out = []
    used = set(forbidden)
    for u, v in positives:
        u, v = int(u), int(v)
        target = deg[v]
        pool = []
        for d in range(max(0, target - tol), target + tol + 1):
            pool.extend(buckets.get(d, []))
        rng.shuffle(pool)
        got = 0
        for w in pool:
            if got >= per_pos:
                break
            if w == u:
                continue
            key = _key(u, w)
            if key in used:
                continue
            used.add(key)
            out.append(key)
            got += 1
        while got < per_pos:
            extra = sample_uniform_negatives(n, used, 1, rng)
            if len(extra) == 0:
                break
            out.append(tuple(extra[0]))
            got += 1
    return np.asarray(out, dtype=np.int64)
