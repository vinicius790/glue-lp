from __future__ import annotations
import numpy as np

def _key(u, v):
    return (int(u), int(v)) if u < v else (int(v), int(u))

def random_edge_split(edges, *, seed, test_frac=0.10, val_frac=0.05):
    rng = np.random.default_rng(seed)
    e = edges[rng.permutation(len(edges))]
    n_test = max(1, int(len(e) * test_frac))
    n_val = max(1, int(len(e) * val_frac))
    return {"test": e[:n_test], "val": e[n_test:n_test+n_val], "train": e[n_test+n_val:]}

def inductive_node_split(n, edges, *, seed, holdout_frac=0.15):
    rng = np.random.default_rng(seed)
    hold = set(rng.choice(n, size=max(1, int(n * holdout_frac)), replace=False).tolist())
    seen = set(range(n)) - hold
    train, test = [], []
    for u, v in edges:
        if int(u) in seen and int(v) in seen:
            train.append((int(u), int(v)))
        else:
            test.append((int(u), int(v)))
    return {"train": np.asarray(train, np.int64), "test": np.asarray(test, np.int64)}
