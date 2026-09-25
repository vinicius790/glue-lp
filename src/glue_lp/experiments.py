from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .data_cora import load_cora
from .heuristics import scores_aa, scores_cn, scores_pa
from .splits import (
    build_adj,
    edge_set,
    inductive_node_split,
    random_edge_split,
    sample_hard_negatives,
    sample_uniform_negatives,
)
from .torch_gcn import metrics_from_scores, score_numpy, train_gcn


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments"


def _pairs_metrics(z, heur, pos, neg) -> dict:
    pp = np.vstack([pos, neg]) if len(neg) else pos
    out = {"gcn": metrics_from_scores(score_numpy(z, pos), score_numpy(z, neg))}
    for name, fn in heur.items():
        s = fn(pp)
        out[name] = metrics_from_scores(s[: len(pos)], s[len(pos) :])
    return out


def run_transductive_cell(
    data: dict,
    *,
    seed: int,
    mode: str,
    negatives: str,
    epochs: int,
    hidden: int,
    neg_per_pos: int,
) -> dict:
    n = data["n"]
    split = random_edge_split(data["edges"], seed=seed)
    train, val, test = split["train"], split["val"], split["test"]
    mp = np.vstack([train, val, test]) if mode == "leaky" else train
    if mode == "valid":
        mp_keys = edge_set(mp)
        if any((int(u), int(v)) in mp_keys or (int(v), int(u)) in mp_keys for u, v in test):
            # undirected keys already min,max
            test_keys = edge_set(test)
            if test_keys & edge_set(mp):
                raise AssertionError("vazamento no modo válido")

    rng = np.random.default_rng(seed + 99)
    forbidden = edge_set(data["edges"])
    if negatives == "hard":
        adj = build_adj(n, mp)
        neg = sample_hard_negatives(n, adj, forbidden, test, neg_per_pos, rng)
    else:
        neg = sample_uniform_negatives(n, set(forbidden), len(test) * neg_per_pos, rng)

    z = train_gcn(
        data["x"],
        mp,
        train,
        n,
        seed=seed,
        hidden=hidden,
        epochs=epochs,
    )
    heur = {
        "cn": lambda p: scores_cn(n, mp, p),
        "aa": lambda p: scores_aa(n, mp, p),
        "pa": lambda p: scores_pa(n, mp, p),
    }
    mets = _pairs_metrics(z, heur, test, neg)
    return {
        "seed": seed,
        "mode": mode,
        "negatives": negatives,
        "split": "random",
        "n": n,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_neg": int(len(neg)),
        "n_mp": int(len(mp)),
        "leakage": mode == "leaky",
        "metrics": mets,
    }


def run_inductive_cell(
    data: dict,
    *,
    seed: int,
    epochs: int,
    hidden: int,
    neg_per_pos: int,
) -> dict:
    n = data["n"]
    split = inductive_node_split(n, data["edges"], seed=seed)
    train, test = split["train"], split["test"]
    if len(test) < 10:
        raise RuntimeError("poucas arestas indutivas")
    mp = train
    rng = np.random.default_rng(seed + 123)
    forbidden = edge_set(data["edges"])
    neg = sample_uniform_negatives(n, set(forbidden), len(test) * neg_per_pos, rng)
    z = train_gcn(data["x"], mp, train, n, seed=seed, hidden=hidden, epochs=epochs)
    heur = {
        "cn": lambda p: scores_cn(n, mp, p),
        "aa": lambda p: scores_aa(n, mp, p),
        "pa": lambda p: scores_pa(n, mp, p),
    }
    mets = _pairs_metrics(z, heur, test, neg)
    return {
        "seed": seed,
        "mode": "valid",
        "negatives": "uniform",
        "split": "inductive-node",
        "n_seen": int(len(split["seen_nodes"])),
        "n_inductive": int(len(split["inductive_nodes"])),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_neg": int(len(neg)),
        "metrics": mets,
    }


def summarize(cells: list[dict], model: str = "gcn", key: str = "auc") -> dict:
    vals = [c["metrics"][model][key] for c in cells]
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "values": [float(v) for v in vals],
    }


def save_json(obj, name: str) -> Path:
    EXP.mkdir(parents=True, exist_ok=True)
    path = EXP / name
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    return path
