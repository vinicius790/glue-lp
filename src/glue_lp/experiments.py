from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .heuristics import scores_aa, scores_cn, scores_pa
from .splits import build_adj, edge_set, inductive_node_split, random_edge_split, sample_hard_negatives, sample_uniform_negatives
from .torch_gcn import metrics_from_scores, score_numpy, train_gcn

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments"

def run_transductive_cell(data, *, seed, mode, negatives, epochs, hidden, neg_per_pos):
    n = data["n"]
    split = random_edge_split(data["edges"], seed=seed)
    train, val, test = split["train"], split["val"], split["test"]
    mp = np.vstack([train, val, test]) if mode == "leaky" else train
    if mode == "valid" and (edge_set(test) & edge_set(mp)):
        raise AssertionError("vazamento no modo valido")
    rng = np.random.default_rng(seed + 99)
    forbidden = edge_set(data["edges"])
    if negatives == "hard":
        neg = sample_hard_negatives(n, build_adj(n, mp), forbidden, test, neg_per_pos, rng)
    else:
        neg = sample_uniform_negatives(n, set(forbidden), len(test)*neg_per_pos, rng)
    z = train_gcn(data["x"], mp, train, n, seed=seed, hidden=hidden, epochs=epochs)
    return {"seed": seed, "mode": mode, "negatives": negatives, "n_test": int(len(test)),
            "metrics": {"gcn": metrics_from_scores(score_numpy(z, test), score_numpy(z, neg))}}

def save_json(obj, name):
    EXP.mkdir(parents=True, exist_ok=True)
    path = EXP / name
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    return path
