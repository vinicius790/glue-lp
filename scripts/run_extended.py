#!/usr/bin/env python3
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
from glue_lp.data_citation import load_citeseer, load_cora
from glue_lp.heuristics import scores_aa
from glue_lp.metrics import roc_auc
from glue_lp.splits import build_adj, edge_set, random_edge_split, sample_uniform_negatives
from glue_lp.stats import paired_t
from glue_lp.torch_models import train_encoder

OUT = Path(__file__).resolve().parents[1] / "experiments"
SEEDS = [0, 1, 2, 3, 4]

def eval_z(z, pos, neg):
    sp = (z[pos[:, 0]] * z[pos[:, 1]]).sum(1)
    sn = (z[neg[:, 0]] * z[neg[:, 1]]).sum(1)
    y = [1] * len(pos) + [0] * len(neg)
    return roc_auc(y, np.concatenate([sp, sn]).tolist())

def cell(data, seed, mode, kind):
    n = data["n"]
    split = random_edge_split(data["edges"], seed=seed)
    train, val, test = split["train"], split["val"], split["test"]
    mp = np.vstack([train, val, test]) if mode == "leaky" else train
    rng = np.random.default_rng(seed + 99)
    neg = sample_uniform_negatives(n, set(edge_set(data["edges"])), len(test), rng)
    packed = train_encoder(data["x"], mp, train, val if mode == "valid" else None,
                           n, seed=seed, kind=kind, hidden=32, epochs=70, patience=12)
    return {"dataset": data["name"], "seed": seed, "mode": mode, "kind": kind,
            "auc": eval_z(packed["z"], test, neg), "epochs_run": packed["epochs_run"]}

def main():
    rows = []
    cora, citeseer = load_cora(), load_citeseer()
    for seed in SEEDS:
        for mode in ("valid", "leaky"):
            rows.append(cell(cora, seed, mode, "gcn"))
        rows.append(cell(cora, seed, "valid", "sage"))
    for seed in SEEDS:
        for mode in ("valid", "leaky"):
            rows.append(cell(citeseer, seed, mode, "gcn"))
    def pick(ds, kind, mode):
        return [r["auc"] for r in rows if r["dataset"]==ds and r["kind"]==kind and r["mode"]==mode]
    summary = {
        "paired_t_cora": paired_t(pick("cora","gcn","leaky"), pick("cora","gcn","valid")),
        "paired_t_citeseer": paired_t(pick("citeseer","gcn","leaky"), pick("citeseer","gcn","valid")),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"extended_gcn_sage.json").write_text(json.dumps({"rows": rows, "summary": summary}, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
