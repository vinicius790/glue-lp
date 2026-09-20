#!/usr/bin/env python3
"""Onda 3: GAT em Cora (5 seeds) + GCN Pubmed (3 seeds) + degree-matched."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from glue_lp.data_citation import load_cora, load_pubmed
from glue_lp.metrics import roc_auc
from glue_lp.splits import build_adj, edge_set, random_edge_split, sample_degree_matched, sample_uniform_negatives
from glue_lp.stats import paired_t
from glue_lp.torch_models import train_encoder

OUT = Path(__file__).resolve().parents[1] / "experiments"

def auc_z(z, pos, neg):
    sp = (z[pos[:,0]] * z[pos[:,1]]).sum(1)
    sn = (z[neg[:,0]] * z[neg[:,1]]).sum(1)
    return roc_auc([1]*len(pos)+[0]*len(neg), np.concatenate([sp,sn]).tolist())

def cell(data, seed, mode, kind, negatives="uniform", epochs=60, patience=10):
    n = data["n"]
    split = random_edge_split(data["edges"], seed=seed)
    train, val, test = split["train"], split["val"], split["test"]
    mp = np.vstack([train, val, test]) if mode == "leaky" else train
    rng = np.random.default_rng(seed + 99)
    forbidden = set(edge_set(data["edges"]))
    if negatives == "degree":
        neg = sample_degree_matched(n, build_adj(n, mp), forbidden, test, 1, rng)
    else:
        neg = sample_uniform_negatives(n, forbidden, len(test), rng)
    packed = train_encoder(data["x"], mp, train, val if mode=="valid" else None,
                           n, seed=seed, kind=kind, hidden=32, epochs=epochs, patience=patience)
    return {"dataset": data.get("name",""), "seed": seed, "mode": mode, "kind": kind,
            "negatives": negatives, "auc": auc_z(packed["z"], test, neg), "epochs_run": packed["epochs_run"]}

def main():
    rows = []
    cora = load_cora()
    for seed in range(5):
        rows.append(cell(cora, seed, "valid", "gat"))
        rows.append(cell(cora, seed, "leaky", "gat"))
        rows.append(cell(cora, seed, "valid", "gcn", negatives="degree", epochs=40, patience=10))
    pub = load_pubmed()
    for seed in range(3):
        rows.append(cell(pub, seed, "valid", "gcn", epochs=25, patience=8))
        rows.append(cell(pub, seed, "leaky", "gcn", epochs=25, patience=8))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"wave3_gat_pubmed.json").write_text(json.dumps({"rows": rows}, indent=2))
    print("wrote", len(rows), "rows")

if __name__ == "__main__":
    main()
