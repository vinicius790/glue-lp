#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from glue_lp.data_citation import load_cora, load_pubmed
from glue_lp.metrics import roc_auc
from glue_lp.splits import (
    build_adj,
    edge_set,
    random_edge_split,
    sample_degree_matched,
    sample_uniform_negatives,
)
from glue_lp.stats import paired_t
from glue_lp.config import DEFAULT_TRAIN
from glue_lp.torch_models import train_encoder

OUT = Path(__file__).resolve().parents[1] / "experiments"
CFG = DEFAULT_TRAIN


def auc_z(z, pos, neg):
    sp = (z[pos[:, 0]] * z[pos[:, 1]]).sum(1)
    sn = (z[neg[:, 0]] * z[neg[:, 1]]).sum(1)
    y = [1] * len(pos) + [0] * len(neg)
    return roc_auc(y, np.concatenate([sp, sn]).tolist())


def run_cell(data, seed, mode, kind, negatives="uniform", epochs=60, patience=10):
    n = data["n"]
    split = random_edge_split(data["edges"], seed=seed)
    train, val, test = split["train"], split["val"], split["test"]
    mp = np.vstack([train, val, test]) if mode == "leaky" else train
    rng = np.random.default_rng(seed + 99)
    forbidden = set(edge_set(data["edges"]))
    if negatives == "degree":
        adj = build_adj(n, mp)
        neg = sample_degree_matched(n, adj, forbidden, test, 1, rng)
    else:
        neg = sample_uniform_negatives(n, forbidden, len(test), rng)
    packed = train_encoder(
        data["x"], mp, train, val,
        n, seed=seed, kind=kind, hidden=CFG.hidden, epochs=epochs, patience=patience,
    )
    return {
        "dataset": data["name"],
        "seed": seed,
        "mode": mode,
        "kind": kind,
        "negatives": negatives,
        "auc": auc_z(packed["z"], test, neg),
        "epochs_run": packed["epochs_run"],
        "n_test": int(len(test)),
    }


def ms(vals):
    a = np.asarray(vals, float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1) if len(a) > 1 else 0.0), "values": [float(x) for x in vals]}


def main():
    t0 = time.time()
    rows = []
    cora = load_cora()
    print("cora", cora["n"], len(cora["edges"]), flush=True)

    # Epochs/patience come from DEFAULT_TRAIN (single source). Values match the
    # measured wave3 JSON; do not silently change without a new experiment file.
    gat_epochs = CFG.epochs  # 50 — measured GAT Cora
    gat_patience = min(CFG.patience, 10)  # measured used 10
    pubmed_epochs = CFG.epochs_cap_pubmed  # 25
    pubmed_patience = 8

    for seed in CFG.seeds:
        for mode in ("valid", "leaky"):
            r = run_cell(cora, seed, mode, "gat", epochs=gat_epochs, patience=gat_patience)
            rows.append(r)
            print("gat", seed, mode, round(r["auc"], 4), "ep", r["epochs_run"], flush=True)
        r = run_cell(cora, seed, "valid", "gcn", negatives="degree", epochs=gat_epochs, patience=gat_patience)
        rows.append(r)
        print("degmatch", seed, round(r["auc"], 4), flush=True)

    print("loading pubmed...", flush=True)
    pubmed = load_pubmed()
    print("pubmed", pubmed["n"], len(pubmed["edges"]), pubmed["n_features"], flush=True)
    for seed in CFG.pubmed_seeds:
        for mode in ("valid", "leaky"):
            r = run_cell(pubmed, seed, mode, "gcn", epochs=pubmed_epochs, patience=pubmed_patience)
            rows.append(r)
            print("pubmed", seed, mode, round(r["auc"], 4), "ep", r["epochs_run"], flush=True)

    gat_v = [r["auc"] for r in rows if r["kind"] == "gat" and r["mode"] == "valid"]
    gat_l = [r["auc"] for r in rows if r["kind"] == "gat" and r["mode"] == "leaky"]
    deg = [r["auc"] for r in rows if r.get("negatives") == "degree"]
    pv = [r["auc"] for r in rows if r["dataset"] == "pubmed" and r["mode"] == "valid"]
    pl = [r["auc"] for r in rows if r["dataset"] == "pubmed" and r["mode"] == "leaky"]
    summary = {
        "cora_gat_valid": ms(gat_v),
        "cora_gat_leaky": ms(gat_l),
        "cora_gcn_degree_matched": ms(deg),
        "paired_t_gat": paired_t(gat_l, gat_v),
        "pubmed_gcn_valid": ms(pv) if pv else None,
        "pubmed_gcn_leaky": ms(pl) if pl else None,
        "paired_t_pubmed": paired_t(pl, pv) if len(pv) > 1 else None,
    }
    out = {"seconds": round(time.time() - t0, 1), "rows": rows, "summary": summary}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "wave3_gat_pubmed.json"
    path.write_text(json.dumps(out, indent=2))
    print("WROTE", path, out["seconds"])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
