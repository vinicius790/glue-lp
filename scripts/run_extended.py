#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from glue_lp.data_citation import load_citeseer, load_cora
from glue_lp.heuristics import scores_aa
from glue_lp.metrics import roc_auc
from glue_lp.splits import (
    build_adj,
    edge_set,
    random_edge_split,
    sample_uniform_negatives,
)
from glue_lp.stats import degree_bins, paired_t
from glue_lp.config import DEFAULT_TRAIN
from glue_lp.torch_models import train_encoder

OUT = Path(__file__).resolve().parents[1] / "experiments"
CFG = DEFAULT_TRAIN
SEEDS = list(CFG.seeds)


def eval_z(z, pos, neg):
    sp = (z[pos[:, 0]] * z[pos[:, 1]]).sum(1)
    sn = (z[neg[:, 0]] * z[neg[:, 1]]).sum(1)
    y = [1] * len(pos) + [0] * len(neg)
    s = np.concatenate([sp, sn]).tolist()
    return roc_auc(y, s), np.concatenate([sp, sn]), y, np.vstack([pos, neg])


def cell(data, seed, mode, kind):
    n = data["n"]
    split = random_edge_split(data["edges"], seed=seed)
    train, val, test = split["train"], split["val"], split["test"]
    mp = np.vstack([train, val, test]) if mode == "leaky" else train
    rng = np.random.default_rng(seed + 99)
    neg = sample_uniform_negatives(n, set(edge_set(data["edges"])), len(test), rng)
    # Always pass val so early-stopping criterion is symmetric (valid vs leaky).
    packed = train_encoder(
        data["x"], mp, train, val,
        n, seed=seed, kind=kind,
        hidden=CFG.hidden, epochs=CFG.epochs_cap, patience=CFG.patience,
    )
    auc, scores, y, pairs = eval_z(packed["z"], test, neg)
    adj = build_adj(n, mp)
    deg = [len(s) for s in adj]
    bins = degree_bins(deg, pairs, scores, y)
    aa = scores_aa(n, mp, pairs)
    aa_auc = roc_auc(y, aa.tolist())
    return {
        "dataset": data["name"],
        "seed": seed,
        "mode": mode,
        "kind": kind,
        "auc": auc,
        "aa_auc": aa_auc,
        "epochs_run": packed["epochs_run"],
        "best_val": packed["best_val"],
        "n_test": int(len(test)),
        "bins": bins,
    }


def main():
    t0 = time.time()
    rows = []
    cora = load_cora()
    citeseer = load_citeseer()
    print("cora", cora["n"], len(cora["edges"]), "citeseer", citeseer["n"], len(citeseer["edges"]), flush=True)

    for seed in SEEDS:
        for mode in ("valid", "leaky"):
            r = cell(cora, seed, mode, "gcn")
            rows.append(r)
            print("cora-gcn", seed, mode, round(r["auc"], 4), "ep", r["epochs_run"], flush=True)
        r = cell(cora, seed, "valid", "sage")
        rows.append(r)
        print("cora-sage", seed, round(r["auc"], 4), flush=True)

    for seed in SEEDS:
        for mode in ("valid", "leaky"):
            r = cell(citeseer, seed, mode, "gcn")
            rows.append(r)
            print("citeseer-gcn", seed, mode, round(r["auc"], 4), flush=True)

    def pick(ds, kind, mode):
        return [r for r in rows if r["dataset"] == ds and r["kind"] == kind and r["mode"] == mode]

    def ms(vals):
        a = np.asarray(vals, float)
        return {"mean": float(a.mean()), "std": float(a.std(ddof=1)), "values": [float(x) for x in vals]}

    cg_v = pick("cora", "gcn", "valid")
    cg_l = pick("cora", "gcn", "leaky")
    cs_v = pick("citeseer", "gcn", "valid")
    cs_l = pick("citeseer", "gcn", "leaky")
    sage = pick("cora", "sage", "valid")
    summary = {
        "cora_gcn_valid": ms([r["auc"] for r in cg_v]),
        "cora_gcn_leaky": ms([r["auc"] for r in cg_l]),
        "citeseer_gcn_valid": ms([r["auc"] for r in cs_v]),
        "citeseer_gcn_leaky": ms([r["auc"] for r in cs_l]),
        "cora_sage_valid": ms([r["auc"] for r in sage]),
        "paired_t_cora": paired_t([r["auc"] for r in cg_l], [r["auc"] for r in cg_v]),
        "paired_t_citeseer": paired_t([r["auc"] for r in cs_l], [r["auc"] for r in cs_v]),
    }
    out = {"seconds": round(time.time() - t0, 1), "rows": rows, "summary": summary}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "extended_gcn_sage.json"
    path.write_text(json.dumps(out, indent=2))
    print("WROTE", path, out["seconds"])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
