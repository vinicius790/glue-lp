#!/usr/bin/env python3
"""End-to-end NumPy-only smoke demo on the synthetic tech tree.

Writes a NEW file ``experiments/smoke_*.json`` with synthetic numbers labeled
as smoke. Never overwrites measured ``cora_*.json`` / ``extended_*.json`` /
``wave3_*.json``.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from glue_lp.config import DEFAULT_PROTOCOL, DEFAULT_TRAIN
from glue_lp.graph import crafting_graph
from glue_lp.metrics import average_precision, hits_at_k, metric_block, mrr, roc_auc
from glue_lp.models import (
    adamic_adar,
    common_neighbors,
    embed_gcn,
    preferential_attachment,
    score_pairs,
)
from glue_lp.protocol import assert_no_leakage, leakage_exists, split_graph

OUT = Path(__file__).resolve().parents[1] / "experiments"
SEEDS = (7, 11, 23)


def _cell(seed: int, mode: str) -> dict:
    nodes, edges = crafting_graph()
    sp = split_graph(
        nodes,
        edges,
        mode=mode,
        split="temporal",
        negatives="hard",
        seed=seed,
    )
    if mode == "valid":
        assert_no_leakage(sp.mp_edges, sp.positives)
    queries = [(e.source, e.target) for e in sp.positives] + list(sp.negatives)
    y = [1] * len(sp.positives) + [0] * len(sp.negatives)
    n = len(nodes)
    Z = embed_gcn(nodes, sp.mp_edges, seed=seed)
    gnn = score_pairs(Z, queries)
    aa = adamic_adar(n, sp.mp_edges, queries)
    cn = common_neighbors(n, sp.mp_edges, queries)
    pa = preferential_attachment(n, sp.mp_edges, queries)
    return {
        "seed": seed,
        "mode": mode,
        "n_pos": len(sp.positives),
        "n_neg": len(sp.negatives),
        "leakage": leakage_exists(sp.mp_edges, sp.positives),
        "gnn": metric_block(y, gnn),
        "aa": {"auc": roc_auc(y, aa), "ap": average_precision(y, aa)},
        "cn": {"auc": roc_auc(y, cn)},
        "pa": {"auc": roc_auc(y, pa)},
        "hits3_gnn": hits_at_k(y, gnn, 3),
        "mrr_gnn": mrr(y, gnn),
    }


def main() -> None:
    t0 = time.time()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rows = []
    for seed in SEEDS:
        for mode in ("valid", "leaky"):
            rows.append(_cell(seed, mode))

    def mean(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else float("nan")

    gnn_v = [r["gnn"]["auc"] for r in rows if r["mode"] == "valid"]
    gnn_l = [r["gnn"]["auc"] for r in rows if r["mode"] == "leaky"]
    out = {
        "label": "smoke",
        "aviso": (
            "smoke-synthetic-only; NOT a measured Cora/Citeseer/Pubmed result; "
            "do not cite as study primary AUC"
        ),
        "generator": "scripts/smoke_synthetic.py",
        "graph": "glue_lp.graph.crafting_graph",
        "n_nodes": 12,
        "seeds": list(SEEDS),
        "train_config_ref": {
            "hidden": DEFAULT_TRAIN.hidden,
            "lr": DEFAULT_TRAIN.lr,
            "note": "smoke uses NumPy embed_gcn, not torch TrainConfig epochs",
        },
        "protocol_ref": {
            "mode_default": DEFAULT_PROTOCOL.mode,
            "assert_invariant": DEFAULT_PROTOCOL.assert_invariant,
        },
        "utc_stamp": stamp,
        "seconds": round(time.time() - t0, 3),
        "rows": rows,
        "summary": {
            "gnn_valid_auc_mean": mean(gnn_v),
            "gnn_leaky_auc_mean": mean(gnn_l),
            "delta_leaky_minus_valid": mean(gnn_l) - mean(gnn_v),
            "all_valid_no_leakage": all(
                (not r["leakage"]) for r in rows if r["mode"] == "valid"
            ),
            "all_leaky_has_leakage": all(
                r["leakage"] for r in rows if r["mode"] == "leaky"
            ),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"smoke_{stamp}.json"
    # also keep a stable alias for CI / quick inspection
    alias = OUT / "smoke_latest.json"
    text = json.dumps(out, indent=2)
    path.write_text(text, encoding="utf-8")
    alias.write_text(text, encoding="utf-8")
    print("WROTE", path)
    print("WROTE", alias)
    print(json.dumps(out["summary"], indent=2))
    assert out["summary"]["all_valid_no_leakage"]
    assert out["summary"]["all_leaky_has_leakage"]


if __name__ == "__main__":
    main()
