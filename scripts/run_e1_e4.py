#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from pathlib import Path

from glue_lp.data_cora import load_cora
from glue_lp.experiments import (
    run_inductive_cell,
    run_transductive_cell,
    save_json,
    summarize,
)
from glue_lp.config import DEFAULT_TRAIN
from glue_lp.run import compare_protocols

CFG = DEFAULT_TRAIN
SEEDS = list(CFG.seeds)
EPOCHS = CFG.epochs
HIDDEN = CFG.hidden
NEG = CFG.neg_per_pos
OUT = Path(__file__).resolve().parents[1] / "experiments"


def main() -> None:
    t0 = time.time()
    data = load_cora()
    log = {
        "dataset": "cora-linqs",
        "n": data["n"],
        "n_edges": int(len(data["edges"])),
        "n_features": data["n_features"],
        "epochs": EPOCHS,
        "hidden": HIDDEN,
        "seeds": SEEDS,
        "encoder": "GCN-2L-SGD-Adam-BCE",
        "aviso_estudo": "Cora link-prediction; sem split temporal nativo",
        "cells": [],
    }

    for seed in SEEDS:
        for mode in ("valid", "leaky"):
            cell = run_transductive_cell(
                data,
                seed=seed,
                mode=mode,
                negatives="uniform",
                epochs=EPOCHS,
                hidden=HIDDEN,
                neg_per_pos=NEG,
            )
            cell["experiment"] = "E1"
            log["cells"].append(cell)
            print("E1", seed, mode, cell["metrics"]["gcn"]["auc"], flush=True)

        cell = run_transductive_cell(
            data,
            seed=seed,
            mode="valid",
            negatives="hard",
            epochs=EPOCHS,
            hidden=HIDDEN,
            neg_per_pos=NEG,
        )
        cell["experiment"] = "E2"
        log["cells"].append(cell)
        print("E2", seed, "hard", cell["metrics"]["gcn"]["auc"], flush=True)

        cell = run_inductive_cell(
            data, seed=seed, epochs=EPOCHS, hidden=HIDDEN, neg_per_pos=NEG
        )
        cell["experiment"] = "E-inductive"
        log["cells"].append(cell)
        print("IND", seed, cell["metrics"]["gcn"]["auc"], flush=True)

    e1_valid = [c for c in log["cells"] if c.get("experiment") == "E1" and c["mode"] == "valid"]
    e1_leaky = [c for c in log["cells"] if c.get("experiment") == "E1" and c["mode"] == "leaky"]
    e2_hard = [c for c in log["cells"] if c.get("experiment") == "E2"]
    e_ind = [c for c in log["cells"] if c.get("experiment") == "E-inductive"]

    log["summary"] = {
        "E1_gcn_valid_auc": summarize(e1_valid, "gcn", "auc"),
        "E1_gcn_leaky_auc": summarize(e1_leaky, "gcn", "auc"),
        "E1_aa_valid_auc": summarize(e1_valid, "aa", "auc"),
        "E1_aa_leaky_auc": summarize(e1_leaky, "aa", "auc"),
        "E2_gcn_hard_auc": summarize(e2_hard, "gcn", "auc"),
        "E2_aa_hard_auc": summarize(e2_hard, "aa", "auc"),
        "E3_valid_uniform": {
            m: summarize(e1_valid, m, "auc") for m in ("gcn", "cn", "aa", "pa")
        },
        "E3_valid_hard": {
            m: summarize(e2_hard, m, "auc") for m in ("gcn", "cn", "aa", "pa")
        },
        "E_inductive_gcn_auc": summarize(e_ind, "gcn", "auc"),
        "E_inductive_aa_auc": summarize(e_ind, "aa", "auc"),
    }
    delta = (
        log["summary"]["E1_gcn_leaky_auc"]["mean"]
        - log["summary"]["E1_gcn_valid_auc"]["mean"]
    )
    log["summary"]["E1_delta_leaky_minus_valid"] = float(delta)

    syn = []
    for seed in (7, 11, 23, 0, 1):
        for split in ("temporal", "random"):
            syn.append(compare_protocols(seed, split, "hard") | {"cell": f"{seed}-{split}-hard"})
            syn.append(compare_protocols(seed, split, "uniform") | {"cell": f"{seed}-{split}-uniform"})
    log["synthetic_e4"] = syn
    log["seconds"] = round(time.time() - t0, 1)
    path = save_json(log, "cora_e1_e4.json")
    print("WROTE", path, "sec", log["seconds"], flush=True)
    print(json.dumps(log["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
