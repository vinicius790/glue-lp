#!/usr/bin/env python3
"""Serie 1 — Cora, 50 epocas, 5 seeds. Requer data/cora LINQS."""
from pathlib import Path
from glue_lp.data_citation import load_cora
from glue_lp.experiments import run_transductive_cell, save_json

def main():
    data = load_cora()
    rows = []
    for seed in range(5):
        for mode in ("valid", "leaky"):
            rows.append(run_transductive_cell(data, seed=seed, mode=mode, negatives="uniform", epochs=50, hidden=32, neg_per_pos=1))
    save_json({"rows": rows}, "cora_e1_e4.json")
    print("ok", len(rows))

if __name__ == "__main__":
    main()
