#!/usr/bin/env python3
"""Relê JSON das séries e imprime as tabelas do artigo. Não treina."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"


def mean_std(vals):
    n = len(vals)
    m = sum(vals) / n
    var = sum((x - m) ** 2 for x in vals) / (n - 1) if n > 1 else 0.0
    return m, var ** 0.5


def main() -> None:
    w3 = json.loads((EXP / "wave3_gat_pubmed.json").read_text())
    rows = w3.get("rows", w3 if isinstance(w3, list) else [])
    print("=== wave3 (arquivo local) ===")
    for kind, mode in (("gat", "valid"), ("gat", "leaky"), ("gcn", "valid"), ("gcn", "leaky")):
        vals = [
            r["auc"]
            for r in rows
            if r.get("kind") == kind and r.get("mode") == mode and r.get("dataset") in ("cora", "pubmed", None)
        ]
        # print by dataset
    by = {}
    for r in rows:
        key = (r.get("dataset"), r.get("kind"), r.get("mode"), r.get("negatives", "uniform"))
        by.setdefault(key, []).append(float(r["auc"]))
    for key, vals in sorted(by.items(), key=lambda kv: str(kv[0])):
        m, s = mean_std(vals)
        print(f"{key}: n={len(vals)}  {m:.3f} ± {s:.3f}")

    ext = EXP / "extended_gcn_sage.json"
    if ext.exists():
        e = json.loads(ext.read_text())
        summary = e.get("summary")
        print("\n=== serie 2 summary ===")
        print(json.dumps(summary, indent=2)[:2000] if summary else "(sem summary; rows=%d)" % len(e.get("rows", [])))

    print("\nFonte: experiments/*.json do ZIP original. Nada foi retreinado.")


if __name__ == "__main__":
    main()
