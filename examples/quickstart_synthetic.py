#!/usr/bin/env python3
"""Quickstart sintético GLUE-LP (NumPy / stdlib) — sem Cora, sem GPU.

1. Assert do protocolo no tech tree (modo valid).
2. Imprime resumo da escada L1/L3/L4 a partir do JSON já medido
   (experiments/leakage_ladder_resumo.json). Se ausente, opcionalmente
   chama scripts/leakage_ladder.py.

Não grava AUC LINQS; não inventa métricas.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from glue_lp.graph import crafting_graph
from glue_lp.protocol import assert_no_leakage, leakage_exists, split_graph

RESUMO = ROOT / "experiments" / "leakage_ladder_resumo.json"
LADDER_SCRIPT = ROOT / "scripts" / "leakage_ladder.py"


def protocol_demo(seed: int = 7) -> dict:
    nodes, edges = crafting_graph()
    valid = split_graph(nodes, edges, mode="valid", seed=seed)
    assert_no_leakage(valid.mp_edges, valid.positives)
    leaky = split_graph(nodes, edges, mode="leaky", seed=seed)
    return {
        "seed": seed,
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "valid": {
            "n_mp": len(valid.mp_edges),
            "n_pos": len(valid.positives),
            "invariant_Qplus_cap_Emp_empty": True,
            "leakage_exists": leakage_exists(valid.mp_edges, valid.positives),
        },
        "leaky": {
            "n_mp": len(leaky.mp_edges),
            "n_pos": len(leaky.positives),
            "leakage_exists": leakage_exists(leaky.mp_edges, leaky.positives),
        },
        "message": "ok: valid satisfaz Q+ ∩ Emp = ∅; leaky intersecta de propósito",
    }


def load_ladder_resumo() -> dict | None:
    if RESUMO.is_file():
        return json.loads(RESUMO.read_text(encoding="utf-8"))
    return None


def ensure_resumo() -> dict:
    data = load_ladder_resumo()
    if data is not None:
        return data
    print(f"[info] {RESUMO.name} ausente — executando leakage_ladder.py …")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    subprocess.check_call([sys.executable, str(LADDER_SCRIPT)], cwd=str(ROOT), env=env)
    data = load_ladder_resumo()
    if data is None:
        raise SystemExit(f"falha: {RESUMO} não gerado")
    return data


def print_ladder(summary: dict) -> None:
    escada = summary.get("escada") or {}
    print("\n=== Escada sintética (JSON medido; L0/L2 não medidos) ===")
    print("fonte:", RESUMO.relative_to(ROOT) if RESUMO.is_file() else "(após re-run)")
    for level in ("L1", "L3", "L4_half_edge"):
        block = escada.get(level)
        if not isinstance(block, dict) or "mean" not in block:
            print(f"  {level}: (ausente neste JSON)")
            continue
        sd = block.get("sd")
        n = block.get("n")
        print(f"  {level}: mean={block['mean']:.4f}  sd={sd}  n={n}")
    print("  L0: proposto / não medido")
    print("  L2: proposto / não medido")
    deltas = [k for k in escada if str(k).startswith("delta_")]
    if deltas:
        print("deltas gravados:", ", ".join(deltas))


def main() -> int:
    demo = protocol_demo()
    print("=== Protocol assert (tech tree) ===")
    print(json.dumps(demo, indent=2, ensure_ascii=False))
    if not demo["leaky"]["leakage_exists"]:
        print("[warn] esperado leakage no modo leaky", file=sys.stderr)
        return 1
    summary = ensure_resumo()
    print_ladder(summary)
    print("\nPronto. Próximos passos: make test | make verify | docs/INDEX.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
