#!/usr/bin/env python3
"""Aplica helpers de glue_lp.stats a arrays .values JÁ PRESENTES nos JSON.

Lê experiments/*.json (ou --file). Não treina, não inventa seeds, não grava
AUC novas. Se um par valid/leaky não existir, apenas reporta e segue.

Uso:
  PYTHONPATH=src python3 scripts/apply_stats_from_json.py
  PYTHONPATH=src python3 scripts/apply_stats_from_json.py --file experiments/extended_gcn_sage.json
  PYTHONPATH=src python3 scripts/apply_stats_from_json.py --n-boot 2000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from glue_lp.stats import bootstrap_ci_delta, cohens_dz, paired_t, paired_t_logit

DEFAULT_FILES = (
    "cora_e1_e4.json",
    "extended_gcn_sage.json",
    "wave3_gat_pubmed.json",
    "heuristics_rescored.json",
)

# Pares (label, chave_a, chave_b) sob summary — só aplicados se ambos tiverem .values
KNOWN_PAIRS: dict[str, list[tuple[str, str, str]]] = {
    "extended_gcn_sage.json": [
        ("cora_gcn_leaky_minus_valid", "cora_gcn_leaky", "cora_gcn_valid"),
        ("citeseer_gcn_leaky_minus_valid", "citeseer_gcn_leaky", "citeseer_gcn_valid"),
    ],
    "wave3_gat_pubmed.json": [
        ("cora_gat_leaky_minus_valid", "cora_gat_leaky", "cora_gat_valid"),
        ("pubmed_gcn_leaky_minus_valid", "pubmed_gcn_leaky", "pubmed_gcn_valid"),
    ],
    "cora_e1_e4.json": [
        ("E1_gcn_leaky_minus_valid", "E1_gcn_leaky_auc", "E1_gcn_valid_auc"),
    ],
}


def _values(summary: dict, key: str) -> list[float] | None:
    block = summary.get(key)
    if not isinstance(block, dict):
        return None
    vals = block.get("values")
    if not isinstance(vals, list) or not vals:
        return None
    if not all(isinstance(x, (int, float)) for x in vals):
        return None
    return [float(x) for x in vals]


def _discover_pairs(summary: dict) -> list[tuple[str, str, str]]:
    """Heurística: chaves *leaky* com par *valid* irmão (mesma prefixo)."""
    keys = [k for k, v in summary.items() if isinstance(v, dict) and "values" in v]
    found: list[tuple[str, str, str]] = []
    for k in keys:
        if "leaky" not in k:
            continue
        sibling = k.replace("leaky", "valid")
        if sibling in summary and sibling != k:
            found.append((f"{k}_minus_{sibling}", k, sibling))
    return found


def analyze_file(path: Path, n_boot: int, alpha: float, seed: int) -> None:
    print(f"\n=== {path.name} ===")
    if not path.is_file():
        print("  [skip] arquivo ausente")
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data.get("summary") if isinstance(data, dict) else None
    if not isinstance(summary, dict):
        print("  [skip] sem bloco summary")
        return

    pairs = list(KNOWN_PAIRS.get(path.name, []))
    if not pairs:
        pairs = _discover_pairs(summary)
    if not pairs:
        print("  [skip] nenhum par valid/leaky com .values")
        return

    for label, key_a, key_b in pairs:
        a = _values(summary, key_a)
        b = _values(summary, key_b)
        if a is None or b is None:
            print(f"  [{label}] ausente: precisa summary.{key_a}.values e .{key_b}.values")
            continue
        if len(a) != len(b):
            print(f"  [{label}] comprimentos diferem n_a={len(a)} n_b={len(b)} — não fabricar")
            continue
        if len(a) < 2:
            print(f"  [{label}] n={len(a)} < 2 — stats pareadas indisponíveis")
            continue

        pt = paired_t(a, b)
        dz = cohens_dz(a, b)
        ci = bootstrap_ci_delta(a, b, n_boot=n_boot, alpha=alpha, seed=seed)
        pt_logit = paired_t_logit(a, b)
        print(f"  [{label}] n={len(a)}  (a={key_a}, b={key_b})")
        print(
            f"    paired_t: mean_diff={pt['mean_diff']:.6f}  t={pt['t']:.4f}  "
            f"df={pt['df']}  se={pt['se']:.6f}"
        )
        print(f"    cohens_dz: {dz:.6f}")
        print(
            f"    bootstrap_CI_delta ({1-alpha:.0%} perc., n_boot={ci['n_boot']}): "
            f"point={ci['point']:.6f}  lo={ci['lo']:.6f}  hi={ci['hi']:.6f}"
        )
        print(
            f"    paired_t_logit: mean_diff={pt_logit['mean_diff']:.6f}  "
            f"t={pt_logit['t']:.4f}  df={pt_logit['df']}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--file",
        action="append",
        dest="files",
        help="JSON relativo à raiz ou a experiments/ (repetível)",
    )
    ap.add_argument("--exp-dir", type=Path, default=ROOT / "experiments")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0, help="semente do bootstrap (reprodutível)")
    args = ap.parse_args()

    paths: list[Path]
    if args.files:
        paths = []
        for f in args.files:
            p = Path(f)
            if not p.is_file():
                alt = args.exp_dir / f
                p = alt if alt.is_file() else ROOT / f
            paths.append(p)
    else:
        paths = [args.exp_dir / name for name in DEFAULT_FILES]

    print(
        "apply_stats_from_json — somente arrays existentes; "
        "não grava JSON; LINQS pode ser pré-correção ES."
    )
    for path in paths:
        analyze_file(path, n_boot=args.n_boot, alpha=args.alpha, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
