"""GLUE-LP console CLI (stdlib argparse — no Typer required).

Entry point: ``glue-lp`` (see pyproject [project.scripts]).

Subcommands
-----------
protocol-check
    Assert Q+ ∩ Emp = ∅ on the synthetic tech tree (valid) and show leaky intersects.
verify-results
    Re-read measured experiments/*.json and print compact tables (no training).
leakage-ladder
    Run / re-print the synthetic L1/L3/L4 ladder (NumPy-only).
export-summary
    Dump a measured summary block as JSON (stdout or --out).
describe-config
    Print DEFAULT_TRAIN / DEFAULT_PROTOCOL / dataset specs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _cmd_protocol_check(args: argparse.Namespace) -> int:
    from .graph import crafting_graph
    from .protocol import assert_no_leakage, leakage_exists, split_graph

    nodes, edges = crafting_graph()
    valid = split_graph(nodes, edges, mode="valid", seed=args.seed)
    assert_no_leakage(valid.mp_edges, valid.positives)
    leaky = split_graph(nodes, edges, mode="leaky", seed=args.seed)
    leaky_ok = leakage_exists(leaky.mp_edges, leaky.positives)
    print(
        json.dumps(
            {
                "seed": args.seed,
                "valid": {
                    "n_mp": len(valid.mp_edges),
                    "n_pos": len(valid.positives),
                    "invariant_held": True,
                    "leakage": False,
                },
                "leaky": {
                    "n_mp": len(leaky.mp_edges),
                    "n_pos": len(leaky.positives),
                    "leakage": leaky_ok,
                },
                "message": "ok: valid without intersection; leaky with intersection",
            },
            indent=2,
        )
    )
    return 0 if leaky_ok else 1



def _print_summary_block(name: str, block: object, *, indent: str = "  ") -> None:
    """Print mean±std leaves; recurse one level for nested heuristic summaries."""
    if isinstance(block, dict) and "mean" in block:
        from .export import format_mean_std
        print(f"{indent}{name}: {format_mean_std(block)}")
    elif isinstance(block, dict) and "t" in block:
        print(
            f"{indent}{name}: t={block['t']:.3f} df={block.get('df')} "
            f"Δ={block.get('mean_diff')}"
        )
    elif isinstance(block, dict):
        # nested e.g. valid_unif -> {aa: {mean,std}, ...}
        print(f"{indent}{name}:")
        for sk, sv in block.items():
            _print_summary_block(sk, sv, indent=indent + "  ")
    elif isinstance(block, (int, float)):
        print(f"{indent}{name}: {block}")


def _cmd_verify_results(args: argparse.Namespace) -> int:
    from .export import KNOWN_FILES, experiments_dir, format_mean_std, load_json

    root = experiments_dir(args.exp_dir)
    print(f"=== GLUE-LP verify-results ({root}) ===")
    missing = []
    for key, fname in KNOWN_FILES.items():
        path = root / fname
        if not path.exists():
            missing.append(fname)
            print(f"[missing] {key}: {fname}")
            continue
        data = load_json(key, exp_dir=root)
        summary = data.get("summary") if isinstance(data, dict) else None
        if isinstance(summary, dict):
            print(f"\n--- {key} ({fname}) ---")
            if key == "serie1":
                print(
                    "  nota: AA/CN/PA ~0.99 no JSON bruto da série 1 são AUC "
                    "ingênua; headline do manuscrito = heuristics_rescored "
                    "(empate=½)."
                )
            for sk, sv in summary.items():
                _print_summary_block(sk, sv)
        elif isinstance(data, dict) and "escada" in data:
            print(f"\n--- {key} escada ---")
            for level, block in data["escada"].items():
                if isinstance(block, dict) and "mean" in block:
                    print(f"  {level}: {format_mean_std(block)}")
        else:
            print(f"[ok] {key}: {fname} ({path.stat().st_size} bytes)")
    print("\nFonte: experiments/*.json. Nada foi retreinado.")
    return 1 if missing else 0


def _cmd_leakage_ladder(args: argparse.Namespace) -> int:
    """Delegate to scripts/leakage_ladder.py logic when --run; else print JSON."""
    from .export import experiments_dir, format_mean_std, load_json

    root = experiments_dir(args.exp_dir)
    if args.run:
        # Import the script module path via subprocess-free exec of public API
        # Prefer reusing measured JSON unless --run forces regeneration.
        scripts = Path(__file__).resolve().parents[2] / "scripts" / "leakage_ladder.py"
        if not scripts.exists():
            print("scripts/leakage_ladder.py not found", file=sys.stderr)
            return 1
        import runpy

        sys.argv = [str(scripts)]  # script has no CLI args
        runpy.run_path(str(scripts), run_name="__main__")
        return 0

    data = load_json("leakage_ladder", exp_dir=root)
    print("=== leakage ladder (measured JSON) ===")
    for level, block in data.get("escada", {}).items():
        if isinstance(block, dict) and "mean" in block:
            print(f"  {level}: {format_mean_std(block)}")
    if args.json:
        print(json.dumps(data.get("escada", {}), indent=2))
    return 0


def _cmd_export_summary(args: argparse.Namespace) -> int:
    from .export import load_summary

    summary = load_summary(args.key, exp_dir=args.exp_dir)
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(text)
    return 0


def _cmd_describe_config(args: argparse.Namespace) -> int:
    from .config import (
        CITESEER,
        CORA,
        DEFAULT_PROTOCOL,
        DEFAULT_SCORERS,
        DEFAULT_TRAIN,
        MEASURED_LEAKAGE_LEVELS,
        PUBMED,
        ProtocolConfig,
        TrainConfig,
    )

    def dc(obj: Any) -> dict[str, Any]:
        if hasattr(obj, "__dataclass_fields__"):
            return {k: getattr(obj, k) for k in obj.__dataclass_fields__}
        return dict(obj)

    payload = {
        "TrainConfig": dc(DEFAULT_TRAIN),
        "ProtocolConfig": dc(DEFAULT_PROTOCOL),
        "ScorerSuite": {
            "encoders": list(DEFAULT_SCORERS.encoders),
            "heuristics": list(DEFAULT_SCORERS.heuristics),
        },
        "datasets": {
            "cora": dc(CORA),
            "citeseer": dc(CITESEER),
            "pubmed": dc(PUBMED),
        },
        "measured_leakage_levels": [lv.value for lv in MEASURED_LEAKAGE_LEVELS],
        "note": (
            "Frozen hyperparameters used in measured Cora/Citeseer/Pubmed runs. "
            "Do not silently change; re-AUC after code fixes is proposto/não medido."
        ),
    }
    if args.as_defaults:
        # prove round-trip constructibility
        assert TrainConfig(**{k: getattr(DEFAULT_TRAIN, k) for k in TrainConfig.__dataclass_fields__})
        assert ProtocolConfig()
    print(json.dumps(payload, indent=2, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="glue-lp",
        description="GLUE-LP — Graph Link-evaluation Under Exclusion (protocol CLI)",
    )
    from . import __version__  # local import keeps argparse build light
    p.add_argument("--version", action="version", version=f"glue-lp {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("protocol-check", help="Assert invariant on synthetic tech tree")
    sp.add_argument("--seed", type=int, default=7)
    sp.set_defaults(func=_cmd_protocol_check)

    sp = sub.add_parser("verify-results", help="Print measured JSON summaries (no train)")
    sp.add_argument("--exp-dir", default=None, help="Override experiments/ directory")
    sp.set_defaults(func=_cmd_verify_results)

    sp = sub.add_parser("leakage-ladder", help="Show or re-run synthetic L1/L3/L4 ladder")
    sp.add_argument("--exp-dir", default=None)
    sp.add_argument("--run", action="store_true", help="Re-execute scripts/leakage_ladder.py")
    sp.add_argument("--json", action="store_true", help="Also dump escada JSON")
    sp.set_defaults(func=_cmd_leakage_ladder)

    sp = sub.add_parser("export-summary", help="Export one measured summary as JSON")
    sp.add_argument("key", help="KNOWN_FILES key or filename (e.g. serie1, onda3)")
    sp.add_argument("--exp-dir", default=None)
    sp.add_argument("--out", default=None, help="Write to file instead of stdout")
    sp.set_defaults(func=_cmd_export_summary)

    sp = sub.add_parser("describe-config", help="Print TrainConfig / ProtocolConfig / datasets")
    sp.add_argument("--as-defaults", action="store_true", help="Also round-trip construct")
    sp.set_defaults(func=_cmd_describe_config)

    return p


def main(argv: list[str] | None = None) -> None:
    """Console-script entry: exits with subcommand status code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    raise SystemExit(int(args.func(args)))


def run(argv: list[str] | None = None) -> int:
    """Programmatic entry used by tests (returns int, does not SystemExit)."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    main()
