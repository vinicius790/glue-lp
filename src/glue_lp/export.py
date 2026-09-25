"""Read-only helpers to load measured experiment JSON summaries."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .logging_util import setup_logging

log = setup_logging()

# Default experiments directory relative to package root (…/GLUE-LP/experiments)
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXP_DIR = _PACKAGE_ROOT / "experiments"

KNOWN_FILES = {
    "serie1": "cora_e1_e4.json",
    "serie2": "extended_gcn_sage.json",
    "onda3": "wave3_gat_pubmed.json",
    "heuristics": "heuristics_rescored.json",
    "leakage_ladder": "leakage_ladder_resumo.json",
    "leakage_ladder_full": "leakage_ladder_sintetico.json",
    "sintetico": "results_sintetico.json",
    "exaustivo": "exhaustive_sintetico.json",
}


def experiments_dir(path: Path | str | None = None) -> Path:
    return Path(path) if path is not None else DEFAULT_EXP_DIR


def load_json(name_or_file: str, *, exp_dir: Path | str | None = None) -> dict[str, Any]:
    """Load an experiment JSON by short key (KNOWN_FILES) or filename."""
    root = experiments_dir(exp_dir)
    fname = KNOWN_FILES.get(name_or_file, name_or_file)
    path = root / fname
    if not path.exists():
        raise FileNotFoundError(f"experiment JSON not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    log.debug("loaded %s (%d bytes)", path.name, path.stat().st_size)
    return data


def load_summary(name_or_file: str, *, exp_dir: Path | str | None = None) -> dict[str, Any]:
    """Return the ``summary`` block when present, else the whole document."""
    data = load_json(name_or_file, exp_dir=exp_dir)
    if isinstance(data, dict) and "summary" in data:
        return data["summary"]
    if isinstance(data, dict) and "escada" in data:
        return data  # leakage_ladder_resumo
    return data


def list_measured_summaries(*, exp_dir: Path | str | None = None) -> list[dict[str, Any]]:
    """Compact list of measured experiment summaries for API / docs."""
    root = experiments_dir(exp_dir)
    out: list[dict[str, Any]] = []
    for key, fname in KNOWN_FILES.items():
        path = root / fname
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        entry: dict[str, Any] = {"key": key, "file": fname, "bytes": path.stat().st_size}
        if isinstance(data, dict):
            if "summary" in data and isinstance(data["summary"], dict):
                # keep only top-level summary keys (no per-seed arrays dump)
                entry["summary_keys"] = sorted(data["summary"].keys())
                # attach small numeric headlines when present
                headlines = {}
                for sk, sv in data["summary"].items():
                    if isinstance(sv, dict) and "mean" in sv and "std" in sv:
                        headlines[sk] = {"mean": sv["mean"], "std": sv["std"]}
                    elif isinstance(sv, dict) and "t" in sv:
                        headlines[sk] = {"t": sv["t"], "df": sv.get("df"), "mean_diff": sv.get("mean_diff")}
                    elif isinstance(sv, (int, float)):
                        headlines[sk] = sv
                entry["headlines"] = headlines
            if "escada" in data:
                entry["escada"] = data["escada"]
            if "exaustivo" in data:
                entry["exaustivo_keys"] = sorted(data["exaustivo"].keys())
            if "dataset" in data:
                entry["dataset"] = data["dataset"]
            if "seconds" in data:
                entry["seconds"] = data["seconds"]
        out.append(entry)
    return out


def format_mean_std(block: dict[str, Any], *, digits: int = 3) -> str:
    mean = float(block["mean"])
    std = float(block.get("std", block.get("sd", 0.0)))
    return f"{mean:.{digits}f} ± {std:.{digits}f}"
