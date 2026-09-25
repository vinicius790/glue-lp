"""Optional thin FastAPI surface (extras: [api]).

Read-only endpoints: health, protocol invariant on the synthetic tech tree,
and measured experiment summaries from JSON. Does NOT train models.

Routes
------
GET /health
GET /protocol/invariant   (alias: /protocol/check)
GET /experiments/summary  (alias list: /experiments)
GET /experiments/{key}
"""
from __future__ import annotations

from typing import Any

try:
    from fastapi import FastAPI, HTTPException
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "glue_lp.api requires FastAPI. Install with: pip install 'glue-lp[api]' "
        "or pip install fastapi uvicorn"
    ) from exc

from . import __version__
from .export import list_measured_summaries, load_summary
from .graph import crafting_graph
from .logging_util import log_kv, setup_logging
from .protocol import assert_no_leakage, leakage_exists, split_graph

log = setup_logging()

app = FastAPI(
    title="GLUE-LP protocol API",
    description=(
        "Optional read-only surface for health, invariant check on the synthetic "
        "tech tree, and measured experiment JSON summaries. Not required for core tests."
    ),
    version=__version__,
)


def _protocol_invariant_payload(seed: int = 7) -> dict[str, Any]:
    nodes, edges = crafting_graph()
    valid = split_graph(nodes, edges, mode="valid", seed=seed)
    assert_no_leakage(valid.mp_edges, valid.positives)
    leaky = split_graph(nodes, edges, mode="leaky", seed=seed)
    leaky_ok = leakage_exists(leaky.mp_edges, leaky.positives)
    log_kv(log, "protocol_invariant", seed=seed, valid_held=True, leaky_intersects=leaky_ok)
    return {
        "seed": seed,
        "invariant": "Q+ ∩ Emp = ∅ (valid)",
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
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "glue-lp", "mode": "read-only", "version": __version__}


@app.get("/protocol/invariant")
def protocol_invariant(seed: int = 7) -> dict[str, Any]:
    """Canonical invariant endpoint (synthetic tech tree; no Cora required)."""
    return _protocol_invariant_payload(seed)


@app.get("/protocol/check")
def protocol_check(seed: int = 7) -> dict[str, Any]:
    """Alias of /protocol/invariant (kept for backward compatibility)."""
    return _protocol_invariant_payload(seed)


@app.get("/experiments/summary")
def experiments_summary() -> dict[str, Any]:
    """Compact measured experiment headlines (read-only JSON)."""
    items = list_measured_summaries()
    return {"n": len(items), "experiments": items}


@app.get("/experiments")
def experiments() -> list[dict[str, Any]]:
    """List measured experiment summaries (read-only JSON)."""
    return list_measured_summaries()


@app.get("/experiments/{key}")
def experiment_summary(key: str) -> dict[str, Any]:
    if key == "summary":
        # defensive: FastAPI matches static /experiments/summary first, but keep safe
        return experiments_summary()
    try:
        return {"key": key, "summary": load_summary(key)}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
