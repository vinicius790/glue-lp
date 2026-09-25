"""Optional NetworkX helpers for the synthetic crafting graph.

Install: ``pip install 'glue-lp[analysis]'`` (networkx).
If networkx is missing, public functions raise a clear ImportError / return None
from the ``try_*`` helpers — never silently invent graph statistics.
"""
from __future__ import annotations

from typing import Any

from .graph import crafting_graph, edge_key

try:
    import networkx as nx  # type: ignore
except ImportError:  # pragma: no cover
    nx = None  # type: ignore


def networkx_available() -> bool:
    return nx is not None


def crafting_nx_graph():
    """Build an undirected NetworkX graph of the synthetic tech tree."""
    if nx is None:
        raise ImportError(
            "networkx is required for graph_analysis. "
            "Install with: pip install 'glue-lp[analysis]' or pip install networkx"
        )
    nodes, edges = crafting_graph()
    g = nx.Graph()
    for n in nodes:
        g.add_node(n.id, name=n.name, kind=n.kind, unlock=n.unlock)
    seen: set[tuple[int, int]] = set()
    for e in edges:
        key = edge_key(e.source, e.target)
        if key in seen:
            continue
        seen.add(key)
        g.add_edge(e.source, e.target, time=e.time, relation=e.relation)
    return g


def degree_histogram(g=None) -> dict[str, Any]:
    """Degree counts for the crafting graph (or a provided undirected graph)."""
    if g is None:
        g = crafting_nx_graph()
    degrees = [d for _, d in g.degree()]
    hist: dict[int, int] = {}
    for d in degrees:
        hist[d] = hist.get(d, 0) + 1
    return {
        "n_nodes": g.number_of_nodes(),
        "n_edges": g.number_of_edges(),
        "degrees": degrees,
        "histogram": dict(sorted(hist.items())),
        "mean_degree": (sum(degrees) / len(degrees)) if degrees else 0.0,
    }


def clustering_summary(g=None) -> dict[str, Any]:
    """Local clustering coefficients on the crafting graph."""
    if g is None:
        g = crafting_nx_graph()
    local = nx.clustering(g)
    vals = list(local.values())
    avg = float(sum(vals) / len(vals)) if vals else 0.0
    return {
        "local": {str(k): float(v) for k, v in local.items()},
        "average_clustering": avg,
        "transitivity": float(nx.transitivity(g)),
    }


def analyze_crafting_graph() -> dict[str, Any]:
    """Full optional analysis bundle (degree hist + clustering)."""
    g = crafting_nx_graph()
    return {
        "backend": "networkx",
        "degree": degree_histogram(g),
        "clustering": clustering_summary(g),
        "aviso": "synthetic-tech-tree-n=12; ilustrativo; nao e estudo principal",
    }


def try_analyze_crafting_graph() -> dict[str, Any] | None:
    """Like analyze_crafting_graph but returns None if networkx is missing."""
    if nx is None:
        return None
    return analyze_crafting_graph()
