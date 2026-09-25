"""Named domain types for GLUE-LP (data shapes first).

These types document the protocol vocabulary. Frozen dataclasses are preferred
for configuration and bundles; TypedDicts describe JSON / dict payloads that
cross module boundaries (loaders, experiment rows).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, NotRequired, TypedDict

ProtocolMode = Literal["valid", "leaky"]
ScorerName = Literal["gcn", "sage", "gat", "cn", "aa", "pa"]
SplitKind = Literal["random", "temporal", "inductive-node"]
NegativePolicy = Literal["uniform", "hard", "degree"]


class LeakageLevel(str, Enum):
    """Leakage ladder on the synthetic tech tree (measured L1/L3/L4).

    L0 — proposed / not measured as a separate coded level in this package.
    L1 — mp = train only ("valid").
    L2 — proposed intermediate (e.g. train+val); not measured on the ladder JSON.
    L3 — mp = train + positives ("leaky").
    L4 — mp = train + half-edge of one target (one direction visible).
    """

    L0 = "L0"
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"
    L4 = "L4"


@dataclass(frozen=True)
class GraphBundle:
    """In-memory citation / synthetic graph used by runners."""

    name: str
    n: int
    n_features: int
    n_edges: int
    x: Any  # np.ndarray float32 [n, d]
    edges: Any  # np.ndarray int64 [m, 2] or list[Edge]
    y: Any | None = None


@dataclass(frozen=True)
class EdgeSplit:
    """Holdout produced by splits / protocol.split_graph."""

    train: Any
    val: Any | None
    test: Any
    mp_edges: Any
    negatives: Any
    mode: ProtocolMode
    split: SplitKind
    seed: int
    cutoff: int | None = None


@dataclass(frozen=True)
class RunResult:
    """One experimental cell (one seed × mode × encoder)."""

    dataset: str
    seed: int
    mode: ProtocolMode
    kind: ScorerName
    auc: float
    epochs_run: int | None = None
    best_val: float | None = None
    n_test: int | None = None
    extras: dict[str, Any] = field(default_factory=dict)


class MetricBlock(TypedDict, total=False):
    auc: float
    ap: float
    hits3: float
    hits10: float
    mrr: float


class SummaryStat(TypedDict, total=False):
    mean: float
    std: float
    sd: float
    values: list[float]
    n: int


class PairedTStat(TypedDict, total=False):
    mean_diff: float
    t: float
    df: int
    n: int
    se: float


class ExperimentSummary(TypedDict, total=False):
    """Loose shape for experiments/*.json summary blocks."""

    dataset: str
    seconds: float
    summary: dict[str, Any]
    rows: list[dict[str, Any]]
    cells: list[dict[str, Any]]
    escada: dict[str, SummaryStat]
    exaustivo: dict[str, Any]
    aviso: NotRequired[str]


@dataclass(frozen=True)
class ProtocolSnapshot:
    """Frozen view of the sacred valid-mode invariant for logging / API."""

    mode: ProtocolMode
    invariant: str = "Q+ ∩ Emp = ∅"
    held: bool = True
    n_mp: int = 0
    n_pos: int = 0
    seed: int = 0

