"""Frozen hyperparameters. Measured runs used these values — do not silently change.

``TrainConfig``, ``ProtocolConfig`` and ``DatasetSpec`` are the **single source**
of hyperparameters / dataset identity for runners under ``scripts/``. Import
``DEFAULT_TRAIN`` rather than duplicating magic numbers.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .types import LeakageLevel, NegativePolicy, ProtocolMode, ScorerName, SplitKind


@dataclass(frozen=True)
class TrainConfig:
    """Encoder training hyperparameters used in measured Cora/Citeseer/Pubmed runs."""

    hidden: int = 32
    epochs: int = 50  # série 1 (fixed); série 2/onda 3 use epochs_cap + patience
    epochs_cap: int = 70  # série 2 sparse + early stopping
    epochs_cap_wave3: int = 40
    epochs_cap_pubmed: int = 25
    lr: float = 0.01
    weight_decay: float = 5e-4  # pass to train_encoder(..., weight_decay=...)
    patience: int = 12
    dropout: float = 0.5  # pass to train_encoder(..., dropout=...) when overriding
    gat_dropout: float = 0.4  # documented; GAT class default=0.4 — not yet passed via train_encoder
    neg_per_pos: int = 1
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    pubmed_seeds: tuple[int, ...] = (0, 1, 2)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProtocolConfig:
    """Protocol regime for a single experimental cell."""

    mode: ProtocolMode = "valid"
    split: SplitKind = "random"
    negatives: NegativePolicy = "uniform"
    test_frac: float = 0.10
    val_frac: float = 0.05
    inductive_node_frac: float = 0.15
    assert_invariant: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetSpec:
    """Dataset identity and reported sizes (LINQS; see data/LINQS.md)."""

    name: str
    n_nodes: int
    n_edges: int
    n_features: int
    source: str = "LINQS"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Canonical dataset specs matching measured JSON headers / manuscript.
CORA = DatasetSpec("cora", 2708, 5278, 1433)
CITESEER = DatasetSpec("citeseer", 3312, 4536, 3703)
PUBMED = DatasetSpec("pubmed", 19717, 44324, 500)

DATASETS: dict[str, DatasetSpec] = {
    CORA.name: CORA,
    CITESEER.name: CITESEER,
    PUBMED.name: PUBMED,
}

DEFAULT_TRAIN = TrainConfig()
DEFAULT_PROTOCOL = ProtocolConfig()

# Leakage ladder levels that have measured numbers in experiments/leakage_ladder_*.json
MEASURED_LEAKAGE_LEVELS: tuple[LeakageLevel, ...] = (
    LeakageLevel.L1,
    LeakageLevel.L3,
    LeakageLevel.L4,
)


@dataclass(frozen=True)
class ScorerSuite:
    """Which scorers share the same Q in a cell."""

    encoders: tuple[ScorerName, ...] = ("gcn",)
    heuristics: tuple[ScorerName, ...] = ("cn", "aa", "pa")


DEFAULT_SCORERS = ScorerSuite()
