from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class TrainConfig:
    hidden: int = 32
    epochs: int = 50
    lr: float = 0.01
    weight_decay: float = 5e-4
    patience: int = 12
    dropout: float = 0.5
    neg_per_pos: int = 1
    seeds: tuple = (0, 1, 2, 3, 4)
