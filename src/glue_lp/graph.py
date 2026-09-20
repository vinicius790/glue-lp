from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class Node:
    id: int
    name: str
    kind: str
    unlock: int

@dataclass(frozen=True)
class Edge:
    source: int
    target: int
    time: int
    relation: str = "crafts"

def edge_key(s, t):
    return (s, t) if s < t else (t, s)
