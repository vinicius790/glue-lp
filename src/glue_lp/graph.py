from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
Kind = Literal["resource", "material", "tool", "station", "goal"]

@dataclass(frozen=True)
class Node:
    id: int
    name: str
    kind: Kind
    unlock: int

@dataclass(frozen=True)
class Edge:
    source: int
    target: int
    time: int
    relation: Literal["crafts", "requires", "unlocks"] = "crafts"

def crafting_graph():
    nodes = [
        Node(0, "Madeira", "resource", 0), Node(1, "Pedra", "resource", 1),
        Node(2, "Carvao", "resource", 2), Node(3, "Ferro", "resource", 5),
        Node(4, "Tabuas", "material", 1), Node(5, "Gravetos", "material", 2),
        Node(6, "Fornalha", "station", 4), Node(7, "Picareta madeira", "tool", 3),
        Node(8, "Picareta pedra", "tool", 4), Node(9, "Picareta ferro", "tool", 7),
        Node(10, "Lingote", "material", 6), Node(11, "Diamante", "goal", 8),
    ]
    e = Edge
    edges = [e(0,4,1), e(4,5,2), e(4,7,3,"requires"), e(5,7,3,"requires"),
        e(1,8,4,"requires"), e(5,8,4,"requires"), e(7,8,4,"unlocks"),
        e(1,6,4,"requires"), e(8,6,4,"unlocks"), e(3,10,6), e(6,10,6,"requires"),
        e(2,10,6,"requires"), e(10,9,7,"requires"), e(5,9,7,"requires"),
        e(8,9,7,"unlocks"), e(9,11,8,"unlocks"), e(10,11,8,"requires"),
        e(4,6,4,"requires"), e(0,5,2), e(1,7,3,"unlocks")]
    return nodes, edges

def edge_key(s, t):
    return (s, t) if s < t else (t, s)
