#!/usr/bin/env python3
"""CLI: quebra se o modo valido vazar. Nao precisa de Cora."""
from glue_lp.graph import crafting_graph
from glue_lp.protocol import assert_no_leakage, leakage_exists, split_graph

def main():
    nodes, edges = crafting_graph()
    ok = split_graph(nodes, edges, mode="valid", seed=7)
    assert_no_leakage(ok.mp_edges, ok.positives)
    leak = split_graph(nodes, edges, mode="leaky", seed=7)
    assert leakage_exists(leak.mp_edges, leak.positives)
    print("ok: valido sem intersecao; leaky com intersecao")

if __name__ == "__main__":
    main()
