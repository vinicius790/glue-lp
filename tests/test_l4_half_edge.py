"""L4 half-edge: message-passing adj is one-way for the target edge."""
from glue_lp.graph import crafting_graph
from glue_lp.models import embed_gcn_directed_half
from glue_lp.protocol import split_graph, undirected_adj


def test_l4_adds_only_one_direction():
    nodes, edges = crafting_graph()
    sp = split_graph(nodes, edges, mode="valid", split="temporal", seed=7)
    assert len(sp.positives) >= 1
    e = sp.positives[0]
    u, v = e.source, e.target
    mp_keys = {(min(x.source, x.target), max(x.source, x.target)) for x in sp.mp_edges}
    assert (min(u, v), max(u, v)) not in mp_keys
    adj = undirected_adj(len(nodes), sp.mp_edges)
    adj_half = [set(s) for s in adj]
    adj_half[u] = set(adj_half[u]) | {v}
    assert v in adj_half[u]
    assert u not in adj_half[v]
    Z = embed_gcn_directed_half(nodes, sp.mp_edges, u, v, seed=7)
    assert Z.shape[0] == len(nodes)
