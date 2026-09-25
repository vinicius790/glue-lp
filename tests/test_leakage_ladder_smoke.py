"""Smoke: leakage ladder script still produces consistent structure."""
import json
from pathlib import Path

from glue_lp.graph import crafting_graph
from glue_lp.protocol import leakage_exists, split_graph

EXP = Path(__file__).resolve().parents[1] / "experiments"


def test_ladder_json_keys():
    data = json.loads((EXP / "leakage_ladder_resumo.json").read_text())
    for level in ("L1", "L3", "L4_half_edge"):
        assert level in data["escada"]
        assert "mean" in data["escada"][level]
    assert "exaustivo" in data
    assert abs(data["escada"]["delta_pa_auc_L3_menos_L1"]["mean"] - 0.4648) < 1e-4


def test_l1_valid_l3_leaky_on_techtree():
    nodes, edges = crafting_graph()
    l1 = split_graph(nodes, edges, mode="valid", split="temporal", negatives="hard", seed=7)
    assert not leakage_exists(l1.mp_edges, l1.positives)
    l3 = split_graph(nodes, edges, mode="leaky", split="temporal", negatives="hard", seed=7)
    assert leakage_exists(l3.mp_edges, l3.positives)
