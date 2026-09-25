import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from glue_lp.graph import crafting_graph
from glue_lp.protocol import assert_no_leakage, leakage_exists, split_graph
from glue_lp.run import compare_protocols, run_experiment


def test_valid_nao_vaza():
    nodes, edges = crafting_graph()
    sp = split_graph(nodes, edges, mode="valid", split="temporal", seed=7)
    assert_no_leakage(sp.mp_edges, sp.positives)
    assert not leakage_exists(sp.mp_edges, sp.positives)


def test_leaky_vaza():
    nodes, edges = crafting_graph()
    sp = split_graph(nodes, edges, mode="leaky", split="temporal", seed=7)
    assert leakage_exists(sp.mp_edges, sp.positives)


def test_valid_raises_se_forcar_intersecao():
    nodes, edges = crafting_graph()
    sp = split_graph(nodes, edges, mode="valid", seed=1)
    with pytest.raises(AssertionError):
        assert_no_leakage(sp.mp_edges + list(sp.positives), sp.positives)


def test_relatorio_deterministico():
    a = run_experiment("valid", "temporal", "hard", 7)
    b = run_experiment("valid", "temporal", "hard", 7)
    assert a == b


def test_compare_tem_aviso():
    r = compare_protocols(7)
    assert r["aviso"] == "ilustrativo-sintetico-n=12"
    assert r["valid"]["invariant_held"] is True
