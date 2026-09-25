"""Optional NetworkX analysis — skip if networkx missing."""
import pytest

from glue_lp import graph_analysis as ga


def test_networkx_available_flag():
    assert isinstance(ga.networkx_available(), bool)


@pytest.mark.skipif(not ga.networkx_available(), reason="networkx not installed")
def test_degree_and_clustering():
    bundle = ga.analyze_crafting_graph()
    assert bundle["degree"]["n_nodes"] == 12
    assert bundle["degree"]["n_edges"] >= 1
    assert "average_clustering" in bundle["clustering"]
    assert bundle["aviso"].startswith("synthetic")


def test_try_analyze_without_crash():
    result = ga.try_analyze_crafting_graph()
    if ga.networkx_available():
        assert result is not None and "degree" in result
    else:
        assert result is None


def test_import_error_message_without_nx(monkeypatch):
    if ga.networkx_available():
        pytest.skip("networkx present — cannot test ImportError path")
    with pytest.raises(ImportError, match="networkx"):
        ga.crafting_nx_graph()
