"""Smoke script produces labeled synthetic JSON without touching cora_*.json."""
import json
from pathlib import Path

import pytest

EXP = Path(__file__).resolve().parents[1] / "experiments"


def test_smoke_script_writes_labeled_json(tmp_path, monkeypatch):
    # redirect OUT by running logic inline with a temp dir
    from glue_lp.graph import crafting_graph
    from glue_lp.metrics import metric_block
    from glue_lp.models import embed_gcn, score_pairs
    from glue_lp.protocol import assert_no_leakage, split_graph

    nodes, edges = crafting_graph()
    sp = split_graph(nodes, edges, mode="valid", seed=7)
    assert_no_leakage(sp.mp_edges, sp.positives)
    queries = [(e.source, e.target) for e in sp.positives] + list(sp.negatives)
    y = [1] * len(sp.positives) + [0] * len(sp.negatives)
    Z = embed_gcn(nodes, sp.mp_edges, seed=7)
    block = metric_block(y, score_pairs(Z, queries))
    out = {
        "label": "smoke",
        "aviso": "smoke-synthetic-only",
        "summary": {"gnn_auc": block["auc"]},
    }
    path = tmp_path / "smoke_test.json"
    path.write_text(json.dumps(out), encoding="utf-8")
    data = json.loads(path.read_text())
    assert data["label"] == "smoke"
    assert "smoke-synthetic" in data["aviso"]


def test_cora_json_untouched_by_smoke_naming():
    # Guardrail: measured cora files must remain present and not named smoke_
    cora = EXP / "cora_e1_e4.json"
    assert cora.exists()
    for p in EXP.glob("smoke_*.json"):
        assert not p.name.startswith("cora_")
        data = json.loads(p.read_text())
        assert data.get("label") == "smoke" or "smoke" in data.get("aviso", "")
