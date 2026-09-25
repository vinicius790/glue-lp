"""Smoke: domain types and config import without torch."""
from glue_lp.config import (
    CORA,
    CITESEER,
    DATASETS,
    PUBMED,
    DEFAULT_PROTOCOL,
    DEFAULT_TRAIN,
    ProtocolConfig,
    TrainConfig,
)
from glue_lp.types import (
    EdgeSplit,
    GraphBundle,
    LeakageLevel,
    ProtocolSnapshot,
    RunResult,
)


def test_train_config_frozen_measured_defaults():
    cfg = TrainConfig()
    assert cfg.hidden == 32
    assert cfg.lr == 0.01
    assert cfg.weight_decay == 5e-4
    assert cfg.patience == 12
    assert cfg.seeds == (0, 1, 2, 3, 4)
    assert DEFAULT_TRAIN.hidden == 32
    assert cfg.to_dict()["epochs_cap_pubmed"] == 25


def test_dataset_specs_match_manuscript():
    assert CORA.n_nodes == 2708 and CORA.n_edges == 5278 and CORA.n_features == 1433
    assert CITESEER.n_nodes == 3312
    assert PUBMED.n_nodes == 19717
    assert DATASETS["cora"] is CORA


def test_leakage_levels():
    assert LeakageLevel.L1.value == "L1"
    assert LeakageLevel.L4.value == "L4"


def test_protocol_config_defaults():
    p = ProtocolConfig()
    assert p.mode == "valid"
    assert p.assert_invariant is True
    assert DEFAULT_PROTOCOL.to_dict()["test_frac"] == 0.10


def test_graph_bundle_and_run_result_constructible():
    g = GraphBundle(name="toy", n=2, n_features=1, n_edges=1, x=None, edges=None)
    assert g.name == "toy"
    r = RunResult(dataset="toy", seed=0, mode="valid", kind="gcn", auc=0.5)
    assert r.auc == 0.5
    s = EdgeSplit(
        train=None, val=None, test=None, mp_edges=None, negatives=None,
        mode="valid", split="random", seed=0,
    )
    assert s.mode == "valid"


def test_protocol_snapshot():
    snap = ProtocolSnapshot(mode="valid", held=True, n_mp=3, n_pos=2, seed=7)
    assert "Emp" in snap.invariant or "∅" in snap.invariant
    assert snap.held is True
