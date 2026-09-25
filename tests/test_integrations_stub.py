import pytest

from glue_lp.integrations import dgl_note, ogb_loader, pyg_adapter


def test_pyg_raises():
    with pytest.raises(NotImplementedError, match="proposed"):
        pyg_adapter.to_pyg_data()
    with pytest.raises(NotImplementedError, match="proposed"):
        pyg_adapter.from_pyg_data()


def test_dgl_raises():
    with pytest.raises(NotImplementedError, match="proposed"):
        dgl_note.to_dgl_graph()
    with pytest.raises(NotImplementedError, match="proposed"):
        dgl_note.from_dgl_graph()


def test_ogb_raises():
    with pytest.raises(NotImplementedError, match="not measured"):
        ogb_loader.load_ogbl_dataset("ogbl-ddi")
    with pytest.raises(NotImplementedError, match="not measured"):
        ogb_loader.load_ogbn_dataset("ogbn-arxiv")
