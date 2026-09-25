"""PyTorch Geometric adapter — PROPOSTO / NÃO MEDIDO.

GLUE-LP's measured runs use a minimal sparse message-passing stack in
``torch_models.py`` (index_add_ spmm), not torch_geometric. A future adapter
could wrap Planetoid / Data objects while preserving Q+ ∩ E_mp = ∅.

How to implement (future)
-------------------------
1. ``pip install torch torch_geometric``
2. Load Planetoid, convert ``edge_index`` → undirected edge list ``[m, 2]``.
3. Build feature matrix ``x`` float32 and call ``glue_lp.splits.random_edge_split``.
4. In mode ``valid``, set ``mp = train`` only; call ``assert_no_leakage``.
5. Do **not** put measured AUCs into experiments/*.json until a versioned runner
   writes them. Mark any draft numbers as proposto / não medido.
"""
from __future__ import annotations


def to_pyg_data(*_args, **_kwargs):  # pragma: no cover
    raise NotImplementedError(
        "PyG adapter is proposed / not measured. Measured encoders live in "
        "glue_lp.torch_models (SparseGCN, GraphSAGE, GAT) without torch_geometric. "
        "How-to: see module docstring and docs/TECNOLOGIAS.md."
    )


def from_pyg_data(*_args, **_kwargs):  # pragma: no cover
    raise NotImplementedError(
        "PyG adapter is proposed / not measured. Convert Planetoid graphs "
        "manually to the dict interface of glue_lp.data_citation, then use "
        "glue_lp.splits with mode='valid'. How-to: see module docstring."
    )
