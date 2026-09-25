"""DGL note — PROPOSTO / NÃO MEDIDO.

Deep Graph Library is out of the measured core. Documented here so readers know
we intentionally kept a single ABI (PyTorch + NumPy) for the Cora/Citeseer/Pubmed runs.

How to wire later (future)
--------------------------
1. ``pip install dgl`` (pick the CUDA/CPU wheel matching your torch).
2. Convert ``dgl.DGLGraph`` edges to a NumPy ``[m, 2]`` edge list.
3. Reuse ``glue_lp.splits`` / ``glue_lp.protocol`` so valid mode still asserts
   Q+ ∩ Emp = ∅ — do not bypass the harness.
4. Keep any DGL numbers out of experiments/*.json until measured by a runner.
"""
from __future__ import annotations


def to_dgl_graph(*_args, **_kwargs):  # pragma: no cover
    raise NotImplementedError(
        "DGL integration is proposed / not measured. GLUE-LP measured runs do not "
        "depend on dgl. How-to: see module docstring; docs/TECNOLOGIAS.md and "
        "docs/ARQUITETURA.md."
    )


def from_dgl_graph(*_args, **_kwargs):  # pragma: no cover
    raise NotImplementedError(
        "DGL integration is proposed / not measured. Convert DGLGraph edges to "
        "NumPy and use glue_lp.splits with mode='valid'."
    )
