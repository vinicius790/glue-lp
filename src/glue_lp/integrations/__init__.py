"""Optional future integrations — stubs only; nothing here is measured.

PyG / DGL / OGB adapters are proposed. Calling them raises NotImplementedError
with a clear message and a short how-to in each module docstring so accidental
use cannot be mistaken for experimental results.

Install hints (when implementing — not required for core tests)::

    pip install torch torch_geometric   # pyg_adapter
    pip install dgl                     # dgl_note
    pip install ogb                     # ogb_loader
"""
from . import dgl_note, ogb_loader, pyg_adapter

__all__ = ["pyg_adapter", "dgl_note", "ogb_loader"]
