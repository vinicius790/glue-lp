"""OGB loader stub — PROPOSTO / NÃO MEDIDO.

Open Graph Benchmark (Hu et al., 2020) splits were never run in this package.
Any call raises so no one can mistake a stub for measured Hits@K / AUC.

How to implement (future)
-------------------------
1. ``pip install ogb``
2. Use ``ogb.linkproppred.LinkPropPredDataset`` for ogbl-* tasks.
3. Map OGB split dicts onto GLUE-LP's train/val/test edge arrays.
4. Enforce ``mode='valid'`` → message-passing edges exclude test (and preferably
   val) positives; call ``assert_no_leakage`` before encoding.
5. Report Hits@K / MRR with ``glue_lp.metrics`` on the **same** Q as heuristics.
6. Until a runner writes JSON under experiments/, mark all OGB figures as
   proposto / não medido. **No OGB numbers appear in this repo's measured JSON.**
"""
from __future__ import annotations


def load_ogbl_dataset(name: str = "ogbl-ddi", *_args, **_kwargs):  # pragma: no cover
    raise NotImplementedError(
        f"OGB loader for {name!r} is proposed / not measured. "
        "No OGB numbers appear in experiments/*.json. Future work: wrap "
        "ogb.linkproppred while enforcing Q+ ∩ Emp = ∅ in valid mode. "
        "How-to: see module docstring; docs/TECNOLOGIAS.md."
    )


def load_ogbn_dataset(name: str = "ogbn-arxiv", *_args, **_kwargs):  # pragma: no cover
    raise NotImplementedError(
        f"OGB node dataset {name!r} is out of scope for link-prediction protocol "
        "and is not measured. Use citation loaders in glue_lp.data_citation for "
        "Cora/Citeseer/Pubmed."
    )
