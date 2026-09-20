from __future__ import annotations
from dataclasses import asdict, dataclass
from .graph import crafting_graph
from .metrics import average_precision, hits_at_k, mrr, roc_auc
from .models import embed_gcn, score_pairs
from .protocol import leakage_exists, split_graph

@dataclass
class Report:
    mode: str
    split: str
    negatives: str
    seed: int
    n_nodes: int
    n_mp: int
    n_pos: int
    n_neg: int
    leakage: bool
    invariant_held: bool
    gnn_auc: float
    gnn_ap: float
    gnn_hits3: float
    gnn_mrr: float
    aa_auc: float

def run_experiment(mode, split, negatives, seed):
    nodes, edges = crafting_graph()
    sp = split_graph(nodes, edges, split=split, negatives=negatives, mode=mode, seed=seed)
    queries = [(e.source, e.target) for e in sp.positives] + list(sp.negatives)
    y = [1] * len(sp.positives) + [0] * len(sp.negatives)
    Z = embed_gcn(nodes, sp.mp_edges, seed=seed)
    gnn = score_pairs(Z, queries)
    leak = leakage_exists(sp.mp_edges, sp.positives)
    return Report(mode=mode, split=split, negatives=negatives, seed=seed, n_nodes=len(nodes),
        n_mp=len(sp.mp_edges), n_pos=len(sp.positives), n_neg=len(sp.negatives), leakage=leak,
        invariant_held=(not leak) if mode == "valid" else True,
        gnn_auc=roc_auc(y, gnn), gnn_ap=average_precision(y, gnn),
        gnn_hits3=hits_at_k(y, gnn, 3), gnn_mrr=mrr(y, gnn), aa_auc=0.0)

def compare_protocols(seed=7, split="temporal", negatives="hard"):
    valid = run_experiment("valid", split, negatives, seed)
    leaky = run_experiment("leaky", split, negatives, seed)
    return {"valid": asdict(valid), "leaky": asdict(leaky),
            "delta_gnn_auc": leaky.gnn_auc - valid.gnn_auc,
            "aviso": "ilustrativo-sintetico-n=12"}
