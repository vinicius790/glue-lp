#!/usr/bin/env python3
"""Escada de vazamento L0-L4 e enumeracao exaustiva de pares no tech tree
sintetico (12 nos). Roda 100% em Python/NumPy puro -- nao depende de PyTorch
nem dos dados LINQS, e por isso e o UNICO experimento deste repositorio em
que a AUC pode ser reportada de forma EXATA (sem ruido de amostragem de
negativos): com n=12, C(12,2)=66 pares cobre o espaco inteiro.

Niveis executados aqui (ver Sec. 3.3 do manuscrito):
  L1 = mp = treino                     ("valido" deste projeto)
  L3 = mp = treino + positivos-alvo    ("leaky" deste projeto)
  L4 = mp = treino + 1 aresta-alvo visivel em UMA SO direcao (bug de simetria)
L0 (vazamento tambem no treino) e L2 (treino+val) nao se aplicam a este
harness sintetico, que nao tem split de validacao proprio nem supervisao de
treino separada da mesma forma que o pipeline PyTorch (Cora/Citeseer/Pubmed);
ficam registrados como N/A e como pendencia explicita para o pipeline real.
"""
from __future__ import annotations
import itertools
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from glue_lp.graph import crafting_graph
from glue_lp.metrics import roc_auc
from glue_lp.models import (
    adamic_adar, common_neighbors, embed_gcn, embed_gcn_directed_half,
    preferential_attachment, score_pairs,
)
from glue_lp.protocol import edge_key, split_graph

OUT = Path(__file__).resolve().parents[1] / "experiments"
SEEDS = list(range(1, 11))  # 10 seeds: grafo de 12 nos, custo desprezivel


def run_ladder(seed, split="temporal", negatives="hard"):
    nodes, edges = crafting_graph()
    sp_valid = split_graph(nodes, edges, split=split, negatives=negatives, mode="valid", seed=seed)
    sp_leaky = split_graph(nodes, edges, split=split, negatives=negatives, mode="leaky", seed=seed)
    queries = [(e.source, e.target) for e in sp_valid.positives] + list(sp_valid.negatives)
    y = [1] * len(sp_valid.positives) + [0] * len(sp_valid.negatives)
    n = len(nodes)

    def scores(mp):
        Z = embed_gcn(nodes, mp, seed=seed)
        return {
            "gnn_auc": roc_auc(y, score_pairs(Z, queries)),
            "aa_auc": roc_auc(y, adamic_adar(n, mp, queries)),
            "cn_auc": roc_auc(y, common_neighbors(n, mp, queries)),
            "pa_auc": roc_auc(y, preferential_attachment(n, mp, queries)),
        }

    out = {"seed": seed, "L1": scores(sp_valid.mp_edges), "L3": scores(sp_leaky.mp_edges)}

    gnn_l4 = []
    for e in sp_valid.positives:
        u, v = e.source, e.target
        Zh = embed_gcn_directed_half(nodes, sp_valid.mp_edges, u, v, seed=seed)
        gnn_l4.append(float(Zh[u] @ Zh[v]))
    Zbase = embed_gcn(nodes, sp_valid.mp_edges, seed=seed)
    gnn_l4_full = gnn_l4 + score_pairs(Zbase, list(sp_valid.negatives))
    out["L4_half_edge"] = {
        "gnn_auc": roc_auc(y, gnn_l4_full),
        "nota": "AA/CN/PA sob L4 nao recalculados: pela invariancia de 1a "
                "ordem (Sec. 4 do manuscrito), coincidem com L1 exceto por "
                "efeito de 2a ordem no grau de vizinhos comuns.",
    }
    return out


def run_exhaustive(seed, split="temporal", negatives="hard"):
    nodes, edges = crafting_graph()
    n = len(nodes)
    sp_valid = split_graph(nodes, edges, split=split, negatives=negatives, mode="valid", seed=seed)
    sp_leaky = split_graph(nodes, edges, split=split, negatives=negatives, mode="leaky", seed=seed)
    train_keys = {edge_key(e.source, e.target) for e in sp_valid.mp_edges}
    pos_keys = {edge_key(e.source, e.target) for e in sp_valid.positives}

    Q, y = [], []
    for u, v in itertools.combinations(range(n), 2):
        k = edge_key(u, v)
        if k in train_keys:
            continue  # ja observado no treino: nao e alvo de predicao
        Q.append((u, v))
        y.append(1 if k in pos_keys else 0)

    out = {
        "seed": seed,
        "n_pares_possiveis_C_12_2": len(list(itertools.combinations(range(n), 2))),
        "n_pares_avaliados": len(Q),
        "n_pos": sum(y),
        "n_neg": len(y) - sum(y),
    }
    for name, sp in (("valido", sp_valid), ("leaky", sp_leaky)):
        mp = sp.mp_edges
        Z = embed_gcn(nodes, mp, seed=seed)
        out[name] = {
            "gnn_auc_exata": roc_auc(y, score_pairs(Z, Q)),
            "aa_auc_exata": roc_auc(y, adamic_adar(n, mp, Q)),
            "cn_auc_exata": roc_auc(y, common_neighbors(n, mp, Q)),
            "pa_auc_exata": roc_auc(y, preferential_attachment(n, mp, Q)),
        }
    return out


def _mean_sd(xs):
    xs = list(xs)
    return {"mean": round(st.mean(xs), 4), "sd": round(st.pstdev(xs), 4) if len(xs) > 1 else 0.0, "n": len(xs)}


def main():
    ladder = [run_ladder(s) for s in SEEDS]
    exhaustive = [run_exhaustive(s) for s in SEEDS]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "leakage_ladder_sintetico.json").write_text(json.dumps({"seeds": SEEDS, "runs": ladder}, indent=2))
    (OUT / "exhaustive_sintetico.json").write_text(json.dumps({"seeds": SEEDS, "runs": exhaustive}, indent=2))

    summary = {"seeds": SEEDS, "escada": {}, "exaustivo": {}}
    for level in ("L1", "L3", "L4_half_edge"):
        summary["escada"][level] = _mean_sd(r[level]["gnn_auc"] for r in ladder)
    for scorer in ("aa_auc", "cn_auc", "pa_auc"):
        summary["escada"][f"delta_{scorer}_L3_menos_L1"] = _mean_sd(
            r["L3"][scorer] - r["L1"][scorer] for r in ladder)
    summary["escada"]["delta_gnn_auc_L3_menos_L1"] = _mean_sd(
        r["L3"]["gnn_auc"] - r["L1"]["gnn_auc"] for r in ladder)
    summary["escada"]["delta_gnn_auc_L4_menos_L1"] = _mean_sd(
        r["L4_half_edge"]["gnn_auc"] - r["L1"]["gnn_auc"] for r in ladder)

    for scorer in ("gnn_auc_exata", "aa_auc_exata", "cn_auc_exata", "pa_auc_exata"):
        summary["exaustivo"][f"{scorer}_valido"] = _mean_sd(r["valido"][scorer] for r in exhaustive)
        summary["exaustivo"][f"{scorer}_leaky"] = _mean_sd(r["leaky"][scorer] for r in exhaustive)
    summary["exaustivo"]["n_pares_avaliados_media"] = round(
        st.mean(r["n_pares_avaliados"] for r in exhaustive), 1)

    (OUT / "leakage_ladder_resumo.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
