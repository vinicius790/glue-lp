#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from glue_lp.graph import crafting_graph
from glue_lp.protocol import split_graph

FIG = Path(__file__).resolve().parents[1] / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.25,
})


def fig_protocol():
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    boxes = [
        (0.2, 1.3, "Grafo\nobservado"),
        (2.1, 1.3, "Split\nregime"),
        (4.0, 1.3, "Mascara\nE_mp cap Q+ = 0"),
        (6.0, 1.3, "Negativos\nunif | hard"),
        (8.0, 1.3, "GCN / CN\nAA / PA"),
    ]
    for i, (x, y, t) in enumerate(boxes):
        ax.add_patch(plt.Rectangle((x, y), 1.7, 1.6, fill=True, facecolor="#f4efe6",
                                   edgecolor="#1c1917", lw=1.2))
        ax.text(x + 0.85, y + 0.8, t, ha="center", va="center", fontsize=9)
        if i < len(boxes) - 1:
            ax.annotate("", xy=(boxes[i + 1][0] - 0.05, 2.1),
                        xytext=(x + 1.7, 2.1),
                        arrowprops=dict(arrowstyle="->", color="#7a2e3a", lw=1.4))
    ax.set_title("GLUE-LP — pipeline (invariante no centro)", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_pipeline.png", dpi=160)
    plt.close()


def fig_leakage():
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))
    for ax, title, extra in (
        (axes[0], "Válido: alvo fora de E_mp", False),
        (axes[1], "Leaky: alvo em E_mp", True),
    ):
        pts = {"u": (0.3, 0.55), "v": (0.7, 0.55), "a": (0.5, 0.85), "b": (0.5, 0.25)}
        for k, (x, y) in pts.items():
            ax.scatter([x], [y], s=400, c="#f4efe6", edgecolors="#1c1917", zorder=3)
            ax.text(x, y, k, ha="center", va="center", fontsize=10, zorder=4)
        ax.plot([0.3, 0.5], [0.55, 0.85], color="#1c1917")
        ax.plot([0.7, 0.5], [0.55, 0.85], color="#1c1917")
        ax.plot([0.3, 0.5], [0.55, 0.25], color="#1c1917")
        if extra:
            ax.plot([0.3, 0.7], [0.55, 0.55], color="#7a2e3a", lw=2.5)
            ax.text(0.5, 0.48, "alvo visível", ha="center", color="#7a2e3a", fontsize=8)
        else:
            ax.plot([0.3, 0.7], [0.55, 0.55], color="#7a2e3a", lw=1.5, ls="--")
            ax.text(0.5, 0.48, "alvo oculto", ha="center", color="#7a2e3a", fontsize=8)
        ax.set_xlim(0.15, 0.85)
        ax.set_ylim(0.12, 0.98)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=10)
    fig.suptitle("Figura 2. Diferença operacional do protocolo", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_protocolo.png", dpi=160)
    plt.close()


def fig_crafting():
    nodes, edges = crafting_graph()
    sp = split_graph(nodes, edges, split="temporal", negatives="hard", mode="valid", seed=7)
    pos = {(e.source, e.target) for e in sp.positives}
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    # layout by unlock
    xs, ys = {}, {}
    buckets: dict[int, list] = {}
    for nd in nodes:
        buckets.setdefault(nd.unlock, []).append(nd)
    for lv, group in buckets.items():
        for i, nd in enumerate(group):
            xs[nd.id] = lv
            ys[nd.id] = i - (len(group) - 1) / 2
    kinds = {"resource": "#b08968", "material": "#7a2e3a", "tool": "#1d4e89",
             "station": "#3f6f4f", "goal": "#111111"}
    for e in edges:
        style = dict(color="#7a2e3a", lw=2.0) if (e.source, e.target) in pos or (e.target, e.source) in pos else dict(color="#1c1917", lw=1.0, alpha=0.7)
        ax.annotate("", xy=(xs[e.target], ys[e.target]), xytext=(xs[e.source], ys[e.source]),
                    arrowprops=dict(arrowstyle="->", **style))
    for nd in nodes:
        ax.scatter([xs[nd.id]], [ys[nd.id]], s=280, c=kinds[nd.kind], zorder=3)
        ax.text(xs[nd.id], ys[nd.id] + 0.28, nd.name, ha="center", fontsize=7)
    ax.set_title("Figura 3. Tech tree sintético — arestas vermelhas = Q+ temporal")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_techtree.png", dpi=160)
    plt.close()


def fig_placeholder_cora():
    """Barras vazias substituídas quando o JSON existir."""
    p = Path(__file__).resolve().parents[1] / "experiments" / "cora_e1_e4.json"
    if not p.exists():
        return
    import json
    log = json.loads(p.read_text())
    s = log["summary"]
    labels = ["GCN válido", "GCN leaky", "AA válido", "GCN hard", "GCN indutivo"]
    means = [
        s["E1_gcn_valid_auc"]["mean"],
        s["E1_gcn_leaky_auc"]["mean"],
        s["E1_aa_valid_auc"]["mean"],
        s["E2_gcn_hard_auc"]["mean"],
        s["E_inductive_gcn_auc"]["mean"],
    ]
    stds = [
        s["E1_gcn_valid_auc"]["std"],
        s["E1_gcn_leaky_auc"]["std"],
        s["E1_aa_valid_auc"]["std"],
        s["E2_gcn_hard_auc"]["std"],
        s["E_inductive_gcn_auc"]["std"],
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=4, color="#7a2e3a", ecolor="#1c1917")
    ax.set_xticks(x, labels, rotation=15)
    ax.set_ylim(0.4, 1.02)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Figura 4. Cora — média ± desvio, 5 seeds")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_cora_auc.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    fig_protocol()
    fig_leakage()
    fig_crafting()
    fig_placeholder_cora()
    print("figures", list(FIG.glob("*.png")))
