#!/usr/bin/env python3
"""Gera figuras 1-2 do protocolo (PNG local). Nao sobe PNG por esta API."""
from pathlib import Path
import matplotlib.pyplot as plt
FIG = Path(__file__).resolve().parents[1] / "figures"
FIG.mkdir(parents=True, exist_ok=True)

def fig_pipeline():
    fig, ax = plt.subplots(figsize=(8.2, 3.2))
    ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 3)
    labels = ["Grafo", "Split", "Mascara", "Negativos", "GCN/AA"]
    for i, t in enumerate(labels):
        x = 0.3 + i*1.9
        ax.add_patch(plt.Rectangle((x, 1.0), 1.6, 1.2, facecolor="#f4efe6", edgecolor="#1c1917"))
        ax.text(x+0.8, 1.6, t, ha="center", va="center")
    fig.savefig(FIG/"fig1_pipeline.png", dpi=140)
    plt.close()

if __name__ == "__main__":
    fig_pipeline()
    print("fig1 em", FIG)
