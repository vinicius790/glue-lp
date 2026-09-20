# Resultados medidos (nao inventados)

## Serie 1 — Cora, GCN denso, 50 epocas, 5 seeds, negativos 1:1

| Condicao | AUC |
|---|---|
| GCN valido uniforme | 0.708 ± 0.020 |
| GCN leaky uniforme | 0.826 ± 0.019 |
| Delta leaky-valido | +0.118 |
| GCN valido hard (CN) | 0.619 ± 0.009 |
| AA valido uniforme | 0.720 ± 0.007 |
| AA valido hard | 0.542 ± 0.016 |
| GCN indutivo 15% nos | 0.725 ± 0.012 |
| AA indutivo | 0.496 ± 0.001 |

## Serie 2 — GCN esparso, early stopping

| Corpus | Valido | Leaky | t pareado |
|---|---|---|---|
| Cora | 0.728 ± 0.020 | 0.782 ± 0.046 | 2.61 (gl=4, seed 4 inverte) |
| Citeseer | 0.733 ± 0.029 | 0.833 ± 0.030 | 5.77 |
| SAGE Cora valido | 0.577 ± 0.025 | — | — |

## Onda 3

| Condicao | Valido | Leaky | t |
|---|---|---|---|
| GAT Cora 5 seeds | 0.782 ± 0.025 | 0.876 ± 0.021 | 20.4 |
| GCN Pubmed 3 seeds | 0.886 ± 0.003 | 0.921 ± 0.009 | 7.75 gl=2 |
| GCN Cora degree-matched | 0.611 ± 0.033 | — | — |

Detalhe por seed do GAT/Pubmed: `wave3_log.txt`.
