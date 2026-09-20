# GLUE-LP: um protocolo executavel para avaliar predicao de ligacoes com GNN sem entregar a aresta-alvo ao modelo

Convertido do PDF e do DOCX. Figuras do PDF nao entram neste Markdown.

## Resumo

Quem implementa predicao de ligacoes com GNN quase sempre deixa a aresta-alvo no message passing. O modelo ve o rotulo. A metrica sobe.

Cora, GCN 50 epocas, 5 seeds: AUC 0.708 ± 0.020 valido vs 0.826 ± 0.019 leaky (Δ = +0.118).
Cora early stopping: Δ +0.054, t = 2.61, gl = 4 (nao cruza 2,78).
Citeseer early stopping: 0.733 vs 0.833, Δ +0.100, t = 5.77.
Hard: GCN 0.619 ± 0.009. AA uniforme 0.720, AA hard 0.542.
Indutivo: GCN 0.725; CN/AA ~0.50.
SAGE Cora: 0.577 ± 0.025.
GAT Cora: 0.782 vs 0.876, t = 20.4.
Pubmed GCN: 0.886 vs 0.921, t = 7.75, gl = 2.

## Abstract

GLUE-LP makes target-edge exclusion an unskippable invariant. No OGB. No new layer.

## 1. Por que este texto existe

O atalho e deixar (u,v) no message passing enquanto se pede para prever (u,v). O assert do modo valido quebra se Q+ intersecta E_mp.

## 2. Protocolo

Cora 2708 nos / 5278 arestas. Citeseer 3312 / 4536. Holdout 10% teste, 5% val.
Valido: MP = treino. Leaky: MP = treino+val+teste.
GCN 2 camadas hidden 32, Adam, BCE.

## 3. Stack

Python 3.11+, PyTorch, NumPy, pytest. Decoder produto interno.
GCN esparso: O((m+n)h + n h d) por epoca.

## 4. Serie 1 Cora

| Condicao | Modelo | AUC |
| --- | --- | --- |
| valido uniforme | GCN | 0.708 ± 0.020 |
| leaky uniforme | GCN | 0.826 ± 0.019 |
| valido hard | GCN | 0.619 ± 0.009 |
| valido uniforme | AA | 0.720 ± 0.007 |
| valido hard | AA | 0.542 ± 0.016 |
| indutivo | GCN | 0.725 ± 0.012 |
| indutivo | AA | 0.496 ± 0.001 |

## 5. Serie 2, GAT, Pubmed

Citeseer t = 5.77. GAT Cora t = 20.4. Pubmed t = 7.75. Degree-matched GCN 0.611.

## 6-10. Limites

LINQS sem tempo. OGB nao rodou. 5 seeds. Hidden 32.
Citeseer ~10 pontos de atalho. Pubmed ~3. GAT Cora ~9.

## Referencias

Kipf & Welling 2017 https://arxiv.org/abs/1609.02907
Hamilton et al. 2017 https://arxiv.org/abs/1706.02216
Velickovic et al. 2018 https://arxiv.org/abs/1710.10903
Li et al. 2023 https://arxiv.org/abs/2306.10453
Dong et al. 2024 https://arxiv.org/abs/2306.00899
Sen et al. 2008 LINQS
