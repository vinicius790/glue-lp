# GLUE-LP: um protocolo executavel para avaliar predicao de ligacoes com GNN sem entregar a aresta-alvo ao modelo

Manuscrito + pacote `glue_lp`. Convertido do PDF/DOCX. Figuras 1-7 nao estao neste Markdown.

## Resumo

Quem implementa predicao de ligacoes com GNN quase sempre deixa a aresta-alvo no message passing. O modelo ve o rotulo. A metrica sobe. Dong et al. e Li et al. descreveram o diagnostico. Aqui a exclusao e um assert: Q+ ∩ E_mp = ∅ no modo valido.

Serie 1, GCN denso, 50 epocas, Cora, 5 seeds: 0.708 ± 0.020 valido vs 0.826 ± 0.019 leaky (Δ +0.118).
Serie 2, GCN esparso + early stopping: Cora t = 2.61 (gl = 4, nao cruza 2,78); Citeseer t = 5.77 (Δ +0.100).
Hard: GCN 0.619. AA uniforme 0.720, AA hard 0.542.
Indutivo 15% nos: GCN 0.725; CN/AA ~0.50.
SAGE Cora valido: 0.577.
GAT Cora: 0.782 vs 0.876, t = 20.4.
Pubmed GCN, 3 seeds: 0.886 vs 0.921, t = 7.75, gl = 2.

Nao e SOTA. OGB nao rodou.

## Abstract

GLUE-LP makes target-edge exclusion an unskippable invariant. Leaky protocol adds 0.118 AUC on fixed-epoch Cora GCN. Early stopping replicates on Citeseer (t = 5.77) and is only suggestive on Cora (t = 2.61). No OGB. No new layer.

## 1. Problema

O default de tutorial reutiliza o Data do classificador de nos e deixa (u,v) em E_mp. O harness desta proposta recusa o run se o modo valido vazar. Heuristicas CN/AA/PA entram no mesmo Q.

## 2. Protocolo

Cora 2708/5278/1433. Citeseer 3312/4536/3703. Pubmed 19717/44324/500. Holdout 10% teste, 5% val.
Valido: E_mp = treino. Leaky: E_mp = treino∪val∪teste.
GCN 2 camadas hidden 32, Adam 0.01, wd 5e-4, BCE. Decoder produto interno.
Serie 1: A denso, 50 epocas. Serie 2: A esparso, early stopping paciencia 12.
GAT 1 cabeca. SAGE media plena da vizinhanca (sem amostragem).

## 3. Complexidade e stack

GCN esparso/epoca: O((m+n)h + n h d). Pubmed nao cabe em A denso.

### Ferramentas do artefato (o que o repo executa de fato)

| Ferramenta | Papel |
|---|---|
| Python 3.11+ | pacote glue_lp |
| PyTorch | GCN/SAGE/GAT, Adam, BCE, index_add_ |
| NumPy | splits, features, RNG |
| pytest | invariante, AUC de empate, shape GAT |
| matplotlib | figuras locais |
| ruff | lint opcional (`ruff.toml`) |
| Makefile | `make test` |
| GitHub Actions | pytest em push (`/.github/workflows/tests.yml`) |
| TrainConfig | hiperparametros congelados em `config.py` |
| scripts/check_protocol.py | CLI do assert no tech tree |

Fora do nucleo de proposito: PyG, DGL, FastAPI, CUDA obrigatoria, OGB loader.

## 4. Serie 1

| Condicao | AUC |
|---|---|
| GCN valido uniforme | 0.708 ± 0.020 |
| GCN leaky uniforme | 0.826 ± 0.019 |
| GCN valido hard | 0.619 ± 0.009 |
| AA valido uniforme | 0.720 ± 0.007 |
| AA valido hard | 0.542 ± 0.016 |
| GCN indutivo | 0.725 ± 0.012 |
| AA indutivo | 0.496 ± 0.001 |

## 5. Serie 2, GAT, Pubmed

Cora early stopping t=2.61 (seed 4 inverte). Citeseer t=5.77. GAT t=20.4. Pubmed t=7.75. Degree-matched GCN 0.611.

## 6. Limites

5 seeds. Hidden 32. Sem ano na aresta LINQS. OGB ausente. gl=2 no Pubmed.

## 7. Reproducao

```bash
pip install -r requirements-dev.txt
make test
PYTHONPATH=src python3 scripts/check_protocol.py
```

Grafos: ver `data/LINQS.md` (este arquivo substitui o antigo `data/README.md`).

## Referencias

Kipf & Welling 2017 https://arxiv.org/abs/1609.02907
Hamilton et al. 2017 https://arxiv.org/abs/1706.02216
Velickovic et al. 2018 https://arxiv.org/abs/1710.10903
Li et al. 2023 https://arxiv.org/abs/2306.10453
Dong et al. 2024 https://arxiv.org/abs/2306.00899
Sen et al. 2008 LINQS
