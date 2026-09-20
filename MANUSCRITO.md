# GLUE-LP: protocolo executavel para predicao de ligacoes com GNN sem entregar a aresta-alvo

## Resumo
Quem implementa link prediction com GNN costuma deixar a aresta-alvo no message passing. O modelo ve o rotulo. A metrica sobe.

Medido neste trabalho (nao inventado):
- Cora, GCN denso, 50 epocas, 5 seeds: AUC 0.708 ± 0.020 (valido) vs 0.826 ± 0.019 (leaky). Delta +0.118.
- Cora, GCN esparso, early stopping: t pareado = 2.61 (gl = 4; nao cruza 2.78).
- Citeseer, GCN esparso, early stopping: 0.733 vs 0.833, delta +0.100, t = 5.77.
- GAT Cora, 5 seeds: 0.782 vs 0.876, t = 20.4.
- Pubmed (19717 nos), GCN esparso, 3 seeds: 0.886 vs 0.921, t = 7.75.
- Negativos hard (CN): GCN 0.619. Adamic-Adar uniforme 0.720, hard 0.542.
- Indutivo (15% nos ocultos): GCN 0.725; CN/AA ~0.50.
- GraphSAGE medio no Cora valido: 0.577.

Nao e SOTA. Nao rodou OGB. Decoder: produto interno.

## Protocolo
Holdout 10% teste, 5% val, resto treino.
Valido: E_mp = E_train. Se Q+ intersecta E_mp, AssertionError.
Leaky: E_mp = train+val+test. Existe so como controle.

## Stack
Python 3.11+, PyTorch (Adam lr=0.01, wd=5e-4, BCE), NumPy, pytest.
GCN 2 camadas hidden 32. SAGE media da vizinhanca. GAT 1 cabeca.

## Como rodar (quando os dados LINQS estiverem em data/)
```
pip install -r requirements.txt
PYTHONPATH=src python3 -m pytest -q
PYTHONPATH=src python3 scripts/run_e1_e4.py
```

## Referencias
Kipf & Welling 2017; Hamilton et al. 2017; Velickovic et al. 2018;
Li et al. 2023 (arXiv:2306.10453); Dong et al. 2024 (arXiv:2306.00899);
Sen et al. 2008 (Cora/Citeseer LINQS).
