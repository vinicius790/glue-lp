# GLUE-LP

Protocolo valido: a aresta-alvo nao entra no message passing.

Resultados medidos:
- Cora GCN 50 epocas: 0.708 valido vs 0.826 leaky
- Citeseer: +0.100 AUC, t = 5.77
- GAT Cora: 0.782 vs 0.876, t = 20.4
- Pubmed GCN: 0.886 vs 0.921, t = 7.75

Stack: Python, PyTorch, NumPy, pytest.
