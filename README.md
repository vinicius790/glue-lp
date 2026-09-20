# GLUE-LP

Predicao de ligacoes sem a aresta-alvo no message passing.

## Estrutura

```
MANUSCRITO.md              texto do artigo
experiments/RESULTADOS.md  tabela de AUC medida
src/glue_lp/               pacote Python
  protocol.py              invariante e split temporal/aleatorio
  splits.py                holdout Cora e split indutivo
  data_citation.py         Cora / Citeseer LINQS
  torch_gcn.py             GCN denso (serie 1)
  torch_models.py          GCN esparso, SAGE, GAT
  metrics.py               AUC Mann-Whitney, AP, Hits, MRR
  heuristics.py            CN, Adamic-Adar, PA
  graph.py                 tech tree de 12 nos
  run.py                   experimento sintetico
scripts/                   runners E1, serie 2, onda 3
tests/                     pytest da invariante
requirements.txt
```

## Resultados
Ver experiments/RESULTADOS.md

## Rodar testes
```
pip install -r requirements.txt
PYTHONPATH=src python3 -m pytest -q
```

PDF/DOCX/PNG e tarballs LINQS nao entram nesta API (binario).
