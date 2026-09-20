# GLUE-LP

Protocolo de predicao de ligacoes em que a aresta-alvo **nao** entra no message passing.

## Ler o artigo
- Texto completo: [docs/ARTIGO.md](docs/ARTIGO.md)
- Tabela de AUC: [experiments/RESULTADOS.md](experiments/RESULTADOS.md)
- Log GAT/Pubmed: [experiments/wave3_log.txt](experiments/wave3_log.txt)

## Layout
```
docs/ARTIGO.md          manuscrito (do PDF/DOCX)
src/glue_lp/            pacote
  protocol.py           invariante
  splits.py             holdout, indutivo, hard, degree-match
  data_citation.py      Cora, Citeseer, Pubmed
  torch_gcn.py          GCN denso serie 1
  torch_models.py       GCN esparso, SAGE, GAT
  metrics.py            AUC Mann-Whitney
  heuristics.py         CN, AA, PA
  experiments.py        celula transdutiva
  graph.py / run.py     tech tree n=12
scripts/run_e1_e4.py    serie 1
scripts/run_extended.py serie 2 (este commit: runner real)
tests/                  invariante + AUC + GAT shape
```

## Instalar e testar
```bash
pip install -r requirements.txt
PYTHONPATH=src python3 -m pytest -q
```

Cora/Citeseer/Pubmed LINQS vao em `data/` (nao versionados aqui).

## Resultado principal
Citeseer GCN: leaky +0.100 AUC (t=5.77). GAT Cora: +0.094 (t=20.4).
Cora GCN com early stopping: t=2.61, uma seed inverte.
