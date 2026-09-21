# Corpora LINQS usados no artigo

Este diretorio **nao** versiona os grafos. Os loaders em `glue_lp.data_citation` esperam:

```
data/cora/cora.content
data/cora/cora.cites
data/citeseer/citeseer.content
data/citeseer/citeseer.cites
data/Pubmed-Diabetes/data/Pubmed-Diabetes.NODE.paper.tab
data/Pubmed-Diabetes/data/Pubmed-Diabetes.DIRECTED.cites.tab
```

Fonte: Sen et al., AI Magazine 2008 (projeto LINQS).

Depois do loader deste repo:

| Corpus | Nos | Arestas nao dirigidas | Features |
|---|---|---|---|
| Cora | 2708 | 5278 | 1433 |
| Citeseer | 3312 | 4536 (orfas fora) | 3703 |
| Pubmed-Diabetes | 19717 | 44324 | 500 |

Nao ha timestamp de aresta em Cora/Citeseer LINQS. Split temporal so no tech tree sintetico.
