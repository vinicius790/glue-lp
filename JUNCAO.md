# Junção dos dois ZIPs (2026-09-24)

## Fontes

| Pacote | O que entra |
|---|---|
| `GLUE-LP-unificado.zip` | PDF/DOCX, figuras 1–7, JSON das séries Cora/Citeseer/Pubmed, `build_manuscript.py`, `verify_results.py`, loaders LINQS |
| `glue-lp-revisado.zip` | `REVISAO.md`, `docs/ARTIGO.md` expandido, `leakage_ladder.py`, JSON L0–L4 no tech tree, correções em `models.py` / `run.py` / `stats.py` / `torch_models.py` |

## Regra de conflito

- Documento diagramado e figuras: unificado (e PDF de 11 páginas se disponível).
- Números Cora/Citeseer/Pubmed: JSON do unificado. Não foram retreinados nesta junção.
- Código do encoder sintético e do `train_encoder`: versão revisada (ReLU só na oculta; AA/CN/PA reais; `pos_val` sempre usado no early stopping).
- `docs/ARTIGO.md`: versão revisada (mais longa).

## O que o leitor não deve misturar

Os AUC de Cora no PDF foram medidos com o `train_encoder` anterior. A revisão documenta o bug e pede reexecução; esta junção **não inventa** AUC nova.
