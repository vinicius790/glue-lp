# GLUE-LP — entrega excelente (2026-09-25)

## Destaques vs. rodada anterior
| | Antes | Agora |
|---|---|---|
| ARTIGO.md | ~4k palavras / 9 pp PDF | **10 221 palavras / 22 pp PDF** |
| SpotTarget | “Dong et al.” (errado) | **Zhu et al., WSDM 2024** |
| AUC LINQS | sem label | **[pré-correção ES]** em banner + tabelas |
| Código | básico | CLI `glue-lp`, 47 testes, API opcional |
| Literatura | ~17 refs no PDF | **~44 no artigo** + 33 anotadas em RELATED_WORK |
| Figs | 1–7 | **+ fig8 arquitetura, fig9 escada** |

## Integridade
Nenhum AUC/t/Δ de Cora/Citeseer/Pubmed foi inventado ou alterado.
Escada L sintética permanece medida (NumPy).
Re-run pós-fix = P0 antes de submissão.

## Como abrir
1. Descompactar `GLUE-LP-excelente.zip`
2. Ler `GLUE-LP/GLUE-LP-artigo-cientifico.pdf`
3. Fonte canônica longa: `docs/ARTIGO.md`
4. `PYTHONPATH=src python3 -m pytest -q`
