# Correções aplicadas na auditoria (2026-09-25)

Nenhum AUC/t/Δ experimental foi inventado. PDF/DOCX regenerados após as mudanças de manuscrito.

## P0 corrigidos

| ID | Fix |
|---|---|
| P0-1 | Versão **0.5.0** em `CITATION.cff`, `docs/ARTIGO.md` (+ `ARTIGO-COMPLETO.md`), footer `scripts/build_manuscript.py`, `scripts/_gen/write_artigo.py` / `write_artigo3.py`. |
| P0-2 | `scripts/_gen/write_artigo.py`: “Dong et al. (SpotTarget)” → **Zhu et al.** (3 ocorrências). Docs canônicos já estavam corretos. |

## P1 corrigidos

| ID | Fix |
|---|---|
| P1-1 | `FIGURES` em `build_manuscript.py` inclui `fig8_arquitetura.png` e `fig9_escada_vazamento.png`; menções fig1–fig9 no ARTIGO/README; **DOCX+PDF regenerados** (raiz + `docs/`). |
| P1-2 | CLI `verify-results`: `_print_summary_block` para summaries aninhados; nota sobre AA ingênua da série 1 vs `heuristics_rescored`. |
| P1-3 | `cli.py` / `api.py` usam `glue_lp.__version__`. |
| P1-4 | `docs/README.md`: entrada `ARTIGO-COMPLETO.md` + fig8/fig9; README raiz `fig1`–`fig9`. |
| P1-5 | `train_encoder(..., weight_decay=5e-4, dropout=None)` — defaults preservam comportamento medido; notas em `TrainConfig`. |
| P1-6 | Removido `_features` morto (`models.py`); `inductive_node_split` sem ramo `test_trans` inalcançável. |
| P1-7 | Novos testes: `test_version_sync.py`, `test_stats_smoke.py`, `test_l4_half_edge.py`. |

## P2 restantes (não corrigidos de propósito)

- Placeholders de autor/afiliação/email no ARTIGO (não inventar nomes).
- JSON série 1 ainda com AA ~0.994 (histórico; headline = rescored).
- `pubmed_patience` / `gat_patience` locais em `run_wave3.py`.
- `gat_dropout` documentado mas não passado via `train_encoder`.
- Itens “proposto / não medido” (OGB, L0–L2, multi-head, re-AUC pós-ES/normalize).

## Verificação pós-fix

- `PYTHONPATH=src python3 -m pytest -q` → **53 passed, 3 skipped**
- `check_protocol.py` / CLI subcomandos → ok
- PDF: `glue_lp v0.5.0`, Figura 8 e Figura 9 presentes; SpotTarget = Zhu
