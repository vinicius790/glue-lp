# Notas — wave de organização / completude (strengthen)

Data (America/Sao_Paulo): 2026-09-25.

Objetivo: tornar o pacote mais **completo, detalhado e navegável** sem inventar AUC/*t*/Δ e sem conflitar com o agente de auditoria em arquivos quentes (`docs/ARTIGO.md`, bugs em `src/`).

## Adicionado

| Arquivo | Função |
|---|---|
| `docs/INDEX.md` | Mapa do pacote (manuscrito, código, experimentos, navegação) |
| `docs/CHECKLIST_SUBMISSAO.md` | Gate pré-submissão (SpotTarget=Zhu, P0, CITATION, OGB, …) |
| `docs/ESTRUTURA_REPOSITORIO.md` | Árvore + responsabilidade por pasta |
| `examples/quickstart_synthetic.py` | Demo NumPy: assert + resumo da escada a partir do JSON medido |
| `scripts/apply_stats_from_json.py` | Cohen *d_z* / *t* pareado / bootstrap CI / *t*-logit **só** de `.values` existentes |
| `.editorconfig` | charset, indent, Makefile tabs, max_line_length 100 |
| `_STRENGTHEN_NOTES.md` | este arquivo |

## Enriquecido / atualizado

| Arquivo | Mudança |
|---|---|
| `docs/PROTOCOLO.md` | Invariante formal, modos, políticas negativas, L0–L4 com ponteiros JSON, ordem da célula, verificação |
| `docs/README.md` | Links para todos os docs major + ponteiros fora de `docs/` |
| `Makefile` | Alvos: `test`, `protocol`, `verify`, `manuscript`, `smoke`, `audit-stats`, `quickstart`, `lint`, `fig`, `cli-check` |
| `ruff.toml` | `examples` no src; exclude experiments/`_gen`; select E/F/I/UP/B/W; per-file E402 em scripts/examples/tests |
| `README.md` (raiz) | Linhas na tabela Artefatos + atalhos `make` / quickstart (aditivo) |

## Não tocado (de propósito)

- `docs/ARTIGO.md` / PDF — agente paralelo de manuscrito / auditoria
- JSON LINQS medidos (`cora_*`, `extended_*`, `wave3_*`, …) — imutáveis nesta wave
- Números de AUC/*t*/Δ — nenhum fabricado; `audit-stats` apenas **relê** arrays
- `CITATION.cff` authors — checklist exige preencher **reais**; não inventamos nomes
- Lógica de treino / fixes P0 — domínio do agente de auditoria

## Comandos verificados nesta wave

| Comando | Resultado |
|---|---|
| `PYTHONPATH=src python3 -m pytest -q` | **53 passed, 3 skipped** |
| `python -m glue_lp.cli protocol-check` + `scripts/check_protocol.py` | OK (invariante) |
| `glue_lp.cli verify-results` + `scripts/verify_results.py` | OK (só relê JSON) |
| `examples/quickstart_synthetic.py` | OK (assert + L1/L3/L4 do resumo) |
| `scripts/apply_stats_from_json.py` | OK (pares com `.values`; heuristics nested skipped) |
| `make -n …` após instalar GNU make | Alvos expandem corretamente |

Nenhum JSON LINQS foi reescrito. Escada lida de `leakage_ladder_resumo.json` existente.

## Próximos passos sugeridos (outro agente / humano)

1. Preencher autores em `CITATION.cff`.
2. Cumprir P0 (`ROTEIRO_REEXECUCAO.md` **ou** banner pré-correção em todas as tabelas LINQS).
3. Opcional: ligar `make audit-stats` na CI como job informativo (não falha o build se JSON faltarem pares).
