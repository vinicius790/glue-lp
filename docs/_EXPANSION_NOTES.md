# Notas de expansão do manuscrito (para o agente pai)

Data: 2026-09-25 (America/Sao_Paulo). Artefato permanece **v0.5.0**. Nenhum AUC/t/Δ/std novo foi inventado; só texto + regeneração DOCX/PDF.

## Contagens
| | Antes | Depois |
|---|---:|---:|
| `docs/ARTIGO.md` (palavras) | ~10 221 | ~14 431 |
| PDF (páginas A4) | 23 | **30** |

## Lacunas endereçadas
1. **§9 Limitações** — expandida com bloqueios P0–P2 e frases que o MS recusa.
2. **§9.2 Trabalho futuro** — backlog priorizado (tabela P0/P1/P2, todos “não medido”).
3. **§10 Reprodutibilidade** — checklist operacional + comandos `make`.
4. **§10.2 Disponibilidade de dados e código** — Planetoid/LINQS (Sen/Yang), MIT, in-repo vs download, OGB não rodou.
5. **§10.3 Conflito de interesses / financiamento** — placeholders honestos.
6. **§11 Conclusão** — expandida (ranking sob negativos, escada, artefato vs fenômeno).
7. **§5 Stack** — medido vs proposto com caminhos de módulo; complexidade; hardware.
8. **Apêndices B, C, F** — mapas e glossário densificados.
9. **Apêndice I** — checklist de submissão condensado de `CHECKLIST_SUBMISSAO.md`.
10. **Apêndice J** — tabela de corpus (contagens já usadas na §6.1 / `data/LINQS.md`).
11. **Pseudocódigo harness** (§4.2b) — módulos reais de `src/glue_lp/`.
12. **Pseudocódigo escada L1/L3/L4** (§4.2c) — alinhado a `scripts/leakage_ladder.py`.
13. **Matriz contribuição ↔ evidência** (§1.3b).
14. **Discussão Figuras 8 e 9** (§7.5.1–7.5.2).
15. **Negativos** (§7.6.2) — definições operacionais + por que o ranking vira (números série 1).
16. **Ameaças** (§8.5) — tabela de mitigação.
17. **§2.6** — 2–3 parágrafos a partir de `RELATED_WORK.md` (sem papers inventados); FakeEdge ≠ SpotTarget.
18. **Resumo/Abstract** — alongados com deltas L / hard-CN / degree-matched já medidos.
19. **README** — landing GitHub a partir do draft em `/workspace/glue-readme-work/`.
20. **CHANGELOG** — entrada 0.5.1 (só manuscrito).
21. **ARTIGO-COMPLETO.md** — cópia sincronizada de ARTIGO.md.

## Verificações
- SpotTarget = Zhu et al.; menções a Dong referem-se só a FakeEdge (distinto).
- Banner **[pré-correção ES]** preservado.
- Versão 0.5.0 no banner/`pyproject`/`CITATION.cff`.
- JSON em `experiments/` não modificados.

## Bloqueios (se houver)
- Nenhum: soffice e python-docx ok; PDF regenerado.
