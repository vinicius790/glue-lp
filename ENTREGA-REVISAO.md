# Entrega — auditoria + reforço (2026-09-25)

## Corrigido (P0+P1)
- Versão alinhada em **0.5.0** (CITATION, ARTIGO, PDF, geradores)
- SpotTarget = Zhu nos geradores legados `_gen/`
- Figuras 8–9 no DOCX/PDF (23 páginas)
- CLI: heurísticas rescored + `__version__`
- `train_encoder` aceita weight_decay/dropout do TrainConfig
- Código morto removido; 53 testes

## Reforçado (organização)
- `docs/INDEX.md`, `CHECKLIST_SUBMISSAO.md`, `ESTRUTURA_REPOSITORIO.md`
- PROTOCOL expandido; Makefile completo; quickstart sintético
- `scripts/apply_stats_from_json.py` (só seeds já no JSON)

## Ainda P2 (não inventamos números)
- Preencher autores no CITATION / manuscrito
- Re-run LINQS pós-fix ES (P0 científico)
- Unificar hiperparâmetros locais do wave3 no TrainConfig
- OGB / L2 / multi-head GAT / SAGE pós-normalize = não medidos
