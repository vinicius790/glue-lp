# Auditoria GLUE-LP (2026-09-25)

Escopo: testes, scripts de protocolo, citações, hotspots de código, drift de versão/docs, manuscrito, cobertura barata de testes.  
**Regra:** nenhum AUC/t/Δ experimental inventado; números LINQS permanecem **[pré-correção ES]**; sem reivindicar re-run pós-fix.

## Ambiente / smoke

| Check | Resultado |
|---|---|
| `PYTHONPATH=src python3 -m pytest -q` | **47 passed, 3 skipped** |
| `scripts/check_protocol.py` | ok (válido sem interseção; leaky com) |
| `scripts/verify_results.py` | ok (lê JSON medidos; não retreina) |
| `python3 -m glue_lp.cli --help` + subcomandos | ok (`protocol-check`, `verify-results`, `leakage-ladder`, `describe-config`, `export-summary`) |

---

## P0 — crítico (corrigir antes de redistribuir)

### P0-1 · Drift de versão (pacote 0.5.0 vs docs 0.4.0)
- **Onde:** `pyproject.toml` / `src/glue_lp/__init__.py` = **0.5.0**; `CITATION.cff`, `docs/ARTIGO.md` (banner + rodapé), `scripts/build_manuscript.py` (footer PDF), `scripts/_gen/write_artigo.py` / `write_artigo3.py` = **0.4.0**.
- **Evidência:** `rg '0\.4\.0|0\.5\.0'` nos arquivos acima; CLI `--version` imprime 0.5.0; PDF extraído ainda diz `glue_lp v0.4.0`.
- **Fix:** alinhar tudo a **0.5.0** (CITATION, ARTIGO, footer do builder, geradores). Não inventar autores.

### P0-2 · Citação SpotTarget errada nos geradores legados
- **Onde:** `scripts/_gen/write_artigo.py` (3 ocorrências: “Dong et al. (SpotTarget)” / “Dong et al. (2024, SpotTarget)”).
- **Evidência:** `rg 'Dong et al.*SpotTarget' scripts/_gen/`. Os docs canônicos (`docs/ARTIGO.md`, `RELATED_WORK.md`, PDF atual) já usam **Zhu et al.**; FakeEdge/Dong 2022 está separado e correto.
- **Fix:** trocar para Zhu et al. nos geradores para não reintroduzir o erro se alguém reexecutar `_gen/`.

---

## P1 — importante (seguro de corrigir)

### P1-1 · fig8 / fig9 existem mas não entram no manuscrito DOCX/PDF
- **Onde:** `figures/fig8_arquitetura.png`, `figures/fig9_escada_vazamento.png` presentes; `scripts/build_manuscript.py` `FIGURES` só lista fig1–fig7; `docs/ARQUITETURA.md` e `ENTREGA.md` já mencionam fig8/9.
- **Evidência:** `ls figures/`; PDF/DOCX sem Figuras 8–9; README raiz ainda diz `fig1`–`fig7`.
- **Fix:** incluir fig8/fig9 em `FIGURES`, atualizar menções fig1–fig7→fig1–fig9 no builder/README/ARTIGO (lista de contribuições / tabela de artefatos), regenerar DOCX+PDF.

### P1-2 · `verify-results` / CLI não exibem heurísticas rescored
- **Onde:** `src/glue_lp/cli.py` `_cmd_verify_results`; `heuristics_rescored.json` tem `summary` aninhado (`valid_unif.aa.mean` …).
- **Evidência:** saída do CLI mostra `--- heuristics ---` vazio; ao mesmo tempo imprime AA “ingênua” ~0.994 da série 1 (JSON bruto), enquanto o manuscrito usa 0.720 (empate=½) — já documentado em ARTIGO § métricas, mas a CLI confunde.
- **Fix:** formatar blocos aninhados com `mean`/`std`; nota explícita de que AA/CN/PA do *headline* vêm de `heuristics`, não do summary bruto da série 1.

### P1-3 · Versão hardcoded em CLI/API (não usa `__version__`)
- **Onde:** `cli.py` (`version="glue-lp 0.5.0"`), `api.py` (`version="0.5.0"` ×2).
- **Fix:** importar `glue_lp.__version__`.

### P1-4 · Índice de docs incompleto
- **Onde:** `docs/README.md` omite `ARTIGO-COMPLETO.md`; README raiz omite fig8/9.
- **Fix:** completar índice e tabela de artefatos.

### P1-5 · `train_encoder` ignora `TrainConfig.weight_decay` / `dropout`
- **Onde:** `torch_models.py` — Adam com `weight_decay=5e-4` fixo; modelos usam dropout default interno; `TrainConfig` declara os campos mas runners não os passam.
- **Evidência:** `rg weight_decay|dropout scripts/run_*.py src/glue_lp/torch_models.py`.
- **Fix:** aceitar kwargs `weight_decay` / `dropout` com os mesmos defaults atuais (sem mudar números medidos). Documentar campos ainda não conectados (`gat_dropout` fica P2 se GAT path não receber).

### P1-6 · Código morto / ramo inalcançável
- **Onde:** `models.py` `_features` (nunca chamado); `splits.py` `inductive_node_split` — `else: test_trans.append(...)` inalcançável (o `elif` já cobre todo o complemento).
- **Fix:** remover `_features` morto; eliminar `test_trans` morto (só retornar `test_ind`).

### P1-7 · Lacunas de teste baratas
- Sincronia `__version__` ↔ `pyproject.toml`.
- `paired_t` / `cohens_dz` com vetores minúsculos.
- Smoke L4: `embed_gcn_directed_half` torna a aresta visível só em uma direção na adj.

---

## P2 — menor / documentar (não bloqueia)

### P2-1 · Placeholders de autoria no manuscrito
- `docs/ARTIGO.md`: `[placeholder — nomes|instituição|email]`. Intencional; **não inventar** nomes.

### P2-2 · JSON série 1 ainda carrega AA ~0.994
- Documentado; `heuristics_rescored.json` é a fonte do manuscrito. Manter JSON histórico; não “corrigir” médias à mão.

### P2-3 · `run_wave3.py`: `pubmed_patience=8` e `gat_patience=min(12,10)` fora de `TrainConfig`
- Drift menor de hiperparâmetros vs “fonte única”; valores batem com o que foi medido. Extrair para config só com cuidado para não sugerir re-AUC.

### P2-4 · `ARTIGO.md` ≈ `ARTIGO-COMPLETO.md` (idênticos)
- Redundância OK se mantidos em sync; índice deve apontar ambos.

### P2-5 · DOCX/PDF vs word count do MD
- ARTIGO.md ~10 221 palavras; texto PDF ~9 8xx (strip LaTeX/`$`). Esperado; regenerar após fixes de fig/versão.

### P2-6 · Sem Hits@K OGB / L0–L2 / multi-head GAT / SAGE pós-normalize
- Já marcados “proposto / não medido” — manter assim.

### P2-7 · `holm_bonferroni` / `paired_t_logit` sem testes e sem uso nas tabelas históricas
- Documentado no ARTIGO; cobertura de teste = P1-7 parcial (`paired_t` basta).

---

## O que já está bom

- Invariante `Q+ ∩ Emp = ∅` com assert no modo `valid`; testes de protocolo verdes.
- Early stopping em `torch_models.train_encoder` corrigido (AUC de val real; `>` estrito; critério simétrico valid/leaky nos runners).
- SpotTarget = **Zhu et al.** nos docs canônicos, RELATED_WORK, COMPARATIVO, GLOSSARIO, ARTIGO, PDF atual; FakeEdge = Dong 2022 separado.
- CLI funcional sem Typer; entry points no pyproject; API opcional read-only.
- Métricas Mann–Whitney com empate=½ + testes manuais minúsculos sólidos.
- Escada L1/L3/L4 sintética medida e verificável (`leakage_ladder_*.json`).
- Integridade experimental: JSON versionados; manuscrito rotula **[pré-correção ES]**; sem OGB fingido.
- 47 testes passam; skips honestos (torch/fastapi/networkx).

---

## Contagens (pré-fix)

| Severidade | Achados |
|---|---|
| P0 | 2 |
| P1 | 7 |
| P2 | 7 |

Nenhum número experimental foi inventado nesta auditoria.


---

## Pós-correção

| | Achados | Corrigidos | Restantes |
|---|---|---|---|
| P0 | 2 | 2 | 0 |
| P1 | 7 | 7 | 0 |
| P2 | 7 | 0 | 7 (intencionais / não seguros sem re-medição) |

- Testes: **53 passed, 3 skipped**
- PDF regenerado: sim (raiz + `docs/`, com fig8/fig9 e v0.5.0)
- Ver `_AUDIT_FIXES.md`
