# Índice do pacote GLUE-LP

Mapa de navegação do repositório científico. **Não inventa métricas** — aponta para fontes já medidas (`experiments/*.json`) ou para documentação.

Versão do pacote: ver `pyproject.toml` / `CITATION.cff`.

---

## Como navegar (caminho rápido)

| Se você quer… | Vá para… |
|---|---|
| Entender o *claim* científico | `docs/ARTIGO.md` (fonte) → PDF/DOCX na raiz e em `docs/` |
| Regras do protocolo (invariante, modos, L0–L4) | `docs/PROTOCOLO.md` + `docs/GLOSSARIO.md` |
| Rodar algo sem Cora/GPU | `examples/quickstart_synthetic.py` ou `make smoke` / `make protocol` |
| Ver números já medidos | `experiments/RESULTADOS.md` + `experiments/*.json` |
| Gate pré-submissão | `docs/CHECKLIST_SUBMISSAO.md` |
| Árvore de pastas e donos | `docs/ESTRUTURA_REPOSITORIO.md` |
| Re-medir LINQS pós-fix ES | `docs/ROTEIRO_REEXECUCAO.md` (**P0**) |
| Contribuir sem quebrar regras | `CONTRIBUTING.md` |

```text
README.md  ──►  docs/INDEX.md (este arquivo)
                  ├─ manuscrito / Design Science
                  ├─ protocolo / ameaças / glossário
                  ├─ código (src/glue_lp) + testes
                  └─ experimentos (JSON imutáveis até re-run)
```

---

## 1. Manuscrito e entrega

| Artefato | Papel |
|---|---|
| `docs/ARTIGO.md` | Fonte Markdown expandida (pt-BR + abstract EN). Agente de manuscrito — **coordenar** antes de editar. |
| `docs/ARTIGO-COMPLETO.md` | Espelho/compilado completo (sincronizar com ARTIGO). |
| `GLUE-LP-artigo-cientifico.pdf` / `.docx` | Build tipográfico (também em `docs/`). |
| `MANUSCRITO.md` / `ENTREGA.md` / `JUNCAO.md` | Ponteiros de entrega / junção de waves. |
| `scripts/build_manuscript.py` | Gera PDF/DOCX a partir do Markdown (requer extras `[docs]`). |

Leitura complementar do manuscrito:

- `docs/DESIGN_SCIENCE.md` — Hevner / Peffers; artefato vs fenômeno  
- `docs/RELATED_WORK.md` — bibliografia anotada (**SpotTarget = Zhu et al.**)  
- `docs/COMPARATIVO_SPOTTARGET_HEART.md` — GLUE-LP vs SpotTarget vs HeaRT vs OGB  
- `docs/AMEACAS_VALIDADE.md` — ameaça **P0** (pré-correção early stopping)  
- `docs/TECNOLOGIAS.md` — stack **[MEDIDO]** vs **[PROPOSTO]**  

---

## 2. Protocolo e conceitos

| Documento | Conteúdo |
|---|---|
| `docs/PROTOCOLO.md` | Invariante formal, modos, políticas de negativos, escada L0–L4, ordem da célula |
| `docs/GLOSSARIO.md` | \(E_{mp}\), \(Q^{+}\), L0–L4, pré-correção, OGB, … |
| `docs/ARQUITETURA.md` | Módulos `glue_lp`, diagramas, fig8 |
| `src/glue_lp/protocol.py` | `assert_no_leakage`, `split_graph`, modos `valid`/`leaky` |
| `src/glue_lp/config.py` | `TrainConfig` / `DEFAULT_TRAIN` (hiperparâmetros canônicos) |

---

## 3. Código (`src/glue_lp`)

| Área | Módulos típicos |
|---|---|
| Domínio / tipos | `types.py`, `graph.py`, `splits.py` |
| Protocolo / métricas | `protocol.py`, `metrics.py`, `stats.py`, `heuristics.py` |
| Modelos | `models.py` (NumPy), `torch_gcn.py` / `torch_models.py` (opcional) |
| Dados | `data_citation.py`, `data_cora.py`, `data_citation` paths em `data/LINQS.md` |
| Orquestração | `run.py`, `experiments.py`, `export.py`, `cli.py`, `api.py` |
| Futuro não medido | `integrations/` (stubs `NotImplementedError`) |

CLI: `glue-lp` / `python -m glue_lp.cli` — ver README raiz.

---

## 4. Experimentos e figuras

| Caminho | Conteúdo |
|---|---|
| `experiments/cora_e1_e4.json` | Série 1 (Cora GCN denso) — **pré-correção ES** |
| `experiments/extended_gcn_sage.json` | Série 2 + `paired_t_*` — **pré-correção ES** |
| `experiments/wave3_gat_pubmed.json` | Onda 3 GAT/Pubmed — **pré-correção ES** |
| `experiments/heuristics_rescored.json` | Heurísticas no mesmo Q |
| `experiments/leakage_ladder_*.json` | Escada L1/L3/L4 **sintético [MEDIDO]** |
| `experiments/smoke_*.json` | Smoke rotulado — **nunca** confundir com LINQS |
| `experiments/RESULTADOS.md` | Tabelas humanas dos JSON |
| `figures/fig1`–`fig9` | Figuras do manuscrito |
| `scripts/run_*.py`, `leakage_ladder.py`, `verify_results.py` | Runners / verificação (só relê ou re-mede) |

**OGB não rodou.** Qualquer Hits@K/MRR OGB é **proposto / não medido**.

---

## 5. Testes, qualidade e exemplos

| Item | Comando / caminho |
|---|---|
| Testes | `make test` → `pytest -q` (`tests/`) |
| Protocolo sintético | `make protocol` |
| Verificar JSON | `make verify` |
| Smoke sintético | `make smoke` |
| Stats a partir de JSON | `make audit-stats` → `scripts/apply_stats_from_json.py` |
| Quickstart | `examples/quickstart_synthetic.py` |
| Lint | `make lint` (`ruff.toml`) |

---

## 6. Documentos operacionais deste pacote

| Documento | Função |
|---|---|
| `docs/CHECKLIST_SUBMISSAO.md` | Gate pré-submissão (P0, SpotTarget, CITATION, OGB, …) |
| `docs/ESTRUTURA_REPOSITORIO.md` | Árvore + responsabilidade por pasta |
| `docs/ROTEIRO_REEXECUCAO.md` | Re-run LINQS pós-fix |
| `docs/README.md` | Catálogo curto de `docs/` |
| `_STRENGTHEN_NOTES.md` | Notas desta wave de organização |
| `CHANGELOG.md` / `REVISAO.md` / `CONTRIBUTING.md` | Histórico, revisão, regras de PR |

---

## 7. Política de números (lembrete)

1. AUC / *t* / Δ **só** dos JSON em `experiments/` gerados por runners versionados.  
2. Não editar médias “na mão” em Markdown.  
3. LINQS atual = **pré-correção** do early stopping até `ROTEIRO_REEXECUCAO.md` completar.  
4. Escada sintética L1/L3/L4 é independente do bug ES e já está medida.  
5. Autores em `CITATION.cff`: preencher com nomes reais — **não inventar**.
