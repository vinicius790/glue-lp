# Estrutura do repositório GLUE-LP

Árvore lógica e **responsabilidade** de cada pasta/arquivo. Complementa `docs/INDEX.md`.

```text
GLUE-LP/
├── README.md                 # Porta de entrada (install, CLI, resumo medido)
├── LICENSE                   # MIT
├── CITATION.cff              # Citação do software (autores: preencher reais)
├── CONTRIBUTING.md           # Regras absolutas + checklist de PR
├── CHANGELOG.md              # Histórico de versões do artefato
├── REVISAO.md                # Notas de revisão / bugs históricos
├── MANUSCRITO.md             # Ponteiro de entrega do manuscrito
├── ENTREGA.md / JUNCAO.md    # Metadados de waves / junção
├── Makefile                  # test, protocol, verify, manuscript, smoke, audit-stats, …
├── pyproject.toml            # Pacote glue-lp + extras + entry points
├── requirements.txt          # Batteries-included (treino)
├── requirements-dev.txt      # Dev pins leves
├── ruff.toml                 # Lint
├── .editorconfig             # Estilo de editor
├── .gitignore
├── .github/workflows/        # CI (testes)
│
├── docs/                     # Documentação científica e operacional
│   ├── INDEX.md              # Mapa do pacote (comece aqui em docs/)
│   ├── README.md             # Catálogo curto de docs/
│   ├── ARTIGO.md             # Fonte do manuscrito (coordenar edições)
│   ├── ARTIGO-COMPLETO.md
│   ├── PROTOCOLO.md          # Invariante, modos, L0–L4, ordem da célula
│   ├── CHECKLIST_SUBMISSAO.md
│   ├── ESTRUTURA_REPOSITORIO.md  # este arquivo
│   ├── ARQUITETURA.md
│   ├── TECNOLOGIAS.md
│   ├── RELATED_WORK.md
│   ├── DESIGN_SCIENCE.md
│   ├── AMEACAS_VALIDADE.md
│   ├── GLOSSARIO.md
│   ├── ROTEIRO_REEXECUCAO.md
│   ├── COMPARATIVO_SPOTTARGET_HEART.md
│   └── GLUE-LP-artigo-cientifico.{pdf,docx}
│
├── src/glue_lp/              # Biblioteca instalável
│   ├── protocol.py           # assert + split_graph
│   ├── config.py             # TrainConfig / DEFAULT_*
│   ├── metrics.py / stats.py
│   ├── models.py             # GCN NumPy + heurísticas
│   ├── torch_*.py            # Encoders PyTorch (extra)
│   ├── data_*.py             # Loaders LINQS
│   ├── cli.py / api.py       # CLI e API opcional
│   ├── export.py             # Leitura tipada dos JSON
│   └── integrations/         # Stubs futuros (OGB/PyG/…) — não medido
│
├── scripts/                  # Runners e utilitários (não são a lib)
│   ├── run_e1_e4.py          # Série 1 → cora_e1_e4.json
│   ├── run_extended.py       # Série 2 → extended_gcn_sage.json
│   ├── run_wave3.py          # Onda 3 → wave3_gat_pubmed.json
│   ├── leakage_ladder.py     # Escada sintética L1/L3/L4
│   ├── smoke_synthetic.py    # Smoke rotulado
│   ├── verify_results.py     # Só relê JSON
│   ├── check_protocol.py
│   ├── apply_stats_from_json.py  # Cohen dz / t / bootstrap a partir de .values
│   ├── make_figures.py
│   ├── build_manuscript.py
│   ├── checksum_data.py
│   ├── serve_protocol.py
│   └── _gen/                 # Geradores auxiliares do manuscrito
│
├── examples/                 # Demos curtas (não gravam LINQS)
│   └── quickstart_synthetic.py
│
├── tests/                    # pytest (protocolo, métricas, smoke, CLI, …)
├── experiments/              # JSON medidos + RESULTADOS.md (+ smoke_*)
├── figures/                  # fig1–fig9 do manuscrito
└── data/                     # LINQS.md + dados locais (não versionar corpora)
```

---

## Responsabilidades por pasta

| Pasta / arquivo | Pode | Não deve |
|---|---|---|
| `src/glue_lp/` | Lógica reutilizável, assert, métricas, loaders | Gravar AUC “de paper” sem passar por `scripts/` |
| `scripts/` | Treinar, escrever JSON, figurar, verificar | Editar médias JSON na mão; sobrescrever LINQS com smoke |
| `experiments/` | Guardar evidência medida; archive pré-fix | Inventar chaves de AUC; apagar histórico sem backup |
| `docs/` | Protocolo, ameaças, índice, checklist | Fabricar números; trocar SpotTarget de autor |
| `examples/` | Demo didática NumPy | Substituir runners oficiais |
| `tests/` | Congelar invariante e contratos | Depender de Cora baixado ou GPU |
| `figures/` | Arte visual do PDF | Legendas com métricas sem fonte JSON |
| `data/` | Instruções + corpora locais | Commitar `.content`/`.cites` grandes sem política |
| `integrations/` | Stubs + how-to até haver medição | Afirmar suporte OGB/PyG como medido |

---

## Donos sugeridos (coordenação multiagente)

| Superfície | Dono típico |
|---|---|
| `docs/ARTIGO.md`, `build_manuscript.py`, PDF | Agente de manuscrito |
| Bugs em `src/`, `tests/`, runners | Agente de auditoria / código |
| Índice, checklist, estrutura, examples, Makefile docs-facing | Wave de organização (este pacote) |

Evitar editar o **mesmo** arquivo quente em paralelo; preferir docs satélite e PRs aditivos.

---

## Arquivos de nota (`_*.md`)

| Arquivo | Uso |
|---|---|
| `_STRENGTHEN_NOTES.md` | Wave de organização / completude |
| `_build_notes_code.md` | Wave de código |
| `_build_notes_docs.md` | Wave de docs |
| `_build_notes_manuscript.md` | Wave de manuscrito |

Não fazem parte do PDF; são trilha de auditoria interna.
