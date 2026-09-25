# GLUE-LP — Graph Link-evaluation Under Exclusion

Protocolo **executável** para avaliar predição de ligações com GNN **sem** entregar a aresta-alvo ao message passing.

**Invariante (modo `valid`):** \(Q^{+} \cap E_{mp} = \emptyset\) — se a interseção não for vazia, o harness levanta `AssertionError`.

Pacote científico em português (pt-BR) + abstract em inglês. Números de Cora / Citeseer / Pubmed e da escada L0–L4 sintético estão em `experiments/*.json` — **não inventamos métricas novas** neste README.

## Artefatos

| Caminho | Conteúdo |
|---|---|
| `GLUE-LP-artigo-cientifico.pdf` / `.docx` | Manuscrito Design Science / IMRaD |
| `docs/ARTIGO.md` | Fonte Markdown expandida (pt-BR) |
| `docs/INDEX.md` | Mapa do pacote (navegação) |
| `docs/CHECKLIST_SUBMISSAO.md` | Gate pré-submissão |
| `docs/ESTRUTURA_REPOSITORIO.md` | Árvore e responsabilidades |
| `docs/PROTOCOLO.md` | Invariante e ordem da célula |
| `docs/ARQUITETURA.md` | Módulos e formas de dados |
| `docs/TECNOLOGIAS.md` | Stack medido vs proposto |
| `figures/fig1`–`fig9` | Figuras do manuscrito |
| `experiments/*.json` | Séries 1–2, onda 3, escada L, heurísticas |

## Instalação (quickstart técnico)

```bash
# núcleo (NumPy) — suficiente para testes de protocolo / métricas / escada / CLI
pip install -e .

# treino GNN (batteries-included)
pip install -r requirements.txt

# opcionais (extras do pyproject)
pip install -e ".[torch]"      # PyTorch
pip install -e ".[docs]"       # python-docx, matplotlib
pip install -e ".[api]"        # FastAPI + uvicorn
pip install -e ".[analysis]"   # networkx (grau / clustering no tech tree)
pip install -e ".[dev]"        # pytest, ruff
pip install -e ".[all]"        # tudo acima
```

Hiperparâmetros medidos vivem em `glue_lp.config.DEFAULT_TRAIN` (`TrainConfig` /
`ProtocolConfig` / `DatasetSpec`) — fonte única para `scripts/run_*.py`.

## CLI (`glue-lp`)

Após `pip install -e .`:

```bash
glue-lp protocol-check          # invariante no tech tree sintético
glue-lp verify-results          # relê experiments/*.json (não treina)
glue-lp leakage-ladder          # imprime escada L1/L3/L4 medida
glue-lp leakage-ladder --run    # reexecuta scripts/leakage_ladder.py
glue-lp export-summary serie1   # dump JSON do summary
glue-lp describe-config         # TrainConfig / datasets
```

Equivalente sem entry-point: `PYTHONPATH=src python3 -m glue_lp.cli …`.

## Testes sem Cora / sem GPU

Atalhos: `make test` · `make protocol` · `make verify` · `make smoke` · `make audit-stats` · `make quickstart` (`examples/quickstart_synthetic.py`).

```bash
pip install -q pytest numpy
PYTHONPATH=src python3 -m pytest -q
PYTHONPATH=src python3 -m glue_lp.cli protocol-check
PYTHONPATH=src python3 scripts/check_protocol.py
PYTHONPATH=src python3 scripts/verify_results.py
PYTHONPATH=src python3 scripts/leakage_ladder.py
PYTHONPATH=src python3 scripts/smoke_synthetic.py   # gera experiments/smoke_*.json
```

Testes que precisam de `torch`, `fastapi` ou `networkx` são **skipped**
automaticamente se o pacote não estiver instalado. Arquivos `smoke_*.json` são
**sintéticos rotulados** — nunca sobrescrevem `cora_*.json`.

## API opcional (somente leitura)

```bash
pip install -e ".[api]"
PYTHONPATH=src python3 scripts/serve_protocol.py --port 8765
# GET /health
# GET /protocol/invariant   (alias: /protocol/check)
# GET /experiments/summary  (alias lista: /experiments)
# GET /experiments/{key}
```

Não é necessária para CI nem para o manuscrito.

## Análise opcional (NetworkX)

```bash
pip install -e ".[analysis]"
PYTHONPATH=src python3 -c "from glue_lp.graph_analysis import analyze_crafting_graph; print(analyze_crafting_graph())"
```

## Resultados já medidos (resumo)

| Condição | AUC / estatística |
|---|---|
| Série 1 GCN Cora válido vs leaky | 0,708 ± 0,020 vs 0,826 ± 0,019 (Δ +0,118) |
| Série 2 Cora t pareado | t = 2,61 (gl = 4) |
| Série 2 Citeseer | t = 5,77 (Δ +0,100) |
| GAT Cora | 0,782 vs 0,876, t = 20,4 |
| Pubmed GCN (3 seeds) | 0,886 vs 0,921, t = 7,75 |
| SAGE Cora válido | 0,577 ± 0,025 |
| Escada L1 / L3 / L4 (sintético) | 0,713 / 0,779 / 0,860 |

Detalhe: `experiments/RESULTADOS.md`. **OGB não rodou.** Re-execução pós-correção de early stopping / `F.normalize` no SAGE: proposto / não medido.

## Reprodução das séries (requer PyTorch + dados LINQS)

```bash
# dados: ver data/LINQS.md
PYTHONPATH=src python3 scripts/run_e1_e4.py
PYTHONPATH=src python3 scripts/run_extended.py
PYTHONPATH=src python3 scripts/run_wave3.py
PYTHONPATH=src python3 scripts/build_manuscript.py
```

## Citação

Ver `CITATION.cff` (sem inventar nomes de autores). Licença MIT.
