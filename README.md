# GLUE-LP — Graph Link-evaluation Under Exclusion

**Versão 0.5.0** · MIT · pacote científico em pt-BR (abstract em inglês)

Protocolo **executável** para avaliar predição de ligações com GNN **sem** entregar a aresta-alvo ao message passing.

**Invariante (modo `valid`):** \(Q^{+} \cap E_{mp} = \emptyset\). Se a interseção não for vazia, o harness levanta `AssertionError` — não é só texto no paper.

Este repositório é o pacote **GLUE-LP-revisado**: manuscrito Design Science / IMRaD (~10 221 palavras / ~22–23 páginas PDF), código Python com CLI `glue-lp`, figuras 1–9, JSONs medidos em Cora / Citeseer / Pubmed e escada sintética de vazamento L0–L4. Os AUCs e estatísticas \(t\) deste README vêm de `experiments/*.json` — **nenhuma métrica LINQS foi inventada aqui**.

Comece por [`docs/INDEX.md`](docs/INDEX.md) se quiser o mapa completo do pacote.

---

## Por que existe

Em link prediction com GNN, é fácil “vazar” a aresta em teste para o grafo de propagação. O resultado sobe, mas a avaliação deixa de medir o que o paper diz medir. GLUE-LP fixa o protocolo (modos `valid` vs `leaky`), testa o invariante em código e reporta o Δ entre condições no mesmo harness.

---

## O que tem neste pacote

| Área | Conteúdo |
|---|---|
| Manuscrito | `GLUE-LP-artigo-cientifico.pdf` / `.docx` + fonte `docs/ARTIGO.md` |
| Protocolo | `docs/PROTOCOLO.md`, glossário, ameaças à validade |
| Código | `src/glue_lp/` · CLI · métricas · config tipada · API opcional |
| Experimentos | `experiments/*.json` + `experiments/RESULTADOS.md` |
| Figuras | `figures/fig1` … `fig9` (pipeline, protocolo, corpora, arquitetura, escada) |
| Qualidade | testes automatizados · `Makefile` · checklist de submissão · CI workflow |

**Ainda aberto (honesto):** re-execução LINQS pós-correção de early stopping / `F.normalize` no SAGE = **proposto, não re-medido**; OGB **não rodou**; autores no `CITATION.cff` ainda a preencher. Ver `docs/ROTEIRO_REEXECUCAO.md` e `docs/CHECKLIST_SUBMISSAO.md`.

---

## Instalação

Python ≥ 3.11.

```bash
# núcleo (NumPy) — protocolo, métricas, escada L, CLI, maioria dos testes
pip install -e .

# treino GNN (batteries-included)
pip install -r requirements.txt

# extras
pip install -e ".[torch]"     # PyTorch
pip install -e ".[docs]"      # python-docx, matplotlib
pip install -e ".[api]"       # FastAPI + uvicorn
pip install -e ".[analysis]"  # networkx
pip install -e ".[dev]"       # pytest, ruff
pip install -e ".[all]"       # tudo acima
```

Hiperparâmetros medidos: `glue_lp.config.DEFAULT_TRAIN` (`TrainConfig` / `ProtocolConfig` / `DatasetSpec`) — fonte usada pelos `scripts/run_*.py`.

---

## Começar em 60 segundos (sem Cora / sem GPU)

```bash
pip install -e ".[dev]"
make test          # ou: PYTHONPATH=src python3 -m pytest -q
make protocol      # invariante no tech tree sintético
make verify        # relê experiments/*.json (não treina)
make quickstart    # examples/quickstart_synthetic.py
```

Atalhos úteis: `make smoke` · `make audit-stats` · `make manuscript` (precisa `[docs]`).

---

## CLI (`glue-lp`)

```bash
pip install -e .
glue-lp protocol-check          # invariante sintético
glue-lp verify-results          # confere JSONs medidos
glue-lp leakage-ladder          # imprime L1 / L3 / L4
glue-lp leakage-ladder --run    # reexecuta scripts/leakage_ladder.py
glue-lp export-summary serie1   # dump do summary
glue-lp describe-config         # TrainConfig / datasets
```

Sem entry-point: `PYTHONPATH=src python3 -m glue_lp.cli …`.

Testes que precisam de `torch`, `fastapi` ou `networkx` são **skipped** se o pacote não estiver instalado. Arquivos `experiments/smoke_*.json` são **sintéticos rotulados** e nunca sobrescrevem `cora_*.json`.

---

## Resultados já medidos (resumo)

Detalhe e sementes: [`experiments/RESULTADOS.md`](experiments/RESULTADOS.md). AUCs LINQS marcados no manuscrito como **[pré-correção ES]** onde aplicável.

| Condição | AUC / estatística |
|---|---|
| Série 1 GCN Cora válido vs leaky | 0,708 ± 0,020 vs 0,826 ± 0,019 (Δ +0,118) |
| AA Cora válido (uniforme) | 0,720 ± 0,007 |
| Série 2 Cora (t pareado) | t = 2,61 (gl = 4) |
| Série 2 Citeseer | t = 5,77 (Δ +0,100) |
| GAT Cora | 0,782 vs 0,876, t = 20,4 |
| Pubmed GCN (3 seeds) | 0,886 vs 0,921, t = 7,75 |
| SAGE Cora válido | 0,577 ± 0,025 |
| Escada L1 / L3 / L4 (sintético) | 0,713 / 0,779 / 0,860 |

**OGB não rodou.** Re-AUC pós-fix ES / SAGE normalize = P0 científico antes de submissão.

---

## Reprodução das séries LINQS

Requer PyTorch + dados Planetoid (ver [`data/LINQS.md`](data/LINQS.md)).

```bash
PYTHONPATH=src python3 scripts/run_e1_e4.py
PYTHONPATH=src python3 scripts/run_extended.py
PYTHONPATH=src python3 scripts/run_wave3.py
PYTHONPATH=src python3 scripts/build_manuscript.py
```

---

## API opcional (somente leitura)

```bash
pip install -e ".[api]"
PYTHONPATH=src python3 scripts/serve_protocol.py --port 8765
# GET /health
# GET /protocol/invariant
# GET /experiments/summary
# GET /experiments/{key}
```

Não é necessária para CI nem para o manuscrito.

---

## Navegação rápida

| Quero… | Abra… |
|---|---|
| Claim científico | `docs/ARTIGO.md` → PDF na raiz |
| Regras do protocolo | `docs/PROTOCOLO.md` |
| Números medidos | `experiments/RESULTADOS.md` |
| Gate pré-submissão | `docs/CHECKLIST_SUBMISSAO.md` |
| Re-medir pós-fix | `docs/ROTEIRO_REEXECUCAO.md` |
| Contribuir | `CONTRIBUTING.md` |
| Changelog | `CHANGELOG.md` |

---

## Citação e licença

Ver [`CITATION.cff`](CITATION.cff) (versão **0.5.0**; preencha autores antes de citar em paper). Licença [MIT](LICENSE).
