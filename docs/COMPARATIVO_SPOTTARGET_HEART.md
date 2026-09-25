# Comparativo honesto: GLUE-LP vs SpotTarget vs HeaRT vs OGB

> SpotTarget = **Zhu et al.** (WSDM 2024, arXiv:2306.00899) — **não** Dong et al.  
> FakeEdge = Dong et al. (2022, arXiv:2211.15899) — trabalho separado.  
> HeaRT = Li et al. (NeurIPS D&B 2023, arXiv:2306.10453).  
> OGB = Hu et al. (NeurIPS 2020, arXiv:2005.00687).  
> Números AUC de GLUE-LP: só os já medidos em `experiments/*.json` (ver ameaça pré-fix).

---

## Tabela de diferenciação (obrigatória)

| Dimensão | **GLUE-LP** | **SpotTarget** (Zhu et al., 2024) | **HeaRT** (Li et al., 2023) | **OGB** (Hu et al., 2020) |
|---|---|---|---|---|
| **Objeto** | Harness / protocolo executável com **assert** | Framework de treino GNN (DGL) que exclui alvos | *Benchmark* + política de negativos difíceis | Suíte de datasets + métricas + *leaderboard* |
| **Problema central** | \(Q^{+} \cap E_{mp} = \emptyset\) incontornável no modo `valid` | Incluir alvo em \(E_{mp}\) → overfit / *shift* / *test leakage*; foco em nós de baixo grau | Negativos fáceis + splits/métricas inconsistentes | Escala, splits realistas, métricas por tarefa |
| **Solução** | `assert_no_leakage`; modos `valid`/`leaky`; escada L0–L4 | Excluir \(T_{low}\) no treino; excluir **todos** os alvos no teste; *leakage check* | HeaRT: \(K\) corruptions hard via RA/PPR/feat. | Splits oficiais + Hits@K / MRR / ROC-AUC |
| **Escopo empírico** | Cora, Citeseer, Pubmed (LINQS) + tech tree 12 nós | ogbl-collab/citation2, USAir, E-commerce (esparso) | Planetoid + ogbl-* com *fair HPO* | ogbn-/ogbl-/ogbg-* (vários domínios) |
| **OGB executado?** | **Não** (stub apenas) | Sim (ogbl-*) | Sim (reavaliação ampla) | É o próprio OGB |
| **Heurísticas no mesmo \(Q\)?** | Sim (CN/AA/PA obrigatórios na célula) | Baselines em experimentos; não é o *claim* central | Sim (CN/AA/RA/Katz no *fair comparison*) | Baselines oficiais nos *leaderboards* |
| **Negativos** | Uniforme 1:1, hard-CN, *degree-matched* | Conforme *setup* DGL/OGB | HeaRT (500 hard/positivo) + *setting* legado | Fixos por dataset (ex.: Hits@50 collab) |
| **Encoder novo?** | Não — GCN/SAGE/GAT rasos (hidden 32) | Não — aplica-se a SAGE/GCN/GATv2/SEAL | Não — reavalia SEAL/BUDDY/NCN/… | Não |
| **Invariante de CI** | `AssertionError` se vazamento no modo `valid` | *Best practice* + sampler; não é assert do usuário final do GLUE-LP | Protocolo de avaliação documentado | Avaliador oficial OGB |
| **Design Science** | Artefato DSR (protocolo) | Paper de *pitfalls* + *framework* | Paper de *benchmarking* | Paper de *datasets* |
| **O que GLUE-LP NÃO é** | — | Não substitui SpotTarget em escala web / DGL | Não substitui HeaRT como *leaderboard* de LP | Não é OGB |

---

## O que GLUE-LP *é*

1. Pacote `glue_lp` com invariante **código-forçada**: no modo `valid`, \(Q^{+} \cap E_{mp} = \emptyset\) ou o processo aborta.
2. Célula experimental única: mesmo \(Q\) para GNN e CN/AA/PA; modos `valid` vs `leaky` para Δ mensurável.
3. Escada L1/L3/L4 **medida no sintético** (L0/L2 propostos).
4. Manuscrito + JSON + testes CI sem GPU.

## O que GLUE-LP *não* é

1. Não é SOTA de LP em OGB (OGB **não rodou**).
2. Não é reimplementação de SpotTarget/HeaRT.
3. Não é *fair comparison* de dezenas de modelos (HeaRT).
4. Não reivindica que o assert sozinho resolve *distribution shift* de *mini-batch* em grafos web-scale (contribuição de SpotTarget).
5. Números Cora/Citeseer/Pubmed publicados = **pré-correção** completa do *early stopping* — ver `ROTEIRO_REEXECUCAO.md` (gate P0).

## Relação FakeEdge → SpotTarget

| | FakeEdge (Dong et al., 2022) | SpotTarget (Zhu et al., 2024) |
|---|---|---|
| Foco | *Dataset shift* em métodos de **subgrafo** (SEAL-like) | Inclusão de alvo em GAE/GNN gerais + baixo grau |
| Venue | arXiv:2211.15899 | WSDM 2024 / arXiv:2306.00899 |
| Em GLUE-LP | Citação de linhagem; **não medido** | Diagnóstico alinhado; implementação distinta (assert) |

**Não** citar SpotTarget como “Dong et al.”

## Mensagem para o manuscrito

> SpotTarget e HeaRT diagnosticam e *benchmarkam*. GLUE-LP transforma a exclusão da aresta-alvo em **teste vermelho no CI** e documenta uma escada de vazamento operacional. São complementares, não concorrentes de *leaderboard*.
