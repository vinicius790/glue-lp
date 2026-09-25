# Changelog

## 0.5.1 — manuscrito denso (2026-09-25)

### Manuscrito (sem novos números experimentais)
- Expansão substancial de `docs/ARTIGO.md` (~10 221 → ~14 431 palavras): §§5, 8–11, Apêndices B/C/F, novos I/J.
- Novos blocos: matriz contribuição↔evidência (§1.3b); pseudocódigo do harness e da escada L (§4.2b–c); discussão Figuras 8–9 (§7.5.x); tabela de negativos com definições operacionais (§7.6.2); tabela de mitigação de ameaças (§8.5); backlog P0–P2 (§9.2); disponibilidade de dados/código + COI (§10.2–10.3).
- Abstract/Resumo alongados só com fatos já medidos (escada L, hard-CN, degree-matched).
- §2.6 enriquecida a partir de `RELATED_WORK.md` (citações já no pacote); FakeEdge (Dong et al., 2022) distinguido de SpotTarget (Zhu et al.).
- `ARTIGO-COMPLETO.md` sincronizado; README de landing atualizado; PDF/DOCX regenerados.

### Não alterado
- Versão do artefato permanece **0.5.0** (`pyproject.toml` / `CITATION.cff`).
- Nenhum valor em `experiments/*.json` foi editado.


## 0.5.0 — wave código / CLI / métricas (2026-09-25)

### Auditoria (2026-09-25)
- Versão alinhada a **0.5.0** em `CITATION.cff`, `docs/ARTIGO.md`, footer do
  `build_manuscript.py` e geradores `_gen/` (antes 0.4.0).
- Gerador legado `scripts/_gen/write_artigo.py`: SpotTarget corrigido Dong→**Zhu**.
- Manuscrito: fig8 (arquitetura) + fig9 (escada L) no builder; DOCX/PDF regenerados.
- CLI `verify-results`: imprime heurísticas aninhadas (AA 0.720) + nota sobre AUC
  ingênua da série 1; CLI/API usam `glue_lp.__version__`.
- `train_encoder`: kwargs `weight_decay` / `dropout` (defaults iguais aos medidos).
- Removidos `_features` morto e ramo inalcançável em `inductive_node_split`.
- Testes: version sync, stats smoke, L4 half-edge (53 passed, 3 skipped).
- **Nenhum AUC/t/Δ LINQS inventado ou alterado.**

- CLI `glue-lp` (`src/glue_lp/cli.py`): `protocol-check`, `verify-results`,
  `leakage-ladder`, `export-summary`, `describe-config` + entry points no pyproject.
- Métricas documentadas (AUC tie=0.5, AP, Hits@K, MRR) + `metric_block`; testes
  com casos manuais minúsculos.
- `graph_analysis.py` opcional (NetworkX): histograma de grau + clustering no
  tech tree sintético; skip se networkx ausente.
- API: `/protocol/invariant`, `/experiments/summary` (aliases mantidos).
- `scripts/smoke_synthetic.py` → `experiments/smoke_*.json` (só sintético rotulado).
- Integrações PyG/DGL/OGB: stubs `NotImplementedError` + how-to nos docstrings.
- Extras pyproject: `torch`, `docs`, `api`, `analysis`, `dev`, `all`.
- Runners `run_e1_e4` / `run_extended` / `run_wave3` usam `DEFAULT_TRAIN`.
- **Nenhum AUC/t/Δ medido de Cora/Citeseer/Pubmed foi inventado ou alterado.**

## 0.4.0 — pacote científico expandido (2026-09-25)

- Domínio tipado: `types.py` (`GraphBundle`, `EdgeSplit`, `LeakageLevel`, `RunResult`, …).
- `TrainConfig` / `ProtocolConfig` / `DatasetSpec` enriquecidos; runners importam `DEFAULT_TRAIN`.
- GraphSAGE: `F.normalize` na saída de `encode` (correção recomendada; **re-AUC não medido**).
- `train_encoder`: early stopping por AUC de validação real (mantido); scripts passam `val` também no leaky.
- `logging_util.py`, `export.py`, API opcional `api.py` + `scripts/serve_protocol.py`.
- Stubs `integrations/` (PyG, DGL, OGB) com `NotImplementedError` claro.
- Docs: `ARQUITETURA.md`, `TECNOLOGIAS.md`, README/PROTOCOLO/CONTRIBUTING expandidos.
- Manuscrito `docs/ARTIGO.md` expandido em estilo Design Science / IMRaD (~3–5×).
- `build_manuscript.py` emite DOCX em raiz e `docs/` (sem path `/home/workdir/artifacts`).
- Testes: types, export, leakage ladder smoke, API smoke (skip sem fastapi), integrations stub.
- **Nenhum AUC/t/Δ de Cora/Citeseer/Pubmed foi inventado ou alterado.**

## 0.3.0 — junção unificado + revisado (2026-09-24)

- Código corrigido da revisão (ReLU, heurísticas, early stopping).
- Escada L0–L4 e JSON sintéticos.
- PDF/figuras/JSON Cora do unificado preservados.

## 0.2.0 — pacote unificado (2026-09-23)

- Fusão do ZIP completo com o repositório GitHub.
- `verify_results.py`, `docs/PROTOCOLO.md`.
- Números de AUC **não** foram alterados.
