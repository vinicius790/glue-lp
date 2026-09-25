# Build notes — code wave 0.5.0 (2026-09-25, America/Sao_Paulo)

Owner: código (`src/`, `tests/`, `scripts/` exceto `build_manuscript.py`,
pyproject/requirements/README técnico). Manuscrito/PDF: outro agente.

## Integrity

- Nenhum AUC/t/Δ em `experiments/cora_*.json`, `extended_*.json`,
  `wave3_*.json`, `leakage_ladder_*.json`, `heuristics_*.json` foi editado.
- Protocolo invariante (`Q+ ∩ Emp = ∅` no modo valid) permanece sagrado.
- `smoke_*.json` é **novo**, sintético, rotulado `label=smoke` — não substitui séries medidas.

## Delivered

1. **CLI** `src/glue_lp/cli.py` (argparse) + `glue-lp` / `glue-lp-check` no pyproject.
   Subcomandos: `protocol-check`, `verify-results`, `leakage-ladder`,
   `export-summary`, `describe-config`.
2. **Domain types / config**: `TrainConfig`/`ProtocolConfig`/`DatasetSpec` com
   `to_dict()`; `DATASETS`; `ProtocolSnapshot`; runners já importam `DEFAULT_TRAIN`
   (`run_e1_e4`, `run_extended`, `run_wave3` alinhado a `CFG.seeds` / epochs).
3. **Metrics**: docs Mann–Whitney tie=0.5, AP, Hits@K, MRR + `metric_block`;
   testes manuais em `tests/test_metrics.py`.
4. **graph_analysis.py**: degree hist + clustering no crafting graph; opcional NetworkX.
5. **integrations/**: NotImplemented + how-to (PyG/DGL/OGB); `from_*` extras.
6. **api.py**: `/health`, `/protocol/invariant`, `/experiments/summary` (+ aliases).
7. **scripts/smoke_synthetic.py**: E2E NumPy → `experiments/smoke_<stamp>.json` + `smoke_latest.json`.
8. **Tests**: cli, metrics ampliados, graph_analysis, smoke guardrail, api routes,
   integrations; torch/gat continua skip sem torch.
9. **pyproject extras**: `torch`, `docs`, `api`, `analysis`, `dev`, `all`; versão 0.5.0.
10. README (quickstart pt-BR), CONTRIBUTING, CHANGELOG 0.5.0.

## Not touched

- `docs/ARTIGO.md`, `scripts/build_manuscript.py`, PDF/DOCX, `figures/*` binários.
- Valores numéricos medidos nos JSON de Cora/Citeseer/Pubmed/escada.

## Verify

```bash
pip install -q pytest numpy
PYTHONPATH=src python3 -m pytest -q
PYTHONPATH=src python3 -m glue_lp.cli protocol-check
PYTHONPATH=src python3 scripts/smoke_synthetic.py
```

Alvo: ≥30 testes passando (skips OK sem torch/fastapi/networkx).
