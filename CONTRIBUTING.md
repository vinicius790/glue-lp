# Contribuir para GLUE-LP

## Regras absolutas

1. **Não inventar métricas.** Qualquer AUC / t / Δ novo precisa de JSON em `experiments/` gerado por um runner versionado.
2. Marcar o que não foi medido como **proposto / não medido** ou **futuro trabalho**.
3. Não reportar modo `leaky` como resultado principal do abstract.
4. Toda célula nova no modo `valid` precisa do assert \(Q^{+} \cap E_{mp} = \emptyset\).
5. Heurísticas CN / AA / PA no **mesmo** conjunto de pares Q do GNN.
6. Hiperparâmetros medidos (hidden=32, lr=0.01, wd=5e-4, patience=12, seeds 0–4) vivem em `TrainConfig` — importar `DEFAULT_TRAIN`; não duplicar constantes mágicas.
7. Não inventar nomes de autores em `CITATION.cff`.
8. Manuscrito primário em **pt-BR**; abstract em inglês.
9. Arquivos `smoke_*.json` são sintéticos rotulados — **nunca** sobrescrever `cora_*.json` / `extended_*.json` / `wave3_*.json`.
10. Não alterar AUCs já medidos em `experiments/*.json` “na mão”.

## Checklist de PR

- [ ] `PYTHONPATH=src python3 -m pytest -q` verde (≥30 testes; skips OK)
- [ ] `glue-lp protocol-check` (ou `python3 -m glue_lp.cli protocol-check`) ok
- [ ] `scripts/check_protocol.py` ok
- [ ] Se alterou números publicados: JSON + `RESULTADOS.md` + `ARTIGO.md` sincronizados
- [ ] Integrações novas em `integrations/` como stub `NotImplementedError` + how-to até haver medição
- [ ] Novos módulos cobertos por testes (torch/fastapi/networkx → skip se ausente)
- [ ] Sem `__pycache__` / `.venv` no commit
- [ ] Entrada no `CHANGELOG.md`

## Onde colocar o quê

| Tipo | Destino |
|---|---|
| Tipos de domínio | `src/glue_lp/types.py` |
| Hiperparâmetros | `src/glue_lp/config.py` (`DEFAULT_TRAIN`) |
| CLI | `src/glue_lp/cli.py` |
| Métricas | `src/glue_lp/metrics.py` |
| Análise NetworkX | `src/glue_lp/graph_analysis.py` |
| Loader / split / modelo | pacote `glue_lp` |
| Runner que grava JSON | `scripts/run_*.py` / `scripts/smoke_synthetic.py` |
| Stub futuro (PyG/DGL/OGB) | `src/glue_lp/integrations/` |
| Texto científico | `docs/ARTIGO.md` (agente de manuscrito) |
| Notas desta wave de código | `_build_notes_code.md` |

## Manuscrito vs código

Outro agente pode estar editando `docs/ARTIGO.md` / `scripts/build_manuscript.py`.
Não conflitar: donos do código cuidam de `src/`, `tests/`, demais `scripts/`,
`pyproject.toml`, partes técnicas do README/CONTRIBUTING/CHANGELOG.
