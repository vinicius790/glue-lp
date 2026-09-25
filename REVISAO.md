# Revisão técnica — changelog

## Bugs corrigidos (código, verificados)
1. `torch_models.train_encoder`: early stopping por validação era um no-op (score sempre 0.0 quando `pos_val` fornecido → `stall` nunca incrementava). Corrigido para AUC real sobre negativos fixos. Callers (`run_extended.py`, `run_wave3.py`) agora passam `val` sempre, não só em modo "valid" — remove a assimetria entre os braços válido/leaky.
2. `models.embed_gcn`: removida ReLU da embedding final (só a camada oculta deve ter ativação antes de um decoder de produto interno).
3. `run.py`: AA estava hardcoded em 0.0; CN e PA nunca eram computados. Agora os três são calculados de verdade sobre o mesmo `mp`/`Q` do GNN.
4. `models.embed_gcn_directed_half`: corrigida semântica (adiciona 1 direção a um grafo que não tem a aresta, em vez de remover 1 direção de um grafo que tem as duas) — implementa L4 (meia-aresta) corretamente.
5. `tests/test_gat.py`: skip condicional se `torch` não estiver instalado (evita quebrar CI, que só instala numpy+pytest).

## Novo
- `scripts/leakage_ladder.py`: escada L0–L4 + enumeração exaustiva de pares no tech tree sintético. **Executado nesta revisão, 10 seeds** — resultados no apêndice de `docs/ARTIGO.md`.
- `scripts/checksum_data.py`: utilitário para o usuário gerar SHA-256 real dos arquivos LINQS locais (nenhum hash foi inventado).
- `stats.py`: `logit`, `cohens_dz`, `bootstrap_ci_delta`, `paired_t_logit`, `holm_bonferroni` — implementadas e testadas.
- `docs/ARTIGO.md`: apêndice novo documentando tudo acima, com os números reais medidos nesta sessão, mais correção de citação (VGAE 2016, não GCN 2017) e nota de desambiguação do nome.

## Verificação
Suítes puro-NumPy (`test_protocol_keys.py`, `test_metrics.py`) executadas manualmente (sem `pytest`, indisponível neste ambiente sem rede): **5/5 passaram**. `test_invariante.py` usa `pytest.raises` e não pôde ser executado aqui (falha de import de `pytest`, não falha de lógica) — rodar `pytest tests/` no seu ambiente para confirmar.

## Pendências reais
- Re-executar Cora/Citeseer/Pubmed com o `train_encoder` corrigido (requer PyTorch; indisponível nesta sessão sem rede).
- SAGE 0,577: aplicar `F.normalize` na saída de `GraphSAGE.encode` e reexecutar — recomendado, não implementado.
- Ligar `TrainConfig` (`config.py`) aos scripts reais, hoje desconectados (constantes duplicadas em cada script).
- Gerar checksums reais de `data/LINQS.md` com `scripts/checksum_data.py` contra os arquivos baixados localmente.
- CITATION.cff sem campo `authors:` — preencher antes de qualquer release/DOI (não preenchido aqui para não inventar autoria).
