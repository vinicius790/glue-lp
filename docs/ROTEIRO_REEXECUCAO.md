# Roteiro de reexecução — Cora / Citeseer / Pubmed (gate P0)

> **P0 antes de qualquer claim de submissão.**  
> Os JSON atuais de LINQS são **pré-correção** do *early stopping* simétrico.  
> Este roteiro re-mede com o código corrigido. **Não invente números** enquanto o re-run não terminar.

---

## 0. O que NÃO reivindicar até re-medir

- Que H1 está “confirmada” em Cora/Citeseer/Pubmed **pós-fix**.
- Que os *t* publicados (2,61 / 5,77 / 20,4 / 7,75) permanecem após o fix.
- Que SAGE 0,577 é o desempenho **com** `F.normalize` (código já normaliza; AUC nova **ausente**).
- Qualquer Hits@K OGB (OGB **não** faz parte deste roteiro).
- Médias editadas à mão em Markdown — só JSON gerados pelos scripts.

Permitido enquanto isso: falar do **artefato** (assert, CI, escada sintética L1/L3/L4 já re-medida), e citar LINQS com *banner* “exploratório / pré-correção”.

---

## 1. Pré-requisitos

```bash
cd /caminho/GLUE-LP
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e ".[torch,dev,docs]"
# ou: pip install -r requirements.txt && pip install -e ".[dev]"
```

Dados LINQS (não versionados) — ver `data/LINQS.md`:

```
data/cora/cora.content
data/cora/cora.cites
data/citeseer/citeseer.content
data/citeseer/citeseer.cites
data/Pubmed-Diabetes/data/Pubmed-Diabetes.NODE.paper.tab
data/Pubmed-Diabetes/data/Pubmed-Diabetes.DIRECTED.cites.tab
```

Opcional mas recomendado:

```bash
PYTHONPATH=src python3 scripts/checksum_data.py   # SHA-256 reais; não inventar
```

---

## 2. Sanidade (sem retreinar LINQS)

```bash
PYTHONPATH=src python3 -m pytest -q
PYTHONPATH=src python3 scripts/check_protocol.py
PYTHONPATH=src python3 scripts/verify_results.py
PYTHONPATH=src python3 scripts/leakage_ladder.py   # sintético; pode sobrescrever JSON da escada
```

Esperado: testes verdes (torch/fastapi skipped se ausentes); assert OK; `verify_results` relê summaries.

---

## 3. Backup dos JSON pré-correção

**Não apague a evidência histórica.**

```bash
mkdir -p experiments/archive_pre_es_fix
cp -n experiments/cora_e1_e4.json \
      experiments/extended_gcn_sage.json \
      experiments/wave3_gat_pubmed.json \
      experiments/heuristics_rescored.json \
      experiments/archive_pre_es_fix/
```

---

## 4. Re-treino (ordem sugerida)

Hiperparâmetros canônicos: `TrainConfig` / `DEFAULT_TRAIN` (`hidden=32`, `lr=0.01`, `wd=5e-4`, `patience=12`, seeds 0–4; Pubmed seeds 0–2).

```bash
# Série 1 — Cora GCN denso, 50 épocas fixas
PYTHONPATH=src python3 scripts/run_e1_e4.py
# → experiments/cora_e1_e4.json

# Série 2 — GCN/SAGE esparso + early stopping (Cora, Citeseer)
PYTHONPATH=src python3 scripts/run_extended.py
# → experiments/extended_gcn_sage.json

# Onda 3 — GAT Cora, degree-matched, Pubmed GCN
PYTHONPATH=src python3 scripts/run_wave3.py
# → experiments/wave3_gat_pubmed.json
```

Tempo: CPU ok para Cora/Citeseer; Pubmed é o mais longo (ainda factível em CPU esparso). GPU opcional.

---

## 5. Artefatos esperados pós-re-run

| Arquivo | Conteúdo |
|---|---|
| `cora_e1_e4.json` | Células E1/E2/indutivo + `summary` |
| `extended_gcn_sage.json` | GCN/SAGE + `paired_t_*` |
| `wave3_gat_pubmed.json` | GAT, degree-matched, Pubmed |
| logs `*_log.txt` | Traço por seed (se o script gravar) |

Atualizar `experiments/RESULTADOS.md` **somente** com médias lidas dos JSON novos.  
Opcional: coluna lado a lado pré vs pós no manuscrito (transparência).

```bash
PYTHONPATH=src python3 scripts/verify_results.py
# regenerar figuras de barras só se make_figures.py ler JSON — senão atualizar à mão com valores verificados
```

---

## 6. Critérios de aceite (gate P0)

1. Cada runner termina sem erro; modo `valid` nunca viola o assert.
2. JSON novos versionados; archive pré-fix preservado.
3. Reportar por seed: AUC valid, AUC leaky, Δ; *t* pareado + gl.
4. Ideal: Cohen *d_z* e IC bootstrap via `stats.py` (já no pacote).
5. Linguagem do paper atualizada: remover “pré-correção” **só** depois destes JSON.
6. SAGE: reportar AUC **nova** com `F.normalize`; arquivar 0,577 como histórico.

Se o re-run **não** for feito: manter banner em resumo, §6, §7, §8 e tabelas.

---

## 7. Fora deste roteiro (não misturar)

- OGB Hits@K (`integrations/ogb_loader.py` ainda stub).
- Optuna / W&B / GAT multi-head / SAGE com amostragem.
- Confirmar P1 (Δ PA) em LINQS — experimento **adicional** após o P0.

---

## 8. Comando mínimo “reproduce read-only” (sem retreino)

```bash
PYTHONPATH=src python3 -m pytest -q
PYTHONPATH=src python3 scripts/check_protocol.py
PYTHONPATH=src python3 scripts/verify_results.py
```

Isso **valida o artefato** e a leitura dos JSON; **não** substitui o gate P0 de re-treino.
