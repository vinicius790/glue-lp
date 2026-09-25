# Protocolo GLUE-LP

Documento operacional do harness. Conceitos curtos: `GLOSSARIO.md`.  
Números: **somente** JSON em `experiments/` (apontados abaixo) — este arquivo **não** inventa AUC / *t* / Δ.

---

## 1. Invariante formal (modo `valid`)

Seja \(E\) o conjunto de arestas do grafo de entrada (não direcionadas sob a chave canônica `edge_key(u,v)`).

Numa célula experimental define-se:

- \(E_{\mathrm{train}}\) — arestas de treino após o split;
- \(Q^{+}\) — positivos de avaliação (val e/ou teste, conforme o runner);
- \(E_{mp}\) — arestas visíveis ao *message passing* do encoder;
- \(Q^{-}\) — negativos amostrados; \(Q = Q^{+} \cup Q^{-}\).

**Invariante (modo `valid`):**

\[
Q^{+} \cap E_{mp} = \emptyset
\]

Implementação: `glue_lp.protocol.assert_no_leakage(mp, positives)` — se a interseção não for vazia, levanta `AssertionError` (`"invariante violada: aresta-alvo em E_mp"`).

Detecção auxiliar: `leakage_exists(mp, positives) -> bool` (sem exceção).

**Consequência operacional:** o decoder pode pontuar o par \((u,v)\in Q^{+}\); o encoder **não** pode ter agregado mensagens ao longo dessa aresta (nem da direção canônica equivalente) via \(E_{mp}\).

No modo `leaky` a invariante **não** é exigida: \(E_{mp}\) inclui positivos-alvo de propósito (controle do atalho).

---

## 2. Modos (`ProtocolMode`)

Tipo literal em `types.py`: `"valid"` | `"leaky"`.

| Modo | Definição de \(E_{mp}\) | Assert | Uso no reporte |
|---|---|---|---|
| `valid` | \(E_{mp} := E_{\mathrm{train}}\) | Obrigatório | **Número do artigo** / *headline* |
| `leaky` | \(E_{mp} := E_{\mathrm{train}} \cup Q^{+}\) (treino ∪ val ∪ teste no pipeline LINQS) | Não aplica | Controle do atalho — **nunca** sozinho no abstract |

No tech tree sintético (`split_graph`):

- `valid` → `mp = train`
- `leaky` → `mp = train + positives`

Configuração: `ProtocolConfig` / `DEFAULT_PROTOCOL` em `config.py`.

---

## 3. Políticas de negativos (e o que **não** fazer)

A amostragem de \(Q^{-}\) é parte da **célula**, não um detalhe cosmético. Políticas usadas neste artefato:

| Política | Ideia | Onde aparece |
|---|---|---|
| `uniform` / uniforme 1:1 | Um (ou *k*) negativo(s) aleatório(s) por positivo | Séries LINQS; splits |
| `hard` / hard-CN | Prioriza pares com alto *common neighbors* sob o adj de \(E_{mp}\) | `split_graph(..., negatives="hard")`; células hard |
| `degree-matched` | Negativo \(w\) com grau próximo ao do extremo alvo | Onda 3 (ver JSON) |

### Políticas negativas (proibições de reporte)

1. **Não** trocar a política de negativos entre GNN e CN/AA/PA na mesma célula e chamar de “comparação justa”.
2. **Não** amostrar \(Q^{-}\) olhando labels de teste além do permitido pelo desenho da célula.
3. **Não** reportar só hard **ou** só uniforme quando a série experimental pedia o par.
4. **Não** promover Hits@K / MRR estilo OGB sem run OGB (este pacote: **OGB não medido**).
5. **Não** silenciar o modo `valid` e publicar só `leaky`.
6. **Não** editar vetores `.values` nos JSON na mão — use runners ou `apply_stats_from_json.py` (só lê).

Heurísticas **sempre** no mesmo \(Q\) do GNN na célula (`CONTRIBUTING.md` regra 5).

---

## 4. Escada de vazamento L0–L4

A escada ordena **quão** a aresta-alvo (ou proxy) entra em \(E_{mp}\). Definições alinhadas a `GLOSSARIO.md` e Sec. 3.3 do manuscrito.

| Nível | Definição operacional | Status neste artefato | Onde olhar números |
|---|---|---|---|
| **L0** | Vazamento também no *treino* / checklist conceitual (baseline degenerada) | **Proposto** — não há célula JSON L0 | — |
| **L1** | \(E_{mp} = E_{\mathrm{train}}\) (= modo `valid`) | **Medido** (tech tree) | `experiments/leakage_ladder_sintetico.json`, `leakage_ladder_resumo.json` |
| **L2** | Intermediário, ex. \(E_{mp} = E_{\mathrm{train}} \cup E_{\mathrm{val}}\) | **Proposto / não medido** na escada JSON | — |
| **L3** | \(E_{mp} = E_{\mathrm{train}} \cup Q^{+}\) (= modo `leaky` na escada) | **Medido** | mesmos JSON + resumo |
| **L4** | \(E_{mp} = E_{\mathrm{train}} \cup\) **meia-aresta** (alvo visível em **uma** direção) | **Medido** (`L4_half_edge`) | mesmos JSON; nota AA/CN/PA sob L4 no JSON |

Runner: `scripts/leakage_ladder.py` (NumPy puro; também `glue-lp leakage-ladder`).  
CLI/`make` **não** devem inventar L0/L2: ao imprimir, marque N/A ou “não medido”.

Resumo humano: `experiments/RESULTADOS.md` (secção da escada) e apêndice de `docs/ARTIGO.md`.  
Smoke rotulado (`smoke_*.json`) **não** substitui a escada oficial.

### Interpretação (sem números novos)

- Subir L1 → L3 → L4 aumenta a informação estrutural ilegal (ou semi-ilegal) disponível ao encoder.
- Em grafo pequeno (tech tree, \(n=12\)), a escada isola o mecanismo; **não** transferir AUC sintética para Cora como magnitude.
- Independência do bug de early stopping LINQS: a escada sintética **não** usa o loop PyTorch de ES — pode ser citada como **[MEDIDO]** mesmo sob banner pré-correção LINQS.

---

## 5. Ordem da célula

1. Carregar Cora / Citeseer / Pubmed (`data_citation.py`) ou tech tree (`graph.crafting_graph`).
2. Split: `random_edge_split` (10% teste, 5% val), `inductive_node_split` (15% nós) ou `protocol.split_graph` (temporal no sintético).
3. Montar \(E_{mp}\) conforme o modo (`valid` / `leaky` / nível L*).
4. Assert de leakage (modo `valid` / L1).
5. Treinar encoder **só** com \(E_{mp}\) (`train_encoder`; early stopping por AUC de validação **real** no código corrigido).
6. Amostrar negativos: uniforme | hard-CN | degree-matched (política da célula).
7. Avaliar AUC Mann–Whitney (e AP/Hits auxiliares se o runner gravar) no **mesmo** \(Q\) para GNN e heurísticas CN/AA/PA.
8. Persistir por seed em JSON (`summary.*.values`); agregar mean/sd / `paired_t_*` nos runners — nunca à mão.

---

## 6. Configuração canônica

Ver `TrainConfig` e `ProtocolConfig` em `src/glue_lp/config.py`.

Runners oficiais importam `DEFAULT_TRAIN` em vez de redefinir `hidden` / `lr` / `patience` à mão:

- `hidden=32`, `lr=0.01`, `wd=5e-4`, `patience=12`
- Seeds LINQS típicas: `0..4` (Pubmed onda 3: `0..2` no JSON existente)

Qualquer desvio vira **nova** célula documentada — não silencioso.

---

## 7. Verificação sem retreino

```bash
make protocol          # assert no tech tree
make verify            # relê experiments/*.json
make audit-stats       # Cohen dz / t / bootstrap CI a partir de .values existentes
PYTHONPATH=src python3 scripts/check_protocol.py
PYTHONPATH=src python3 examples/quickstart_synthetic.py
```

`verify_results.py` e `glue_lp.export` **só relêem**. Re-treino pós-correção de código LINQS = `ROTEIRO_REEXECUCAO.md` (futuro / P0).

---

## 8. O que este pacote não muda

Os JSON em `experiments/` (exceto novos `smoke_*.json` rotulados e re-runs explícitos) são os runs **já medidos**.  
LINQS atual: tratar como **pré-correção ES** até o gate P0 (`CHECKLIST_SUBMISSAO.md`, `AMEACAS_VALIDADE.md`).
