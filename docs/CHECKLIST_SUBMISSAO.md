# Checklist de submissão — GLUE-LP

Gate pré-submissão (artigo + artefato). Marque só o que for verdadeiro **agora**.  
Referências: `AMEACAS_VALIDADE.md` (P0), `ROTEIRO_REEXECUCAO.md`, `RELATED_WORK.md`, `CITATION.cff`.

> **Regra de ouro:** não inventar AUC / *t* / Δ; não promover OGB ou SAGE pós-normalize sem JSON novo; não citar SpotTarget como “Dong et al.”.

---

## A. Identidade científica e citações

- [ ] **SpotTarget = Zhu et al.** (WSDM 2024) em PDF, `ARTIGO.md`, `RELATED_WORK.md`, tabelas comparativas — **zero** “Dong et al.” residual
- [ ] HeaRT = Li et al.; OGB = Hu et al.; Planetoid/LINQS = Sen / Yang — grafias e anos conferidos
- [ ] Abstract **não** apresenta modo `leaky` como *headline*
- [ ] H1 em Cora+ES: linguagem **sugestiva** (não “confirmada”) enquanto *t* < crítico ou seed inverter
- [ ] H2 / P1 (PA sob leakage): só **sintético**; marcado como não confirmado em LINQS
- [ ] Design Science: artefato (assert, harness) separado de fenômeno (Δ leaky−valid em corpora)

## B. Autores e citação do software

- [ ] `CITATION.cff` contém **autores reais** (não deixar vazio na versão pública; **não inventar** nomes)
- [ ] `version` alinhada a `pyproject.toml`
- [ ] LICENSE MIT e mensagem de citação coerentes com o PDF

## C. Gate P0 — pré-correção do early stopping (LINQS)

Os JSON `cora_e1_e4.json`, `extended_gcn_sage.json`, `wave3_gat_pubmed.json` (e rescored associado) foram medidos **antes** da correção completa do early stopping simétrico.

Escolher **uma** linha e cumprir:

- [ ] **Opção 1 — Re-run completo** segundo `docs/ROTEIRO_REEXECUCAO.md`, backup em `experiments/archive_pre_es_fix/`, novos JSON + `RESULTADOS.md` + tabelas do ARTIGO sincronizados  
  **ou**
- [ ] **Opção 2 — Downgrade explícito:** banner “exploratório / pré-correção ES” em **todas** as tabelas/figuras LINQS do PDF e do README; claims H1 LINQS não definitivos

Sem (1) ou (2): **não submeter** como evidência definitiva de H1.

Também:

- [ ] SAGE Cora 0,577: **não** como veredito pós-`F.normalize` (AUC nova ausente)
- [ ] Escada L1/L3/L4 sintética citada como **[MEDIDO]** e independente do ES

## D. Escopo negativo (o que NÃO afirmar)

- [ ] **OGB:** apenas “não medido / não executado” — sem Hits@K/MRR inventados
- [ ] Sem bipartidos / ogbl-* / temporal real em citação como se tivessem rodado
- [ ] L0 e L2: **propostos / não medidos** na escada JSON (não misturar com L1/L3/L4)
- [ ] `smoke_*.json`: nunca misturados com tabelas LINQS do paper

## E. Reprodutibilidade do artefato (sem retreinar LINQS)

Rodar na máquina de entrega:

```bash
make test
make protocol
make verify
make smoke          # opcional; gera smoke_*.json rotulado
make audit-stats    # relê arrays .values existentes; não fabrica seeds
```

- [ ] `pytest` verde (skips de torch/fastapi/networkx OK)
- [ ] `protocol-check` / `check_protocol.py`: invariante OK no tech tree
- [ ] `verify_results.py`: summaries batem com o texto citado
- [ ] Checksums de dados LINQS (se for reivindicar re-run): `scripts/checksum_data.py` — SHA reais

## F. Manuscrito e figuras

- [ ] PDF/DOCX regenerados (`make manuscript` ou `scripts/build_manuscript.py`) após mudanças textuais
- [ ] Figuras `figures/fig1`–`fig9` referenciadas com legenda e fonte (JSON / script)
- [ ] Apêndice de ameaças menciona P0
- [ ] Checklist estilo ACM / Pineau (secção 10 do ARTIGO) preenchida honestamente

## G. Código e organização

- [ ] Hiperparâmetros via `DEFAULT_TRAIN` — sem constantes mágicas divergentes nos runners
- [ ] Heurísticas CN/AA/PA no **mesmo** Q do GNN em cada célula reportada
- [ ] Modo `valid`: assert \(Q^{+} \cap E_{mp} = \emptyset\) no harness
- [ ] `docs/INDEX.md` e `docs/ESTRUTURA_REPOSITORIO.md` atualizados se pastas mudarem
- [ ] `CHANGELOG.md` com a versão a submeter

## H. Ética de reporte (Kapoor & Narayanan)

- [ ] Nenhuma tabela promove leaky como resultado principal sem o par `valid`
- [ ] Seed que inverte (ex.: Cora Série 2) **mencionada**, não omitida
- [ ] Banner pré-fix visível se Opção 2
- [ ] Diffs de JSON só via runners — sem edição manual de AUC

---

## Assinatura interna (opcional)

| Campo | Valor |
|---|---|
| Data (America/Sao_Paulo) | |
| Opção P0 (1 ou 2) | |
| Commit / tag | |
| Responsável | |

Quando tudo acima estiver marcado, o pacote está **elegível** a submissão sob as premissas documentadas — não “provado SOTA”.
