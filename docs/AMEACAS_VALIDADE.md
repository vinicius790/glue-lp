# Ameaças à validade — GLUE-LP (expandido)

Taxonomia inspirada em Wohlin et al. (experimentação em SE) + Kapoor & Narayanan (leakage em ML).  
Mitigações concretas no pacote; o que **ainda** está aberto.

---

## P0 — Validade interna: medição pré-correção do early stopping

> **Esta é a ameaça #1 do artefato atual. Não é nota de rodapé.**

| Item | Detalhe |
|---|---|
| **Fato** | AUC / *t* de Cora, Citeseer e Pubmed em `experiments/*.json` foram medidos sob `train_encoder` **antes** da correção completa do *early stopping* simétrico (`score` de validação morto / assimetria valid vs leaky). Ver `REVISAO.md`, Apêndice A do ARTIGO. |
| **Efeito** | O *headline* Série 1 (Δ +0,118) e os *t* Série 2 / Onda 3 podem **mudar** após re-execução. A direção (leaky ≥ valid) é plausível, mas a magnitude e a significância **não** estão blindadas. |
| **Mitigação atual** | Código corrigido; documentado; *banner* de “pré-correção” no README/RESULTADOS. |
| **Mitigação obrigatória pré-submissão** | Re-run completo segundo `ROTEIRO_REEXECUCAO.md` **ou** downgrade explícito de todos os claims LINQS a “exploratório / pré-fix”. |
| **Gate** | Sem re-run (ou banner inequívoco), **não** submeter como evidência definitiva de H1. |

Ameaça relacionada (também interna): SAGE Cora 0,577 medido **antes** de `F.normalize`; re-AUC **não medido** — não promover como conclusão sobre GraphSAGE.

---

## 1. Validade de construto

| Ameaça | Por que importa | Mitigação | Residual |
|---|---|---|---|
| AUC 1:1 ≠ Hits@K / MRR OGB | Construto “desempenho de LP” não é único | Declarar métrica; **não** equivaler a OGB | Alto se leitor espera Hits@50 |
| “Válido” = só excluir teste? | SpotTarget distingue treino vs teste | Modo `valid` = \(E_{mp}\) = treino; leaky = treino∪val∪teste | L2 (treino∪val) proposto / não medido |
| Leakage = só \(e\in Q^{+}\cap E_{mp}\)? | SpotTarget também fala overfit / *shift* | Definição operacional no assert; escada L4 cobre meia-aresta | Sem lema formal \(\Rightarrow\langle z_u,z_v\rangle\) |
| Heurísticas “justas” | Hard-CN redefine o que AA/CN veem | Reportar política de negativos; AA hard ~0,54 no JSON | Hard pode parecer “ataque” a heurísticas |

---

## 2. Validade interna (além do P0)

| Ameaça | Detalhe | Mitigação | Residual |
|---|---|---|---|
| **P0 pré-fix** | Ver caixa acima | Re-run | **Crítico** |
| Poucas seeds | 5 (Cora/Citeseer/GAT); 3 (Pubmed, gl=2) | Seeds fixas 0–4; reportar seed que inverte | Alto — meta ≥10–30 pós-re-run |
| Cora Série 2 *t*=2,61 < 2,78 | Não cruzar α=5% bilateral | Texto: **sugestivo**, não “confirmado” | — |
| `stats.py` não aplicado | Cohen *d_z*, bootstrap, Holm existem mas tabelas usam só média±sd e *t* | Futuro: reprocessar JSON | Médio |
| Assimetria histórica ES | Valid parava “errado”; leaky não | Corrigido no código | Só resolve após re-run |
| Contaminação de implementação | Bug L4 meia-aresta; AA=0 no sintético | Corrigidos e medidos no sintético | LINQS não re-medido |

---

## 3. Validade externa

| Ameaça | Detalhe | Mitigação | Residual |
|---|---|---|---|
| Só LINQS + tech tree 12 nós | Sem bipartidos, sem ogbl-*, sem tempo real em citação | Escopo negativo explícito; OGB stub | **Alto** |
| Hidden 32, 2L, GAT 1 cabeça | Configuração rasa | Objeto = protocolo, não SOTA | Médio |
| Split aleatório 10%/5% | ≠ splits OGB/HeaRT unificados | Documentar; não comparar *rankings* cruzados | Médio |
| Domínio crafting sintético | n=12, variância alta | Qualitativo (L4 ≈ L3); não transferir AUC | Alto se generalizar números |

---

## 4. Validade de conclusão

| Ameaça | Detalhe | Mitigação |
|---|---|---|
| Confundir artefato com fenômeno | “O assert funciona” ≠ “H1 sempre significativa” | `DESIGN_SCIENCE.md` §3 |
| *p-hacking* de protocolo | Escolher só leaky ou só uniforme | Célula manda ambos os modos + heurísticas |
| Generalizar P1 (PA) | Só sintético | Marcar H2 como **não confirmada** em LINQS |
| Citação errônea SpotTarget | “Dong et al.” no rascunho | **Corrigido** nos docs deste pacote → Zhu et al. |

---

## 5. Ética de reporte (Kapoor & Narayanan)

Reportar leaky como *headline*, omitir a seed que inverte, ou silenciar o *banner* pré-fix é o **mesmo gênero** de atalho que o paper critica. Mitigação: JSON imutáveis + `verify_results.py` + este documento.

---

## Checklist rápido pré-submissão

- [ ] P0: re-run LINQS **ou** banner “exploratório / pré-fix” em **todas** as tabelas LINQS
- [ ] SpotTarget = Zhu et al. em todo o PDF
- [ ] H1 Cora+ES: linguagem “sugestiva”
- [ ] H2/P1: só sintético
- [ ] OGB: “não medido”
- [ ] SAGE 0,577: não como veredito pós-normalize
