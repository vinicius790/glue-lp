# Build notes — manuscrito GLUE-LP

- **Data (America/Sao_Paulo):** 2026-09-25
- **ARTIGO.md word count:** 10221
- **ARTIGO-COMPLETO.md:** cópia idêntica da forma longa canônica
- **PDF pages (A4):** 22
- **DOCX paths:** `GLUE-LP-artigo-cientifico.docx`, `docs/GLUE-LP-artigo-cientifico.docx`
- **PDF paths:** `GLUE-LP-artigo-cientifico.pdf`, `docs/GLUE-LP-artigo-cientifico.pdf`
- **Formatação:** Times New Roman 11, line spacing 1.15, margens ~2.3 cm, figuras fig1–fig7
- **Builder:** `scripts/build_manuscript.py` parseia o bulk de `docs/ARTIGO.md` (headings, parágrafos, tabelas, code fences)

## Integridade (review brief)

1. SpotTarget citado como **Zhu et al.** (WSDM 2024, arXiv:2306.00899) — nunca Dong et al.
2. Todos AUC/t/Δ de Cora/Citeseer/Pubmed rotulados **pré-correção ES**; banner no topo + ameaça interna §8.2.
3. H1 clara em série 1 / Citeseer / GAT / Pubmed; **apenas sugestiva** em Cora+ES (t=2.61 < 2.78).
4. SAGE 0.577 **não** promovido como pós-normalize.
5. OGB / L2 / multi-head / SAGE amostrado / SEAL: **não medidos**.
6. Cohen d_z + IC bootstrap calculados na discussão a partir dos vetores por seed dos JSON; Holm/logit disponíveis em `stats.py` mas não reescritos nas tabelas históricas.
7. Zero AUC inventados.

## Contagens auxiliares

- Referências numeradas no ARTIGO: ver seção Referências (~33+ entradas distintas)
- Menções "pré-correção": elevadas (banner + tabelas + ameaças)
- PDF alvo 12–16+ páginas corpo: alcançado (22 pp. A4 incluindo apêndices)
