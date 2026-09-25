# Notas de build — pacote de docs científicos (docs agent)

Data local (America/Sao_Paulo): 2026-09-25.

## Escopo

Fortalecer `/workspace/artigo-src/GLUE-LP` com docs novos + expansão de arquitetura/tecnologias + figuras esquemáticas.  
**Não** editados: `docs/ARTIGO.md`, `scripts/build_manuscript.py` (agente paralelo).  
**Não** alterados: loops de treino (exceto leitura; bugs já corrigidos em sessão anterior).

## Correções do brief (`glue-lp-brief-revisao.md`)

1. SpotTarget = **Zhu et al.** (WSDM 2024) em RELATED_WORK e COMPARATIVO — **não** Dong.
2. FakeEdge (Dong et al., 2022) citado como linhagem distinta.
3. AMEACAS: medição pré-fix elevada a **validade interna P0**.
4. DESIGN_SCIENCE: Peffers (6 passos) + Hevner (7 guidelines) + separação artefato vs fenômeno.
5. ROTEIRO: re-run LINQS como **gate P0** pré-submissão.
6. Tabela de diferenciação GLUE-LP × SpotTarget × HeaRT × OGB (obrigatória).

## Docs criados / atualizados

| Caminho | Ação |
|---|---|
| `docs/RELATED_WORK.md` | **novo** — 33 entradas anotadas |
| `docs/DESIGN_SCIENCE.md` | **novo** |
| `docs/AMEACAS_VALIDADE.md` | **novo** |
| `docs/GLOSSARIO.md` | **novo** |
| `docs/ROTEIRO_REEXECUCAO.md` | **novo** |
| `docs/COMPARATIVO_SPOTTARGET_HEART.md` | **novo** |
| `docs/TECNOLOGIAS.md` | **expandido** (mermaid) |
| `docs/ARQUITETURA.md` | **expandido** (mermaid + sequência) |
| `docs/README.md` | índice atualizado |
| `_build_notes_docs.md` | este arquivo |

## Figuras novas (conceituais — sem AUC inventada)

| Caminho | Conteúdo |
|---|---|
| `figures/fig8_arquitetura.png` | Esquema de módulos do pacote |
| `figures/fig9_escada_vazamento.png` | Escada L0–L4 com status medido/proposto |

## Citações verificadas (amostra + lista do brief)

Verificadas via WebSearch/WebFetch nesta sessão, entre outras:

- Kipf GCN arXiv:1609.02907; VGAE arXiv:1611.07308
- Hamilton GraphSAGE arXiv:1706.02216; Veličković GAT arXiv:1710.10903
- Gilmer MPNN arXiv:1704.01212; Zhang SEAL arXiv:1802.09691
- Liben-Nowell & Kleinberg 2007; Adamic & Adar 2003
- Lü & Zhou survey arXiv:1010.0725
- **Zhu et al. SpotTarget** arXiv:2306.00899 / DOI 10.1145/3616855.3635786
- Li et al. HeaRT arXiv:2306.10453
- Hu OGB arXiv:2005.00687; Sen AI Mag 2008; Yang Planetoid arXiv:1603.08861
- Shchur arXiv:1811.05868; Errica arXiv:1912.09893; Kapoor arXiv:2207.07048
- Dwivedi arXiv:2003.00982; He LightGCN arXiv:2002.02126
- Chamberlain BUDDY arXiv:2209.15486; Wang NCNC arXiv:2302.00890
- Fey PyG arXiv:1903.02428; Pineau JMLR 2021; Hevner 2004; Peffers 2007; Gregor & Hevner 2013
- Grover node2vec arXiv:1607.00653; Yun Neo-GNN NeurIPS 2021

**Contagem no RELATED_WORK.md: 33 entradas anotadas** (todas com venue/arXiv/DOI verificáveis).  
Flag residual: DOI MISQ Gregor & Hevner 2013 — confirmar no publisher antes da biblio final do PDF.

## Integridade

- Nenhum AUC/t novo inventado.
- Resultados LINQS tratados como pré-correção ES.
- Escada sintética: remeter a `leakage_ladder_resumo.json` existente.
