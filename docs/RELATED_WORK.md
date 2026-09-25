# Trabalhos relacionados — bibliografia anotada (GLUE-LP)

> **Integridade.** Títulos, anos e ids arXiv/DOI abaixo foram verificados via WebSearch/WebFetch (2026-09-25).  
> **Correção de autoria:** SpotTarget = **Zhu et al.** (WSDM 2024), **não** Dong et al.  
> FakeEdge (Dong et al., 2022) é trabalho anterior distinto. Notas em pt-BR.

Cada entrada tem 3–6 frases sobre relevância para *target-edge leakage*, avaliação de LP ou encoders GNN no escopo do harness `glue_lp`.

---

## A. Predição clássica e surveys

### 1. Liben-Nowell, D., & Kleinberg, J. (2007). The link-prediction problem for social networks. *JASIST*, 58(7), 1019–1031. DOI: 10.1002/asi.20591
(Também versão CIKM 2003.) Sistematiza a tarefa de prever ligações futuras a partir da topologia. Populariza índices de vizinhança como *baseline*. Em GLUE-LP, motiva a obrigação de reportar CN/AA/PA no **mesmo** \(Q\) do GNN — sem isso o *run* é incompleto.

### 2. Adamic, L. A., & Adar, E. (2003). Friends and neighbors on the Web. *Social Networks*, 25(3), 211–230. DOI: 10.1016/S0378-8733(03)00009-1
Pondera vizinhos comuns por raridade (\(1/\log \mathrm{deg}\)). Implementado em `heuristics.py` como scorer AA. Na escada sintética, AA é menos sensível ao vazamento que preferential attachment — evidência qualitativa de que o atalho **não** é uniforme entre scorers (P1 do manuscrito; só medido no tech tree).

### 3. Newman, M. E. J. (2001). Clustering and preferential attachment in growing networks. *Physical Review E*, 64, 025102.
Motiva contagem de vizinhos comuns e crescimento preferencial. Preferential attachment (PA) em GLUE-LP escoreia \(\mathrm{deg}(u)\cdot\mathrm{deg}(v)\) sobre \(E_{mp}\): incluir \(Q^{+}\) em \(E_{mp}\) aumenta graus **só** nos positivos — bump aditivo exclusivo. Δ PA ≫ Δ CN/AA no sintético; confirmação em LINQS = **futuro trabalho**.

### 4. Barabási, A.-L., & Albert, R. (1999). Emergence of scaling in random networks. *Science*, 286, 509–512.
Modelo canônico de *preferential attachment*. Justifica por que PA é o scorer heurístico mais vulnerável a vazamento de grau quando positivos entram em \(E_{mp}\).

### 5. Lü, L., & Zhou, T. (2011). Link prediction in complex networks: A survey. *Physica A*, 390(6), 1150–1170. arXiv:1010.0725 · DOI: 10.1016/j.physa.2010.11.027
Survey clássico (similaridade local/global, *random walks*, máxima verossimilhança). Posiciona CN/AA/RA/Katz como família de *baselines* que GLUE-LP trata como obrigatórias no mesmo protocolo. Não discute GNN nem *target-edge leakage* — lacuna que SpotTarget/HeaRT/GLUE-LP cobrem décadas depois.

### 6. Martínez, V., Berzal, F., & Cubero, J.-C. (2016). A survey of link prediction in complex networks. *ACM Computing Surveys*, 49(4). DOI: 10.1145/3012704
Survey complementar (taxonomia ampla). Útil para mapear heurísticas vs métodos probabilísticos; GLUE-LP fica no eixo *evaluation protocol*, não no eixo *novo scorer*.

---

## B. Message passing e encoders medidos / próximos

### 7. Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., & Dahl, G. E. (2017). Neural message passing for quantum chemistry. *ICML*. arXiv:1704.01212
Formaliza MPNNs (mensagem + agregação + *update*). Vocabulário \(E_{mp}\) de GLUE-LP herda essa abstração: o que entra no grafo de agregação **é** o que o encoder vê. Sem separar aresta-alvo de mensagem, o decoder recebe atalho estrutural.

### 8. Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. *ICLR*. arXiv:1609.02907
GCN para classificação de nós (aproximação espectral de 1ª ordem). Encoder medido nas séries 1–2 e Pubmed. **Não** é o paper metodológico de *link prediction* — não confundir com VGAE.

### 9. Kipf, T. N., & Welling, M. (2016). Variational graph auto-encoders. *NIPS Workshop on Bayesian Deep Learning*. arXiv:1611.07308
**Referência metodológica principal** do harness: encoder GCN + decoder produto interno \(\langle z_u,z_v\rangle\), avaliados com AUC/AP em Cora/Citeseer. GLUE-LP mede o efeito de incluir/excluir a aresta-alvo em \(E_{mp}\) sob esse *template* GAE (versão determinística, sem KL).

### 10. Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive representation learning on large graphs. *NeurIPS*. arXiv:1706.02216
GraphSAGE: agregação amostrada / média e generalização indutiva. Em GLUE-LP: variante média plena (sem amostragem), medido em Cora; AUC 0,577 **pré-**`F.normalize` — re-AUC **não medido**. Split indutivo de nós (15%) é célula separada, inspirada nesta linha.

### 11. Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). Graph attention networks. *ICLR*. arXiv:1710.10903
Atenção mascarada na vizinhança. GAT 1 cabeça medido em Cora (onda 3): vazamento permanece (*t* alto no JSON publicado). Atenção **não** imuniza contra \(Q^{+}\subseteq E_{mp}\).

### 12. Zhang, M., & Chen, Y. (2018). Link prediction based on graph neural networks (SEAL). *NeurIPS*. arXiv:1802.09691
*Enclosing subgraph* + DRNL. Já discute remoção/injeção de aresta-alvo no subgrafo. **Fora do núcleo medido** de GLUE-LP (**proposto / não medido**). SpotTarget cita SEAL como precursor do diagnóstico de inclusão da aresta-alvo.

### 13. Berg, R. van den, Kipf, T. N., & Welling, M. (2017). Graph convolutional matrix completion. arXiv:1706.02263
Decoder bilinear em grafos bipartidos (recomendação). GLUE-LP permanece no produto interno homogêneo; GC-MC ilustra que o decoder pode mudar sem resolver o vazamento de \(E_{mp}\).

### 14. Grover, A., & Leskovec, J. (2016). node2vec: Scalable feature learning for networks. *KDD*. arXiv:1607.00653
*Walk*-based embeddings. Baseline clássico de LP/KGC em OGB/HeaRT. **Não medido** em GLUE-LP; útil como contraste futuro (encoder sem *message passing* explícito sobre \(E_{mp}\)).

### 15. He, X., Deng, K., Wang, X., Li, Y., Zhang, Y., & Wang, M. (2020). LightGCN: Simplifying and powering GCN for recommendation. *SIGIR*. arXiv:2002.02126 · DOI: 10.1145/3397271.3401063
GCN linear sem transformação/ativação (recomendação). Fora do núcleo medido; mostra que simplificações de agregação também dependem de quais arestas entram no grafo de treino.

### 16. Yun, S., Kim, S., Lee, J., Kang, J., & Kim, H. J. (2021). Neo-GNNs: Neighborhood overlap-aware GNNs for link prediction. *NeurIPS*.
Emparelha sobreposição de vizinhança com GNN. Fora do núcleo medido; HeaRT o inclui no *fair comparison*. Relevância: *pairwise structural features* ainda pressupõem um grafo de mensagem — vazamento de alvo continua sendo questão de protocolo.

### 17. Chamberlain, B. P., et al. (2023). Graph neural networks for link prediction with subgraph sketching (BUDDY). *ICLR*. arXiv:2209.15486
SF + MPNN escalável. Avaliado em HeaRT; **não medido** em GLUE-LP. Diferenciação: BUDDY é *modelo*; GLUE-LP é *harness*.

### 18. Wang, X., Yang, H., & Zhang, M. (2024). Neural common neighbor with completion (NCN/NCNC). *ICLR*. arXiv:2302.00890
MPNN-then-SF; depende fortemente de CN. **Não medido** aqui. Completar vizinhos comuns em grafo incompleto é ortogonal ao assert \(Q^{+}\cap E_{mp}=\emptyset\), mas reforça que a estrutura de avaliação importa.

---

## C. Avaliação, leakage, negativos e benchmarks

### 19. Zhu, J., Zhou, Y., Ioannidis, V. N., Qian, S., Ai, W., Song, X., & Koutra, D. (2024). Pitfalls in link prediction with GNNs: Understanding the impact of target-link inclusion & better practices (SpotTarget). *WSDM*. arXiv:2306.00899 · DOI: 10.1145/3616855.3635786
**Autoria correta: Zhu et al. — NÃO Dong et al.** Diagnostica overfitting, *distribution shift* e *test leakage* ao incluir arestas-alvo em \(E_{mp}\); propõe SpotTarget (excluir \(T_{\mathrm{low}}\) no treino; excluir todos os alvos no teste; integração DGL). GLUE-LP compartilha o diagnóstico e diverge no desenho: **assert incontornável** + escada L0–L4 + heurísticas no mesmo \(Q\) + domínio LINQS/sintético pequeno — **não** é *benchmark* amplo nem *framework* de amostragem DGL. Ver `COMPARATIVO_SPOTTARGET_HEART.md`.

### 20. Dong, K., Tian, Y., Guo, Z., Yang, Y., & Chawla, N. V. (2022). FakeEdge: Alleviate dataset shift in link prediction. arXiv:2211.15899
Precursor citado por SpotTarget: *shift* treino/teste por presença/ausência da aresta-alvo (foco em métodos de subgrafo tipo SEAL). Distinto de SpotTarget; possível origem da citação errônea “Dong et al. (SpotTarget)” em rascunhos anteriores do projeto.

### 21. Li, J., Shomer, H., Mao, H., Zeng, S., Ma, Y., Shah, N., Tang, J., & Yin, D. (2023). Evaluating graph neural networks for link prediction: Current pitfalls and new benchmarking (HeaRT). *NeurIPS Datasets and Benchmarks*. arXiv:2306.10453
*Fair comparison* + HeaRT (negativos personalizados via heurísticas RA/PPR/similaridade). Foca *easy negatives* e métricas inconsistentes. GLUE-LP foca *target-edge exclusion* como invariante de código; negativos hard-CN / *degree-matched* são políticas locais, **não** o *benchmark* HeaRT. OGB/Planetoid reavaliados em HeaRT — em GLUE-LP, OGB **não rodou**.

### 22. Hu, W., Fey, M., Zitnik, M., Dong, Y., Ren, H., Liu, B., Catasta, M., & Leskovec, J. (2020). Open Graph Benchmark: Datasets for machine learning on graphs. *NeurIPS*. arXiv:2005.00687
Splits oficiais, Hits@K/MRR, *leaderboards*. Stub `integrations/ogb_loader.py` — **OGB não foi executado**. Não reivindicar Hits@50 OGB a partir de AUC 1:1 LINQS.

### 23. Sen, P., Namata, G., Bilgic, M., Getoor, L., Galligher, B., & Eliassi-Rad, T. (2008). Collective classification in network data. *AI Magazine*, 29(3), 93–106. DOI: 10.1609/aimag.v29i3.2157
Fonte LINQS de Cora/CiteSeer (classificação coletiva). Corpora usados por GLUE-LP via `data_citation.py` / `data/LINQS.md`. Pubmed-Diabetes entra na suíte Planetoid popularizada depois (Yang et al.).

### 24. Yang, Z., Cohen, W., & Salakhutdinov, R. (2016). Revisiting semi-supervised learning with graph embeddings (Planetoid). *ICML*. arXiv:1603.08861
Populariza Cora/CiteSeer/Pubmed com *splits* e *embeddings*. Tutoriais Planetoid/PyG de LP frequentemente reutilizam `edge_index` completo no encoder — motivação prática do assert de GLUE-LP.

### 25. Shchur, O., Mumme, M., Bojchevski, A., & Günnemann, S. (2018). Pitfalls of graph neural network evaluation. arXiv:1811.05868
Pitfalls de avaliação em classificação de nós (splits, variância). Analogia direta: métricas de LP também colapsam sob protocolo complacente. Justifica *early stopping* simétrico e múltiplas seeds.

### 26. Errica, F., Podda, M., Bacciu, D., & Micheli, A. (2020). A fair comparison of graph neural networks for graph classification. *ICLR*. arXiv:1912.09893
>47k experimentos sob protocolo uniforme. Mensagem transferível: sem protocolo fixo, *rankings* de GNN são frágeis — o mesmo princípio aplica-se a LP com/sem exclusão de alvo.

### 27. Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in ML-based science. *Patterns*. arXiv:2207.07048 · DOI: 10.1016/j.patter.2023.100804
Taxonomia de *leakage* em ciência baseada em ML. *Target-edge leakage* é uma instância grafos-específica do mesmo gênero de contaminação treino/teste. Reforça o framing ético de não reportar leaky como *headline*.

### 28. Dwivedi, V. P., Joshi, C. K., Luu, A. T., Laurent, T., Bengio, Y., & Bresson, X. (2023). Benchmarking graph neural networks. *JMLR*, 24(43). arXiv:2003.00982
*Benchmark* de blocos GNN (muitos dados sintéticos + realistas). Lembra que GLUE-LP **não** é *leaderboard* de expressividade — é harness de regime de avaliação.

### 29. Fey, M., & Lenssen, J. E. (2019). Fast graph representation learning with PyTorch Geometric. *ICLR Workshop*. arXiv:1903.02428
PyG: SpotTarget nota que, por anos, exemplos/loaders não excluíam arestas-alvo no *mini-batch*. Stub `integrations/pyg_adapter.py` em GLUE-LP — **proposto / não medido**.

### 30. Pineau, J., et al. (2021). Improving reproducibility in machine learning research (NeurIPS 2019 reproducibility program). *JMLR*, 22(164). https://jmlr.org/papers/v22/20-303.html
Checklist de reprodutibilidade. GLUE-LP alinha: código + JSON versionados + `verify_results.py` + testes CI; falha ainda: re-run pós-fix e checksums LINQS pendentes.

---

## D. Design Science

### 31. Hevner, A. R., March, S. T., Park, J., & Ram, S. (2004). Design science in information systems research. *MIS Quarterly*, 28(1), 75–105.
Guidelines: relevância, rigor, desenho como busca, avaliação, contribuição, comunicação. Ver `DESIGN_SCIENCE.md` para mapa honesto artefato↔guidelines.

### 32. Peffers, K., Tuunanen, T., Rothenberger, M. A., & Chatterjee, S. (2007). A design science research methodology for information systems research. *JMIS*, 24(3), 45–77. DOI: 10.2753/MIS0742-1222240302
DSRM: identificação → objetivos → desenho/desenvolvimento → demonstração → avaliação → comunicação. GLUE-LP deve relatar o ciclo, não só o harness.

### 33. Gregor, S., & Hevner, A. R. (2013). Positioning and presenting design science research for maximum impact. *MIS Quarterly*, 37(2), 337–355.
Posiciona tipos de contribuição DSR (*improvement* / *invention* / *exaptation*). GLUE-LP = **improvement** de protocolo de avaliação (assert + escada), não invenção de encoder. DOI MISQ a confirmar no publisher antes da submissão final.

---

## Contagem

| Bloco | Entradas |
|---|---|
| A clássica/surveys | 6 |
| B encoders / LP-GNN | 12 |
| C avaliação / leakage / dados | 12 |
| D Design Science | 3 |
| **Total anotado** | **33** |

Todas as 33 entradas foram cruzadas com arXiv/DOI/venue verificáveis. SpotTarget credita **Zhu et al.** em todas as menções deste pacote de docs.
