# Glossário — GLUE-LP

Termos do protocolo e do pacote `glue_lp`. pt-BR.

---

| Termo | Definição |
|---|---|
| **GLUE-LP** | *Graph Link-evaluation Under Exclusion*. **Não** confundir com GLUE (benchmark de PLN). |
| **\(E_{mp}\)** | Conjunto de arestas visíveis ao *message passing* (índice de agregação do encoder). |
| **\(Q^{+}\)** | Positivos de avaliação (arestas em val/teste cujo escore entra na métrica). |
| **\(Q^{-}\)** | Negativos de avaliação (pares não-aresta amostrados). |
| **\(Q\)** | \(Q^{+} \cup Q^{-}\) — o mesmo para GNN e heurísticas numa célula. |
| **Invariante (modo `valid`)** | \(Q^{+} \cap E_{mp} = \emptyset\). Se falsa → `AssertionError`. |
| **Modo `valid`** | \(E_{mp}\) = arestas de **treino** apenas. Número do artigo. |
| **Modo `leaky`** | \(E_{mp}\) = treino ∪ val ∪ teste. Controle do atalho; **não** é *headline*. |
| **Target-edge leakage** | Existe \(e \in Q^{+}\) com \(e \in E_{mp}\) (ou direção correspondente). |
| **ProtocolMode** | Tipo literal `"valid"` \| `"leaky"` (`types.py`). |
| **L0** | Nível conceitual / *checklist*; **proposto**, não medido como célula JSON. |
| **L1** | \(E_{mp}\) = treino (= `valid`). **Medido** no tech tree. |
| **L2** | Intermediário (ex.: treino∪val). **Proposto / não medido** na escada JSON. |
| **L3** | \(E_{mp}\) = treino ∪ positivos (= `leaky` na escada). **Medido**. |
| **L4** | \(E_{mp}\) = treino ∪ **meia-aresta** (uma direção do alvo). **Medido**. |
| **Hard negative** | Negativo amostrado para ser difícil (ex.: alto CN com um extremo). Política `hard`. |
| **Degree-matched** | Negativo \(w\) com \(\lvert\mathrm{deg}(w)-\mathrm{deg}(v)\rvert \le 2\) (aprox.). |
| **Uniforme 1:1** | Um negativo aleatório por positivo (treino/avaliação conforme script). |
| **Split indutivo (nós)** | Fração de nós (15%) ocultos; arestas incidentes fora do treino. |
| **Split transdutivo / aleatório** | Holdout de arestas (10% teste, 5% val) sobre o mesmo conjunto de nós. |
| **Split temporal** | Por *timestamp* de aresta — só no tech tree sintético (LINQS sem tempo). |
| **CN / AA / PA** | Common Neighbors; Adamic–Adar; Preferential Attachment. |
| **Encoder** | GCN / GraphSAGE / GAT → embeddings \(z_i\). |
| **Decoder** | Produto interno \(\langle z_u, z_v \rangle\) (GAE-style). |
| **Early stopping** | Para por AUC de validação real (paciência 12); bug histórico documentado. |
| **Célula** | Uma combinação seed × mode × encoder × política de negativos. |
| **Harness** | Orquestração: split → mp → assert → treino → \(Q^{-}\) → métricas. |
| **SpotTarget** | Framework Zhu et al. (WSDM 2024) — **não** “Dong et al.”. |
| **HeaRT** | Política/benchmark de negativos difíceis (Li et al., 2023). |
| **OGB** | Open Graph Benchmark (Hu et al., 2020). **Não executado** neste artefato. |
| **LINQS / Planetoid** | Corpora Cora/Citeseer/Pubmed (Sen 2008; Yang et al. 2016). |
| **Tech tree** | Grafo sintético de *crafting* (12 nós, 20 arestas) em `graph.crafting_graph`. |
| **Pré-correção** | Resultados LINQS medidos antes do fix completo de `train_encoder`. |
| **\[MEDIDO\] / \[PROPOSTO\]** | Legenda de `TECNOLOGIAS.md`: executado nos JSON vs documentado sem run. |
