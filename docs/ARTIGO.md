# GLUE-LP: um protocolo executavel para avaliar predicao de ligacoes com GNN sem entregar a aresta-alvo ao modelo

Manuscrito de Design Science + pacote `glue_lp`. Convertido do PDF/DOCX. Figuras 1-7 nao estao neste arquivo.

## Resumo

Quem implementa predicao de ligacoes com GNN pela primeira vez quase sempre deixa a aresta que quer prever dentro do grafo de message passing. O modelo ve o rotulo. A metrica sobe. Dong et al. e Li et al. ja descreveram isso. Aqui a exclusao e um assert: se Q+ intersecta E_mp no modo valido, o processo falha.

Serie 1, GCN denso, 50 epocas, Cora, 5 seeds: AUC 0.708 ± 0.020 (valido) vs 0.826 ± 0.019 (leaky), Δ = +0.118.
Serie 2, GCN esparso + early stopping: Cora Δ +0.054, t = 2.61, gl = 4 (nao cruza 2,78). Citeseer Δ +0.100, t = 5.77.
Hard: GCN 0.619 ± 0.009. Adamic-Adar uniforme 0.720 ± 0.007, hard 0.542 ± 0.016.
Indutivo (15% nos ocultos): GCN 0.725 ± 0.012; CN/AA ~0.50.
GraphSAGE Cora valido: 0.577 ± 0.025.
GAT Cora: 0.782 ± 0.025 vs 0.876 ± 0.021, t = 20.4.
Pubmed GCN esparso, 3 seeds: 0.886 ± 0.003 vs 0.921 ± 0.009, t = 7.75, gl = 2.

Nao e SOTA. OGB nao rodou.

## Abstract

GLUE-LP makes target-edge exclusion an unskippable invariant. On Cora with a fixed-epoch GCN the leaky protocol adds 0.118 AUC. Early stopping replicates the direction on Citeseer (t = 5.77) but is only suggestive on Cora (t = 2.61, df = 4). Hard negatives and inductive splits change the ranking between GCN and neighborhood heuristics. No OGB. No new layer.

## 1. Por que este texto existe

Dado um grafo incompleto, quais pares deveriam ser aresta? O atalho e deixar (u, v) no indice da agregacao enquanto se pede ao modelo que adivinhe (u, v). Nao e malicia: e o default de um tutorial que reutiliza o Data do classificador de nos.

SpotTarget e HeaRT ja fizeram o diagnostico com mais folego de benchmark. A decisao daqui: o assert quebra se Q+ intersecta E_mp. Relatorio sem regime declarado nao conta. Heuristica classica entra no mesmo Q.

## 2. Protocolo de laboratorio

Cora LINQS: 2708 papers, 5278 arestas nao dirigidas, 1433 features. Citeseer: 3312 papers, 4536 arestas apos orfas. Features bag-of-words normalizadas por linha. Holdout 10% teste, 5% val, resto treino. Seed no gerador.

- Valido: E_mp = E_train. Se teste intersecta E_mp, AssertionError.
- Leaky: E_mp = treino ∪ val ∪ teste. Existe para medir o atalho, nao para o abstract.

GCN Kipf-Welling, 2 camadas, hidden 32, Adam lr=0.01, wd=5e-4, BCE, 1 negativo por positivo no treino. Serie 1: A denso, 50 epocas. Serie 2: A esparso (D^{-1/2} A D^{-1/2} por indice), early stopping na AUC de validacao do modo valido, paciencia 12, teto 70. SAGE: concat(self, media da vizinhanca). GAT: 1 cabeca, LeakyReLU 0.2, ELU. Decoder: s(u,v) = <z_u, z_v>.

## 3. Complexidade e Python

Cora/Citeseer cabem em A denso. Pubmed n=19717: n² ≈ 4e8, so esparso.
Tempo GCN esparso por epoca: O((m+n)h + n h d).
Hard negatives: O(|Q+| · P · grau), P=80.
AUC Mann-Whitney: O(|Q+| · |Q-|).

Pacote `glue_lp`, Python ≥ 3.11. PyTorch no SGD. NumPy no split. pytest na invariante. Sem PyG/DGL no nucleo. Sem sklearn: empate da AUC e 1/2 por construcao em `metrics.py`.

Ordem obrigatoria da celula: load → split → escolher E_mp → assert → treinar → amostrar Q- → AUC no mesmo Q das heuristicas. Inverter mascara e treino e o atalho.

## 4. Serie 1 — Cora, epocas fixas

| Condicao | Modelo | AUC |
| --- | --- | --- |
| valido / uniforme | GCN | 0.708 ± 0.020 |
| leaky / uniforme | GCN | 0.826 ± 0.019 |
| Δ leaky − valido | GCN | +0.118 |
| valido / hard | GCN | 0.619 ± 0.009 |
| valido / uniforme | AA | 0.720 ± 0.007 |
| valido / hard | AA | 0.542 ± 0.016 |
| indutivo / uniforme | GCN | 0.725 ± 0.012 |
| indutivo / uniforme | AA | 0.496 ± 0.001 |

No uniforme GCN e AA quase empatam. No hard o GCN passa. No indutivo CN/AA sao constantes (~0.50); PA cai a ~0.16 porque o positivo toca grau zero. O GCN ainda le o bag-of-words (MLP no vetor do paper + self-loop).

## 5. Serie 2, GAT, grau, Pubmed

Early stopping no valido: Cora GCN 0.728 ± 0.020 vs leaky 0.782 ± 0.046. Seed 4 inverte (leaky 0.729 vs valido 0.735). t = 2.61. Corte bilateral 5% com gl=4 e 2,78. H1 nao e lei neste recorte.

Citeseer: 0.733 ± 0.029 vs 0.833 ± 0.030, t = 5.77. Cinco seeds acima da diagonal.

SAGE Cora valido 0.577 ± 0.025. Esta instancia (media plena, 2 camadas, decoder bilinear) nao ajudou. Nao amostramos vizinhanca.

GAT 1 cabeca, Cora 5 seeds: 0.782 vs 0.876, Δ +0.094, t = 20.4. Atencao nao imuniza: a aresta-alvo vira canal com peso aprendido.

Degree-matched |deg(w)-deg(v)|≤2: GCN valido 0.611 ± 0.033, mesmo patamar do hard por CN.

Pubmed-Diabetes: 19717 nos, 44324 arestas, 500 features. 3 seeds, teto 25 epocas. 0.886 vs 0.921, Δ +0.035, t = 7.75, gl = 2.

Bins de grau (serie 2, Cora GCN valido): baixo ~0.70, medio ~0.73, alto ~0.76 (bin alto pequeno).

Cora/Citeseer LINQS nao tem ano na aresta. E4 temporal ficou no tech tree de 12 nos.

## 6. O que um revisor pode recusar

Engenharia sobre SpotTarget: concordamos no diagnostico; o objeto e o harness + assert + Cora/Citeseer/Pubmed. 5 seeds sao pouco — por isso o t vem com gl. Hidden 32 e raso. AUC 1:1 nao e Hits@50 do OGB. OGB nao rodou.

## 7. Como repetir

```bash
pip install numpy torch pytest matplotlib python-docx
PYTHONPATH=src python3 -m pytest -q
PYTHONPATH=src python3 scripts/run_e1_e4.py
PYTHONPATH=src python3 scripts/run_extended.py
```

Tarballs LINQS em `data/`. Sem conta em nuvem no protocolo.

## 8. Fora de proposito

OGB oficial, GAT multi-cabeca, amostragem SAGE, Hits@50 HeaRT, plugin de engine, LLM.

## 9. Fechamento

Clonar e rodar so o leaky produz numero mais bonito e sistema pior. A contribuicao e essa frase virar teste vermelho no CI. Citeseer ~10 pontos. Pubmed ~3. GAT Cora ~9. Cora-GCN com early stopping depende de como se para o treino.

## Referencias

1. Kipf & Welling, ICLR 2017. https://arxiv.org/abs/1609.02907
2. Hamilton, Ying, Leskovec, NeurIPS 2017. https://arxiv.org/abs/1706.02216
3. Velickovic et al., ICLR 2018. https://arxiv.org/abs/1710.10903
4. Gilmer et al., ICML 2017.
5. Liben-Nowell & Kleinberg, JASIST 2007.
6. Li et al., 2023. https://arxiv.org/abs/2306.10453
7. Dong et al., WSDM 2024. https://arxiv.org/abs/2306.00899
8. Hu et al., OGB 2020. https://arxiv.org/abs/2005.00687
9. Hevner et al., MISQ 2004.
10. Peffers et al., JMIS 2007.
11. Adamic & Adar, Social Networks 2003.
12. Newman, PRE 2001.
13. Sen et al., AI Magazine 2008 (LINQS).
14. Barabasi & Albert, Science 1999.
