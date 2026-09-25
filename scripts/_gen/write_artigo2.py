#!/usr/bin/env python3
from __future__ import annotations
import json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments"
OUT = ROOT / "docs" / "ARTIGO.md"

A = json.loads((EXP / "cora_e1_e4.json").read_text())
H = json.loads((EXP / "heuristics_rescored.json").read_text())["summary"]
E = json.loads((EXP / "extended_gcn_sage.json").read_text())
W = json.loads((EXP / "wave3_gat_pubmed.json").read_text())["summary"]
L = json.loads((EXP / "leakage_ladder_resumo.json").read_text())
S, X = A["summary"], E["summary"]

def ms(d, prec=3):
    std = d.get("std", d.get("sd", 0.0))
    return f"{d['mean']:.{prec}f} ± {std:.{prec}f}"

parts = []
def add(s=""):
    parts.append(s)

add("")
add("---")
add("")
add("## 3. Fundamentação formal")
add("")
add("### 3.1 Predição de ligações")
add("")
add(
    "Seja \\(G = (V, E)\\) um grafo simples não dirigido (após simetrização das "
    "citações LINQS), com atributos de nó \\(X \\in \\mathbb{R}^{n \\times d}\\). "
    "Um *split* de arestas particiona \\(E\\) em treino \\(E_{\\mathrm{train}}\\), "
    "validação \\(E_{\\mathrm{val}}\\) e teste \\(E_{\\mathrm{test}}\\). Os "
    "positivos de avaliação são \\(Q^{+} = E_{\\mathrm{test}}\\) (ou o subconjunto "
    "indutivo correspondente). Os negativos \\(Q^{-}\\) são pares não-aresta "
    "amostrados sob uma política declarada. O conjunto de avaliação é "
    "\\(Q = Q^{+} \\cup Q^{-}\\)."
)
add("")
add(
    "Um encoder \\(f_\\theta\\) produz embeddings \\(z_i = f_\\theta(X, E_{mp})_i\\), "
    "onde \\(E_{mp} \\subseteq E\\) é o conjunto de arestas visíveis ao "
    "*message passing*. O decoder escoreia pares "
    "\\(s(u,v) = \\langle z_u, z_v \\rangle\\) (produto interno; bilinear diagonal). "
    "O treino minimiza BCE com logits em positivos de treino contra negativos "
    "amostrados (1:1 nas séries medidas). A métrica principal de avaliação é a "
    "AUC Mann–Whitney em \\(Q\\), com empate contado como \\(\\tfrac{1}{2}\\) "
    "(implementação em `glue_lp.metrics.roc_auc`). Métricas auxiliares já "
    "presentes no código — *average precision* (AP), Hits@\\(K\\) e MRR — são "
    "registradas nos JSON da série 1; o *headline* do manuscrito permanece a AUC "
    "porque é a série completa e comparável entre GNN e heurísticas."
)
add("")
add("### 3.2 Definição de target-edge leakage")
add("")
add(
    "**Definição (leakage de aresta-alvo).** Dizemos que há *target-edge leakage* "
    "quando existe \\(e \\in Q^{+}\\) tal que \\(e \\in E_{mp}\\) (ou a versão "
    "dirigida correspondente após simetrização incompleta). No modo `valid`, o "
    "harness exige a negação dessa condição via `assert_no_leakage` "
    "(`glue_lp.protocol`). No modo `leaky`, a interseção é intencional e "
    "documentada: \\(E_{mp} = E_{\\mathrm{train}} \\cup E_{\\mathrm{val}} \\cup "
    "E_{\\mathrm{test}}\\) (nas células medidas)."
)
add("")
add(
    "A definição é operacional, não metafísica. Ela não cobre todos os "
    "*leakages* imagináveis (por exemplo, vazamento via hiperparâmetros "
    "escolhidos no teste), mas cobre o atalho estrutural que tutoriais "
    "reproduzem: o encoder agrega sobre a aresta que o decoder deveria "
    "adivinhar."
)
add("")
add("### 3.3 Escada L0–L4")
add("")
add(
    "Para tornar o vazamento *graduável* — e não um booleano único — adotamos "
    "uma escada operacional. Os níveis L1, L3 e L4 foram **medidos** no "
    "*tech tree* sintético (`leakage_ladder_resumo.json`); L0 e L2 são "
    "propostos / não medidos na escada JSON."
)
add("")
add("| Nível | Definição operacional | Status neste pacote |")
add("|---|---|---|")
add("| L0 | Conceito / *checklist* sem mp especial; documentação apenas | proposto |")
add("| L1 | \\(E_{mp} = E_{\\mathrm{train}}\\) (*valid*) | **medido** (sintético) |")
add("| L2 | \\(E_{mp} = E_{\\mathrm{train}} \\cup E_{\\mathrm{val}}\\) (intermediário) | proposto / não medido na escada JSON |")
add("| L3 | \\(E_{mp} = E_{\\mathrm{train}} \\cup Q^{+}\\) (*leaky* pleno de positivos) | **medido** |")
add("| L4 | \\(E_{mp} = E_{\\mathrm{train}} \\cup\\) meia-aresta (uma direção do alvo) | **medido** |")
add("")
add(
    "L4 modela um bug frequente ao portar código dirigido ↔ não dirigido: "
    "simetrização incompleta. Qualitativamente, L4 **não** é leakage "
    "\"pequena\": no sintético, a média L4 superou L3 "
    f"({ms(L['escada']['L4_half_edge'], 4)} vs {ms(L['escada']['L3'], 4)}), "
    "com menor desvio-padrão. Proposição qualitativa: *bugs silenciosos de "
    "meia-aresta podem inflar a métrica tanto quanto o vazamento completo, "
    "às vezes com variância menor* — argumentada pelos números medidos da "
    "escada, sem extrapolar para LINQS sem medição."
)
add("")
add("### 3.4 Proposições qualitativas sobre leakage (P1–P3)")
add("")
add(
    "**P1 (não-uniformidade).** O deslocamento Δ AUC sob vazamento não é "
    "uniforme entre scorers. Preferential attachment escoreia "
    "\\(\\mathrm{deg}(u)\\cdot\\mathrm{deg}(v)\\). Incluir positivos em "
    "\\(E_{mp}\\) aumenta os graus dos *endpoints* **somente** nos pares "
    "positivos — um *bump* aditivo exclusivo de \\(Q^{+}\\). CN e AA dependem "
    "de vizinhos comuns; o efeito de primeira ordem em \\(Q\\) amostrado pode "
    "ser menor, com efeitos de segunda ordem via grau dos intermediários. "
    f"**P1 confirmada no tech tree:** Δ(L3−L1) PA = +{L['escada']['delta_pa_auc_L3_menos_L1']['mean']:.4f}, "
    f"AA = +{L['escada']['delta_aa_auc_L3_menos_L1']['mean']:.4f}, "
    f"CN = +{L['escada']['delta_cn_auc_L3_menos_L1']['mean']:.4f}. "
    "Reexecução da escada em Cora/Citeseer/Pubmed: **futuro trabalho**."
)
add("")
add(
    "**P2 (visibilidade estatística depende do protocolo de treino).** O mesmo "
    "atalho estrutural pode ser *claro* ou *sugestivo* conforme o critério de "
    "parada e o corpus. Evidência: série 1 (épocas fixas) vs série 2 "
    "(*early stopping*) em Cora — ver §7. Não tratamos P2 como lei universal; "
    "tratamos como achado de laboratório que desarma a frase \"H1 sempre "
    "confirmada\"."
)
add("")
add(
    "**P3 (atenção não imuniza).** Se \\((u,v) \\in E_{mp}\\), o mecanismo de "
    "atenção pode *aumentar* o peso da mensagem no canal do rótulo. Evidência "
    f"medida: GAT-Cora *t* = {W['paired_t_gat']['t']:.1f}. Não afirmamos que "
    "multi-head se comporta igual (não medido)."
)
add("")
add("### 3.5 Métricas: Mann–Whitney, AP, Hits@K, MRR")
add("")
add(
    "A AUC implementada é a probabilidade de um positivo ranqueado acima de um "
    "negativo, com empates como meio. Isso importa para heurísticas: CN e AA "
    "produzem muitos empates em zero; uma AUC que ignore empates ou que use "
    "ordenação estável arbitrária infla o número. A série 1 gravou AUC "
    "\"ingênua\" ~1,0 para CN/AA no JSON bruto; `heuristics_rescored.json` "
    "recalcula com empate = ½ — **estes** são os números do manuscrito "
    f"({ms(H['valid_unif']['aa'])} no uniforme válido). AP, Hits@3, Hits@10 e "
    "MRR existem em `glue_lp.metrics` e no JSON da série 1; não os usamos como "
    "*headline* porque (i) Hits@K em amostragem 1:1 não é Hits@K OGB e "
    "(ii) a comparação GNN↔heurística é mais limpa em AUC Mann–Whitney."
)
add("")
add("---")
add("")
add("## 4. Método / artefato de Design Science")
add("")
add("### 4.1 Requisitos do artefato")
add("")
add(
    "Derivamos requisitos a partir do problema (Seção 1) e do enquadramento "
    "Hevner/Peffers:"
)
add("")
add(
    "- **R1 (invariante).** No modo válido, \\(Q^{+} \\cap E_{mp} = \\emptyset\\) "
    "é assert, não documentação."
)
add(
    "- **R2 (controle leaky).** O modo leaky existe, é explícito, e nunca é o "
    "número de *abstract* sem o par válido."
)
add(
    "- **R3 (mesmo Q).** GNN e heurísticas CN/AA/PA escoreiam o mesmo \\(Q\\) "
    "na mesma célula."
)
add(
    "- **R4 (políticas de negativos declaradas).** Uniforme, hard-CN e "
    "*degree-matched* são primeira classe."
)
add(
    "- **R5 (splits declarados).** Aleatório, indutivo (nós ocultos) e temporal "
    "(sintético) são suportados; LINQS não tem *timestamp* de aresta nativo."
)
add(
    "- **R6 (reprodutibilidade).** Seeds fixas, `TrainConfig` congelado, JSON "
    "versionado, testes CI sem GPU."
)
add(
    "- **R7 (honestidade).** O que não rodou (OGB, PyG, DGL, Optuna, W&B, "
    "SEAL, GAT multi-head, SAGE amostrado) está marcado como proposto / não medido."
)
add("")
add("### 4.2 Harness e invariante")
add("")
add("```text")
add("1. data ← load_cora | load_citeseer | load_pubmed | crafting_graph")
add("2. split ← random_edge_split | inductive_node_split | split_graph")
add("3. mp ← train                 se modo = valid")
add("   mp ← train ∪ val ∪ test    se modo = leaky")
add("4. se valid e (Q+ ∩ mp) ≠ ∅: raise AssertionError")
add("5. z ← train_encoder(X, mp, …)   # early stopping por AUC de val real")
add("6. Q- ← uniforme | hard-CN | degree-matched")
add("7. AUC(GNN, Q) e AUC(CN/AA/PA, mesmo Q)")
add("8. gravar célula em experiments/*.json")
add("```")
add("")
add(
    "A máscara não é comentário: `mp_edges` é argumento de `train_encoder` / "
    "`split_graph`. `assert_no_leakage` quebra o processo antes do treino. "
    "Testes em `tests/test_invariante.py` e `tests/test_leakage_ladder_smoke.py` "
    "cobrem o caminho feliz e o caminho que deve falhar."
)
add("")
add("### 4.3 Splits")
add("")
add(
    "**Aleatório (transdutivo de arestas).** Holdout 10% teste / 5% validação / "
    "resto treino, seed no gerador NumPy (`ProtocolConfig.test_frac=0.10`, "
    "`val_frac=0.05`). Usado nas séries 1–2 e onda 3 (exceto indutivo)."
)
add("")
add(
    "**Indutivo (nós ocultos).** 15% dos nós são removidos do grafo de treino "
    "(`inductive_node_frac=0.15`); positivos de teste são arestas que tocam "
    "nós ocultos. Medido em Cora (série 1). Interpretação: heurísticas de "
    "vizinhança perdem o suporte local; o GCN ainda lê *bag-of-words* via "
    "self-loop — não é milagre espectral."
)
add("")
add(
    "**Temporal sintético.** O *tech tree* (`graph.crafting_graph`) carrega "
    "*timestamps* de aresta; `protocol.split_graph` corta no tempo. LINQS "
    "não oferece relógio de aresta — não inventamos."
)
add("")
add("### 4.4 Negative sampling")
add("")
add(
    "- **Uniforme 1:1:** um negativo por positivo, amostrado uniformemente "
    "entre não-arestas. Célula mais complacente."
)
add(
    "- **Hard-CN:** negativos escolhidos entre pares de alto *common neighbors* "
    "(difíceis para índices de vizinhança). Célula que troca o ranking GCN↔AA."
)
add(
    "- **Degree-matched:** para cada positivo \\((u,v)\\), amostra-se \\(w\\) "
    "com \\(|\\mathrm{deg}(w)-\\mathrm{deg}(v)| \\le 2\\) (implementação do "
    "pacote). Alinha empiricamente com hard-CN no Cora válido "
    f"({ms(W['cora_gcn_degree_matched'])} vs {ms(S['E2_gcn_hard_auc'])})."
)
add("")
add("### 4.5 Encoders e decoder")
add("")
add(
    "- **GCN** (série 1: A denso; série 2 / onda 3: A esparso via "
    "`index_add_`): duas camadas, hidden 32, ReLU + dropout 0,5 na oculta, "
    "saída linear. Renormalização "
    "\\(\\hat{{A}} = D^{{-1/2}}(A+I)D^{{-1/2}}\\)."
)
add(
    "- **GraphSAGE:** concat \\([h_v, \\mathrm{mean}(N(v))]\\); média plena "
    "(sem amostragem). Saída com **`F.normalize` L2** aplicada no código como "
    "correção recomendada pelo paper; a AUC "
    f"{ms(X['cora_sage_valid'])} foi medida **antes** desta normalização — "
    "re-run **não medido**."
)
add(
    "- **GAT:** uma cabeça, LeakyReLU(0,2), ELU, dropout de atenção 0,4. "
    "Multi-head: proposto / não medido."
)
add(
    "- **Decoder:** produto interno \\(\\langle z_u, z_v \\rangle\\)."
)
add(
    "- **Otimização:** Adam, lr = 0,01, weight decay \\(5\\cdot 10^{{-4}}\\), "
    "BCE com logits, 1 negativo de treino por positivo."
)
add("")
add(
    "Série 1: 50 épocas fixas, sem *early stopping*. Série 2 / onda 3: "
    "*early stopping* por AUC de validação **real** (paciência 12; teto 70 "
    "épocas na série 2, 40 na onda 3 Cora, 25 no Pubmed). **Nota de "
    "integridade:** um *bug* histórico fazia `score = 0.0` sempre que "
    "`pos_val` era fornecido, matando o *early stopping* no modo válido "
    "(ver Apêndice E). O código foi corrigido; os números publicados de "
    "Cora/Citeseer/Pubmed foram medidos sob o código *anterior* à correção "
    "completa do critério simétrico. Re-run: **futuro trabalho** — não "
    "substituímos JSON à mão."
)
add("")
add("### 4.6 Domínio ilustrativo (tech tree)")
add("")
add(
    "O grafo de *crafting* (`crafting_graph`) tem 12 nós e 20 arestas com "
    "*timestamps*. Serve a três papéis: (i) *split* temporal honesto; "
    "(ii) escada L1/L3/L4 barata em CPU; (iii) enumeração exaustiva de pares "
    "(\\(\\binom{{12}}{{2}} = 66\\), ~54 avaliados após remover treino). "
    "Não é o estudo principal — é o laboratório de *leakage* controlado."
)
add("")
add("### 4.7 Critérios de avaliação do artefato (Design Science)")
add("")
add(
    "Seguindo Hevner, avaliamos o artefato por: **fidelidade** (o assert "
    "reproduz a definição formal), **completude** (modos, splits, negativos, "
    "heurísticas), **usabilidade** (CI sem GPU; `check_protocol.py`), "
    "**impacto epistêmico** (séries medidas que mostram o atalho e seus "
    "limites estatísticos). Não avaliamos por posição em *leaderboard* OGB."
)
add("")

# Section 5
add("---")
add("")
add("## 5. Stack tecnológico detalhado")
add("")
add(
    "Esta seção separa honestamente o que **rodou** nos JSON medidos do que "
    "está apenas proposto. Detalhe operacional adicional em "
    "`docs/TECNOLOGIAS.md` e `docs/ARQUITETURA.md`."
)
add("")
add("### 5.1 O que de fato rodou")
add("")
add("| Camada | Tecnologia | Papel | Status |")
add("|---|---|---|---|")
add("| Linguagem | Python ≥ 3.11 | pacote `glue_lp` | **medido** |")
add("| Tabular / RNG | NumPy ≥ 1.26 | splits, features, GNN sintético | **medido** |")
add("| Tensor / SGD | PyTorch ≥ 2.1 | GCN/SAGE/GAT, Adam, BCE, `index_add_` | **medido** (séries 1–2, onda 3) |")
add("| Teste | pytest | invariante, AUC, types, export, stubs | **medido** (CI) |")
add("| Figura | matplotlib | fig1–fig7 | **medido** (artefato) |")
add("| Manuscrito | python-docx + LibreOffice | DOCX → PDF | **medido** (build) |")
add("| Lint | ruff | estilo opcional | suporte |")
add("| CI | GitHub Actions | `pytest` em push | suporte |")
add("| API | FastAPI + uvicorn | health/check/experiments (somente leitura) | opcional |")
add("")
add("### 5.2 Proposto / não medido")
add("")
add("| Tecnologia | Intenção | Status |")
add("|---|---|---|")
add("| PyTorch Geometric (PyG) | adapter Planetoid → protocolo | stub `integrations/pyg_adapter.py` |")
add("| DGL | segundo ABI de convolução | stub `integrations/dgl_note.py` |")
add("| OGB (`ogbl-*`) | Hits@K oficial, splits oficiais | stub — **OGB não rodou** |")
add("| NetworkX | análise exploratória | proposto |")
add("| Optuna | HPO de hidden/lr/dropout | proposto |")
add("| Weights & Biases | tracking remoto | proposto |")
add("| GAT multi-head (4–8) | atenção plena | proposto (medido: 1 cabeça) |")
add("| GraphSAGE amostrado | S vizinhos/camada | proposto (medido: média plena) |")
add("| SEAL / enclosing subgraphs | decoder estrutural | proposto |")
add("| CUDA obrigatória | treino GPU | não exigido (Cora CPU ok) |")
add("")
add("### 5.3 Complexidade e memória")
add("")
add(
    "GCN esparso por época: \\(O((m+n)h + n h d)\\) com implementação por "
    "`index_add_`. GCN denso (série 1) materializa \\(A \\in \\mathbb{R}^{n "
    "\\times n}\\) — viável em Cora (\\(n=2708\\)), irresponsável em Pubmed "
    "(\\(n^2 \\approx 4\\cdot 10^8\\)). Por isso Pubmed só entra na onda 3 "
    "com encoder esparso. O *tech tree* (\\(n=12\\) ) cabe em NumPy puro e "
    "não depende de PyTorch para a escada."
)
add("")
add("### 5.4 Hardware (honestidade)")
add("")
add(
    "Os JSON medidos registram tempos de parede aproximados da série 2 "
    f"(~{E.get('seconds', '—')} s no log agregado) e da onda 3 "
    f"(~{W.get('seconds', json.loads((EXP/'wave3_gat_pubmed.json').read_text()).get('seconds','—'))} s). "
    "Não reivindicamos *benchmark* de hardware: as máquinas de medição "
    "originais foram ambientes CPU/GPU modestos de desenvolvimento. O CI "
    "público roda só testes leves (sem retreinar Cora). Reproduzir as tabelas "
    "completas exige PyTorch e os dados LINQS em `data/` (ver `data/LINQS.md`)."
)
add("")

text = OUT.read_text(encoding="utf-8") + "\n".join(parts)
OUT.write_text(text, encoding="utf-8")
print("after chunk2 words", len(re.findall(r"\S+", text)))
