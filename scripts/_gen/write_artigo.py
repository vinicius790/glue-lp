#!/usr/bin/env python3
"""Generate the dense Design Science ARTIGO.md (≥10k words) from measured JSON only."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments"
OUT = ROOT / "docs" / "ARTIGO.md"
OUT2 = ROOT / "docs" / "ARTIGO-COMPLETO.md"

A = json.loads((EXP / "cora_e1_e4.json").read_text())
H = json.loads((EXP / "heuristics_rescored.json").read_text())["summary"]
E = json.loads((EXP / "extended_gcn_sage.json").read_text())
W = json.loads((EXP / "wave3_gat_pubmed.json").read_text())["summary"]
L = json.loads((EXP / "leakage_ladder_resumo.json").read_text())
S, X = A["summary"], E["summary"]

def ms(d, prec=3):
    std = d.get("std", d.get("sd", 0.0))
    return f"{d['mean']:.{prec}f} ± {std:.{prec}f}"

def f3(x):
    return f"{x:.3f}"

# Per-seed strings for transparency
def vals(d, key="values"):
    if key in d:
        return ", ".join(f"{v:.4f}" for v in d[key])
    return "—"

parts: list[str] = []

def add(s: str = ""):
    parts.append(s)

# ============================================================================
add("# GLUE-LP: um protocolo executável para avaliar predição de ligações com GNN sem entregar a aresta-alvo ao modelo")
add("")
add("**Graph Link-evaluation Under Exclusion**")
add("")
add("*(Título disambiguado do *benchmark* GLUE de PLN — Wang et al., 2018 — com o qual este projeto não tem relação.)*")
add("")
add("**Autores:** *[placeholder — nomes]*  ")
add("**Afiliação:** *[placeholder — instituição / laboratório / grupo]*  ")
add("**Contato:** *[placeholder — email]*  ")
add("**Versão do artefato:** 0.5.0 · **Idioma principal:** pt-BR · **Abstract:** English  ")
add("**Tipo:** manuscrito de *Design Science* (IMRaD) acompanhado do pacote reproduzível `glue_lp`")
add("")
add("> **Integridade experimental.** Todos os AUC, *t* de Student pareado, desvios-padrão e Δ citados neste manuscrito foram lidos de `experiments/*.json`, `experiments/RESULTADOS.md` ou do apêndice medido da escada de vazamento. O que não foi medido está marcado explicitamente como **proposto / não medido** ou **futuro trabalho**. Não inventamos números. Não afirmamos SOTA. OGB **não rodou**.")
add("")
add("---")
add("")

# ========== RESUMO ==========
add("## Resumo")
add("")
add(
    "A predição de ligações (*link prediction*) pergunta, dado um grafo incompleto, "
    "quais pares de vértices deveriam ser aresta. Em recomendação, em redes de citação "
    "científica e em árvores de *crafting* de jogos, a pergunta é a mesma; o instrumento "
    "moderno é uma rede neural de grafos (GNN). O atalho moderno — e não raro o *default* "
    "de tutoriais que reutilizam o objeto `Data` do classificador de nós — é deixar o par "
    "alvo \\((u,v)\\) no índice de *message passing* enquanto se pede ao modelo que "
    "adivinhe \\((u,v)\\). O encoder vê o rótulo. A métrica sobe. O artigo parece melhor "
    "do que o sistema em produção, onde a aresta futura ainda não existe."
)
add("")
add(
    "Zhu et al. (SpotTarget) e Li et al. (HeaRT) descreveram o diagnóstico com fôlego "
    "de *benchmark*. A decisão de desenho deste artefato é outra: a exclusão da aresta-alvo "
    "não é um parágrafo de *related work* nem um *flag* silencioso — é um **assert** que "
    "quebra o processo. Formalmente, seja \\(E_{mp}\\) o conjunto de arestas visíveis ao "
    "*message passing* e \\(Q^{+}\\) o conjunto de positivos de avaliação. O protocolo "
    "**válido** exige \\(Q^{+} \\cap E_{mp} = \\emptyset\\). O protocolo **leaky** admite "
    "\\(Q^{+} \\subseteq E_{mp}\\) como controle do atalho. O relatório sem regime declarado "
    "não conta. Heurísticas clássicas (vizinhos comuns, Adamic–Adar, *preferential attachment*) "
    "entram no mesmo \\(Q\\), ou o *run* é incompleto."
)
add("")
add(
    f"**Série 1** (GCN denso, 50 épocas fixas, Cora, 5 seeds, negativos 1:1): "
    f"AUC {ms(S['E1_gcn_valid_auc'])} no protocolo válido contra "
    f"{ms(S['E1_gcn_leaky_auc'])} no leaky "
    f"(Δ = +{S['E1_delta_leaky_minus_valid']:.3f}). "
    f"Negativos hard por vizinhos comuns: GCN {ms(S['E2_gcn_hard_auc'])}. "
    f"Adamic–Adar (AUC Mann–Whitney com empate = ½): "
    f"{ms(H['valid_unif']['aa'])} no uniforme e {ms(H['valid_hard']['aa'])} no hard. "
    f"Split indutivo (15% dos nós ocultos): GCN {ms(S['E_inductive_gcn_auc'])}; "
    f"CN/AA caem à chance ({ms(H['inductive']['aa'])})."
)
add("")
add(
    f"**Série 2** (GCN esparso + *early stopping*, paciência 12): "
    f"Cora válido {ms(X['cora_gcn_valid'])} vs leaky {ms(X['cora_gcn_leaky'])}, "
    f"*t* pareado = {X['paired_t_cora']['t']:.2f} (gl = 4; a seed 4 inverte o sinal; "
    f"o limiar bilateral 5% ≈ 2,78 **não** é atravessado). "
    f"Citeseer válido {ms(X['citeseer_gcn_valid'])} vs leaky {ms(X['citeseer_gcn_leaky'])}, "
    f"*t* = {X['paired_t_citeseer']['t']:.2f} (Δ médio +{X['paired_t_citeseer']['mean_diff']:.3f}). "
    f"GraphSAGE (média plena, Cora válido): {ms(X['cora_sage_valid'])}."
)
add("")
add(
    f"**Onda 3:** GAT (1 cabeça) em Cora — válido {ms(W['cora_gat_valid'])} vs "
    f"leaky {ms(W['cora_gat_leaky'])}, *t* = {W['paired_t_gat']['t']:.1f}. "
    f"GCN com negativos *degree-matched*: {ms(W['cora_gcn_degree_matched'])}. "
    f"Pubmed (GCN esparso, 3 seeds): válido {ms(W['pubmed_gcn_valid'])} vs "
    f"leaky {ms(W['pubmed_gcn_leaky'])}, *t* = {W['paired_t_pubmed']['t']:.2f} (gl = 2)."
)
add("")
add(
    f"**Escada de vazamento** no *tech tree* sintético (12 nós, 10 seeds, NumPy): "
    f"L1 {ms(L['escada']['L1'], 4)}, L3 {ms(L['escada']['L3'], 4)}, "
    f"L4 (meia-aresta) {ms(L['escada']['L4_half_edge'], 4)}. "
    f"*Preferential attachment* sobe Δ(L3−L1) = +{L['escada']['delta_pa_auc_L3_menos_L1']['mean']:.3f}. "
    "Um bug silencioso de meia-aresta (L4) pode inflar a AUC tanto quanto ou mais que o vazamento completo (L3)."
)
add("")
add(
    "Não é SOTA. **OGB não rodou.** Não propomos nova camada de convolução — propomos um "
    "*harness* de Design Science que recusa o atalho: a exclusão vira invariante executável, "
    "o mesmo \\(Q\\) alimenta GNN e heurísticas, e o CI fica vermelho quando alguém esquece."
)
add("")
add(
    "**Palavras-chave:** GNN; predição de ligações; vazamento (*leakage*); Cora; Citeseer; "
    "Pubmed; avaliação indutiva; Design Science; protocolo executável; Adamic–Adar; "
    "GraphSAGE; GAT; VGAE."
)
add("")

# ========== ABSTRACT ==========
add("## Abstract")
add("")
add(
    f"GLUE-LP (*Graph Link-evaluation Under Exclusion*) is a Design Science artifact that "
    f"makes target-edge exclusion an unskippable invariant of link-prediction evaluation with "
    f"graph neural networks: \\(Q^{{+}} \\cap E_{{mp}} = \\emptyset\\) in valid mode, enforced "
    f"by an assert that aborts the run. The same evaluation set \\(Q\\) is scored by GCN, "
    f"GraphSAGE, GAT and classical heuristics (common neighbors, Adamic–Adar, preferential "
    f"attachment) under uniform, hard-CN and degree-matched negative sampling, with random, "
    f"inductive-node and synthetic-temporal splits."
)
add("")
add(
    f"On Cora with a fixed-epoch dense GCN (Series 1, 5 seeds) the leaky protocol adds "
    f"+{S['E1_delta_leaky_minus_valid']:.3f} AUC "
    f"({ms(S['E1_gcn_valid_auc'])} valid vs {ms(S['E1_gcn_leaky_auc'])} leaky). "
    f"A second series with sparse GCN and early stopping replicates the direction on Citeseer "
    f"(paired \\(t = {X['paired_t_citeseer']['t']:.2f}\\), Δ = +{X['paired_t_citeseer']['mean_diff']:.3f}) "
    f"but is only suggestive on Cora (\\(t = {X['paired_t_cora']['t']:.2f}\\), df = 4; one seed "
    f"inverts). GAT on Cora yields \\(t = {W['paired_t_gat']['t']:.1f}\\); Pubmed GCN "
    f"(3 seeds) yields \\(t = {W['paired_t_pubmed']['t']:.2f}\\). A synthetic leakage ladder "
    f"(L1/L3/L4, 10 seeds) shows that a silent half-edge bug (L4) can inflate AUC as much as "
    f"full leakage (L3), while preferential attachment is far more sensitive "
    f"(Δ = +{L['escada']['delta_pa_auc_L3_menos_L1']['mean']:.3f}) than CN/AA. "
    "No OGB runs. No new message-passing layer. All numbers come from measured "
    "`experiments/*.json`; unmeasured items are marked as proposed."
)
add("")
add(
    "**Keywords:** GNN; link prediction; leakage; Cora; Citeseer; Pubmed; inductive "
    "evaluation; design science; executable protocol; Adamic–Adar; GraphSAGE; GAT."
)
add("")
add("---")
add("")

# ========== 1 INTRODUÇÃO ==========
add("## 1. Introdução")
add("")
add("### 1.1 Problema e motivações")
add("")
add(
    "A tarefa de predição de ligações é antiga na ciência de redes. Liben-Nowell e Kleinberg "
    "(2007) a sistematizaram em redes sociais; Newman (2001) e Barabási e Albert (1999) "
    "ligaram crescimento preferencial a padrões observáveis de grau; Adamic e Adar (2003) "
    "ponderaram vizinhos comuns pela raridade. O que mudou na última década não foi a "
    "pergunta — *quais pares deveriam ser aresta?* — e sim o instrumento: *message passing* "
    "aprendido (Gilmer et al., 2017; Kipf e Welling, 2017; Hamilton et al., 2017; "
    "Veličković et al., 2018) e o decoder bilinear popularizado por autoencoders de grafo "
    "(Kipf e Welling, 2016, VGAE/GAE)."
)
add("")
add(
    "Três domínios motivam o artefato. **Citação científica:** Cora, Citeseer e Pubmed "
    "(Sen et al., 2008, via LINQS) são o *Planetoid* clássico: documentos como nós, "
    "citações como arestas, *bag-of-words* como atributos. Prever uma citação futura é "
    "predição de ligação; o modelo em produção não pode ter visto a aresta que ainda não "
    "existe. **Recomendação:** o mesmo encoder–decoder em grafos bipartidos usuário–item "
    "(Berg, Kipf e Welling, 2017, GC-MC) enfrenta a mesma tensão entre *message passing* "
    "e rótulo. **Tech tree / crafting:** em jogos e sistemas de progressão, a árvore de "
    "receitas é um grafo pequeno com *timestamps* naturais — domínio em que *split* temporal "
    "é honesto sem inventar relógio em LINQS. O pacote inclui um *tech tree* sintético "
    "(12 nós, 20 arestas) exatamente para essa ilustração."
)
add("")
add(
    "O problema de engenharia que este manuscrito ataca não é *qual GNN é SOTA*, e sim "
    "*como garantir que a avaliação não entregue o rótulo ao encoder*. Formalmente:"
)
add("")
add("\\[")
add("Q^{+} \\cap E_{mp} = \\emptyset \\quad \\text{(modo válido)}.")
add("\\]")
add("")
add(
    "Quando a interseção não é vazia, o encoder pode propagar informação do próprio rótulo "
    "para a embedding usada pelo decoder. A AUC Mann–Whitney sobe. O *paper* reporta um "
    "número que o sistema em produção — onde \\((u,v)\\) ainda não está no grafo — não "
    "reproduz. Não é necessariamente malícia: é o *default* de um *notebook* que reutiliza "
    "`edge_index` completo e mascara apenas o decoder."
)
add("")
add("### 1.2 Lacuna: diagnóstico sem harness incontornável")
add("")
add(
    "A literatura recente de avaliação já documentou o problema. Li et al. (2023, HeaRT, "
    "arXiv:2306.10453) argumentam que políticas de negativos e *splits* inadequados "
    "distorcem *rankings* de GNN para *link prediction*. Zhu et al. (2024, SpotTarget, "
    "WSDM / arXiv:2306.00899) isolam *pitfalls* incluindo vazamento de aresta-alvo. "
    "Hu et al. (2020, OGB) padronizam *splits* oficiais e Hits@K — padrão que este "
    "artefato **não executou** (marcado como proposto / não medido). Zhang e Chen "
    "(2018, SEAL) mostram que *subgraphs enclosing* são poderosos, mas estão fora do "
    "núcleo medido aqui."
)
add("")
add(
    "A lacuna que GLUE-LP preenche não é *mais um paper de diagnóstico*, e sim um "
    "**artefato executável** no sentido de Design Science (Hevner et al., 2004; "
    "Peffers et al., 2007): protocolo + código + invariante + evidência medida + "
    "checklist de reprodutibilidade. A exclusão deixa de ser conselho e vira "
    "`AssertionError` no CI."
)
add("")
add("### 1.3 Contribuições")
add("")
add(
    "Enquadramos o trabalho como *Design Science Research* em engenharia de ML / "
    "sistemas de informação: o artefato é o protocolo `glue_lp`; a avaliação é a "
    "série de experimentos medidos; a utilidade é tornar o atalho *incontornável* "
    "no processo de desenvolvimento."
)
add("")
add("1. **Protocolo executável** com modos `valid` e `leaky`, invariante "
   "\\(Q^{+} \\cap E_{mp} = \\emptyset\\) no modo válido, e relatório incompleto "
   "se heurísticas clássicas não compartilham o mesmo \\(Q\\).")
add("2. **Harness único** para GCN, GraphSAGE e GAT (decoder bilinear) e para "
   "CN / Adamic–Adar / preferential attachment, com políticas de negativos "
   "(uniforme, hard-CN, *degree-matched*) e *splits* aleatório, indutivo e "
   "temporal sintético.")
add("3. **Evidência medida** em Cora, Citeseer e Pubmed (séries 1–2, onda 3) "
   "mais escada L1/L3/L4 e enumeração exaustiva no *tech tree* sintético — "
   "todos os números em `experiments/*.json`.")
add("4. **Pacote reproduzível** (JSON versionados, figuras fig1–fig7, testes "
   "pytest em CI sem GPU, manuscrito DOCX/PDF gerado por script, API opcional "
   "somente leitura).")
add("")
add("### 1.4 Hipóteses (honestas)")
add("")
add(
    "Formulamos hipóteses *a priori* e reportamos o que os dados medidos permitem "
    "concluir — inclusive quando a conclusão é *sugestiva* e não *definitiva*."
)
add("")
add(
    "- **H1 (inflação leaky).** O protocolo leaky infla a AUC do GNN relativamente "
    "ao válido no mesmo *seed* e no mesmo \\(Q^{-}\\). *Esperado:* efeito positivo "
    f"e estável. *Observado:* claro na série 1 (Cora, Δ = +{S['E1_delta_leaky_minus_valid']:.3f}, "
    "cinco seeds no mesmo sentido), em Citeseer "
    f"(*t* = {X['paired_t_citeseer']['t']:.2f}), em GAT-Cora "
    f"(*t* = {W['paired_t_gat']['t']:.1f}) e em Pubmed "
    f"(*t* = {W['paired_t_pubmed']['t']:.2f}); apenas *sugestivo* em Cora com "
    f"*early stopping* (*t* = {X['paired_t_cora']['t']:.2f} < 2,78; seed 4 inverte)."
)
add(
    "- **H2 (não-uniformidade entre scorers).** O vazamento **não** é uniforme: "
    "*preferential attachment* é mais sensível que CN/AA porque o grau dos "
    "*endpoints* sobe **somente** nos positivos quando estes entram em \\(E_{mp}\\). "
    f"*Observado no sintético:* Δ(L3−L1) PA = +{L['escada']['delta_pa_auc_L3_menos_L1']['mean']:.3f} "
    f"vs AA +{L['escada']['delta_aa_auc_L3_menos_L1']['mean']:.3f} e "
    f"CN +{L['escada']['delta_cn_auc_L3_menos_L1']['mean']:.3f}. "
    "**Proposto / não medido** como confirmação formal da mesma proposição em "
    "Cora/Citeseer/Pubmed sob a escada L."
)
add(
    "- **H3 (política de negativos troca o ranking).** Em Cora válido, a política "
    "de negativos altera quem vence entre GCN e Adamic–Adar. *Observado:* no "
    f"uniforme, AA ({ms(H['valid_unif']['aa'])}) ≈ GCN ({ms(S['E1_gcn_valid_auc'])}); "
    f"no hard, GCN ({ms(S['E2_gcn_hard_auc'])}) > AA ({ms(H['valid_hard']['aa'])}). "
    "Quem publica só o uniforme escolhe o \\(Q\\) mais complacente."
)
add("")
add(
    "Não afirmamos que hidden=32 é ótimo. Não afirmamos que GAT de uma cabeça "
    "representa atenção em geral. Afirmamos que o atalho é mensurável, que sua "
    "visibilidade estatística depende de corpus / critério de parada / encoder, "
    "e que o *harness* torna essa mensuração *incontornável*."
)
add("")
add("### 1.5 Organização do manuscrito")
add("")
add(
    "A Seção 2 posiciona o trabalho frente à literatura clássica e moderna. A Seção 3 "
    "formaliza LP, leakage e a escada L0–L4. A Seção 4 descreve o artefato de Design "
    "Science. A Seção 5 detalha a *stack* tecnológica (medido vs proposto). A Seção 6 "
    "dá o protocolo experimental completo. A Seção 7 reporta e discute todos os "
    "resultados medidos. As Seções 8–11 cobrem ameaças à validade, limitações, "
    "reprodutibilidade e conclusão. Apêndices trazem revisão técnica, mapa de módulos, "
    "tabelas JSON, glossário e a ameaça do *bug* histórico de *early stopping*."
)
add("")

# ========== 2 TRABALHOS RELACIONADOS ==========
add("## 2. Trabalhos relacionados")
add("")
add("### 2.1 Predição de ligações clássica")
add("")
add(
    "Liben-Nowell e Kleinberg (2007) formularam o *link-prediction problem* para redes "
    "sociais e compararam índices de vizinhança e caminhos. O trabalho permanece a "
    "referência conceitual: a tarefa é ranquear pares não observados; a avaliação "
    "deve separar treino e teste de forma que o futuro não vaze para o passado. "
    "Adamic e Adar (2003) introduziram a ponderação de vizinhos comuns por "
    "\\(1/\\log \\mathrm{deg}(w)\\), capturando a intuição de que vizinhos raros "
    "carregam mais evidência. Newman (2001) estudou *clustering* e *preferential "
    "attachment* em redes de coautoria; Barabási e Albert (1999) derivaram a lei de "
    "potência de grau a partir do crescimento preferencial. No *harness* GLUE-LP, "
    "CN, Adamic–Adar e preferential attachment **não são opcionais**: compartilham "
    "o mesmo \\(Q\\) do GNN, ou a célula experimental é incompleta."
)
add("")
add(
    "Essa insistência tem motivo empírico. Na série 1 (Cora, negativos uniformes), "
    f"Adamic–Adar com AUC de empate = ½ marca {ms(H['valid_unif']['aa'])}, "
    f"numericamente lado a lado do GCN válido ({ms(S['E1_gcn_valid_auc'])}). "
    "Omitir a heurística no mesmo \\(Q\\) ocultaria que o GNN, nessa célula, "
    "não ganha de um índice de 2003. No regime hard, a história inverte — e "
    "omitir o hard ocultaria a outra metade."
)
add("")
add("### 2.2 Message passing e GNN para arestas")
add("")
add(
    "Gilmer et al. (2017) formalizaram *message passing neural networks* (MPNN) "
    "como framework unificador: mensagens \\(m_{v}^{{(t)}}\\) agregadas de vizinhos, "
    "atualização de estado, *readout*. Kipf e Welling (2017, arXiv:1609.02907, ICLR) "
    "propuseram o GCN com renormalização "
    "\\(\\hat{{A}} = D^{{-1/2}}(A+I)D^{{-1/2}}\\) para classificação semi-supervisionada "
    "de nós. **Para *link prediction*, a referência metodológica principal deste "
    "harness é outra:** Kipf e Welling (2016, VGAE/GAE, arXiv:1611.07308), que "
    "treinam um encoder GCN e um decoder de produto interno (ou bilinear) sobre "
    "pares. Distinguir os dois papers evita o erro comum de citar só o ICLR 2017 "
    "quando o objeto é aresta."
)
add("")
add(
    "Hamilton et al. (2017, GraphSAGE, arXiv:1706.02216, NeurIPS) introduziram "
    "agregação amostrada e concatenação \\([h_v \\,\\|\\, \\mathrm{{AGG}}(N(v))]\\) "
    "para aprendizado indutivo. Neste pacote medimos a variante de média plena "
    "(sem amostragem de vizinhos) — **GraphSAGE com amostragem permanece proposto "
    "/ não medido**. Veličković et al. (2018, GAT, arXiv:1710.10903, ICLR) "
    "substituem a agregação uniforme por atenção; medimos GAT de **uma cabeça** "
    "(multi-head: proposto / não medido). Zhang e Chen (2018, SEAL, "
    "arXiv:1802.09691, NeurIPS) argumentam que *enclosing subgraphs* com "
    "rotulagem de Double-Radius Node Labeling superam produto interno puro — "
    "extensão natural, **não medida** aqui. Berg, Kipf e Welling (2017, GC-MC) "
    "ilustram decoders bilineares em grafos bipartidos; nosso decoder permanece "
    "\\(s(u,v) = \\langle z_u, z_v \\rangle\\)."
)
add("")
add("### 2.3 Avaliação com e sem vazamento")
add("")
add(
    "Li et al. (2023, HeaRT, arXiv:2306.10453) mostram que *rankings* de métodos "
    "de *link prediction* mudam drasticamente sob *hard negative sampling* "
    "estruturado e *splits* mais realistas. Zhu et al. (2024, SpotTarget) "
    "catalogam *pitfalls* de GNN-LP, incluindo o vazamento de aresta-alvo no "
    "*message passing*. GLUE-LP opera na mesma família de preocupação, mas "
    "com ênfase de Design Science: a exclusão é assert, o leaky é controle "
    "explícito (nunca o *headline*), e o mesmo \\(Q\\) alimenta heurísticas."
)
add("")
add(
    "Hu et al. (2020, Open Graph Benchmark, arXiv:2005.00687) padronizam "
    "datasets `ogbl-*`, *splits* oficiais e Hits@K / MRR. **OGB não foi "
    "executado neste artefato**; `integrations/ogb_loader.py` é stub. "
    "Sen et al. (2008) disponibilizam Cora, Citeseer e Pubmed via LINQS — "
    "os três corpora medidos. O *Planetoid* de Yang, Cohen e Salakhutdinov "
    "(2016) popularizou os *splits* de classificação de nós; para LP, "
    "reutilizamos os grafos e atributos, com *holdout* de arestas próprio "
    "do protocolo (10% teste, 5% validação)."
)
add("")
add("### 2.4 Design Science e artefatos de avaliação")
add("")
add(
    "Hevner et al. (2004) definem Design Science em sistemas de informação "
    "como ciclo de construção e avaliação de artefatos que resolvem problemas "
    "de classe. Peffers et al. (2007) propõem metodologia em etapas "
    "(identificação do problema, objetivos, *design*, demonstração, avaliação, "
    "comunicação). GLUE-LP trata o protocolo como artefato: relevância "
    "(tutoriais vazam; produção não pode), rigor (assert + JSON + testes), "
    "utilidade (CI sem GPU; *runners* determinísticos por seed). Não "
    "confundimos o artefato com o *leaderboard*: a contribuição é a "
    "invariante, não um novo encoder."
)
add("")
add("### 2.5 Posicionamento")
add("")
add(
    "Em relação a HeaRT e SpotTarget, somos mais estreitos em *benchmark* "
    "(LINQS + sintético; sem OGB) e mais rígidos em engenharia (assert, "
    "mesmo \\(Q\\), modos explícitos). Em relação a VGAE/GAE, reutilizamos "
    "o encoder–decoder bilinear sem reivindicar novelty arquitetural. Em "
    "relação a SEAL, apontamos a extensão e a marcamos como não medida. "
    "Em relação ao GLUE de PLN (Wang et al., 2018), a homonímia é acidental: "
    "nosso acrônimo é *Graph Link-evaluation Under Exclusion*."
)
add("")

# Continue in next chunk - write what we have so far and append
text_so_far = "\n".join(parts)
OUT.write_text(text_so_far, encoding="utf-8")
print("chunk1 words", len(re.findall(r"\S+", text_so_far)))
print("written", OUT)
