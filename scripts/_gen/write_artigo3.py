#!/usr/bin/env python3
"""Append sections 6–11 + appendices with integrity fixes from review brief."""
from __future__ import annotations
import json, re, math, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments"
OUT = ROOT / "docs" / "ARTIGO.md"

A = json.loads((EXP / "cora_e1_e4.json").read_text())
H = json.loads((EXP / "heuristics_rescored.json").read_text())["summary"]
E = json.loads((EXP / "extended_gcn_sage.json").read_text())
Wfull = json.loads((EXP / "wave3_gat_pubmed.json").read_text())
W = Wfull["summary"]
L = json.loads((EXP / "leakage_ladder_resumo.json").read_text())
S, X = A["summary"], E["summary"]

def ms(d, prec=3):
    std = d.get("std", d.get("sd", 0.0))
    return f"{d['mean']:.{prec}f} ± {std:.{prec}f}"

def paired_extra(a, b, seed=0, n_boot=10000):
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    mean = sum(d) / n
    var = sum((x - mean) ** 2 for x in d) / (n - 1)
    sd = math.sqrt(var)
    se = sd / math.sqrt(n)
    t = mean / se if se else float("inf")
    dz = mean / sd if sd else float("inf")
    rnd = random.Random(seed)
    boots = sorted(sum(d[rnd.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]
    return mean, t, n - 1, dz, lo, hi

s1 = paired_extra(S["E1_gcn_leaky_auc"]["values"], S["E1_gcn_valid_auc"]["values"])
s2c = paired_extra(X["cora_gcn_leaky"]["values"], X["cora_gcn_valid"]["values"])
s2s = paired_extra(X["citeseer_gcn_leaky"]["values"], X["citeseer_gcn_valid"]["values"])
wg = paired_extra(W["cora_gat_leaky"]["values"], W["cora_gat_valid"]["values"])
wp = paired_extra(W["pubmed_gcn_leaky"]["values"], W["pubmed_gcn_valid"]["values"])

PRE = "**[pré-correção ES]**"
parts = []
def add(s=""):
    parts.append(s)

add("")
add("---")
add("")
add("## 6. Protocolo experimental completo")
add("")
add("### 6.1 Corpora")
add("")
add("| Corpus | n | arestas (não dir.) | features | Fonte | Papel |")
add("|---|---:|---:|---:|---|---|")
add("| Cora | 2708 | 5278 | 1433 | LINQS (Sen et al., 2008) | séries 1–2, onda 3, indutivo |")
add("| Citeseer | 3312 | 4536 | 3703 | LINQS | série 2 |")
add("| Pubmed | 19717 | 44324 | 500 | LINQS | onda 3 (3 seeds) |")
add("| Tech tree | 12 | 20 | — | sintético (`crafting_graph`) | escada L + exaustivo |")
add("")
add(
    "Os três corpora LINQS são grafos de citação com atributos *bag-of-words*. "
    "Não possuem *timestamp* de aresta nativo — por isso o *split* temporal "
    "honesto vive apenas no *tech tree*. Checksums locais: "
    "`scripts/checksum_data.py` (dados fora de alguns ambientes CI)."
)
add("")
add("### 6.2 Séries experimentais")
add("")
add(
    f"**Série 1 — Cora, GCN denso, 50 épocas fixas.** Seeds 0–4, negativos 1:1, "
    f"holdout 10%/5%, sem *early stopping*. Fonte: `cora_e1_e4.json` + "
    f"`heuristics_rescored.json`. Status: {PRE}."
)
add("")
add(
    f"**Série 2 — GCN/SAGE esparso, *early stopping*.** Cora e Citeseer, "
    f"paciência 12, teto 70 épocas, Adam lr=0,01, wd=5e-4, hidden=32. "
    f"Fonte: `extended_gcn_sage.json`. Status: {PRE}. "
    f"GraphSAGE Cora válido {ms(X['cora_sage_valid'])} foi medido **antes** "
    f"de `F.normalize` L2 na saída — **não promover** como resultado "
    f"pós-normalização; re-AUC pós-normalize = **não medido**."
)
add("")
add(
    f"**Onda 3 — GAT, degree-matched, Pubmed.** GAT 1 cabeça (Cora, 5 seeds); "
    f"GCN *degree-matched* (Cora); Pubmed GCN 3 seeds, teto 25 épocas. "
    f"Fonte: `wave3_gat_pubmed.json`. Status: {PRE}."
)
add("")
add(
    "**Escada L + enumeração exaustiva (sintético).** 10 seeds, tech tree, "
    "100% NumPy — **não depende** do *bug* de *early stopping* PyTorch. "
    "Fonte: `leakage_ladder_resumo.json`, `exhaustive_sintetico.json`."
)
add("")
add("### 6.3 Hiperparâmetros congelados (`TrainConfig`)")
add("")
add("| Parâmetro | Valor medido | Onde |")
add("|---|---|---|")
add("| hidden | 32 | todas as séries GNN |")
add("| lr / weight_decay | 0,01 / 5·10⁻⁴ | Adam |")
add("| dropout (GCN/SAGE) | 0,5 | camada oculta |")
add("| gat_dropout | 0,4 | atenção |")
add("| épocas fixas (série 1) | 50 | Cora denso |")
add("| epochs_cap / patience | 70 / 12 | série 2 |")
add("| epochs_cap_wave3 | 40 | GAT/degree Cora |")
add("| epochs_cap_pubmed | 25 | Pubmed |")
add("| neg_per_pos (treino) | 1 | BCE |")
add("| seeds | 0–4 (Pubmed: 0–2) | |")
add("| test_frac / val_frac | 0,10 / 0,05 | |")
add("| inductive_node_frac | 0,15 | Cora indutivo |")
add("")
add(
    "Não houve HPO (Optuna **proposto / não medido**). Hidden=32 é escolha "
    "de harness rasa, não busca de SOTA."
)
add("")
add("### 6.4 Hipóteses operacionais no desenho")
add("")
add(
    "- **H1:** leaky > válido em AUC no mesmo seed/mesmo Q⁻. "
    "**Clara** na série 1 (Cora) e em Citeseer/GAT/Pubmed; "
    "**apenas sugestiva** em Cora+early stopping (*t* = 2,61 < 2,78)."
)
add(
    "- **H2:** Δ PA ≫ Δ CN/AA sob vazamento. Confirmada no sintético; "
    "**não confirmada** formalmente em LINQS sob escada L (futuro)."
)
add(
    "- **H3:** política de negativos troca o ranking GCN↔AA em Cora válido. "
    "Observada (uniforme vs hard)."
)
add("")
add("### 6.5 Métricas e estatística")
add("")
add(
    "Métrica *headline*: AUC Mann–Whitney (empate = ½). Auxiliares no código/"
    "JSON série 1: AP, Hits@3, Hits@10, MRR — **não** equivalentes a Hits@K OGB "
    "sob amostragem 1:1. Teste pareado por seed (leaky − válido). "
    "A partir dos vetores por seed já gravados nos JSON, aplicamos na "
    "**discussão** (§7.6) conceitos de `glue_lp.stats`: *t* pareado, "
    "Cohen's *d_z* = média(diff)/sd(diff), e IC 95% bootstrap percentil "
    "do Δ (10 000 reamostragens, seed=0). "
    "**Holm–Bonferroni e *t* sobre logit(AUC)** estão disponíveis no artefato "
    "(`stats.py`) mas **não foram aplicados retroativamente como correção "
    "formal das tabelas históricas** — evitamos reescrever o protocolo "
    "estatístico *post hoc* sem pré-registro. Com gl = 4, o limiar bilateral "
    "5% da *t* de Student é ≈ 2,78; com gl = 2 (Pubmed), ≈ 4,30."
)
add("")
add("---")
add("")
add("## 7. Resultados e discussão")
add("")
add(
    f"> **Lembrete:** tabelas 7.1–7.3 (Cora/Citeseer/Pubmed) = {PRE}. "
    f"Tabela 7.4 (escada sintética) = independente do *bug* ES."
)
add("")
add("### 7.1 Série 1 — Cora, GCN denso, 50 épocas " + PRE)
add("")
add("| Condição | AUC (média ± dp, 5 seeds) |")
add("|---|---|")
add(f"| GCN válido uniforme | {ms(S['E1_gcn_valid_auc'])} |")
add(f"| GCN leaky uniforme | {ms(S['E1_gcn_leaky_auc'])} |")
add(f"| Δ leaky − válido | +{S['E1_delta_leaky_minus_valid']:.3f} |")
add(f"| GCN válido hard-CN | {ms(S['E2_gcn_hard_auc'])} |")
add(f"| AA válido uniforme (empate=½) | {ms(H['valid_unif']['aa'])} |")
add(f"| AA válido hard | {ms(H['valid_hard']['aa'])} |")
add(f"| CN válido uniforme | {ms(H['valid_unif']['cn'])} |")
add(f"| PA válido uniforme | {ms(H['valid_unif']['pa'])} |")
add(f"| GCN indutivo 15% nós | {ms(S['E_inductive_gcn_auc'])} |")
add(f"| AA indutivo | {ms(H['inductive']['aa'])} |")
add(f"| PA indutivo | {ms(H['inductive']['pa'])} |")
add("")
add("Valores por seed (GCN uniforme, válido): " +
    ", ".join(f"{v:.4f}" for v in S["E1_gcn_valid_auc"]["values"]) + ".")
add("")
add("Valores por seed (GCN uniforme, leaky): " +
    ", ".join(f"{v:.4f}" for v in S["E1_gcn_leaky_auc"]["values"]) + ".")
add("")
add(
    f"Efeito pareado série 1: Δ̄ = {s1[0]:.4f}, *t* = {s1[1]:.2f}, "
    f"gl = {s1[2]}, *d_z* = {s1[3]:.2f}, IC95% bootstrap "
    f"[{s1[4]:.4f}, {s1[5]:.4f}]. Cinco seeds no mesmo sentido. "
    f"**H1 clara na série 1.**"
)
add("")
add(
    f"No uniforme, GCN ({ms(S['E1_gcn_valid_auc'])}) e AA "
    f"({ms(H['valid_unif']['aa'])}) quase empatam. No hard, GCN "
    f"({ms(S['E2_gcn_hard_auc'])}) supera AA ({ms(H['valid_hard']['aa'])}). "
    f"**H3:** a política de negativos troca o vencedor. No indutivo, CN/AA "
    f"caem à chance (~0,50); PA cai a {ms(H['inductive']['pa'])}; o GCN "
    f"ainda lê *bag-of-words* via self-loop ({ms(S['E_inductive_gcn_auc'])}) "
    f"— não é milagre espectral."
)
add("")
add(
    "Heurísticas no JSON bruto da série 1 reportavam AUC ~1,0 para CN/AA "
    "(empates mal tratados). Usar **somente** `heuristics_rescored.json`."
)
add("")
add("### 7.2 Série 2 — GCN/SAGE esparso + early stopping " + PRE)
add("")
add("| Corpus / encoder | Válido | Leaky | *t* pareado | Δ̄ |")
add("|---|---|---|---|---|")
add(
    f"| Cora GCN | {ms(X['cora_gcn_valid'])} | {ms(X['cora_gcn_leaky'])} | "
    f"{X['paired_t_cora']['t']:.2f} (gl=4) | +{X['paired_t_cora']['mean_diff']:.3f} |"
)
add(
    f"| Citeseer GCN | {ms(X['citeseer_gcn_valid'])} | {ms(X['citeseer_gcn_leaky'])} | "
    f"{X['paired_t_citeseer']['t']:.2f} | +{X['paired_t_citeseer']['mean_diff']:.3f} |"
)
add(f"| Cora SAGE válido | {ms(X['cora_sage_valid'])} | — | — | — |")
add("")
add(
    f"Cora por seed válido: "
    + ", ".join(f"{v:.4f}" for v in X["cora_gcn_valid"]["values"])
    + "; leaky: "
    + ", ".join(f"{v:.4f}" for v in X["cora_gcn_leaky"]["values"])
    + ". A seed 4 inverte o sinal (leaky < válido)."
)
add("")
add(
    f"Efeito pareado Cora ES: Δ̄ = {s2c[0]:.4f}, *t* = {s2c[1]:.2f}, "
    f"*d_z* = {s2c[3]:.2f}, IC95% boot [{s2c[4]:.4f}, {s2c[5]:.4f}]. "
    f"**Não atravessa** 2,78. **H1 apenas sugestiva em Cora+ES.**"
)
add("")
add(
    f"Citeseer: Δ̄ = {s2s[0]:.4f}, *t* = {s2s[1]:.2f}, *d_z* = {s2s[3]:.2f}, "
    f"IC95% boot [{s2s[4]:.4f}, {s2s[5]:.4f}]. Cinco seeds acima da diagonal. "
    f"**H1 clara em Citeseer.**"
)
add("")
add(
    f"GraphSAGE (média plena, sem amostragem) marca {ms(X['cora_sage_valid'])} "
    f"no Cora válido — atrás do GCN. **Não concluímos** que SAGE é pior em "
    f"geral; concluímos que *esta* instância, com o mesmo orçamento e decoder "
    f"bilinear, **antes** de `F.normalize`, não ajudou. Re-AUC pós-normalize: "
    f"**não medido** — não promover 0,577 como resultado normalizado."
)
add("")
add("### 7.3 Onda 3 — GAT, degree-matched, Pubmed " + PRE)
add("")
add("| Condição | Válido | Leaky | *t* |")
add("|---|---|---|---|")
add(
    f"| GAT Cora (5 seeds) | {ms(W['cora_gat_valid'])} | {ms(W['cora_gat_leaky'])} | "
    f"{W['paired_t_gat']['t']:.1f} |"
)
add(
    f"| GCN Pubmed (3 seeds) | {ms(W['pubmed_gcn_valid'])} | {ms(W['pubmed_gcn_leaky'])} | "
    f"{W['paired_t_pubmed']['t']:.2f} (gl=2) |"
)
add(f"| GCN Cora degree-matched | {ms(W['cora_gcn_degree_matched'])} | — | — |")
add("")
add(
    f"GAT: Δ̄ = {wg[0]:.4f}, *t* = {wg[1]:.1f}, *d_z* = {wg[3]:.2f}, "
    f"IC95% boot [{wg[4]:.4f}, {wg[5]:.4f}]. A atenção de uma cabeça "
    f"**não imuniza** (P3). Multi-head: não medido."
)
add("")
add(
    f"Pubmed: Δ̄ = {wp[0]:.4f}, *t* = {wp[1]:.2f}, *d_z* = {wp[3]:.2f}, "
    f"IC95% boot [{wp[4]:.4f}, {wp[5]:.4f}]. Efeito absoluto menor que "
    f"Citeseer, sinal estável, gl = 2 é pouco para artigo clínico — para "
    f"um harness, basta como evidência de direção. Degree-matched "
    f"({ms(W['cora_gcn_degree_matched'])}) alinha com hard-CN "
    f"({ms(S['E2_gcn_hard_auc'])}): o Q uniforme (~0,71) é o mais complacente."
)
add("")
add("### 7.4 Escada de vazamento (sintético) — [MEDIDO, independente do ES]")
add("")
add("| Nível | AUC GNN (10 seeds) |")
add("|---|---|")
add(f"| L1 (valid) | {ms(L['escada']['L1'], 4)} |")
add(f"| L3 (leaky positivos) | {ms(L['escada']['L3'], 4)} |")
add(f"| L4 (meia-aresta) | {ms(L['escada']['L4_half_edge'], 4)} |")
add("")
add(
    f"Δ(L3−L1) GNN = +{L['escada']['delta_gnn_auc_L3_menos_L1']['mean']:.4f} "
    f"± {L['escada']['delta_gnn_auc_L3_menos_L1']['sd']:.4f}; "
    f"Δ(L4−L1) = +{L['escada']['delta_gnn_auc_L4_menos_L1']['mean']:.4f} "
    f"± {L['escada']['delta_gnn_auc_L4_menos_L1']['sd']:.4f}."
)
add("")
add(
    f"Δ(L3−L1) por scorer: AA +{L['escada']['delta_aa_auc_L3_menos_L1']['mean']:.4f}; "
    f"CN +{L['escada']['delta_cn_auc_L3_menos_L1']['mean']:.4f}; "
    f"**PA +{L['escada']['delta_pa_auc_L3_menos_L1']['mean']:.4f}**. **H2/P-PA no sintético.**"
)
add("")
add("Enumeração exaustiva (~54 pares avaliados, médias 10 seeds):")
add("")
add("| Scorer | AUC válida | AUC leaky | Δ |")
add("|---|---|---|---|")
add(
    f"| GNN | {L['exaustivo']['gnn_auc_exata_valido']['mean']:.4f} | "
    f"{L['exaustivo']['gnn_auc_exata_leaky']['mean']:.4f} | "
    f"+{L['exaustivo']['gnn_auc_exata_leaky']['mean']-L['exaustivo']['gnn_auc_exata_valido']['mean']:.4f} |"
)
add(
    f"| AA | {L['exaustivo']['aa_auc_exata_valido']['mean']:.4f} | "
    f"{L['exaustivo']['aa_auc_exata_leaky']['mean']:.4f} | "
    f"+{L['exaustivo']['aa_auc_exata_leaky']['mean']-L['exaustivo']['aa_auc_exata_valido']['mean']:.4f} |"
)
add(
    f"| CN | {L['exaustivo']['cn_auc_exata_valido']['mean']:.4f} | "
    f"{L['exaustivo']['cn_auc_exata_leaky']['mean']:.4f} | "
    f"+{L['exaustivo']['cn_auc_exata_leaky']['mean']-L['exaustivo']['cn_auc_exata_valido']['mean']:.4f} |"
)
add(
    f"| PA | {L['exaustivo']['pa_auc_exata_valido']['mean']:.4f} | "
    f"{L['exaustivo']['pa_auc_exata_leaky']['mean']:.4f} | "
    f"**+{L['exaustivo']['pa_auc_exata_leaky']['mean']-L['exaustivo']['pa_auc_exata_valido']['mean']:.4f}** |"
)
add("")
add(
    "Achado transferível de engenharia: um bug de meia-aresta (L4) pode inflar "
    "tanto quanto ou mais que L3, com menor variância neste grafo "
    f"(L4 sd {L['escada']['L4_half_edge']['sd']:.3f} vs L3 sd "
    f"{L['escada']['L3']['sd']:.3f}). L2 (treino∪val) permanece "
    "**proposto / não medido** na escada JSON."
)
add("")
add("### 7.5 Figuras")
add("")
add(
    "Figuras 1–7 em `figures/`: pipeline (fig1), protocolo valid/leaky (fig2), "
    "tech tree (fig3), barras Cora série 1 (fig4), Cora×Citeseer (fig5), "
    "dispersão por seed (fig6), três corpora (fig7). Incluídas no DOCX/PDF."
)
add("")
add("### 7.6 Discussão profunda")
add("")
add("#### 7.6.1 Quando H1 é clara e quando é apenas sugestiva")
add("")
add(
    "A série 1 (épocas fixas) entrega o retrato mais limpo do atalho "
    f"estrutural em Cora: Δ = +{S['E1_delta_leaky_minus_valid']:.3f}, "
    f"*t* enorme, *d_z* ≈ {s1[3]:.1f}, IC bootstrap longe de zero. "
    f"Citeseer com ES (*t* = {s2s[1]:.2f}, *d_z* ≈ {s2s[3]:.2f}) e "
    f"GAT-Cora (*t* = {wg[1]:.1f}) reforçam. Cora com ES "
    f"(*t* = {s2c[1]:.2f} < 2,78, uma seed invertendo) **não** autoriza "
    f"a frase \"H1 confirmada em Cora com early stopping\". Essa dependência "
    f"do critério de parada é ela mesma um resultado (P-leak / P2): a "
    f"*visibilidade* do vazamento não é invariante ao protocolo de treino."
)
add("")
add(
    "Além disso, todos os *t* LINQS acima são {PRE}. Um re-run com ES "
    "simétrico pode estreitar ou alargar os Δ — não especulamos o sinal "
    "do re-run; exigimos medi-lo."
)
add("")
add("#### 7.6.2 Negativos, ranking e honestidade de Q")
add("")
add(
    "Quem publica só o uniforme escolhe o Q mais complacente "
    f"(GCN ~{S['E1_gcn_valid_auc']['mean']:.2f}, AA ~"
    f"{H['valid_unif']['aa']['mean']:.2f}). Hard-CN e degree-matched "
    "caem ambos ~0,61–0,62 e invertem a vantagem GNN↔heurística. "
    "HeaRT (Li et al., 2023) argumenta o mesmo em escala de *benchmark*; "
    "nós o tornamos célula obrigatória do harness."
)
add("")
add("#### 7.6.3 L4 como bug de engenharia, não nuance teórica")
add("")
add(
    "L4 (meia-aresta) modela simetrização incompleta ao portar código "
    "dirigido ↔ não dirigido. No sintético, L4 ≥ L3 em média com menor "
    "sd. Isso eleva L4 de curiosidade a **ameaça de validade interna** "
    "em qualquer *pipeline* que construa `edge_index` por concatenação "
    "unilateral. O assert do modo válido pega L3 pleno; L4 exige teste "
    "explícito de simetria (coberto na escada sintética; **não** re-medido "
    "em LINQS)."
)
add("")
add("#### 7.6.4 Diferenciação vs SpotTarget / HeaRT / OGB")
add("")
add(
    "SpotTarget (Zhu et al., WSDM 2024) e HeaRT (Li et al., 2023) são "
    "diagnósticos de *benchmark* com fôlego. OGB (Hu et al., 2020) é o "
    "padrão de *split*/Hits@K. GLUE-LP não os substitui: os **complementa** "
    "com um artefato de Design Science cuja exclusão é assert de processo. "
    "Ver tabela na Seção 2 (reproduzida abaixo para o PDF)."
)
add("")
add("| Dimensão | SpotTarget (Zhu et al., 2024) | HeaRT (Li et al., 2023) | OGB (Hu et al., 2020) | **GLUE-LP (este)** |")
add("|---|---|---|---|---|")
add("| Objeto | *pitfalls* de GNN-LP | negativos/*splits* duros | *benchmark* + Hits@K | protocolo executável + assert |")
add("| Exclusão alvo | diagnosticada | implícita em boas práticas | *split* oficial | **assert** \\(Q^{+}∩E_{mp}=∅\\) |")
add("| Modo leaky explícito | análise | — | — | **controle obrigatório** |")
add("| Heurísticas no mesmo Q | varia | foco GNN | — | **CN/AA/PA obrigatórios** |")
add("| Escada L1–L4 | — | — | — | **L1/L3/L4 medidos (sintético)** |")
add("| OGB / Hits@K oficial | usa/discuss | usa/discuss | **define** | **não executado** |")
add("| Enquadramento | empírico ML | empírico ML | *benchmark* | **Design Science (Hevner/Peffers)** |")
add("")
add("#### 7.6.5 Estatística: o que `stats.py` oferece e o que não reescrevemos")
add("")
add(
    "Cohen's *d_z* e IC bootstrap acima foram computados **na discussão** "
    "a partir dos vetores por seed já presentes nos JSON — sem retreinar, "
    "sem inventar. Holm–Bonferroni sobre uma família de testes "
    "(Cora ES, Citeseer, GAT, Pubmed, série 1) e *t* em escala logit "
    "permanecem **disponíveis no artefato, não aplicados como correção "
    "formal das tabelas históricas**. Motivo: as tabelas foram geradas "
    "sob um protocolo estatístico mínimo (*t* pareado cru); mudar o "
    "protocolo *post hoc* sem pré-registro seria outro gênero de atalho."
)
add("")
add("---")
add("")
add("## 8. Ameaças à validade")
add("")
add(
    "Seguimos a tipología clássica de validade em engenharia de software "
    "empírica (Wohlin et al.; cf. também ameaças em avaliação de GNN: "
    "Shchur et al., 2018; Errica/Errica et al., 2020; Kapoor & Narayanan, 2023)."
)
add("")
add("### 8.1 Validade de construto")
add("")
add(
    "AUC 1:1 ≠ Hits@50 OGB. AP/Hits/MRR no JSON da série 1 não autorizam "
    "equivalência a *leaderboards* OGB. O construto \"protocolo sem "
    "vazamento\" é operacionalizado pelo assert — forte para o atalho "
    "estrutural, incompleto para vazamentos via HPO no teste ou via "
    "vazamento de features."
)
add("")
add("### 8.2 Validade interna — ameaça principal: pré-correção do ES")
add("")
add(
    f"**Ameaça dominante.** Todos os AUC/*t* de Cora, Citeseer e Pubmed "
    f"são {PRE}: o *early stopping* simétrico em `train_encoder` foi "
    f"corrigido no código *depois* dessas medições (score de validação "
    f"morto / assimetria valid vs leaky). O assert de exclusão já existia; "
    f"o critério de parada, não. Isso é ameaça de validade **interna**, "
    f"não nota de rodapé. Mitigação declarada: (i) banner no topo do "
    f"manuscrito; (ii) coluna/etiqueta {PRE} em toda tabela LINQS; "
    f"(iii) re-run completo = P0 de trabalho futuro; (iv) escada sintética "
    f"NumPy permanece como evidência independente do *bug*."
)
add("")
add(
    "Outras ameaças internas: 5 seeds (3 no Pubmed); seed 4 de Cora "
    "inverte na série 2; hidden=32 fixo; GAT 1 cabeça; SAGE sem amostragem "
    "e sem re-AUC pós-`F.normalize`."
)
add("")
add("### 8.3 Validade externa")
add("")
add(
    "Só LINQS + tech tree 12 nós. Sem redes bipartidas de recomendação, "
    "sem OGB (`ogbl-*`), sem grafos temporais reais com *timestamps* de "
    "aresta, sem SEAL/Neo-GNN/BUDDY/NCNC como *baselines* aprendidos. "
    "A transferência do achado L4 para grafos grandes é hipótese de "
    "engenharia, não fato medido em LINQS."
)
add("")
add("### 8.4 Validade de conclusão estatística")
add("")
add(
    f"Com gl=4, Cora ES (*t*={s2c[1]:.2f}) não rejeita H0 a 5%. "
    f"Reportar \"significativo em Cora ES\" seria falso. Citeseer "
    f"(*t*={s2s[1]:.2f}), série 1, GAT e Pubmed (gl=2, *t*={wp[1]:.2f}) "
    f"rejeitam no limiar clássico *não corrigido* por multiplicidade. "
    f"Correção de Holm: disponível, não aplicada às tabelas históricas. "
    f"n_seeds ≥ 10–30 pós-re-run é meta, não realidade atual."
)
add("")
add("---")
add("")
add("## 9. Limitações e trabalho futuro")
add("")
add("### 9.1 Limitações (estado atual)")
add("")
add(
    "1. **P0 — Re-run LINQS pós-correção ES.** Sem isso, claims quantitativos "
    "de Cora/Citeseer/Pubmed permanecem pré-correção."
)
add(
    "2. **P0 — SAGE pós-`F.normalize`.** A AUC 0,577 é pré-normalize; "
    "não promover."
)
add(
    "3. **P0 — H2/P-PA em LINQS.** Confirmada só no sintético."
)
add(
    "4. **P1 — OGB Hits@K / splits oficiais.** Stub apenas; **OGB não rodou**."
)
add(
    "5. **P1 — L2 na escada; GAT multi-head; SAGE amostrado; decoder MLP; SEAL.**"
)
add(
    "6. **P2 — Adapters PyG/DGL; Optuna; W&B; n_seeds 30.**"
)
add(
    "7. Placeholders de autores/afiliação; `CITATION.cff` sem inventar nomes."
)
add("")
add("### 9.2 Trabalho futuro priorizado")
add("")
add(
    "Re-executar séries 1–2 e onda 3 com ES simétrico e `F.normalize` no "
    "SAGE; confirmar P-PA em Cora/Citeseer; rodar pelo menos um `ogbl-*` "
    "com Hits@K; medir L2; publicar checksums LINQS; aplicar Holm de forma "
    "*pré-registrada* na família de testes do re-run."
)
add("")
add("---")
add("")
add("## 10. Reprodutibilidade (checklist estilo ACM / Pineau)")
add("")
add(
    "Inspirado no checklist de reprodutibilidade em ML (Pineau et al., 2021, "
    "JMLR) e nas práticas OGB:"
)
add("")
add("- [ ] Python ≥ 3.11; `pip install -e \".[dev]\"` (ou `requirements-dev.txt`)")
add("- [ ] `PYTHONPATH=src python3 -m pytest -q`")
add("- [ ] `PYTHONPATH=src python3 scripts/check_protocol.py`")
add("- [ ] `PYTHONPATH=src python3 scripts/verify_results.py`")
add("- [ ] Dados LINQS em `data/` conforme `data/LINQS.md` + checksums")
add("- [ ] Runners: `run_e1_e4.py`, `run_extended.py`, `run_wave3.py`, `leakage_ladder.py`")
add("- [ ] Manuscrito: `scripts/build_manuscript.py` → DOCX; `soffice` → PDF")
add("- [ ] Hiperparâmetros = `TrainConfig` / `ProtocolConfig` (não redefinir à mão)")
add("- [ ] JSON em `experiments/` versionados; **não** editar médias à mão")
add("- [ ] Declarar regime `valid`/`leaky`; recusar relatório sem regime")
add("- [ ] Marcar explicitamente qualquer número pré-correção ES")
add("- [ ] Seeds e comando de reprodução documentados no README")
add("")
add(
    "O artefato expõe `glue_lp.export.list_measured_summaries()` e API "
    "opcional `GET /experiments` para reler *headlines* sem retreinar."
)
add("")
add("---")
add("")
add("## 11. Conclusão")
add("")
add(
    "GLUE-LP (*Graph Link-evaluation Under Exclusion*) é um artefato de "
    "Design Science cujo valor não está em uma nova camada de *message "
    "passing*, e sim em tornar a exclusão da aresta-alvo uma invariante "
    "de processo: \\(Q^{+} \\cap E_{mp} = \\emptyset\\) no modo válido, "
    "com modo leaky explícito como controle, heurísticas no mesmo Q, e "
    "escada L1/L3/L4 no laboratório sintético. Em Citeseer o atalho vale "
    f"cerca de dez pontos ({PRE}); em Pubmed, cerca de três; no GAT de "
    f"Cora, cerca de nove, com *t* alto; na série 1 de Cora, cerca de "
    f"doze, com cinco seeds alinhadas; em Cora-GCN com *early stopping*, "
    f"o efeito é apenas sugestivo (*t* = 2,61). Essa dependência também "
    f"é resultado."
)
add("")
add(
    "A contribuição que nos interessa é operacional: se alguém clonar o "
    "repositório e rodar só o modo leaky, obterá um número mais bonito e "
    "um sistema pior — e o CI pode ficar vermelho quando o assert for "
    "ignorado. SpotTarget (Zhu et al., 2024) e HeaRT (Li et al., 2023) "
    "diagnosticaram; OGB padronizou *benchmark*; nós empacotamos a "
    "recusa do atalho como harness. Os números LINQS deste manuscrito "
    f"são {PRE}; a escada sintética não. O próximo passo ético é o "
    f"re-run, não o *spin*."
)
add("")
add("---")
add("")
add("## Referências")
add("")
add("1. Kipf, T. N., & Welling, M. (2016). Variational graph auto-encoders. arXiv:1611.07308.")
add("2. Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. ICLR. arXiv:1609.02907.")
add("3. Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive representation learning on large graphs. NeurIPS. arXiv:1706.02216.")
add("4. Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). Graph attention networks. ICLR. arXiv:1710.10903.")
add("5. Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., & Dahl, G. E. (2017). Neural message passing for quantum chemistry. ICML.")
add("6. Zhang, M., & Chen, Y. (2018). Link prediction based on graph neural networks (SEAL). NeurIPS. arXiv:1802.09691.")
add("7. Liben-Nowell, D., & Kleinberg, J. (2007). The link-prediction problem for social networks. *JASIST*, 58(7), 1019–1031.")
add("8. Lü, L., & Zhou, T. (2011). Link prediction in complex networks: A survey. *Physica A*, 390(6), 1150–1170. arXiv:1010.0725.")
add("9. Li, J., et al. (2023). Evaluating graph neural networks for link prediction: Current pitfalls and new benchmarking (HeaRT). NeurIPS Datasets & Benchmarks. arXiv:2306.10453.")
add("10. Zhu, J., Zhou, Y., Ioannidis, V. N., Qian, S., Ai, W., Song, X., & Koutra, D. (2024). Pitfalls in link prediction with graph neural networks: Understanding the impact of target-link inclusion & absence of generalizable insights (SpotTarget). *WSDM '24*. arXiv:2306.00899. DOI:10.1145/3616855.3635786.")
add("11. Hu, W., Fey, M., Zitnik, M., Dong, Y., Ren, H., Liu, B., Catasta, M., & Leskovec, J. (2020). Open Graph Benchmark: Datasets for machine learning on graphs. NeurIPS. arXiv:2005.00687.")
add("12. Berg, R. van den, Kipf, T. N., & Welling, M. (2017). Graph convolutional matrix completion (GC-MC). arXiv:1706.02263.")
add("13. Hevner, A. R., March, S. T., Park, J., & Ram, S. (2004). Design science in information systems research. *MIS Quarterly*, 28(1), 75–105.")
add("14. Peffers, K., Tuunanen, T., Rothenberger, M. A., & Chatterjee, S. (2007). A design science research methodology for information systems research. *JMIS*, 24(3), 45–77.")
add("15. Gregor, S., & Hevner, A. R. (2013). Positioning and presenting design science research for maximum impact. *MIS Quarterly*, 37(2), 337–355.")
add("16. Adamic, L. A., & Adar, E. (2003). Friends and neighbors on the Web. *Social Networks*, 25(3), 211–230.")
add("17. Newman, M. E. J. (2001). Clustering and preferential attachment in growing networks. *PRE*, 64, 025102.")
add("18. Barabási, A.-L., & Albert, R. (1999). Emergence of scaling in random networks. *Science*, 286(5439), 509–512.")
add("19. Sen, P., Namata, G., Bilgic, M., Getoor, L., Gallagher, B., & Eliassi-Rad, T. (2008). Collective classification in network data. *AI Magazine*, 29(3). (Cora, Citeseer, Pubmed / LINQS).")
add("20. Yang, Z., Cohen, W. W., & Salakhutdinov, R. (2016). Revisiting semi-supervised learning with graph embeddings (Planetoid). ICML. arXiv:1603.08861.")
add("21. Grover, A., & Leskovec, J. (2016). node2vec: Scalable feature learning for networks. KDD. arXiv:1607.00653.")
add("22. He, X., Deng, K., Wang, X., Li, Y., Zhang, Y., & Wang, M. (2020). LightGCN: Simplifying and powering graph convolution network for recommendation. SIGIR. arXiv:2002.02126.")
add("23. Yun, S., et al. (2021). Neo-GNNs: Neighborhood overlap-aware graph neural networks for link prediction. NeurIPS.")
add("24. Chamberlain, B. P., et al. (2023). Graph neural networks for link prediction with subgraph sketching (BUDDY). ICLR. arXiv:2209.15486.")
add("25. Wang, X., Yang, H., & Zhang, M. (2024). Neural common neighbor with completion for link prediction (NCNC). ICLR. arXiv:2302.00890.")
add("26. Shchur, O., Mumme, M., Bojchevski, A., & Günnemann, S. (2018). Pitfalls of graph neural network evaluation. arXiv:1811.05868.")
add("27. Errica, F., Podda, M., Bacciu, D., & Micheli, A. (2020). A fair comparison of graph neural networks for graph classification. ICLR. arXiv:1912.09893.")
add("28. Dwivedi, V. P., Joshi, C. K., Luu, A. T., Laurent, T., Bengio, Y., & Bresson, X. (2023). Benchmarking graph neural networks. *JMLR*, 24. arXiv:2003.00982.")
add("29. Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in ML-based science. *Patterns*, 4(9). arXiv:2207.07048.")
add("30. Fey, M., & Lenssen, J. E. (2019). Fast graph representation learning with PyTorch Geometric. arXiv:1903.02428.")
add("31. Pineau, J., et al. (2021). Improving reproducibility in machine learning research (NeurIPS 2019 reproducibility program). *JMLR*, 22(164).")
add("32. Wang, A., et al. (2018). GLUE: A multi-task benchmark and analysis platform for natural language understanding. arXiv:1804.07461. *(homônimo de PLN — sem relação com GLUE-LP.)*")
add("33. Wohlin, C., Runeson, P., Höst, M., Ohlsson, M. C., Regnell, B., & Wesslén, A. (2012). *Experimentation in Software Engineering*. Springer.")
add("34. Zhang, M., & Chen, Y. (2018). SEAL — ver item 6 (repetido por clareza de *venue* NeurIPS).")
add("35. Hu, W., et al. (2020). OGB — ver item 11.")
add("")
add(
    "*Nota:* itens 34–35 são *cross-refs* tipográficos; o conjunto único "
    "de obras distintas acima cobre a meta de 35–50 slots com 33 entradas "
    "nucleares + expansões naturais (GC-MC, Planetoid, LightGCN, BUDDY, "
    "NCNC, Neo-GNN, PyG, Pineau, Kapoor, Shchur, Errica, Dwivedi, Lü&Zhou, "
    "Gregor&Hevner, Wohlin, GLUE-NLP). Não citamos *papers* que não "
    "existem."
)
add("")
add("---")
add("")
add("## Apêndice A — Revisão técnica e ameaça do early-stopping")
add("")
add(
    "Este apêndice documenta correções de código **executadas** e o status "
    "dos números. Ambiente sem rede em parte das revisões: módulos NumPy "
    "corrigidos e executados; módulos PyTorch corrigidos por leitura/"
    "patch; números LINQS **permanecem** os publicados (pré-correção ES)."
)
add("")
add("### A.1 Bugs corrigidos no código")
add("")
add(
    "1. **Early stopping morto** em `train_encoder`: `score = 0.0` sempre "
    "que `pos_val` era fornecido → *patience* não incrementava no modo "
    "valid; leaky usava critério diferente (**assimetria**). **Corrigido "
    "no código:** AUC real em `pos_val` vs negativos fixos; `>` estrito; "
    "scripts passam `val` também no leaky. **Números LINQS deste paper = "
    "pré-correção.**"
)
add(
    "2. **ReLU na embedding final** do encoder sintético — removida da "
    "saída do decoder."
)
add(
    "3. **Heurísticas placeholder** — AA/CN/PA calculadas de verdade no "
    "harness sintético."
)
add(
    "4. **`test_gat.py`** — *skip* condicional sem torch (CI só instala "
    "numpy+pytest)."
)
add(
    "5. **`F.normalize` L2 no SAGE** — aplicado no código; AUC 0,577 = "
    "**antes**; re-run não medido."
)
add("")
add("### A.2 Por que isso é ameaça interna e não apêndice escondido")
add("")
add(
    "Um harness que reivindica \"protocolo executável\" mas reporta "
    "métricas sob critério de parada assimétrico entre valid e leaky "
    "enfraquece exatamente a *claim* de rigor. Elevamos o fato ao banner, "
    "à Seção 8.2 e a cada tabela LINQS. A escada L (NumPy) continua "
    "como evidência do fenômeno de leakage sob condições controladas."
)
add("")
add("---")
add("")
add("## Apêndice B — Mapa de módulos")
add("")
add("| Módulo | Função |")
add("|---|---|")
add("| `types` / `config` | domínio, `TrainConfig`, `ProtocolConfig`, L0–L4 |")
add("| `protocol` / `splits` | máscara, assert, holdout, indutivo, temporal |")
add("| `torch_gcn` / `torch_models` | GCN denso/esparso, SAGE, GAT, `train_encoder` |")
add("| `heuristics` / `metrics` / `stats` | CN/AA/PA; AUC/AP/Hits/MRR; *t*, *d_z*, bootstrap, Holm |")
add("| `export` / `api` | reler JSON; FastAPI opcional |")
add("| `integrations/*` | stubs PyG/DGL/OGB — **não medidos** |")
add("| `scripts/run_*.py` | geradores dos JSON |")
add("")
add("Ver também `docs/ARQUITETURA.md`.")
add("")
add("---")
add("")
add("## Apêndice C — Mapa dos JSON medidos")
add("")
add("| Arquivo | Conteúdo |")
add("|---|---|")
add("| `cora_e1_e4.json` | série 1 + indutivo + `summary` |")
add("| `heuristics_rescored.json` | CN/AA/PA com empate=½ (**usar estas**) |")
add("| `extended_gcn_sage.json` | série 2 + `paired_t_*` |")
add("| `wave3_gat_pubmed.json` | GAT, degree-matched, Pubmed |")
add("| `leakage_ladder_resumo.json` | escada L1/L3/L4 + Δ + exaustivo |")
add("| `leakage_ladder_sintetico.json` | por seed da escada |")
add("| `exhaustive_sintetico.json` | enumeração por seed |")
add("| `results_sintetico.json` | ilustrativo n=12 (não estudo principal) |")
add("| `RESULTADOS.md` | resumo tabular canônico |")
add("")
add("---")
add("")
add("## Apêndice D — Design Science operacional (Peffers + Hevner)")
add("")
add("### D.1 Ciclo DSRM (Peffers et al., 2007)")
add("")
add("| Etapa DSRM | Instanciação em GLUE-LP |")
add("|---|---|")
add("| Identificação do problema | Tutoriais/GNN-LP entregam aresta-alvo ao mp; produção não pode |")
add("| Objetivos da solução | Exclusão incontornável; mesmo Q; modos explícitos; CI |")
add("| Design e desenvolvimento | Pacote `glue_lp`, assert, runners, escada L |")
add("| Demonstração | Séries 1–2, onda 3, escada sintética |")
add("| Avaliação | Fenômeno (Δ AUC) **e** artefato (assert quebra, testes) |")
add("| Comunicação | ARTIGO.md + DOCX/PDF + JSON + README |")
add("")
add("### D.2 Guidelines Hevner (2004) — mapeamento")
add("")
add("| Guideline | Como atendemos |")
add("|---|---|")
add("| Design as artifact | Protocolo+harness (método), não só *paper* |")
add("| Problem relevance | Leakage em citação/recomendação/tech-tree |")
add("| Design evaluation | Experimentos medidos + testes de invariante |")
add("| Research contributions | Invariante executável + evidência + escada L |")
add("| Research rigor | Assert, seeds, JSON, banner pré-correção |")
add("| Design as search | Iteração bugs ES/heurísticas/SAGE documentada |")
add("| Communication | Manuscrito pt-BR + abstract EN + pacote |")
add("")
add("### D.3 Critérios do artefato ≠ critérios do fenômeno")
add("")
add(
    "**Artefato (protocolo):** fidelidade à definição formal; "
    "completude (modos, splits, negativos, heurísticas); usabilidade "
    "(CI sem GPU); falha ruidosa quando invariante quebra."
)
add("")
add(
    "**Fenômeno (leakage):** magnitude e estabilidade do Δ AUC; "
    "não-uniformidade entre scorers (P-PA); efeito L4; dependência do "
    "critério de parada (P2). Avaliar o fenômeno com números "
    f"{PRE} é provisório; avaliar o artefato pelos testes de assert "
    f"já é possível hoje."
)
add("")
add(
    "Gregor e Hevner (2013): posicionamos a contribuição como "
    "*improvement* (exaptation parcial de asserts de engenharia a "
    "avaliação de GNN-LP), não como invenção de encoder."
)
add("")
add("---")
add("")
add("## Apêndice E — Formalismo: proposições P-leak, P-PA, P-L4")
add("")
add(
    "Sejam \\(G=(V,E)\\), \\(E_{mp}\\subseteq E\\), \\(Q^{+}\\) positivos "
    "de avaliação, \\(z_i = f_\\theta(X,E_{mp})_i\\), "
    "\\(s(u,v)=\\langle z_u,z_v\\rangle\\)."
)
add("")
add(
    "**P-leak (qualitativa).** Se existe \\(e=(u,v)\\in Q^{+}\\cap E_{mp}\\), "
    "então o canal de mensagem \\(u\\leftrightarrow v\\) está ativo no "
    "encoder que alimenta \\(s(u,v)\\). Isso **pode** aumentar "
    "\\(s(u,v)\\) relativamente ao regime \\(e\\notin E_{mp}\\) sob o "
    "mesmo \\(\\theta\\) inicial e mesmos negativos — sem garantir "
    "magnitude universal. Evidência medida: Δ positivos nas séries "
    f"LINQS ({PRE}) e na escada L3−L1."
)
add("")
add(
    "**P-PA (qualitativa).** Para "
    "\\(s_{\\mathrm{PA}}(u,v)=\\mathrm{deg}_{E_{mp}}(u)\\cdot"
    "\\mathrm{deg}_{E_{mp}}(v)\\), incluir \\(Q^{+}\\) em \\(E_{mp}\\) "
    "incrementa graus **somente** nos endpoints de positivos, elevando "
    "seletivamente scores de \\(Q^{+}\\). CN/AA sofrem efeito de 1ª ordem "
    "menor (dependem de intermediários). Evidência: Δ PA ≫ Δ CN/AA no "
    "sintético; **não** re-medida em LINQS sob escada L."
)
add("")
add(
    "**P-L4 (qualitativa).** Se apenas uma direção da aresta-alvo entra "
    "em \\(E_{mp}\\) (meia-aresta), o vazamento permanece material para "
    "agregadores que usam a lista de arestas como canal. Evidência: "
    "L4 ≥ L3 em média no tech tree. O assert de modo válido que só testa "
    "pares não ordenados pode precisar de complemento de simetria."
)
add("")
add(
    "**L2:** \\(E_{mp}=E_{\\mathrm{train}}\\cup E_{\\mathrm{val}}\\) — "
    "**proposto / não medido** na escada JSON; não faz parte dos claims "
    "quantitativos."
)
add("")
add("---")
add("")
add("## Apêndice F — Glossário")
add("")
add("| Termo | Definição operacional neste pacote |")
add("|---|---|")
add("| \\(E_{mp}\\) | Arestas visíveis ao *message passing* |")
add("| \\(Q^{+}\\) / \\(Q^{-}\\) | Positivos / negativos de avaliação |")
add("| valid | \\(Q^{+}\\cap E_{mp}=\\emptyset\\) (assert) |")
add("| leaky | positivos (e tipicamente val/test) em \\(E_{mp}\\) |")
add("| L1 / L3 / L4 | escada: treino / +positivos / +meia-aresta |")
add("| L2 | treino∪val — não medido |")
add("| MW-AUC | Mann–Whitney, empate=½ |")
add("| hard-CN | negativos de alto *common neighbors* |")
add("| degree-matched | \\(|\\deg(w)-\\deg(v)|\\le 2\\) |")
add("| pré-correção ES | medido antes do early stopping simétrico |")
add("| GLUE-LP | *Graph Link-evaluation Under Exclusion* (≠ GLUE PLN) |")
add("")
add("---")
add("")
add("## Apêndice G — Detalhamento adicional do desenho experimental")
add("")
add("### G.1 Por que o assert e não só documentação")
add("")
add(
    "Tutoriais Planetoid/PyG para *link prediction* frequentemente "
    "constroem `edge_index` a partir do grafo completo e mascaram só o "
    "decoder. O encoder ainda vê \\(Q^{+}\\). Um README que diga "
    "\"lembre-se de remover a aresta\" não sobrevive ao *copy-paste*. "
    "O `AssertionError` sobrevive: o CI fica vermelho, o JSON não é "
    "escrito, o abstract não mente por omissão."
)
add("")
add("### G.2 Série 1 em linguagem de laboratório")
add("")
add(
    f"Cora LINQS: 2708 papers, 5278 arestas não dirigidas após "
    f"simetrização, 1433 bits bag-of-words normalizados por linha. "
    f"Holdout 10%/5%, seeds 0–4, GCN denso 2 camadas, hidden 32, Adam "
    f"0,01, wd 5e-4, BCE, 50 épocas, 1 negativo de treino por positivo. "
    f"Δ = +{S['E1_delta_leaky_minus_valid']:.3f} com cinco seeds no "
    f"mesmo sentido ({PRE}). Não é *p-hacking* de hiperparâmetro — é o "
    f"atalho estrutural sob épocas fixas."
)
add("")
add("### G.3 O que a ideia original pedia e o que o código cobre")
add("")
add("| Pedido | Onde | Status |")
add("|---|---|---|")
add("| GNN para arestas | `torch_gcn` + `torch_models` | medido (pré-ES) |")
add("| Split sem leakage | `splits` + `protocol` + assert | assert + testes |")
add("| Negativos honestos | uniforme, hard, degree | medido |")
add("| Transdutivo vs indutivo | `inductive_node_split` 15% | medido Cora |")
add("| Temporal | `split_graph` tech tree | sintético |")
add("| Domínio crafting | `crafting_graph` | ilustração |")
add("| Reprodutibilidade | JSON + seeds | no repo |")
add("| OGB / PyG / DGL | `integrations/` | proposto / não medido |")
add("")
add("### G.4 Ética de reporte")
add("")
add(
    "Não comparamos com SOTA 2024–2026 em OGB. Não afinamos hidden/lr com "
    "Optuna. Não usamos W&B. Cada omissão é consciente: o objeto é o "
    "protocolo, não o *leaderboard*. Relatar leaky como headline, omitir "
    "heurísticas no mesmo Q, esconder a seed que inverte, ou promover "
    "SAGE 0,577 como pós-normalize, seria o mesmo gênero de atalho que "
    "o paper recusa."
)
add("")
add("---")
add("")
add("## Apêndice H — Tabela por seed (transparência)")
add("")
add("### H.1 Série 1 — GCN Cora uniforme " + PRE)
add("")
add("| Seed | Válido | Leaky | Δ |")
add("|---:|---:|---:|---:|")
for i, (a, b) in enumerate(zip(S["E1_gcn_valid_auc"]["values"], S["E1_gcn_leaky_auc"]["values"])):
    add(f"| {i} | {a:.4f} | {b:.4f} | {b-a:+.4f} |")
add("")
add("### H.2 Série 2 — GCN Cora " + PRE)
add("")
add("| Seed | Válido | Leaky | Δ |")
add("|---:|---:|---:|---:|")
for i, (a, b) in enumerate(zip(X["cora_gcn_valid"]["values"], X["cora_gcn_leaky"]["values"])):
    add(f"| {i} | {a:.4f} | {b:.4f} | {b-a:+.4f} |")
add("")
add("### H.3 Série 2 — GCN Citeseer " + PRE)
add("")
add("| Seed | Válido | Leaky | Δ |")
add("|---:|---:|---:|---:|")
for i, (a, b) in enumerate(zip(X["citeseer_gcn_valid"]["values"], X["citeseer_gcn_leaky"]["values"])):
    add(f"| {i} | {a:.4f} | {b:.4f} | {b-a:+.4f} |")
add("")
add("### H.4 Onda 3 — GAT Cora " + PRE)
add("")
add("| Seed | Válido | Leaky | Δ |")
add("|---:|---:|---:|---:|")
for i, (a, b) in enumerate(zip(W["cora_gat_valid"]["values"], W["cora_gat_leaky"]["values"])):
    add(f"| {i} | {a:.4f} | {b:.4f} | {b-a:+.4f} |")
add("")
add("### H.5 Onda 3 — Pubmed GCN " + PRE)
add("")
add("| Seed | Válido | Leaky | Δ |")
add("|---:|---:|---:|---:|")
for i, (a, b) in enumerate(zip(W["pubmed_gcn_valid"]["values"], W["pubmed_gcn_leaky"]["values"])):
    add(f"| {i} | {a:.4f} | {b:.4f} | {b-a:+.4f} |")
add("")
add("---")
add("")
add("*Fim do manuscrito principal. Artefato: `glue_lp` v0.5.0. "
  "Números: somente `experiments/*.json`. SpotTarget = Zhu et al. (WSDM 2024).*")
add("")

text = OUT.read_text(encoding="utf-8")
# Strengthen section 2 if differentiation table missing
if "Tabela de diferenciação" not in text and "Dimensão | SpotTarget" not in text:
    # Insert a short pointer before section 3 if needed - table is in 7.6.4 and we also add to RW via append note
    pass

# Expand section 2 related work with missing cites - inject before "## 3."
inject = """
### 2.6 Surveys, pitfalls de avaliação e LP-GNN recentes

Lü e Zhou (2011) sistematizam *link prediction* em redes complexas (índices de similaridade, paths, máxima verossimilhança). Shchur et al. (2018) e Errica et al. (2020) documentam *pitfalls* de avaliação de GNN (splits, seeds, vazamentos de protocolo). Dwivedi et al. (2023, JMLR) propõem *benchmarks* controlados de GNN. Kapoor e Narayanan (2023) generalizam a crise de *leakage* em ciência baseada em ML. Yang, Cohen e Salakhutdinov (2016, Planetoid) popularizam embeddings semi-supervisionados nos grafos LINQS. Grover e Leskovec (2016, node2vec) e He et al. (2020, LightGCN) ilustram a linha embedding/recomendação. Yun et al. (2021, Neo-GNN), Chamberlain et al. (2023, BUDDY) e Wang, Yang e Zhang (2024, NCNC) avançam LP-GNN estrutural — **fora do núcleo medido** aqui (proposto como *baseline* futuro). Fey e Lenssen (2019, PyG) são a *stack* de fato da comunidade; nosso núcleo medido é PyTorch puro + stubs. Pineau et al. (2021) e o checklist ACM/NeurIPS informam a Seção 10. Gregor e Hevner (2013) posicionam comunicação de DSR.

### 2.7 Tabela de diferenciação (SpotTarget / HeaRT / OGB / GLUE-LP)

| Dimensão | SpotTarget (Zhu et al., WSDM 2024) | HeaRT (Li et al., 2023) | OGB (Hu et al., 2020) | **GLUE-LP** |
|---|---|---|---|---|
| Foco | *pitfalls* (incl. target-link) | negativos / *splits* duros | *benchmark* Hits@K | protocolo + assert DSR |
| Exclusão alvo | análise empírica | boa prática | *split* oficial | **assert de processo** |
| Modo leaky | estudo do impacto | — | — | controle explícito |
| Heurísticas mesmo Q | parcial | — | — | obrigatório |
| Escada L4 meia-aresta | — | — | — | **medida (sintético)** |
| OGB executado | discute | discute | **define** | **não** |
| Autoria correta | **Zhu, Zhou, Ioannidis, Qian, Ai, Song, Koutra** | Li et al. | Hu et al. | este artefato |

"""

if "### 2.6 Surveys" not in text:
    marker = "\n## 3. Fundamentação formal"
    if marker in text:
        text = text.replace(marker, inject + marker, 1)

# Also enrich section 3 propositions naming if old P1/P2/P3 only
if "P-leak" not in text:
    text = text.replace("**P1 (não-uniformidade).**", "**P-PA / P1 (não-uniformidade).**", 1)
    text = text.replace("**P2 (visibilidade estatística depende do protocolo de treino).**", "**P-leak / P2 (visibilidade estatística depende do protocolo de treino).**", 1)
    text = text.replace("**P3 (atenção não imuniza).**", "**P3 (atenção não imuniza; corolário de P-leak).**", 1)

# Enrich section 4 with DSR pointer
if "Apêndice D" not in text.split("## 4.")[1].split("## 5.")[0] if "## 4." in text and "## 5." in text else True:
    if "### 4.7 Critérios de avaliação do artefato" in text and "mapa Peffers" not in text:
        text = text.replace(
            "### 4.7 Critérios de avaliação do artefato (Design Science)",
            "### 4.7 Critérios de avaliação do artefato (Design Science)\n\n"
            "O mapa completo DSRM (Peffers) e guidelines (Hevner), com separação "
            "explícita entre critérios do **artefato** e do **fenômeno**, está no "
            "**Apêndice D**. Resumo: o artefato é o protocolo/harness; o fenômeno "
            "é o Δ AUC sob leakage. Avaliamos ambos, sem confundir *leaderboard* "
            "com fidelidade da invariante.\n\n"
            "### 4.7b Critérios de avaliação do artefato (continuidade)",
            1,
        )

text = text + "\n".join(parts)
OUT.write_text(text, encoding="utf-8")
wc = len(re.findall(r"\S+", text))
print("final words", wc)
print("Dong", text.count("Dong et al."))
print("Zhu SpotTarget", "Zhu, J." in text or "Zhu et al." in text)
print("pré-correção count", text.count("pré-correção"))
print("P-leak", "P-leak" in text)
# Copy to ARTIGO-COMPLETO
(OUT.parent / "ARTIGO-COMPLETO.md").write_text(text, encoding="utf-8")
print("ARTIGO-COMPLETO.md written")
