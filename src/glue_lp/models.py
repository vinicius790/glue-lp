from __future__ import annotations
import math
import numpy as np
from .protocol import undirected_adj
from .rng import mulberry32

KINDS = ["resource", "material", "tool", "station", "goal"]


def embed_gcn(nodes, mp, hidden=8, seed=7):
    """Encoder GCN de 1 camada (self+vizinhanca, media, ReLU na camada oculta).

    Correcao (2026): a versao anterior aplicava ReLU tambem na embedding final,
    antes do decoder de produto interno. Isso confina a embedding ao ortante
    nao-negativo e degrada o poder discriminativo do produto interno -- o
    padrao correto (GAE/VGAE, Kipf & Welling 2016) so ativa camadas ocultas,
    nunca a saida usada pelo decoder.
    """
    rnd = mulberry32(seed)
    adj = undirected_adj(len(nodes), mp)
    max_u = max(nd.unlock for nd in nodes) or 1
    X = []
    for nd in nodes:
        oh = [1.0 if nd.kind == k else 0.0 for k in KINDS]
        X.append(oh + [nd.unlock / max_u, len(adj[nd.id]) / max(len(nodes) - 1, 1), 1.0])
    X = np.asarray(X)
    in_dim = X.shape[1]
    scale = math.sqrt(2 / (hidden + in_dim))
    W1 = np.array([[(rnd() * 2 - 1) * scale for _ in range(in_dim)] for _ in range(hidden)])

    def agg(feat, adjacency):
        out = np.zeros_like(feat)
        for v in range(len(nodes)):
            acc = feat[v].copy()
            for u in adjacency[v]:
                acc += feat[u]
            out[v] = acc / (len(adjacency[v]) + 1)
        return out

    hidden_repr = np.maximum(agg(X, adj) @ W1.T, 0)  # ReLU e correta AQUI (camada oculta)
    return hidden_repr  # sem ativacao na saida: e o embedding usado pelo decoder


def embed_gcn_directed_half(nodes, mp, half_source, half_target, hidden=8, seed=7):
    """Como embed_gcn, mas com `mp` = grafo SEM a aresta-alvo (u,v) em nenhuma
    direcao, e entao adiciona a aresta de volta em APENAS uma direcao:
    half_source passa a enxergar half_target como vizinho, mas half_target
    nao enxerga half_source. Reproduz o nivel L4 da escada de vazamento --
    o bug classico de grafo nao-direcionado onde so uma das duas linhas do
    CSV foi removida ao montar o holdout."""
    rnd = mulberry32(seed)
    n = len(nodes)
    adj = undirected_adj(n, mp)  # mp = treino (SEM a aresta-alvo em nenhuma direcao)
    adj[half_source] = set(adj[half_source]) | {half_target}  # bug: so uma direcao foi adicionada de volta
    max_u = max(nd.unlock for nd in nodes) or 1
    X = []
    for nd in nodes:
        oh = [1.0 if nd.kind == k else 0.0 for k in KINDS]
        deg_sym = len(undirected_adj(n, mp)[nd.id])
        X.append(oh + [nd.unlock / max_u, deg_sym / max(n - 1, 1), 1.0])
    X = np.asarray(X)
    in_dim = X.shape[1]
    scale = math.sqrt(2 / (hidden + in_dim))
    W1 = np.array([[(rnd() * 2 - 1) * scale for _ in range(in_dim)] for _ in range(hidden)])

    def agg(feat, adjacency):
        out = np.zeros_like(feat)
        for v in range(n):
            acc = feat[v].copy()
            for u in adjacency[v]:
                acc += feat[u]
            out[v] = acc / (len(adjacency[v]) + 1)
        return out

    return np.maximum(agg(X, adj) @ W1.T, 0)


def score_pairs(Z, pairs):
    return [float(Z[u] @ Z[v]) for u, v in pairs]


def adamic_adar(n, mp, pairs):
    adj = undirected_adj(n, mp)
    deg = [len(s) for s in adj]
    out = []
    for u, v in pairs:
        a, b = adj[u], adj[v]
        small, large = (a, b) if len(a) <= len(b) else (b, a)
        s = 0.0
        for x in small:
            if x in large:
                s += 1.0 / math.log(max(deg[x], 2))
        out.append(s)
    return out


def common_neighbors(n, mp, pairs):
    adj = undirected_adj(n, mp)
    out = []
    for u, v in pairs:
        a, b = adj[u], adj[v]
        small, large = (a, b) if len(a) <= len(b) else (b, a)
        out.append(float(sum(1 for x in small if x in large)))
    return out


def preferential_attachment(n, mp, pairs):
    adj = undirected_adj(n, mp)
    deg = [len(s) for s in adj]
    return [float(deg[u] * deg[v]) for u, v in pairs]
