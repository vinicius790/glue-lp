from __future__ import annotations
import math
from .metrics import roc_auc

def paired_t(a, b):
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    mean = sum(d) / n
    var = sum((x - mean) ** 2 for x in d) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n) if n else float("nan")
    t = mean / se if se else float("inf")
    return {"mean_diff": mean, "t": t, "df": n - 1, "n": n, "se": se}

def degree_bins(deg, pairs, scores, y):
    buckets = {"low": ([], []), "mid": ([], []), "high": ([], [])}
    for (u, v), s, lab in zip(pairs, scores, y):
        d = min(deg[int(u)], deg[int(v)])
        key = "low" if d < 2 else "mid" if d <= 5 else "high"
        buckets[key][0].append(int(lab))
        buckets[key][1].append(float(s))
    return {k: {"n": len(yy), "auc": roc_auc(yy, ss) if yy else None, "n_pos": int(sum(yy))} for k, (yy, ss) in buckets.items()}

def logit(p, eps=1e-4):
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))

def paired_t_logit(a, b):
    """Teste t pareado sobre AUC transformada por logit. AUC e limitada em
    [0,1]; perto de 0.9 a distribuicao amostral fica assimetrica e o t
    perde calibracao. Logit remove a restricao de dominio."""
    return paired_t([logit(x) for x in a], [logit(x) for x in b])

def cohens_dz(a, b):
    """Tamanho de efeito para medidas pareadas: d_z = media(diff) / desvio(diff)."""
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    if n < 2:
        return float("nan")
    mean = sum(d) / n
    var = sum((x - mean) ** 2 for x in d) / (n - 1)
    sd = math.sqrt(var)
    return mean / sd if sd else float("inf")

def bootstrap_ci_delta(a, b, n_boot=10000, alpha=0.05, seed=0):
    """IC bootstrap (percentil) sobre delta = media(a-b), pareado por indice
    (mesma seed -> mesmo split -> a[i] e b[i] comparaveis)."""
    import random
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    if n == 0:
        return {"lo": float("nan"), "hi": float("nan"), "point": float("nan")}
    rnd = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        sample = [d[rnd.randrange(n)] for _ in range(n)]
        boots.append(sum(sample) / n)
    boots.sort()
    lo_idx = int((alpha / 2) * n_boot)
    hi_idx = int((1 - alpha / 2) * n_boot) - 1
    return {"lo": boots[max(lo_idx, 0)], "hi": boots[min(hi_idx, n_boot - 1)],
            "point": sum(d) / n, "n_boot": n_boot}

def holm_bonferroni(named_pvalues, alpha=0.05):
    """named_pvalues: dict nome->p. Retorna dict nome->(p, p_ajustado, rejeita)
    sob o procedimento step-down de Holm (menos conservador que Bonferroni
    simples, mesmo controle de FWER)."""
    items = sorted(named_pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out = {}
    max_adj_so_far = 0.0
    for i, (name, p) in enumerate(items):
        adj = (m - i) * p
        max_adj_so_far = max(max_adj_so_far, adj)
        adj_capped = min(max_adj_so_far, 1.0)
        out[name] = {"p": p, "p_holm": adj_capped, "rejeita_h0": adj_capped < alpha}
    return out
