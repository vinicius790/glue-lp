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
