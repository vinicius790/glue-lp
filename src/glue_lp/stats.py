from __future__ import annotations
import math

def paired_t(a, b):
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    mean = sum(d) / n
    var = sum((x - mean) ** 2 for x in d) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n) if n else float("nan")
    t = mean / se if se else float("inf")
    return {"mean_diff": mean, "t": t, "df": n - 1, "n": n, "se": se}
