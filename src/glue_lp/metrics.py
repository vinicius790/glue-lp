from __future__ import annotations

def roc_auc(y: list[int], s: list[float]) -> float:
    pos = [sc for sc, lab in zip(s, y) if lab == 1]
    neg = [sc for sc, lab in zip(s, y) if lab == 0]
    if not pos or not neg:
        return 0.5
    gt = eq = 0
    for p in pos:
        for n in neg:
            if p > n:
                gt += 1
            elif p == n:
                eq += 1
    return (gt + 0.5 * eq) / (len(pos) * len(neg))

def average_precision(y: list[int], s: list[float]) -> float:
    pairs = sorted(zip(s, y), key=lambda p: -p[0])
    p = sum(y)
    if p == 0:
        return 0.0
    tp = seen = acc = 0.0
    for _, lab in pairs:
        seen += 1
        if lab != 1:
            continue
        tp += 1
        acc += tp / seen
    return acc / p

def hits_at_k(y: list[int], s: list[float], k: int) -> float:
    pos = [i for i, lab in enumerate(y) if lab == 1]
    if not pos:
        return 0.0
    hit = 0
    for i in pos:
        better = sum(1 for j, lab in enumerate(y) if lab == 0 and s[j] > s[i])
        if better < k:
            hit += 1
    return hit / len(pos)

def mrr(y: list[int], s: list[float]) -> float:
    pos = [i for i, lab in enumerate(y) if lab == 1]
    if not pos:
        return 0.0
    tot = 0.0
    for i in pos:
        better = sum(1 for j, lab in enumerate(y) if lab == 0 and s[j] > s[i])
        tot += 1.0 / (better + 1)
    return tot / len(pos)
