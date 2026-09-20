from __future__ import annotations

def roc_auc(y, s):
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

def average_precision(y, s):
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
