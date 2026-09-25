"""Ranking metrics for link prediction (NumPy-free pure Python).

All functions take parallel lists ``y`` (0/1 labels) and ``s`` (scores).
Higher score = more likely positive.

Definitions (hand-checkable; see tests/test_metrics.py)
-------------------------------------------------------
roc_auc
    Mann–Whitney / Wilcoxon form of ROC-AUC. For every (pos, neg) pair:
    contribute 1 if score_pos > score_neg, 0.5 if equal, else 0.
    Result = mean over |pos| × |neg| pairs. Ties therefore count **0.5**.
    Empty pos or empty neg → 0.5 by convention.

average_precision (AP)
    Sort by descending score. At each positive hit, add precision@rank
    (tp / seen). Mean over positives. Identical to the usual sklearn
    ``average_precision_score`` for binary labels without sample weights.

hits_at_k
    Per positive: count negatives with *strictly higher* score as ``better``.
    Hit if ``better < k`` (rank among negatives is in the top-k).
    Mean over positives. Ties with negatives do **not** push the positive down
    (only strict ``>`` counts).

mrr
    Mean Reciprocal Rank. Per positive: rank = better + 1 where ``better`` is
    the number of negatives with strictly higher score; contribute 1/rank.
    Mean over positives.
"""
from __future__ import annotations


def roc_auc(y: list[int], s: list[float]) -> float:
    """Mann–Whitney AUC; ties contribute 0.5. Order-independent."""
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
    """Average precision (AP) over positives, sorted by descending score."""
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
    """Fraction of positives with fewer than k negatives scoring strictly higher."""
    if k <= 0:
        return 0.0
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
    """Mean reciprocal rank of positives among negatives (strict >)."""
    pos = [i for i, lab in enumerate(y) if lab == 1]
    if not pos:
        return 0.0
    tot = 0.0
    for i in pos:
        better = sum(1 for j, lab in enumerate(y) if lab == 0 and s[j] > s[i])
        tot += 1.0 / (better + 1)
    return tot / len(pos)


def metric_block(y: list[int], s: list[float], *, ks: tuple[int, ...] = (3, 10)) -> dict[str, float]:
    """Convenience: AUC, AP, Hits@K for each k in ``ks``, and MRR."""
    out: dict[str, float] = {
        "auc": roc_auc(y, s),
        "ap": average_precision(y, s),
        "mrr": mrr(y, s),
    }
    for k in ks:
        out[f"hits{k}"] = hits_at_k(y, s, k)
    return out
