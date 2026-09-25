"""Hand-computed tiny cases for AUC (tie=0.5), AP, Hits@K, MRR."""
from glue_lp.metrics import (
    average_precision,
    hits_at_k,
    metric_block,
    mrr,
    roc_auc,
)


def test_empate_e_meio():
    # todos iguais → 0.5
    assert abs(roc_auc([1, 1, 0, 0], [0, 0, 0, 0]) - 0.5) < 1e-9


def test_separacao_perfeita():
    assert roc_auc([1, 1, 0, 0], [2, 3, 0, 1]) == 1.0


def test_inverso():
    assert roc_auc([1, 0], [0, 1]) == 0.0


def test_auc_partial_tie_hand():
    # pos={1.0, 0.5}, neg={0.5, 0.0}
    # pairs: (1>0.5)=1, (1>0)=1, (0.5==0.5)=0.5, (0.5>0)=1 → (1+1+0.5+1)/4 = 0.875
    y = [1, 1, 0, 0]
    s = [1.0, 0.5, 0.5, 0.0]
    assert abs(roc_auc(y, s) - 0.875) < 1e-12


def test_auc_empty_side():
    assert roc_auc([1, 1], [0.2, 0.3]) == 0.5
    assert roc_auc([0, 0], [0.2, 0.3]) == 0.5


def test_ap_perfect():
    # scores descending already separate: pos first
    y = [1, 1, 0, 0]
    s = [4.0, 3.0, 2.0, 1.0]
    # precisions at pos ranks 1 and 2: 1/1 + 2/2 = 2 → AP=1
    assert abs(average_precision(y, s) - 1.0) < 1e-12


def test_ap_hand_mixed():
    # order by score: pos(3), neg(2), pos(1) → prec@1=1/1, prec@3=2/3 → AP=(1+2/3)/2 = 5/6
    y = [1, 0, 1]
    s = [3.0, 2.0, 1.0]
    assert abs(average_precision(y, s) - (1.0 + 2.0 / 3.0) / 2.0) < 1e-12


def test_ap_no_positives():
    assert average_precision([0, 0], [1.0, 0.5]) == 0.0


def test_hits_at_k_hand():
    # one positive score 0.5; negatives 0.9 and 0.1
    # better=1 (only 0.9) → Hits@1 = 0 (better < 1? no); Hits@2 = 1
    y = [1, 0, 0]
    s = [0.5, 0.9, 0.1]
    assert hits_at_k(y, s, 1) == 0.0
    assert hits_at_k(y, s, 2) == 1.0
    assert hits_at_k(y, s, 0) == 0.0


def test_hits_two_positives():
    # pos A=0.8 (0 negs better), pos B=0.2 (1 neg better: 0.5)
    y = [1, 1, 0]
    s = [0.8, 0.2, 0.5]
    assert abs(hits_at_k(y, s, 1) - 0.5) < 1e-12  # only A hits@1
    assert hits_at_k(y, s, 2) == 1.0


def test_mrr_hand():
    # pos score 0.5; one neg 0.9 → better=1 → RR=1/2
    y = [1, 0]
    s = [0.5, 0.9]
    assert abs(mrr(y, s) - 0.5) < 1e-12


def test_mrr_perfect():
    assert abs(mrr([1, 0], [1.0, 0.0]) - 1.0) < 1e-12


def test_metric_block_keys():
    block = metric_block([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2], ks=(1, 3))
    assert set(block) >= {"auc", "ap", "mrr", "hits1", "hits3"}
    assert block["auc"] == 1.0
