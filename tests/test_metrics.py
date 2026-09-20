from glue_lp.metrics import roc_auc

def test_perfect():
    assert roc_auc([1, 0], [1.0, 0.0]) == 1.0

def test_tie():
    assert roc_auc([1, 0], [0.5, 0.5]) == 0.5
