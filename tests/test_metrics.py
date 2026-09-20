from glue_lp.metrics import roc_auc

def test_empate_e_meio():
    assert abs(roc_auc([1, 1, 0, 0], [0, 0, 0, 0]) - 0.5) < 1e-9

def test_separacao_perfeita():
    assert roc_auc([1, 1, 0, 0], [2, 3, 0, 1]) == 1.0

def test_inverso():
    assert roc_auc([1, 0], [0, 1]) == 0.0
