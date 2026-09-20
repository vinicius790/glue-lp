from glue_lp.protocol import leakage_exists, assert_no_leakage
from glue_lp.graph import Edge

def test_keys_independem_de_ordem():
    mp = [Edge(0, 1, 0)]
    pos = [Edge(1, 0, 1)]
    assert leakage_exists(mp, pos) is True

def test_par_disjunto():
    mp = [Edge(0, 1, 0)]
    pos = [Edge(2, 3, 1)]
    assert leakage_exists(mp, pos) is False
    assert_no_leakage(mp, pos)
