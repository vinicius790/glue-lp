"""Tiny hand checks for paired_t / cohens_dz."""
from glue_lp.stats import cohens_dz, paired_t


def test_paired_t_identical():
    a = [0.7, 0.8, 0.9]
    r = paired_t(a, a)
    assert abs(r["mean_diff"]) < 1e-12
    assert r["df"] == 2
    assert r["n"] == 3


def test_paired_t_known_delta():
    a = [0.8, 0.9, 1.0]
    b = [0.5, 0.6, 0.7]
    r = paired_t(a, b)
    assert abs(r["mean_diff"] - 0.3) < 1e-12
    assert r["df"] == 2
    assert r["t"] > 0  # all diffs equal → huge t (fp may not be exact inf)


def test_cohens_dz_positive():
    a = [1.0, 2.0, 4.0]
    b = [0.0, 1.0, 2.0]
    d = cohens_dz(a, b)
    assert d > 0
