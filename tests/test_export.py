"""Load measured experiment JSON summaries (read-only)."""
from pathlib import Path

from glue_lp.export import (
    KNOWN_FILES,
    format_mean_std,
    list_measured_summaries,
    load_json,
    load_summary,
)

EXP = Path(__file__).resolve().parents[1] / "experiments"


def test_known_files_exist():
    for fname in KNOWN_FILES.values():
        assert (EXP / fname).exists(), fname


def test_load_serie1_summary():
    s = load_summary("serie1", exp_dir=EXP)
    assert "E1_gcn_valid_auc" in s
    assert abs(s["E1_gcn_valid_auc"]["mean"] - 0.708) < 0.01
    assert abs(s["E1_delta_leaky_minus_valid"] - 0.118) < 0.01


def test_load_leakage_ladder():
    data = load_json("leakage_ladder", exp_dir=EXP)
    assert "escada" in data
    assert abs(data["escada"]["L1"]["mean"] - 0.7133) < 1e-4


def test_list_measured():
    items = list_measured_summaries(exp_dir=EXP)
    keys = {i["key"] for i in items}
    assert "serie1" in keys and "onda3" in keys and "leakage_ladder" in keys


def test_format_mean_std():
    assert format_mean_std({"mean": 0.708, "std": 0.02}) == "0.708 ± 0.020"
