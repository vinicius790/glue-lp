"""Package version must stay aligned with pyproject.toml."""
from pathlib import Path

import glue_lp


def test_version_matches_pyproject():
    root = Path(__file__).resolve().parents[1]
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    line = next(ln for ln in text.splitlines() if ln.startswith("version"))
    # version = "0.5.0"
    py_ver = line.split("=", 1)[1].strip().strip('"').strip("'")
    assert glue_lp.__version__ == py_ver


def test_citation_cff_version():
    root = Path(__file__).resolve().parents[1]
    text = (root / "CITATION.cff").read_text(encoding="utf-8")
    line = next(ln for ln in text.splitlines() if ln.startswith("version:"))
    cff_ver = line.split(":", 1)[1].strip()
    assert cff_ver == glue_lp.__version__
