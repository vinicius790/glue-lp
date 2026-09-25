#!/usr/bin/env python3
"""SHA-256 dos arquivos LINQS locais. Não inventa hash: só lê o disco."""
from __future__ import annotations
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data"
CANDIDATES = [
    ROOT / "cora" / "cora.content",
    ROOT / "cora" / "cora.cites",
    ROOT / "citeseer" / "citeseer.content",
    ROOT / "citeseer" / "citeseer.cites",
    ROOT / "Pubmed-Diabetes" / "data" / "Pubmed-Diabetes.NODE.paper.tab",
    ROOT / "Pubmed-Diabetes" / "data" / "Pubmed-Diabetes.DIRECTED.cites.tab",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    found = 0
    for p in CANDIDATES:
        if p.is_file():
            print(f"{sha256(p)}  {p.relative_to(ROOT.parent)}")
            found += 1
        else:
            print(f"AUSENTE  {p}")
    if found == 0:
        raise SystemExit("Nenhum arquivo LINQS no disco. Ver data/LINQS.md.")


if __name__ == "__main__":
    main()
