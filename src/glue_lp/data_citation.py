from __future__ import annotations

from pathlib import Path

import numpy as np


def load_linqs_content(content: Path, cites: Path) -> dict:
    ids, feats, labels = [], [], []
    with content.open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            ids.append(parts[0])
            try:
                row = [float(x) for x in parts[1:-1]]
            except ValueError:
                continue
            feats.append(row)
            labels.append(parts[-1])
    index = {pid: i for i, pid in enumerate(ids)}
    n = len(ids)
    x = np.asarray(feats, dtype=np.float32)
    row = x.sum(axis=1, keepdims=True)
    row[row == 0] = 1.0
    x = x / row
    seen: set[tuple[int, int]] = set()
    edges: list[tuple[int, int]] = []
    skipped = 0
    with cites.open() as f:
        for line in f:
            bits = line.strip().split()
            if len(bits) < 2:
                continue
            a, b = bits[0], bits[1]
            if a not in index or b not in index:
                skipped += 1
                continue
            i, j = index[a], index[b]
            if i == j:
                continue
            key = (i, j) if i < j else (j, i)
            if key in seen:
                continue
            seen.add(key)
            edges.append(key)
    y_names = sorted(set(labels))
    y = np.array([y_names.index(s) for s in labels], dtype=np.int64)
    return {
        "x": x,
        "edges": np.asarray(edges, dtype=np.int64),
        "y": y,
        "n": n,
        "n_features": int(x.shape[1]),
        "n_classes": len(y_names),
        "skipped_cites": skipped,
        "source": str(content),
    }


def load_cora(root: Path | None = None) -> dict:
    base = Path(root) if root else Path(__file__).resolve().parents[2] / "data" / "cora"
    d = load_linqs_content(base / "cora.content", base / "cora.cites")
    d["name"] = "cora"
    return d


def load_citeseer(root: Path | None = None) -> dict:
    base = Path(root) if root else Path(__file__).resolve().parents[2] / "data" / "citeseer"
    d = load_linqs_content(base / "citeseer.content", base / "citeseer.cites")
    d["name"] = "citeseer"
    return d


def load_pubmed(root: Path | None = None) -> dict:
    base = Path(root) if root else Path(__file__).resolve().parents[2] / "data" / "Pubmed-Diabetes" / "data"
    node_p = base / "Pubmed-Diabetes.NODE.paper.tab"
    cite_p = base / "Pubmed-Diabetes.DIRECTED.cites.tab"
    feat_names: list[str] = []
    rows = []
    ids = []
    labels = []
    with node_p.open() as f:
        next(f)
        header = next(f).split("\t")
        feat_names = [h.split(":")[1] for h in header if h.startswith("numeric:")]
        index_f = {name: i for i, name in enumerate(feat_names)}
        d = len(feat_names)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            ids.append(parts[0])
            vec = np.zeros(d, dtype=np.float32)
            lab = 0
            for tok in parts[1:]:
                if tok.startswith("label="):
                    lab = int(tok.split("=", 1)[1])
                elif tok.startswith("w-") and "=" in tok:
                    k, v = tok.split("=", 1)
                    j = index_f.get(k)
                    if j is not None:
                        vec[j] = float(v)
            rows.append(vec)
            labels.append(lab)
    x = np.vstack(rows)
    s = x.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    x = x / s
    index = {pid: i for i, pid in enumerate(ids)}
    seen: set[tuple[int, int]] = set()
    edges = []
    skipped = 0
    with cite_p.open() as f:
        for line in f:
            if "paper:" not in line or "|" not in line:
                continue
            left, right = line.split("|", 1)
            a = left.split("paper:")[-1].strip()
            b = right.split("paper:")[-1].strip()
            if a not in index or b not in index:
                skipped += 1
                continue
            i, j = index[a], index[b]
            if i == j:
                continue
            key = (i, j) if i < j else (j, i)
            if key in seen:
                continue
            seen.add(key)
            edges.append(key)
    return {
        "x": x,
        "edges": np.asarray(edges, dtype=np.int64),
        "y": np.asarray(labels, dtype=np.int64),
        "n": len(ids),
        "n_features": int(x.shape[1]),
        "n_classes": int(len(set(labels))),
        "skipped_cites": skipped,
        "source": str(node_p),
        "name": "pubmed",
    }
