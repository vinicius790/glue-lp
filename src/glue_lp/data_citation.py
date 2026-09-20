from __future__ import annotations
from pathlib import Path
import numpy as np

def load_linqs_content(content, cites):
    ids, feats, labels = [], [], []
    with Path(content).open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                feats.append([float(x) for x in parts[1:-1]])
            except ValueError:
                continue
            ids.append(parts[0]); labels.append(parts[-1])
    index = {pid: i for i, pid in enumerate(ids)}
    x = np.asarray(feats, np.float32)
    s = x.sum(1, keepdims=True); s[s==0] = 1; x = x / s
    seen, edges = set(), []
    with Path(cites).open() as f:
        for line in f:
            bits = line.strip().split()
            if len(bits) < 2 or bits[0] not in index or bits[1] not in index:
                continue
            i, j = index[bits[0]], index[bits[1]]
            if i == j: continue
            key = (i, j) if i < j else (j, i)
            if key not in seen:
                seen.add(key); edges.append(key)
    ymap = {n:i for i,n in enumerate(sorted(set(labels)))}
    return {"x": x, "edges": np.asarray(edges, np.int64), "y": np.array([ymap[s] for s in labels], np.int64), "n": len(ids), "name": "linqs"}

def load_cora(root=None):
    base = Path(root) if root else Path(__file__).resolve().parents[2] / "data" / "cora"
    d = load_linqs_content(base/"cora.content", base/"cora.cites"); d["name"]="cora"; return d

def load_citeseer(root=None):
    base = Path(root) if root else Path(__file__).resolve().parents[2] / "data" / "citeseer"
    d = load_linqs_content(base/"citeseer.content", base/"citeseer.cites"); d["name"]="citeseer"; return d
