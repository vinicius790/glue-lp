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
    return {"x": x, "edges": np.asarray(edges, np.int64), "y": np.array([ymap[s] for s in labels], np.int64), "n": len(ids)}

def load_cora(root=None):
    base = Path(root) if root else Path(__file__).resolve().parents[2] / "data" / "cora"
    d = load_linqs_content(base/"cora.content", base/"cora.cites"); d["name"]="cora"; return d

def load_citeseer(root=None):
    base = Path(root) if root else Path(__file__).resolve().parents[2] / "data" / "citeseer"
    d = load_linqs_content(base/"citeseer.content", base/"citeseer.cites"); d["name"]="citeseer"; return d

def load_pubmed(root=None):
    base = Path(root) if root else Path(__file__).resolve().parents[2] / "data" / "Pubmed-Diabetes" / "data"
    node_p = base / "Pubmed-Diabetes.NODE.paper.tab"
    cite_p = base / "Pubmed-Diabetes.DIRECTED.cites.tab"
    ids, rows, labels = [], [], []
    with node_p.open() as f:
        next(f)
        header = next(f).split("\t")
        feat_names = [h.split(":")[1] for h in header if h.startswith("numeric:")]
        index_f = {name: i for i, name in enumerate(feat_names)}
        d = len(feat_names)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2: continue
            ids.append(parts[0])
            vec = np.zeros(d, np.float32); lab = 0
            for tok in parts[1:]:
                if tok.startswith("label="):
                    lab = int(tok.split("=",1)[1])
                elif tok.startswith("w-") and "=" in tok:
                    k, v = tok.split("=",1)
                    j = index_f.get(k)
                    if j is not None: vec[j] = float(v)
            rows.append(vec); labels.append(lab)
    x = np.vstack(rows); s = x.sum(1, keepdims=True); s[s==0]=1; x = x/s
    index = {pid:i for i,pid in enumerate(ids)}
    seen, edges = set(), []
    with cite_p.open() as f:
        for line in f:
            if "paper:" not in line or "|" not in line: continue
            left, right = line.split("|",1)
            a = left.split("paper:")[-1].strip(); b = right.split("paper:")[-1].strip()
            if a not in index or b not in index: continue
            i, j = index[a], index[b]
            if i==j: continue
            key = (i,j) if i<j else (j,i)
            if key not in seen:
                seen.add(key); edges.append(key)
    return {"x": x, "edges": np.asarray(edges, np.int64), "y": np.asarray(labels, np.int64), "n": len(ids), "n_features": int(x.shape[1]), "name": "pubmed"}
