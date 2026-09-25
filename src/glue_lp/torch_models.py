from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .metrics import roc_auc

def device_of():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def edge_index_sym(n, edges, device):
    src = [int(u) for u,v in edges] + [int(v) for u,v in edges] + list(range(n))
    dst = [int(v) for u,v in edges] + [int(u) for u,v in edges] + list(range(n))
    src_t = torch.tensor(src, dtype=torch.long, device=device)
    dst_t = torch.tensor(dst, dtype=torch.long, device=device)
    deg = torch.zeros(n, device=device)
    deg.index_add_(0, src_t, torch.ones(len(src), device=device))
    dinv = torch.pow(deg.clamp(min=1.0), -0.5)
    w = dinv[src_t] * dinv[dst_t]
    return src_t, dst_t, w

def spmm(src, dst, w, x, n):
    msg = x[src] * w.unsqueeze(-1)
    out = torch.zeros(n, x.size(1), device=x.device, dtype=x.dtype)
    out.index_add_(0, dst, msg)
    return out

class SparseGCN(nn.Module):
    def __init__(self, in_dim, hidden=32, dropout=0.5):
        super().__init__()
        self.w1 = nn.Linear(in_dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, hidden, bias=False)
        self.dropout = dropout
    def encode(self, x, src, dst, w, n):
        h = F.relu(self.w1(spmm(src, dst, w, x, n)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.w2(spmm(src, dst, w, h, n))

class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden=32, dropout=0.5):
        super().__init__()
        self.w1 = nn.Linear(in_dim*2, hidden, bias=True)
        self.w2 = nn.Linear(hidden*2, hidden, bias=True)
        self.dropout = dropout
    def _sage(self, x, src, dst, n, linear):
        msg = torch.zeros(n, x.size(1), device=x.device, dtype=x.dtype)
        msg.index_add_(0, dst, x[src])
        deg = torch.zeros(n, device=x.device)
        deg.index_add_(0, dst, torch.ones(src.numel(), device=x.device))
        neigh = msg / deg.clamp(min=1.0).unsqueeze(-1)
        return F.relu(linear(torch.cat([x, neigh], 1)))
    def encode(self, x, src, dst, w, n):
        # L2-normalize final embedding (GraphSAGE paper practice; recommended fix
        # for the low Cora SAGE AUC 0.577 measured under the prior code — re-run
        # after this change is proposed / not yet measured).
        h = F.dropout(self._sage(x, src, dst, n, self.w1), p=self.dropout, training=self.training)
        z = self._sage(h, src, dst, n, self.w2)
        return F.normalize(z, p=2, dim=-1)

class GAT(nn.Module):
    def __init__(self, in_dim, hidden=32, dropout=0.4):
        super().__init__()
        self.w1 = nn.Linear(in_dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, hidden, bias=False)
        self.a1 = nn.Linear(2*hidden, 1, bias=False)
        self.a2 = nn.Linear(2*hidden, 1, bias=False)
        self.dropout = dropout
    def _gat(self, x, src, dst, n, lin, att):
        h = lin(x)
        e = F.leaky_relu(att(torch.cat([h[src], h[dst]], 1)).squeeze(-1), 0.2)
        exp = torch.exp(e - e.max())
        den = torch.zeros(n, device=x.device, dtype=x.dtype)
        den.index_add_(0, dst, exp)
        alpha = exp / den[dst].clamp(min=1e-12)
        out = torch.zeros(n, h.size(1), device=x.device, dtype=x.dtype)
        out.index_add_(0, dst, h[src]*alpha.unsqueeze(-1))
        return F.elu(out)
    def encode(self, x, src, dst, w, n):
        h = F.dropout(self._gat(x, src, dst, n, self.w1, self.a1), p=self.dropout, training=self.training)
        return self._gat(h, src, dst, n, self.w2, self.a2)

def _sample_negatives(rng, n, forbidden, k):
    neg, tries = [], 0
    while len(neg) < k and tries < k * 200:
        a, b = int(rng.integers(0, n)), int(rng.integers(0, n))
        tries += 1
        if a == b:
            continue
        key = (a, b) if a < b else (b, a)
        if key in forbidden:
            continue
        neg.append(key)
    return np.array(neg, dtype=np.int64) if neg else np.zeros((0, 2), dtype=np.int64)


def train_encoder(x_np, mp_edges, pos_train, pos_val, n, *, seed, kind="gcn", hidden=32, epochs=80, lr=0.01, patience=15, weight_decay=5e-4, dropout=None):
    """Correcao (2026): a versao anterior calculava `score = 0.0` sempre que
    `pos_val` era fornecido (linha morta: `... if pos_val is None else 0.0`),
    o que tornava a condicao `score >= best_val` verdadeira em TODO epoch e
    fazia `stall` nunca incrementar. Resultado: o early stopping baseado em
    validacao nunca disparava no modo "valid" (patience virava no-op e
    best_z era sempre o ultimo epoch, sem selecao real de modelo), enquanto
    o modo "leaky" usava um criterio diferente (perda de treino). Isso e
    uma assimetria que invalida a comparacao pareada valido-vs-leaky.
    Agora `pos_val` e usado de fato: AUC real em negativos amostrados uma
    unica vez (fixos ao longo do treino, para a serie de "score" por epoch
    ser comparavel), com `>` estrito para contagem correta de patience.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    dev = device_of()
    x = torch.from_numpy(np.ascontiguousarray(x_np)).to(dev)
    src, dst, w = edge_index_sym(n, mp_edges, dev)
    ctor = {"gcn": SparseGCN, "sage": GraphSAGE, "gat": GAT}[kind]
    # dropout=None → keep each class default (GCN/SAGE 0.5, GAT 0.4) to match measured runs
    model = (ctor(x.size(1), hidden, dropout=dropout) if dropout is not None else ctor(x.size(1), hidden)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(seed + 17)
    forbidden = {(int(a), int(b)) if a < b else (int(b), int(a)) for a, b in mp_edges}

    has_val = pos_val is not None and len(pos_val) > 0
    val_neg = _sample_negatives(rng, n, forbidden, max(len(pos_val), 8)) if has_val else None

    best_z, best_val, stall, epoch, z_np = None, -1.0, 0, 0, None
    for epoch in range(epochs):
        model.train()
        neg = _sample_negatives(rng, n, forbidden, max(len(pos_train), 8))
        z = model.encode(x, src, dst, w, n)
        pt = torch.as_tensor(pos_train, device=dev)
        nt = torch.as_tensor(neg, device=dev)
        loss = F.binary_cross_entropy_with_logits(
            torch.cat([(z[pt[:, 0]] * z[pt[:, 1]]).sum(-1),
                       (z[nt[:, 0]] * z[nt[:, 1]]).sum(-1)]),
            torch.cat([torch.ones(len(pos_train), device=dev), torch.zeros(len(neg), device=dev)]))
        opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            z_np = model.encode(x, src, dst, w, n).cpu().numpy()
        if has_val and len(val_neg):
            sp = (z_np[pos_val[:, 0]] * z_np[pos_val[:, 1]]).sum(1)
            sn = (z_np[val_neg[:, 0]] * z_np[val_neg[:, 1]]).sum(1)
            score = roc_auc([1] * len(pos_val) + [0] * len(val_neg), np.concatenate([sp, sn]).tolist())
        else:
            score = -float(loss.detach().cpu())
        if score > best_val:
            best_val, best_z, stall = score, z_np, 0
        else:
            stall += 1
            if stall >= patience:
                break
    return {"z": best_z if best_z is not None else z_np, "best_val": best_val, "epochs_run": epoch + 1}
