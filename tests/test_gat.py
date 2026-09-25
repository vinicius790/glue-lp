import numpy as np

try:
    import torch
except ImportError:
    import pytest
    pytest.skip("torch nao instalado neste ambiente", allow_module_level=True)

from glue_lp.torch_models import GAT, edge_index_sym

def test_gat_shape():
    n, d = 6, 4
    x = torch.randn(n, d)
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=np.int64)
    src, dst, w = edge_index_sym(n, edges, torch.device("cpu"))
    z = GAT(d, hidden=8, dropout=0.0).encode(x, src, dst, w, n)
    assert z.shape == (n, 8)
    assert torch.isfinite(z).all()
