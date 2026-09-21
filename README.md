# GLUE-LP

Predicao de ligacoes sem a aresta-alvo no message passing.

- Artigo: [docs/ARTIGO.md](docs/ARTIGO.md)
- Resultados: [experiments/RESULTADOS.md](experiments/RESULTADOS.md)
- Dados LINQS: [data/LINQS.md](data/LINQS.md)

```bash
pip install -r requirements-dev.txt
make test
PYTHONPATH=src python3 scripts/check_protocol.py
```

CI: `.github/workflows/tests.yml` (pytest, sem Cora).
