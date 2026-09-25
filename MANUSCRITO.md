# Manuscrito GLUE-LP

- **Fonte canônica (pt-BR, ≥10k palavras):** [`docs/ARTIGO.md`](docs/ARTIGO.md)
- **Cópia longa:** [`docs/ARTIGO-COMPLETO.md`](docs/ARTIGO-COMPLETO.md)
- **DOCX / PDF:** [`GLUE-LP-artigo-cientifico.docx`](GLUE-LP-artigo-cientifico.docx) · [`GLUE-LP-artigo-cientifico.pdf`](GLUE-LP-artigo-cientifico.pdf)
  - Cópias em [`docs/`](docs/)
- **Build:** `python3 scripts/build_manuscript.py` (python-docx) + LibreOffice `soffice --headless --convert-to pdf`
- **Integridade:** somente números de `experiments/*.json` / `experiments/RESULTADOS.md`. SpotTarget = **Zhu et al.** (WSDM 2024, arXiv:2306.00899). AUC/t LINQS = **pré-correção** do early-stopping simétrico. OGB não rodou. SAGE 0,577 = pré-`F.normalize`.
- **Notas de build:** [`_build_notes_manuscript.md`](_build_notes_manuscript.md)
