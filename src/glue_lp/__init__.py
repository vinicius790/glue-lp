"""GLUE-LP: protocolo de avaliacao de predicao de ligacoes sem leakage."""
from .protocol import assert_no_leakage, leakage_exists, split_graph
from .run import compare_protocols, run_experiment
__all__ = ["assert_no_leakage", "leakage_exists", "split_graph", "run_experiment", "compare_protocols"]
__version__ = "0.1.0"
