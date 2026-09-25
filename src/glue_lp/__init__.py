"""GLUE-LP: Graph Link-evaluation Under Exclusion."""
from .protocol import assert_no_leakage, leakage_exists, split_graph
from .run import compare_protocols, run_experiment
from .metrics import roc_auc, average_precision, hits_at_k, mrr, metric_block
from .config import TrainConfig, ProtocolConfig, DatasetSpec, DEFAULT_TRAIN

__all__ = [
    "assert_no_leakage", "leakage_exists", "split_graph",
    "run_experiment", "compare_protocols",
    "roc_auc", "average_precision", "hits_at_k", "mrr", "metric_block",
    "TrainConfig", "ProtocolConfig", "DatasetSpec", "DEFAULT_TRAIN",
]
__version__ = "0.5.0"
