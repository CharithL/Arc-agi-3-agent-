"""The Hum — enforced predict -> compare -> update falsification loop (S3)."""

from charith.hum.comparator import Outcome, score_prediction
from charith.hum.ledger import (
    Ledger,
    PredictionRecord,
    ExpansionGate,
    gate_expansion,
)

__all__ = [
    "Outcome",
    "score_prediction",
    "Ledger",
    "PredictionRecord",
    "ExpansionGate",
    "gate_expansion",
]
