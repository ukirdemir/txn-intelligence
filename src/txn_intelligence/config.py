"""Central configuration (cost values are assumptions)."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class CostMatrix:
    """Monetary cost of each decision outcome (assumed values)."""

    false_negative: float = 100.0
    false_positive: float = 5.0
    true_positive: float = 1.0


COSTS = CostMatrix()
MODEL_PATH = os.environ.get("MODEL_PATH", "artifacts/model.joblib")
