"""Epistemic uncertainty tracking — stub for Phase 2."""


class ConfidenceTracker:
    """Tracks overall epistemic uncertainty of the agent."""

    def __init__(self):
        self._uncertainty = 1.0

    def update(self, error_magnitude: float, rule_count: int):
        alpha = 0.1
        self._uncertainty = alpha * error_magnitude + (1 - alpha) * self._uncertainty

    @property
    def uncertainty(self) -> float:
        return self._uncertainty

    def reset(self):
        self._uncertainty = 1.0
