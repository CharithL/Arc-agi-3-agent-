"""Action Sequence Memory -- bigram model over action sequences."""
from collections import defaultdict
from typing import Dict, Tuple, Optional

import numpy as np


class ActionSequenceMemory:
    """Tracks bigram success rates over (prev_action, curr_action) pairs.

    Used to give a small boost to actions that historically follow well
    after a given previous action.  Bigrams persist across level resets
    (soft reset) so cross-level transfer learning is possible.
    """

    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        self._bigrams: Dict[Tuple[int, int], Tuple[float, float]] = defaultdict(
            lambda: (1.0, 2.0)  # Prior: 50% success rate
        )
        self._prev_action: Optional[int] = None

    def suggest_action(self, prev_action: int) -> Optional[int]:
        """Return best follow-up action if clearly better than chance (>0.5)."""
        best_action = None
        best_rate = 0.5
        for a in range(self.n_actions):
            succ, total = self._bigrams[(prev_action, a)]
            rate = succ / total
            if rate > best_rate:
                best_rate = rate
                best_action = a
        return best_action

    def update(self, prev_action: int, curr_action: int, reward: float):
        """Update bigram counts for (prev_action, curr_action)."""
        key = (prev_action, curr_action)
        succ, total = self._bigrams[key]
        total += 1
        if reward > 0:
            succ += 1
        self._bigrams[key] = (succ, total)

    def get_sequence_boost(self, prev_action: int, candidate_action: int) -> float:
        """Small boost for actions that historically follow well.

        Returns a value in [0, 0.15] -- positive only when the bigram
        success rate exceeds chance (0.5).
        """
        succ, total = self._bigrams[(prev_action, candidate_action)]
        rate = succ / total
        return max(0, rate - 0.5) * 0.3  # Small boost, max 0.15

    def reset(self):
        """Soft reset -- keep bigrams for cross-level transfer."""
        pass

    def hard_reset(self):
        """Full reset -- clear all learned bigrams."""
        self._bigrams.clear()
        self._prev_action = None
