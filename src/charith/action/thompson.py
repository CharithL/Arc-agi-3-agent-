"""Thompson Sampling with Beta posteriors per (context, action) pair."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from charith.action.action_space import N_ACTIONS
from charith.memory.sequences import ActionSequenceMemory


@dataclass
class ActionStats:
    """Beta-distributed posterior for a single action."""

    alpha: float = 1.0
    beta: float = 1.0
    times_taken: int = 0
    total_reward: float = 0.0
    info_gain_history: List[float] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    @property
    def uncertainty(self) -> float:
        return math.sqrt(self.variance)

    def sample(self) -> float:
        """Draw a single sample from the Beta posterior."""
        return float(np.random.beta(self.alpha, self.beta))

    def update(self, reward: float, info_gain: float = 0.0) -> None:
        """Bayesian update of Beta posterior with observed reward."""
        self.times_taken += 1
        self.total_reward += reward
        # Treat reward as pseudo-count for Beta update
        self.alpha += reward
        self.beta += (1.0 - reward)
        if info_gain > 0:
            self.info_gain_history.append(info_gain)


class ThompsonSampler:
    """Thompson Sampling action selector with optional sequence memory boost.

    Maintains per-(context, action) Beta posteriors and global action
    statistics.  Supports goal-directed bias, information-gain bonuses,
    and bigram sequence boosts from ActionSequenceMemory.
    """

    def __init__(
        self,
        n_actions: int = N_ACTIONS,
        explore_threshold: float = 0.3,
        info_gain_weight: float = 0.5,
    ):
        self.n_actions = n_actions
        self.explore_threshold = explore_threshold
        self.info_gain_weight = info_gain_weight

        # context_hash -> action_id -> ActionStats
        self._stats: Dict[int, Dict[int, ActionStats]] = {}
        # action_id -> ActionStats (aggregated across all contexts)
        self._global_stats: Dict[int, ActionStats] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_stats(self, context_hash: int, action: int) -> ActionStats:
        """Lazily initialise and return stats for (context, action)."""
        if context_hash not in self._stats:
            self._stats[context_hash] = {}
        ctx = self._stats[context_hash]
        if action not in ctx:
            ctx[action] = ActionStats()
        return ctx[action]

    def _get_global(self, action: int) -> ActionStats:
        if action not in self._global_stats:
            self._global_stats[action] = ActionStats()
        return self._global_stats[action]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_action(
        self,
        context_hash: int,
        available_actions: Optional[List[int]] = None,
        goal_directed: bool = False,
        goal_action: Optional[int] = None,
        prev_action: Optional[int] = None,
        sequence_memory: Optional[ActionSequenceMemory] = None,
    ) -> int:
        """Select an action via Thompson Sampling.

        Parameters
        ----------
        context_hash : int
            Hash of the current observation / state.
        available_actions : list[int] | None
            Subset of actions to consider.  *None* means all actions.
        goal_directed : bool
            When True and *goal_action* is set, return the goal action
            with ~50 % probability.
        goal_action : int | None
            The preferred action when goal-directed.
        prev_action : int | None
            The previous action taken (for sequence boost).
        sequence_memory : ActionSequenceMemory | None
            Bigram model to provide sequence boosts.

        Returns
        -------
        int
            The selected action index.
        """
        if available_actions is None:
            available_actions = list(range(self.n_actions))

        # Goal-directed shortcut: 50 % chance of returning goal_action
        if goal_directed and goal_action is not None:
            if goal_action in available_actions and np.random.random() < 0.5:
                return int(goal_action)

        # Thompson sample each available action
        best_score = -float("inf")
        best_action = available_actions[0]

        for a in available_actions:
            stats = self._get_stats(context_hash, a)
            score = stats.sample()

            # Information-gain bonus
            if stats.info_gain_history:
                avg_ig = sum(stats.info_gain_history) / len(stats.info_gain_history)
                score += self.info_gain_weight * avg_ig

            # Sequence boost
            if prev_action is not None and sequence_memory is not None:
                score += sequence_memory.get_sequence_boost(prev_action, a)

            if score > best_score:
                best_score = score
                best_action = a

        return int(best_action)

    def update(
        self,
        context_hash: int,
        action: int,
        reward: float,
        info_gain: float = 0.0,
    ) -> None:
        """Update both context-specific and global stats."""
        self._get_stats(context_hash, action).update(reward, info_gain)
        self._get_global(action).update(reward, info_gain)

    def get_average_uncertainty(self) -> float:
        """Average uncertainty across all global action posteriors."""
        if not self._global_stats:
            # Uniform Beta(1,1) uncertainty
            return ActionStats().uncertainty
        return sum(s.uncertainty for s in self._global_stats.values()) / len(
            self._global_stats
        )

    def is_exploring(self) -> bool:
        """True when we have no data or average uncertainty exceeds threshold."""
        if not self._global_stats:
            return True
        return self.get_average_uncertainty() > self.explore_threshold

    def get_action_summary(self) -> Dict[int, Dict[str, float]]:
        """Return a summary dict of global stats per action."""
        return {
            a: {
                "mean": s.mean,
                "uncertainty": s.uncertainty,
                "times_taken": s.times_taken,
                "total_reward": s.total_reward,
            }
            for a, s in self._global_stats.items()
        }

    def reset_context(self) -> None:
        """Clear context-specific stats but keep global knowledge."""
        self._stats.clear()

    def hard_reset(self) -> None:
        """Clear everything -- full tabula rasa."""
        self._stats.clear()
        self._global_stats.clear()
