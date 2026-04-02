"""Goal Discovery with discriminating hypothesis predictions (Amendment 3).

Each GoalHypothesis subclass makes DIFFERENT predictions about what
constitutes reward, allowing the system to discriminate between competing
goal models based on observed outcomes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from charith.perception.core_knowledge import (
    CoreKnowledgePerception,
    StructuredPercept,
)


# ---------------------------------------------------------------------------
# GoalHypothesis base class
# ---------------------------------------------------------------------------

@dataclass
class GoalHypothesis:
    """Base class for goal hypotheses.

    Each subclass must implement ``predict_reward`` with a DIFFERENT
    prediction strategy so hypotheses can be discriminated by accuracy.
    """

    description: str
    confidence: float = 0.0
    prediction_correct: int = 0
    prediction_total: int = 0
    first_proposed: int = 0

    def predict_reward(
        self,
        prev_state: np.ndarray,
        curr_state: np.ndarray,
        percept_prev: Optional[StructuredPercept],
        percept_curr: Optional[StructuredPercept],
    ) -> float:
        """Predict the reward for transitioning from *prev_state* to *curr_state*.

        Subclasses MUST override this.
        """
        raise NotImplementedError

    def update_accuracy(self, predicted_reward: float, actual_reward: float) -> None:
        """Update accuracy tracking.  A prediction is 'correct' when the
        absolute error is < 0.3."""
        self.prediction_total += 1
        if abs(predicted_reward - actual_reward) < 0.3:
            self.prediction_correct += 1
        self.confidence = self.prediction_correct / max(self.prediction_total, 1)


# ---------------------------------------------------------------------------
# Hypothesis 1: ReduceColorsHypothesis
# ---------------------------------------------------------------------------

@dataclass
class ReduceColorsHypothesis(GoalHypothesis):
    """Predicts reward based on reducing the number of unique colours.

    fewer unique colours -> +0.5
    more unique colours  -> -0.3
    same                 ->  0.0
    """

    def __init__(self, tick: int = 0) -> None:
        super().__init__(
            description="Goal: reduce number of unique colours in the grid",
            first_proposed=tick,
        )

    def predict_reward(
        self,
        prev_state: np.ndarray,
        curr_state: np.ndarray,
        percept_prev: Optional[StructuredPercept] = None,
        percept_curr: Optional[StructuredPercept] = None,
    ) -> float:
        prev_colors = len(set(int(v) for v in np.unique(prev_state)))
        curr_colors = len(set(int(v) for v in np.unique(curr_state)))
        if curr_colors < prev_colors:
            return 0.5
        elif curr_colors > prev_colors:
            return -0.3
        return 0.0


# ---------------------------------------------------------------------------
# Hypothesis 2: CreateSymmetryHypothesis
# ---------------------------------------------------------------------------

@dataclass
class CreateSymmetryHypothesis(GoalHypothesis):
    """Predicts reward based on increasing grid symmetry.

    Symmetry score = fraction of applicable symmetries that are True.
    Applicable symmetries: h_symmetric, v_symmetric, plus rot_90 if the
    grid is square.

    increased symmetry  -> +0.5
    decreased symmetry  -> -0.3
    same                ->  0.0
    """

    def __init__(self, tick: int = 0) -> None:
        super().__init__(
            description="Goal: increase grid symmetry",
            first_proposed=tick,
        )

    def predict_reward(
        self,
        prev_state: np.ndarray,
        curr_state: np.ndarray,
        percept_prev: Optional[StructuredPercept] = None,
        percept_curr: Optional[StructuredPercept] = None,
    ) -> float:
        prev_score = self._symmetry_score_from_percept(percept_prev, prev_state)
        curr_score = self._symmetry_score_from_percept(percept_curr, curr_state)
        if curr_score > prev_score:
            return 0.5
        elif curr_score < prev_score:
            return -0.3
        return 0.0

    # ------------------------------------------------------------------

    @staticmethod
    def _symmetry_score_from_percept(
        percept: Optional[StructuredPercept],
        grid: np.ndarray,
    ) -> float:
        """Compute fractional symmetry score from a percept (or grid)."""
        if percept is not None:
            sym = percept.symmetry
        else:
            # Fallback: compute directly
            from charith.perception.core_knowledge import SpatialPrior
            sym = SpatialPrior.detect_grid_symmetry(grid)

        checks = [sym.get("h_symmetric", False), sym.get("v_symmetric", False)]
        rows, cols = grid.shape
        if rows == cols:
            checks.append(sym.get("rot_90", False))

        return sum(1 for c in checks if c) / max(len(checks), 1)


# ---------------------------------------------------------------------------
# Hypothesis 3: MoveToTargetHypothesis (placeholder -- Phase 2)
# ---------------------------------------------------------------------------

@dataclass
class MoveToTargetHypothesis(GoalHypothesis):
    """Placeholder: needs object tracker data (Phase 2).

    Always returns 0.0 for now.
    """

    def __init__(self, tick: int = 0) -> None:
        super().__init__(
            description="Goal: move objects toward target positions (placeholder)",
            first_proposed=tick,
        )

    def predict_reward(
        self,
        prev_state: np.ndarray,
        curr_state: np.ndarray,
        percept_prev: Optional[StructuredPercept] = None,
        percept_curr: Optional[StructuredPercept] = None,
    ) -> float:
        return 0.0


# ---------------------------------------------------------------------------
# Hypothesis 4: MatchTemplateHypothesis
# ---------------------------------------------------------------------------

@dataclass
class MatchTemplateHypothesis(GoalHypothesis):
    """Predicts reward when the grid moves closer to a template.

    If a template is set and grid difference decreased -> +0.5,
    increased -> -0.3.  If no template is set, returns 0.0.
    """

    template: Optional[np.ndarray] = field(default=None, repr=False)

    def __init__(self, tick: int = 0) -> None:
        super().__init__(
            description="Goal: match a target template grid",
            first_proposed=tick,
        )
        self.template: Optional[np.ndarray] = None

    def set_template(self, template: np.ndarray) -> None:
        """Set the target template grid."""
        self.template = template.copy()

    def predict_reward(
        self,
        prev_state: np.ndarray,
        curr_state: np.ndarray,
        percept_prev: Optional[StructuredPercept] = None,
        percept_curr: Optional[StructuredPercept] = None,
    ) -> float:
        if self.template is None:
            return 0.0
        if prev_state.shape != self.template.shape or curr_state.shape != self.template.shape:
            return 0.0

        prev_diff = int(np.sum(prev_state != self.template))
        curr_diff = int(np.sum(curr_state != self.template))

        if curr_diff < prev_diff:
            return 0.5
        elif curr_diff > prev_diff:
            return -0.3
        return 0.0


# ---------------------------------------------------------------------------
# Hypothesis 5: SortObjectsHypothesis
# ---------------------------------------------------------------------------

@dataclass
class SortObjectsHypothesis(GoalHypothesis):
    """Predicts reward when objects become more sorted by colour + position.

    Order score = 1 - (inversions / max_inversions) where inversions are
    counted on colours sorted by centroid column position (left-to-right).

    More ordered -> +0.5, less ordered -> -0.3.
    """

    def __init__(self, tick: int = 0) -> None:
        super().__init__(
            description="Goal: sort objects by colour and position",
            first_proposed=tick,
        )

    def predict_reward(
        self,
        prev_state: np.ndarray,
        curr_state: np.ndarray,
        percept_prev: Optional[StructuredPercept] = None,
        percept_curr: Optional[StructuredPercept] = None,
    ) -> float:
        prev_score = self._order_score(percept_prev, prev_state)
        curr_score = self._order_score(percept_curr, curr_state)

        if curr_score > prev_score:
            return 0.5
        elif curr_score < prev_score:
            return -0.3
        return 0.0

    # ------------------------------------------------------------------

    @staticmethod
    def _order_score(
        percept: Optional[StructuredPercept],
        grid: np.ndarray,
    ) -> float:
        """Compute an order score based on inversion count.

        Objects are sorted by centroid column position; the score measures
        how close the resulting colour sequence is to being sorted.
        """
        if percept is not None:
            objects = percept.objects
        else:
            # Fallback: detect objects directly
            from charith.perception.core_knowledge import ObjectnessPrior, NumberPrior
            np_ = NumberPrior()
            cc = np_.count_by_color(grid)
            bg = max(cc, key=lambda c: cc[c])
            op = ObjectnessPrior()
            objects = op.detect(grid, background_color=bg)

        if len(objects) < 2:
            return 1.0  # trivially sorted

        # Sort objects by centroid column position (left-to-right)
        sorted_by_col = sorted(objects, key=lambda o: o.centroid[1])
        colors = [o.color for o in sorted_by_col]

        # Count inversions
        n = len(colors)
        inversions = 0
        for i in range(n):
            for j in range(i + 1, n):
                if colors[i] > colors[j]:
                    inversions += 1

        max_inversions = n * (n - 1) // 2
        if max_inversions == 0:
            return 1.0

        return 1.0 - (inversions / max_inversions)


# ---------------------------------------------------------------------------
# GoalDiscovery -- main orchestrator
# ---------------------------------------------------------------------------

class GoalDiscovery:
    """Discovers the goal of the current ARC task by maintaining and
    scoring discriminating hypotheses.

    Call ``update()`` each tick with the current state, action, and
    optional score / percept information.
    """

    def __init__(self) -> None:
        self._hypotheses: List[GoalHypothesis] = []
        self._score_history: List[float] = []
        self._state_history: List[np.ndarray] = []
        self._action_history: List[str] = []
        self._reward_signals: List[float] = []
        self._tick: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        state: np.ndarray,
        action: str,
        score: Optional[float] = None,
        level_complete: bool = False,
        game_over: bool = False,
        percept_prev: Optional[StructuredPercept] = None,
        percept_curr: Optional[StructuredPercept] = None,
    ) -> float:
        """Update goal model.  Returns estimated reward for this tick."""
        self._state_history.append(state.copy())
        self._action_history.append(action)
        self._tick += 1

        # ----- External reward from score changes -----
        external_reward = 0.0
        if score is not None:
            self._score_history.append(score)
            if len(self._score_history) >= 2:
                external_reward = (self._score_history[-1] - self._score_history[-2]) * 10.0
        if level_complete:
            external_reward += 1.0

        # ----- Intrinsic reward -----
        prev_state = self._state_history[-2] if len(self._state_history) >= 2 else state
        intrinsic = self._compute_intrinsic_reward(prev_state, state)

        actual_reward = external_reward + intrinsic * 0.1

        # ----- Score each hypothesis -----
        if len(self._state_history) >= 2:
            prev = self._state_history[-2]
            for h in self._hypotheses:
                predicted = h.predict_reward(prev, state, percept_prev, percept_curr)
                h.update_accuracy(predicted, actual_reward)

        # ----- Generate hypotheses at tick 20 -----
        if self._tick == 20 and not self._hypotheses:
            self._generate_initial_hypotheses()

        self._reward_signals.append(actual_reward)
        return actual_reward

    # ------------------------------------------------------------------

    def get_best_hypothesis(self) -> Optional[GoalHypothesis]:
        """Return the hypothesis with highest confidence, or None."""
        if not self._hypotheses:
            return None
        return max(self._hypotheses, key=lambda h: h.confidence)

    def get_reward_signal(self) -> float:
        """Return exponential moving average of reward signals."""
        if not self._reward_signals:
            return 0.0
        alpha = 0.3
        ema = self._reward_signals[0]
        for r in self._reward_signals[1:]:
            ema = alpha * r + (1 - alpha) * ema
        return ema

    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear state/action/score/reward histories but keep hypotheses."""
        self._state_history.clear()
        self._action_history.clear()
        self._score_history.clear()
        self._reward_signals.clear()
        self._tick = 0

    def hard_reset(self) -> None:
        """Full reset including hypotheses."""
        self.reset()
        self._hypotheses.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_intrinsic_reward(
        self,
        prev: np.ndarray,
        curr: np.ndarray,
    ) -> float:
        """Intrinsic reward: entropy reduction + symmetry increase + change bonus."""
        reward = 0.0

        # 1. Entropy reduction
        prev_entropy = self._grid_entropy(prev)
        curr_entropy = self._grid_entropy(curr)
        if curr_entropy < prev_entropy:
            reward += (prev_entropy - curr_entropy)

        # 2. Symmetry increase
        prev_sym = self._symmetry_score(prev)
        curr_sym = self._symmetry_score(curr)
        if curr_sym > prev_sym:
            reward += (curr_sym - prev_sym)

        # 3. Any-change bonus (small)
        if not np.array_equal(prev, curr):
            reward += 0.01

        return reward

    # ------------------------------------------------------------------

    @staticmethod
    def _grid_entropy(grid: np.ndarray) -> float:
        """Shannon entropy of colour distribution."""
        _, counts = np.unique(grid, return_counts=True)
        total = counts.sum()
        if total == 0:
            return 0.0
        probs = counts / total
        # Avoid log(0)
        probs = probs[probs > 0]
        return -float(np.sum(probs * np.log2(probs)))

    @staticmethod
    def _symmetry_score(grid: np.ndarray) -> float:
        """Fraction of symmetries satisfied (h, v, rot_90 if square)."""
        checks = [
            bool(np.array_equal(grid, np.flipud(grid))),
            bool(np.array_equal(grid, np.fliplr(grid))),
        ]
        rows, cols = grid.shape
        if rows == cols:
            checks.append(bool(np.array_equal(grid, np.rot90(grid))))
        return sum(1 for c in checks if c) / max(len(checks), 1)

    # ------------------------------------------------------------------

    def _generate_initial_hypotheses(self) -> None:
        """Create the initial set of discriminating hypotheses."""
        self._hypotheses = [
            ReduceColorsHypothesis(self._tick),
            CreateSymmetryHypothesis(self._tick),
            MoveToTargetHypothesis(self._tick),
            MatchTemplateHypothesis(self._tick),
            SortObjectsHypothesis(self._tick),
        ]
