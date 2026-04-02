"""Tests for goal discovery with discriminating hypothesis predictions (Amendment 3)."""

import numpy as np
import pytest

from charith.perception.core_knowledge import CoreKnowledgePerception
from charith.metacognition.goal_discovery import (
    GoalHypothesis,
    ReduceColorsHypothesis,
    CreateSymmetryHypothesis,
    MoveToTargetHypothesis,
    MatchTemplateHypothesis,
    SortObjectsHypothesis,
    GoalDiscovery,
)


# ---------------------------------------------------------------------------
# Helper: build percepts from grids via the real perception pipeline
# ---------------------------------------------------------------------------

def _percept(grid: np.ndarray):
    """Return a StructuredPercept for the given grid."""
    ckp = CoreKnowledgePerception()
    return ckp.perceive(grid)


# ---------------------------------------------------------------------------
# ReduceColorsHypothesis
# ---------------------------------------------------------------------------

class TestReduceColorsHypothesis:

    def test_reduce_colors_hypothesis_predicts_positive(self):
        """Fewer unique colors in curr -> predicted reward > 0."""
        h = ReduceColorsHypothesis(tick=0)
        prev = np.array([[1, 2, 3], [4, 5, 6]], dtype=int)  # 6 colors
        curr = np.array([[1, 1, 1], [2, 2, 2]], dtype=int)  # 2 colors
        reward = h.predict_reward(prev, curr, None, None)
        assert reward > 0, f"Expected positive reward, got {reward}"

    def test_reduce_colors_hypothesis_predicts_negative(self):
        """More unique colors in curr -> predicted reward < 0."""
        h = ReduceColorsHypothesis(tick=0)
        prev = np.array([[1, 1, 1], [1, 1, 1]], dtype=int)  # 1 color
        curr = np.array([[1, 2, 3], [4, 5, 6]], dtype=int)  # 6 colors
        reward = h.predict_reward(prev, curr, None, None)
        assert reward < 0, f"Expected negative reward, got {reward}"

    def test_reduce_colors_same_count(self):
        """Same number of unique colors -> predicted reward == 0."""
        h = ReduceColorsHypothesis(tick=0)
        prev = np.array([[1, 2], [3, 0]], dtype=int)  # 4 colors
        curr = np.array([[4, 5], [6, 7]], dtype=int)  # 4 colors
        reward = h.predict_reward(prev, curr, None, None)
        assert reward == 0.0


# ---------------------------------------------------------------------------
# CreateSymmetryHypothesis
# ---------------------------------------------------------------------------

class TestCreateSymmetryHypothesis:

    def test_create_symmetry_hypothesis_symmetric_grid(self):
        """Grid that is fully symmetric should get >= 0 reward when
        transitioning from a non-symmetric grid."""
        h = CreateSymmetryHypothesis(tick=0)

        # Non-symmetric prev
        prev = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=int)
        # Fully symmetric curr (h, v, rot180 all True)
        curr = np.array([[1, 2, 1], [3, 4, 3], [1, 2, 1]], dtype=int)

        percept_prev = _percept(prev)
        percept_curr = _percept(curr)

        reward = h.predict_reward(prev, curr, percept_prev, percept_curr)
        assert reward >= 0, f"Expected non-negative reward for symmetric grid, got {reward}"

    def test_symmetry_decrease_predicts_negative(self):
        """Moving from symmetric to non-symmetric -> negative reward."""
        h = CreateSymmetryHypothesis(tick=0)

        # Symmetric prev
        prev = np.array([[1, 2, 1], [3, 4, 3], [1, 2, 1]], dtype=int)
        # Non-symmetric curr
        curr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=int)

        percept_prev = _percept(prev)
        percept_curr = _percept(curr)

        reward = h.predict_reward(prev, curr, percept_prev, percept_curr)
        assert reward < 0, f"Expected negative reward for decreased symmetry, got {reward}"


# ---------------------------------------------------------------------------
# Hypothesis accuracy tracking
# ---------------------------------------------------------------------------

class TestHypothesisAccuracyTracking:

    def test_hypothesis_accuracy_tracking(self):
        """Correct prediction -> confidence increases; incorrect -> tracked."""
        h = ReduceColorsHypothesis(tick=0)

        # Correct prediction: predicted 0.5, actual 0.5 => |diff| < 0.3
        h.update_accuracy(predicted_reward=0.5, actual_reward=0.5)
        assert h.prediction_total == 1
        assert h.prediction_correct == 1
        assert h.confidence == 1.0

        # Incorrect prediction: predicted 0.5, actual -1.0 => |diff| = 1.5
        h.update_accuracy(predicted_reward=0.5, actual_reward=-1.0)
        assert h.prediction_total == 2
        assert h.prediction_correct == 1
        assert h.confidence == 0.5

    def test_initial_confidence_zero(self):
        """Before any predictions, confidence should be 0."""
        h = ReduceColorsHypothesis(tick=0)
        assert h.confidence == 0.0
        assert h.prediction_total == 0


# ---------------------------------------------------------------------------
# GoalDiscovery -- external reward
# ---------------------------------------------------------------------------

class TestGoalDiscoveryExternalReward:

    def test_goal_discovery_external_reward(self):
        """Score going up -> positive reward returned by update."""
        gd = GoalDiscovery()
        state = np.array([[0, 0], [0, 0]], dtype=int)

        # First call: establish baseline score
        r1 = gd.update(state, "noop", score=0.0)

        # Second call: score increased
        r2 = gd.update(state, "noop", score=1.0)
        assert r2 > 0, f"Expected positive reward for score increase, got {r2}"

    def test_level_complete_reward(self):
        """Level completion should add +1.0 to reward."""
        gd = GoalDiscovery()
        state = np.array([[0, 0], [0, 0]], dtype=int)

        reward = gd.update(state, "noop", score=0.0, level_complete=True)
        assert reward >= 1.0, f"Expected reward >= 1.0 for level_complete, got {reward}"


# ---------------------------------------------------------------------------
# GoalDiscovery -- hypothesis generation
# ---------------------------------------------------------------------------

class TestGoalDiscoveryHypothesisGeneration:

    def test_goal_discovery_generates_hypotheses(self):
        """After 25 ticks, hypotheses should exist."""
        gd = GoalDiscovery()
        state = np.array([[1, 2], [3, 4]], dtype=int)

        for i in range(25):
            gd.update(state, "noop", score=0.0)

        assert len(gd._hypotheses) > 0, "Expected hypotheses after 25 ticks"
        # Should have all 5 hypothesis types
        assert len(gd._hypotheses) == 5

    def test_hypotheses_not_generated_before_tick_20(self):
        """Before tick 20, no hypotheses should exist."""
        gd = GoalDiscovery()
        state = np.array([[1, 2], [3, 4]], dtype=int)

        for i in range(19):
            gd.update(state, "noop", score=0.0)

        assert len(gd._hypotheses) == 0


# ---------------------------------------------------------------------------
# GoalDiscovery -- best hypothesis selection
# ---------------------------------------------------------------------------

class TestBestHypothesisSelection:

    def test_best_hypothesis_selection(self):
        """get_best_hypothesis should return the hypothesis with highest confidence."""
        gd = GoalDiscovery()

        # Manually inject hypotheses with known confidences
        h1 = ReduceColorsHypothesis(tick=0)
        h1.confidence = 0.3

        h2 = CreateSymmetryHypothesis(tick=0)
        h2.confidence = 0.8

        h3 = MoveToTargetHypothesis(tick=0)
        h3.confidence = 0.5

        gd._hypotheses = [h1, h2, h3]

        best = gd.get_best_hypothesis()
        assert best is h2, f"Expected CreateSymmetryHypothesis (0.8), got {best.description}"

    def test_best_hypothesis_none_when_empty(self):
        """get_best_hypothesis returns None when no hypotheses exist."""
        gd = GoalDiscovery()
        assert gd.get_best_hypothesis() is None


# ---------------------------------------------------------------------------
# GoalDiscovery -- reset behaviour
# ---------------------------------------------------------------------------

class TestGoalDiscoveryReset:

    def test_reset_keeps_hypotheses(self):
        """reset() clears histories but keeps hypotheses."""
        gd = GoalDiscovery()
        state = np.array([[1, 2], [3, 4]], dtype=int)

        # Generate hypotheses
        for i in range(25):
            gd.update(state, "noop", score=0.0)
        assert len(gd._hypotheses) == 5

        gd.reset()
        assert len(gd._hypotheses) == 5
        assert len(gd._state_history) == 0
        assert len(gd._score_history) == 0

    def test_hard_reset_clears_everything(self):
        """hard_reset() clears hypotheses as well."""
        gd = GoalDiscovery()
        state = np.array([[1, 2], [3, 4]], dtype=int)

        for i in range(25):
            gd.update(state, "noop", score=0.0)
        assert len(gd._hypotheses) == 5

        gd.hard_reset()
        assert len(gd._hypotheses) == 0
        assert len(gd._state_history) == 0


# ---------------------------------------------------------------------------
# MoveToTargetHypothesis -- placeholder
# ---------------------------------------------------------------------------

class TestMoveToTargetHypothesis:

    def test_move_to_target_returns_zero(self):
        """Placeholder hypothesis always returns 0.0."""
        h = MoveToTargetHypothesis(tick=0)
        prev = np.array([[1, 2], [3, 4]], dtype=int)
        curr = np.array([[0, 0], [0, 0]], dtype=int)
        assert h.predict_reward(prev, curr, None, None) == 0.0


# ---------------------------------------------------------------------------
# SortObjectsHypothesis
# ---------------------------------------------------------------------------

class TestSortObjectsHypothesis:

    def test_sort_objects_more_ordered(self):
        """Objects becoming more sorted by color+position -> positive reward."""
        h = SortObjectsHypothesis(tick=0)

        # prev: objects in random order (colors: 3, 1, 2 left-to-right)
        prev = np.array([
            [3, 0, 1, 0, 2],
        ], dtype=int)
        # curr: objects in sorted order (colors: 1, 2, 3 left-to-right)
        curr = np.array([
            [1, 0, 2, 0, 3],
        ], dtype=int)

        percept_prev = _percept(prev)
        percept_curr = _percept(curr)

        reward = h.predict_reward(prev, curr, percept_prev, percept_curr)
        assert reward >= 0, f"Expected non-negative reward for more sorted objects, got {reward}"
