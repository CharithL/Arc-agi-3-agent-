"""Tests for Phase 3 — Verifier (the critical novel phase)."""
import numpy as np
from charith.perception.core_knowledge import CoreKnowledgePerception
from charith.causal_engine.table_model import ArcTableModel
from charith.causal_engine.error_analyzer import ArcErrorAnalyzer
from charith.alfa_loop.verifier import Verifier
from charith.full_stack.hypothesis_schema import Hypothesis, ExpectedOutcome
from tests.fixtures.mock_env import MockArcEnv, move_obj_by


def _env_action1_moves_up():
    grid = np.zeros((20, 20), dtype=int)
    grid[10, 10] = 12
    return MockArcEnv(
        initial_grid=grid,
        rules={1: move_obj_by(12, -1, 0), 2: move_obj_by(12, 1, 0)},
    )


def test_verify_confirms_correct_hypothesis():
    env = _env_action1_moves_up()
    env.reset()
    perception = CoreKnowledgePerception()
    table = ArcTableModel(num_actions=8)
    analyzer = ArcErrorAnalyzer()
    v = Verifier(env, perception, table, analyzer)

    h = Hypothesis(
        rule="action 1 moves up",
        confidence=0.8,
        test_action=1,
        expected=ExpectedOutcome(direction="up", magnitude_cells=1),
    )
    verified = v.verify([h])
    assert verified[0].status == "confirmed"
    assert verified[0].match_score >= 0.70


def test_verify_refutes_wrong_direction():
    env = _env_action1_moves_up()
    env.reset()
    perception = CoreKnowledgePerception()
    table = ArcTableModel(num_actions=8)
    analyzer = ArcErrorAnalyzer()
    v = Verifier(env, perception, table, analyzer)

    h = Hypothesis(
        rule="action 1 moves DOWN",
        confidence=0.8,
        test_action=1,
        expected=ExpectedOutcome(direction="down", magnitude_cells=1),
    )
    verified = v.verify([h])
    assert verified[0].status == "ambiguous"


def test_verify_records_errors_in_analyzer():
    env = _env_action1_moves_up()
    env.reset()
    perception = CoreKnowledgePerception()
    table = ArcTableModel(num_actions=8)
    analyzer = ArcErrorAnalyzer()
    v = Verifier(env, perception, table, analyzer)

    h = Hypothesis(
        rule="action 1 moves up",
        confidence=0.8,
        test_action=1,
        expected=ExpectedOutcome(direction="up", magnitude_cells=1),
    )
    v.verify([h])
    assert len(analyzer.errors) == 1


def test_wall_collision_becomes_ambiguous_not_refuted():
    """
    Regression test for the ls20 wall-collision bug.

    Scenario: action 1 was observed moving the sprite up by 1 during
    Phase 1 exploration (table already has that evidence). Now Phase 3
    verifies the same hypothesis, but the sprite has drifted to the top
    wall, so the move_obj_by rule clips and produces 'no change'. The
    naive match-score logic would compute score=0 and mark the hypothesis
    refuted — permanently losing the 'up' rule for this attempt.

    The fix: when the expected outcome specified a direction AND
    Phase 1 exploration evidence in table.single_table[action] already
    contains a 'moved' entry, override the verdict from 'refuted' to
    'ambiguous' (match_score_raw preserved for diagnostics). This tells
    the planner 'this rule is real, just not applicable from the
    current position — try it again from a different position.'
    """
    # Sprite pinned to row 0 — action 1 'up' rule will clip to no change
    grid = np.zeros((20, 20), dtype=int)
    grid[0, 10] = 12
    env = MockArcEnv(
        initial_grid=grid,
        rules={1: move_obj_by(12, -1, 0), 2: move_obj_by(12, 1, 0)},
    )
    env.reset()

    perception = CoreKnowledgePerception()
    table = ArcTableModel(num_actions=8)
    analyzer = ArcErrorAnalyzer()

    # Seed Phase 1 evidence: the explorer already recorded that action 1
    # moved the sprite up by 1 (from some other starting row, not at the wall).
    table.record(action=1, changes=["action 1: moved up by 1 cells"])

    v = Verifier(env, perception, table, analyzer)

    h = Hypothesis(
        rule="action 1 moves up",
        confidence=0.8,
        test_action=1,
        expected=ExpectedOutcome(direction="up", magnitude_cells=1),
    )
    verified = v.verify([h])

    # The rule is real but temporarily not applicable at the top wall.
    # The wall-collision fix should override the verdict from 'refuted'
    # to 'ambiguous' so the planner can still use it after moving away.
    assert verified[0].status == "ambiguous", (
        f"expected status='ambiguous' after wall collision, "
        f"got status={verified[0].status!r} match_score={verified[0].match_score}"
    )


def test_verify_skips_untestable_hypotheses():
    env = _env_action1_moves_up()
    env.reset()
    perception = CoreKnowledgePerception()
    table = ArcTableModel(num_actions=8)
    analyzer = ArcErrorAnalyzer()
    v = Verifier(env, perception, table, analyzer)

    h = Hypothesis(
        rule="bad",
        confidence=0.5,
        test_action=99,
        expected=ExpectedOutcome(direction="up"),
    )
    h.status = "untestable"
    verified = v.verify([h])
    assert verified[0].status == "untestable"
