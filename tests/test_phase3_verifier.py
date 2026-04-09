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
