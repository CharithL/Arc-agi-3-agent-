"""Tests for Phase 5 — Planner."""
from charith.causal_engine.table_model import ArcTableModel
from charith.alfa_loop.planner import Planner, _coerce_action
from charith.full_stack.hypothesis_schema import Hypothesis, ExpectedOutcome
from tests.fixtures.mock_llm import MockOllamaReasoner


def test_coerce_action_accepts_int():
    assert _coerce_action(3, 8) == 3


def test_coerce_action_accepts_numeric_string():
    assert _coerce_action(" 3 ", 8) == 3


def test_coerce_action_accepts_prose_action_n():
    assert _coerce_action("action 5", 8) == 5
    assert _coerce_action("ACTION_3", 8) == 3
    assert _coerce_action("action8: move up", 8) == 8


def test_coerce_action_rejects_out_of_range():
    assert _coerce_action(99, 8) is None
    assert _coerce_action("action 99", 8) is None
    assert _coerce_action(0, 8) is None


def test_coerce_action_rejects_non_numeric():
    assert _coerce_action("foo", 8) is None
    assert _coerce_action(None, 8) is None


def test_coerce_action_digs_into_dict():
    assert _coerce_action({"action": 4}, 8) == 4


def _confirmed_hyp():
    return Hypothesis(
        rule="action 1 moves up",
        confidence=0.8,
        test_action=1,
        expected=ExpectedOutcome(direction="up"),
        status="confirmed",
    )


def test_plan_calls_llm_once_with_verified_rules():
    llm = MockOllamaReasoner(plan_response={"plan": [1, 1, 3], "reasoning": "r"})
    table = ArcTableModel(num_actions=8)
    planner = Planner(llm, table)
    plan = planner.plan([_confirmed_hyp()], goal="reach target", state_desc="test", num_actions=8)
    assert plan == [1, 1, 3]
    assert llm.call_count == 1


def test_plan_filters_invalid_actions():
    llm = MockOllamaReasoner(plan_response={"plan": [1, 99, 3], "reasoning": "r"})
    table = ArcTableModel(num_actions=8)
    planner = Planner(llm, table)
    plan = planner.plan([_confirmed_hyp()], goal="reach target", state_desc="test", num_actions=8)
    assert plan == [1, 3]


def test_plan_accepts_prose_action_strings():
    """LLM sometimes emits ['action 1', 'action 3'] instead of [1, 3]."""
    llm = MockOllamaReasoner(
        plan_response={"plan": ["action 1", "action 3", "ACTION_4"], "reasoning": "r"}
    )
    table = ArcTableModel(num_actions=8)
    planner = Planner(llm, table)
    plan = planner.plan([_confirmed_hyp()], goal="reach target", state_desc="test", num_actions=8)
    assert plan == [1, 3, 4]


def test_plan_zero_confirmed_triggers_emergency_fallback():
    llm = MockOllamaReasoner()
    table = ArcTableModel(num_actions=8)
    for _ in range(5):
        table.record(action=1, changes=["change"])
        table.record(action=2, changes=["change"])
    planner = Planner(llm, table)
    plan = planner.plan([], goal="?", state_desc="test", num_actions=8)
    assert llm.call_count == 0
    assert len(plan) > 0
    for a in plan:
        assert 1 <= a <= 8
