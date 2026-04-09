"""Tests for Phase 5 — Planner."""
from charith.causal_engine.table_model import ArcTableModel
from charith.alfa_loop.planner import Planner
from charith.full_stack.hypothesis_schema import Hypothesis, ExpectedOutcome
from tests.fixtures.mock_llm import MockOllamaReasoner


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
