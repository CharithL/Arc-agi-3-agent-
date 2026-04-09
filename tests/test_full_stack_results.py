"""Tests for budgets and result dataclasses."""
from charith.full_stack.budgets import AgentBudgets
from charith.full_stack.results import AttemptResult, LevelResult


def test_budgets_default_values():
    b = AgentBudgets()
    assert b.max_actions_per_level == 150
    assert b.max_llm_calls_per_level == 8
    assert b.max_plan_length == 20
    assert b.consecutive_surprises_to_halt == 3


def test_attempt_result_construction():
    r = AttemptResult(
        completed=False, actions_taken=25, llm_calls=3,
        reason="delta_spike", phase_reached=6,
        hypotheses_generated=4, hypotheses_confirmed=2,
        hypotheses_refuted=1, expansions_triggered=[],
        final_error_summary="ok",
    )
    assert r.completed is False
    assert r.reason == "delta_spike"


def test_level_result_totals():
    a1 = AttemptResult(
        completed=True, actions_taken=20, llm_calls=3,
        reason="success", phase_reached=6,
        hypotheses_generated=3, hypotheses_confirmed=3,
        hypotheses_refuted=0, expansions_triggered=[],
        final_error_summary="ok",
    )
    level = LevelResult(
        completed=True, attempts=[a1],
        total_actions=20, total_llm_calls=3, final_table_stats={}
    )
    assert level.completed is True
    assert level.total_actions == 20
