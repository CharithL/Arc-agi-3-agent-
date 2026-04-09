"""
End-to-end integration tests with MockArcEnv + MockOllamaReasoner.
The agent runs the full 6-phase loop without any real deps.
"""
import numpy as np

from charith.full_stack.charith_full_stack_agent import CharithFullStackAgent
from tests.fixtures.mock_env import MockArcEnv, move_obj_by
from tests.fixtures.mock_llm import MockOllamaReasoner


def _env_basic():
    grid = np.zeros((20, 20), dtype=int)
    grid[10, 10] = 12
    return MockArcEnv(
        initial_grid=grid,
        rules={
            1: move_obj_by(12, -1, 0),
            2: move_obj_by(12, 1, 0),
            3: move_obj_by(12, 0, -1),
            4: move_obj_by(12, 0, 1),
        },
    )


def test_full_loop_runs_without_crashing():
    env = _env_basic()
    env.reset()
    llm = MockOllamaReasoner()
    agent = CharithFullStackAgent(env, llm, num_actions=8)

    result = agent.play_level()
    assert result is not None
    assert len(result.attempts) >= 1


def test_full_loop_respects_llm_budget():
    env = _env_basic()
    env.reset()
    llm = MockOllamaReasoner()
    agent = CharithFullStackAgent(env, llm, num_actions=8)

    agent.play_level()
    assert llm.call_count <= 20
    assert llm.call_count >= 1


def test_play_game_runs_multiple_levels_without_crashing():
    env = _env_basic()
    env.reset()
    llm = MockOllamaReasoner()
    agent = CharithFullStackAgent(env, llm, num_actions=8)

    result = agent.play_game(game_id="mock_basic", max_levels=3)
    assert result.levels_attempted >= 1
    assert result.levels_attempted <= 3
    assert result.total_actions >= 0
    assert result.stopped_reason in {"max_levels_reached", "level_failed"}
    assert result.game_id == "mock_basic"


def test_replan_loop_fires_when_plan_exhausts_without_completion():
    """
    After Phase 6 executes a plan and finishes without a level transition,
    the orchestrator must re-observe the scene and call Phase 5 again,
    up to budgets.max_replans_per_attempt times within a single attempt.

    Setup: a sprite and a target that are far apart, rules that only move
    the sprite by 1 cell, and an LLM whose plan is always just [1, 1, 1]
    (never enough to reach the target in one shot). We expect the planner
    to be invoked multiple times within one attempt.
    """
    from charith.full_stack.budgets import AgentBudgets

    grid = np.zeros((20, 20), dtype=int)
    grid[2, 2] = 12   # sprite / controllable
    grid[18, 18] = 7  # target
    env = MockArcEnv(
        initial_grid=grid,
        rules={1: move_obj_by(12, 1, 0), 2: move_obj_by(12, 0, 1)},
    )
    env.reset()

    llm = MockOllamaReasoner(
        hypothesize_response={
            "hypotheses": [
                {"rule": "action 1 moves down 1", "confidence": 0.9,
                 "test_action": 1,
                 "expected": {"direction": "down", "magnitude_cells": 1,
                              "object_ref": "controllable"}},
                {"rule": "action 2 moves right 1", "confidence": 0.9,
                 "test_action": 2,
                 "expected": {"direction": "right", "magnitude_cells": 1,
                              "object_ref": "controllable"}},
            ],
            "goal_guess": "reach target",
        },
        plan_response={"plan": [1, 1, 1], "reasoning": "short plan"},
    )

    budgets = AgentBudgets(
        max_attempts_per_level=1,
        max_expansion_cycles_per_attempt=1,
        max_replans_per_attempt=3,
    )
    agent = CharithFullStackAgent(env, llm, num_actions=8, budgets=budgets)

    agent.play_level()

    # Count planning-system-prompt LLM calls. Pre-replan baseline was 1
    # per attempt. With the loop, we expect >1 (at least the initial plan
    # plus one re-plan). Use "planning" in lowercased system prompt per
    # MockOllamaReasoner's router.
    plan_calls = sum(
        1 for sys_prompt, _user in llm.calls
        if "planning" in sys_prompt.lower()
    )
    assert plan_calls >= 2, f"expected >= 2 planner calls, got {plan_calls}"


def test_full_loop_refuted_hypotheses_fall_back_to_emergency():
    env = _env_basic()
    env.reset()
    llm = MockOllamaReasoner(
        hypothesize_response={
            "hypotheses": [
                {
                    "rule": "action 1 moves DOWN by 5",
                    "confidence": 0.9,
                    "test_action": 1,
                    "expected": {"direction": "down", "magnitude_cells": 5},
                }
            ],
            "goal_guess": "unknown",
        },
        plan_response={"plan": [], "reasoning": "empty"},
    )
    agent = CharithFullStackAgent(env, llm, num_actions=8)

    result = agent.play_level()
    assert len(result.attempts) == 3
