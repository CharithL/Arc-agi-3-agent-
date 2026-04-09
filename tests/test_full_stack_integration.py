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
