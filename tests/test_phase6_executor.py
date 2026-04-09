"""Tests for Phase 6 — Executor."""
import numpy as np
from charith.perception.core_knowledge import CoreKnowledgePerception
from charith.causal_engine.table_model import ArcTableModel
from charith.causal_engine.error_analyzer import ArcErrorAnalyzer
from charith.alfa_loop.executor import Executor
from tests.fixtures.mock_env import MockArcEnv, move_obj_by


def _env():
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


def test_execute_happy_path_completes_plan():
    env = _env()
    env.reset()
    perception = CoreKnowledgePerception()
    table = ArcTableModel(num_actions=8)
    analyzer = ArcErrorAnalyzer()
    for _ in range(3):
        table.record(action=1, changes=["direction=up,mag=1"])
    executor = Executor(env, perception, table, analyzer)
    result = executor.execute([1, 1, 1])
    assert result["actions_taken"] == 3
    assert result["completed"] is False


def test_execute_returns_on_done():
    env = _env()
    env.done_on_action = 2
    env.reset()
    perception = CoreKnowledgePerception()
    table = ArcTableModel(num_actions=8)
    analyzer = ArcErrorAnalyzer()
    executor = Executor(env, perception, table, analyzer)
    result = executor.execute([1, 2, 3])
    assert result["completed"] is True
    assert result["actions_taken"] == 2
