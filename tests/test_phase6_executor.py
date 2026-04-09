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


class _EmptyFrameAfterNEnv:
    """
    Mock env whose step() returns a frame with an empty .frame list after
    N successful steps. Simulates a real arc_agi SDK quirk observed on ls20
    where the game stops producing grids mid-episode without setting any
    terminal state flag.
    """

    def __init__(self, initial_grid, empty_after: int = 1):
        from tests.fixtures.mock_env import MockFrame
        self._MockFrame = MockFrame
        self._grid = initial_grid.copy()
        self._empty_after = empty_after
        self._count = 0
        self._latest = self._MockFrame(self._grid.copy())

    def reset(self):
        self._count = 0
        self._latest = self._MockFrame(self._grid.copy())
        return self._latest

    def get_observation(self):
        return self._latest

    def step(self, action_id: int):
        self._count += 1
        if self._count > self._empty_after:
            bad = self._MockFrame(self._grid.copy())
            bad.frame = []  # SDK returned no grid — reproduces the crash
            self._latest = bad
            return bad, 0.0, False, {}
        self._latest = self._MockFrame(self._grid.copy())
        return self._latest, 0.0, False, {}


def test_execute_survives_step_returning_empty_frame():
    """
    When env.step() returns a frame with an empty .frame list, the
    Executor must not crash. It should stop executing, return a structured
    result with a clear reason, and leave the table+analyzer consistent.

    Regression test for the IndexError at `grid_after = obs_after.frame[0]`
    observed on real ls20 at step 151 of the re-plan loop.
    """
    grid = np.zeros((10, 10), dtype=int)
    grid[5, 5] = 12
    env = _EmptyFrameAfterNEnv(grid, empty_after=2)
    env.reset()

    perception = CoreKnowledgePerception()
    table = ArcTableModel(num_actions=8)
    analyzer = ArcErrorAnalyzer()
    executor = Executor(env, perception, table, analyzer)

    # Plan is 5 actions long but the env goes bad after 2 successful steps.
    result = executor.execute([1, 1, 1, 1, 1])

    assert result is not None
    # We took 2 good steps + 1 that hit the empty frame = 3
    assert result["actions_taken"] <= 5
    assert result["actions_taken"] >= 2
    assert "reason" in result
    assert "empty_frame" in result["reason"] or "env_empty_frame" == result["reason"]
