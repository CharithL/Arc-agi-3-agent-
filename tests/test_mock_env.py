"""Smoke test for MockArcEnv — the mock must behave deterministically."""
import numpy as np
from tests.fixtures.mock_env import MockArcEnv, move_obj_by


def test_reset_returns_initial_grid():
    grid = np.zeros((10, 10), dtype=int)
    grid[4, 4] = 12
    env = MockArcEnv(initial_grid=grid, rules={})
    frame = env.reset()
    assert frame.grid[4, 4] == 12


def test_step_applies_rule():
    grid = np.zeros((10, 10), dtype=int)
    grid[5, 5] = 12
    env = MockArcEnv(
        initial_grid=grid,
        rules={1: move_obj_by(color=12, dr=-1, dc=0)},
    )
    env.reset()
    frame, _, done, _ = env.step(1)
    # Sprite at (5,5) -> (4,5)
    assert frame.grid[4, 5] == 12
    assert frame.grid[5, 5] == 0
    assert done is False


def test_step_with_no_rule_is_noop():
    grid = np.zeros((10, 10), dtype=int)
    grid[5, 5] = 12
    env = MockArcEnv(initial_grid=grid, rules={})
    env.reset()
    frame, _, done, _ = env.step(99)
    assert frame.grid[5, 5] == 12
