"""
Deterministic mock for arc_agi.Environment.

Designed to exercise the full 6-phase loop without a real game.
Rules are callables: (grid, state) -> (new_grid, new_state).
The `state` dict lets rules remember things (e.g., prev_action).
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class MockFrame:
    grid: np.ndarray
    win_levels: List[int] = field(default_factory=lambda: [1])
    score: int = 0
    frame: Optional[list] = None  # populated by __post_init__

    def __post_init__(self):
        # Expose frame[0] as the grid for compatibility with arc_agi
        self.frame = [self.grid]


RuleFn = Callable[[np.ndarray, dict], Tuple[np.ndarray, dict]]


class MockArcEnv:
    """Deterministic fake environment driven by a dict of grid transforms."""

    def __init__(
        self,
        initial_grid: np.ndarray,
        rules: Dict[int, RuleFn],
        score_triggers: Optional[Dict[int, int]] = None,
        done_on_action: Optional[int] = None,
    ):
        self.initial_grid = initial_grid.copy()
        self.rules = rules
        self.score_triggers = score_triggers or {}
        self.done_on_action = done_on_action
        self.grid: np.ndarray = self.initial_grid.copy()
        self.score: int = 0
        self.done: bool = False
        self._internal_state: dict = {}

    def reset(self) -> MockFrame:
        self.grid = self.initial_grid.copy()
        self.score = 0
        self.done = False
        self._internal_state = {}
        return MockFrame(self.grid.copy(), [1], self.score)

    def get_observation(self) -> MockFrame:
        return MockFrame(self.grid.copy(), [1], self.score)

    def step(self, action_id: int) -> Tuple[MockFrame, float, bool, dict]:
        if action_id in self.rules:
            self.grid, self._internal_state = self.rules[action_id](
                self.grid, self._internal_state
            )
        if action_id in self.score_triggers:
            self.score += self.score_triggers[action_id]
        if self.done_on_action is not None and action_id == self.done_on_action:
            self.done = True
        return MockFrame(self.grid.copy(), [1], self.score), 0.0, self.done, {}


def move_obj_by(color: int, dr: int, dc: int) -> RuleFn:
    """Factory: returns a rule that moves all pixels of `color` by (dr, dc)."""
    def rule(grid: np.ndarray, state: dict) -> Tuple[np.ndarray, dict]:
        new_grid = grid.copy()
        rows, cols = np.where(grid == color)
        # Clear old positions
        for r, c in zip(rows, cols):
            new_grid[r, c] = 0
        # Set new positions (clipped to grid)
        h, w = grid.shape
        for r, c in zip(rows, cols):
            nr, nc = max(0, min(h - 1, r + dr)), max(0, min(w - 1, c + dc))
            new_grid[nr, nc] = color
        return new_grid, state
    return rule
