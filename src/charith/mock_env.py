"""MockEnvironment: Test harness simulating ARC-AGI-3 grid games.

Provides mock environments so all agent modules can be tested without the
real SDK. Each environment models a different challenge:
  - DeterministicMovementEnv: deterministic transition rules
  - HiddenGoalEnv: goal discovery
  - ContextDependentEnv: context-dependent action semantics
  - MultiLevelEnv: cross-level transfer

Grid colors: 0=black, 1=blue, 2=red, 3=green, 5=grey, 8=cyan
"""

from __future__ import annotations

import numpy as np
from typing import Any


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class MockEnvironment:
    """Base class for mock ARC-AGI-3 environments.

    All observations are np.ndarray with integer dtype and values in [0, 9].
    All step results are dicts with keys: score, level_complete, game_over.
    """

    def __init__(self, rows: int = 10, cols: int = 10) -> None:
        self._grid: np.ndarray = np.zeros((rows, cols), dtype=np.int64)
        self._step_count: int = 0
        self._max_steps: int = 1000
        self._game_over: bool = False
        self._arcade: MockArcade | None = None  # set by MockArcade.make

    def get_observation(self) -> np.ndarray:
        """Return a copy of the current grid as an int numpy array."""
        return self._grid.copy()

    def step(self, action: int) -> dict[str, Any]:
        """Execute *action* and return {score, level_complete, game_over}."""
        raise NotImplementedError

    def _make_result(
        self,
        score: float = 0.0,
        level_complete: bool = False,
        game_over: bool = False,
    ) -> dict[str, Any]:
        """Build a canonical step-result dict."""
        self._step_count += 1
        if self._arcade is not None:
            self._arcade._total_actions += 1
        if self._step_count >= self._max_steps:
            game_over = True
        self._game_over = game_over
        return {
            "score": float(score),
            "level_complete": bool(level_complete),
            "game_over": bool(game_over),
        }


# ---------------------------------------------------------------------------
# DeterministicMovementEnv
# ---------------------------------------------------------------------------

class DeterministicMovementEnv(MockEnvironment):
    """10x10 grid with a 1x1 blue block that moves deterministically.

    Actions: 0=up, 1=down, 2=left, 3=right.
    Wall collision keeps the block in place.
    Score is always 0.0; game_over after 1000 steps.
    """

    def __init__(self) -> None:
        super().__init__(10, 10)
        self._max_steps = 1000
        self._block_row: int = 5
        self._block_col: int = 5
        self._update_grid()

    def _update_grid(self) -> None:
        self._grid[:] = 0
        self._grid[self._block_row, self._block_col] = 1

    def step(self, action: int) -> dict[str, Any]:
        new_row, new_col = self._block_row, self._block_col
        if action == 0:    # up
            new_row -= 1
        elif action == 1:  # down
            new_row += 1
        elif action == 2:  # left
            new_col -= 1
        elif action == 3:  # right
            new_col += 1

        # Wall collision check
        if 0 <= new_row < 10 and 0 <= new_col < 10:
            self._block_row = new_row
            self._block_col = new_col

        self._update_grid()
        return self._make_result(score=0.0, level_complete=False)


# ---------------------------------------------------------------------------
# HiddenGoalEnv
# ---------------------------------------------------------------------------

class HiddenGoalEnv(MockEnvironment):
    """10x10 grid with a blue block and a hidden green target.

    Block at (5,5), target at (2,8).  Same movement as DeterministicMovement.
    When block reaches the target: score=1.0, level_complete=True, and the
    target resets.  game_over after 3 levels completed or 1000 steps.
    """

    def __init__(self) -> None:
        super().__init__(10, 10)
        self._max_steps = 1000
        self._block_row: int = 5
        self._block_col: int = 5
        self._levels_completed: int = 0
        self._max_levels: int = 3
        # Target positions for each level
        self._target_positions = [(2, 8), (7, 1), (0, 9)]
        self._target_row, self._target_col = self._target_positions[0]
        self._update_grid()

    def _update_grid(self) -> None:
        self._grid[:] = 0
        self._grid[self._target_row, self._target_col] = 3  # green target
        self._grid[self._block_row, self._block_col] = 1    # blue block (overrides target if overlapping)

    def step(self, action: int) -> dict[str, Any]:
        new_row, new_col = self._block_row, self._block_col
        if action == 0:    # up
            new_row -= 1
        elif action == 1:  # down
            new_row += 1
        elif action == 2:  # left
            new_col -= 1
        elif action == 3:  # right
            new_col += 1

        if 0 <= new_row < 10 and 0 <= new_col < 10:
            self._block_row = new_row
            self._block_col = new_col

        # Check goal
        score = 0.0
        level_complete = False
        game_over = False
        if self._block_row == self._target_row and self._block_col == self._target_col:
            score = 1.0
            level_complete = True
            self._levels_completed += 1
            if self._levels_completed >= self._max_levels:
                game_over = True
            else:
                # Reset block and set next target
                self._block_row = 5
                self._block_col = 5
                self._target_row, self._target_col = self._target_positions[
                    self._levels_completed
                ]

        self._update_grid()
        return self._make_result(
            score=score, level_complete=level_complete, game_over=game_over
        )


# ---------------------------------------------------------------------------
# ContextDependentEnv
# ---------------------------------------------------------------------------

class ContextDependentEnv(MockEnvironment):
    """10x10 grid where action semantics depend on background color.

    Blue block (1) at (5,5).
    Background alternates: white (0) for first 20 steps, grey (5) for next 20.
    On WHITE: action 0 moves block RIGHT (+1 col).
    On GREY:  action 0 moves block LEFT  (-1 col).
    Actions 1-3 always map to down/left/right regardless of background.
    Score always 0.0; game_over after 500 steps.
    """

    def __init__(self) -> None:
        super().__init__(10, 10)
        self._max_steps = 500
        self._block_row: int = 5
        self._block_col: int = 5
        self._update_grid()

    @property
    def _background_color(self) -> int:
        """Background is white (0) for steps 0-19, grey (5) for 20-39, etc."""
        cycle_pos = self._step_count % 40
        return 0 if cycle_pos < 20 else 5

    def _update_grid(self) -> None:
        bg = self._background_color
        self._grid[:] = bg
        self._grid[self._block_row, self._block_col] = 1

    def step(self, action: int) -> dict[str, Any]:
        new_row, new_col = self._block_row, self._block_col
        bg = self._background_color

        if action == 0:
            # Context-dependent
            if bg == 0:   # white background -> right
                new_col += 1
            else:         # grey background -> left
                new_col -= 1
        elif action == 1:  # always down
            new_row += 1
        elif action == 2:  # always left
            new_col -= 1
        elif action == 3:  # always right
            new_col += 1

        if 0 <= new_row < 10 and 0 <= new_col < 10:
            self._block_row = new_row
            self._block_col = new_col

        result = self._make_result(score=0.0, level_complete=False)
        self._update_grid()
        return result


# ---------------------------------------------------------------------------
# MultiLevelEnv
# ---------------------------------------------------------------------------

class MultiLevelEnv(MockEnvironment):
    """Multi-level environment testing cross-level transfer.

    Level 1: blue block (1) at (5,5), target (3) at (2,8).
    Level 2: blue (1) at (5,5) + red (2) at (7,7), target at (1,1).
             Only blue moves.
    Level 3: blue (1) at (5,5) + red (2) at (7,7) + cyan wall (8) across row 3.
             Target at (1,5).  Blue must navigate around the wall.
    game_over after all 3 levels or 2000 total steps.
    """

    def __init__(self) -> None:
        super().__init__(10, 10)
        self._max_steps = 2000
        self._current_level: int = 1
        self._levels_completed: int = 0
        self._setup_level(1)

    def _setup_level(self, level: int) -> None:
        self._current_level = level
        self._block_row = 5
        self._block_col = 5

        if level == 1:
            self._target_row = 2
            self._target_col = 8
            self._red_row: int | None = None
            self._red_col: int | None = None
            self._has_wall = False
        elif level == 2:
            self._target_row = 1
            self._target_col = 1
            self._red_row = 7
            self._red_col = 7
            self._has_wall = False
        elif level == 3:
            self._target_row = 1
            self._target_col = 5
            self._red_row = 7
            self._red_col = 7
            self._has_wall = True

        self._update_grid()

    def _update_grid(self) -> None:
        self._grid[:] = 0

        # Wall (cyan=8) across row 3
        if self._has_wall:
            self._grid[3, :] = 8

        # Target (green=3)
        self._grid[self._target_row, self._target_col] = 3

        # Red block if present
        if self._red_row is not None and self._red_col is not None:
            self._grid[self._red_row, self._red_col] = 2

        # Blue block on top of everything
        self._grid[self._block_row, self._block_col] = 1

    def step(self, action: int) -> dict[str, Any]:
        new_row, new_col = self._block_row, self._block_col
        if action == 0:    # up
            new_row -= 1
        elif action == 1:  # down
            new_row += 1
        elif action == 2:  # left
            new_col -= 1
        elif action == 3:  # right
            new_col += 1

        # Bounds check
        if 0 <= new_row < 10 and 0 <= new_col < 10:
            # Wall collision: cyan wall (8) blocks movement
            if not (self._has_wall and new_row == 3 and self._grid[new_row, new_col] == 8):
                self._block_row = new_row
                self._block_col = new_col

        # Check goal
        score = 0.0
        level_complete = False
        game_over = False
        if self._block_row == self._target_row and self._block_col == self._target_col:
            score = 1.0
            level_complete = True
            self._levels_completed += 1
            if self._levels_completed >= 3:
                game_over = True
            else:
                self._setup_level(self._current_level + 1)

        self._update_grid()
        return self._make_result(
            score=score, level_complete=level_complete, game_over=game_over
        )


# ---------------------------------------------------------------------------
# MockArcade
# ---------------------------------------------------------------------------

_GAME_REGISTRY: dict[str, type[MockEnvironment]] = {
    "deterministic_movement": DeterministicMovementEnv,
    "hidden_goal": HiddenGoalEnv,
    "context_dependent": ContextDependentEnv,
    "multi_level": MultiLevelEnv,
}


class MockArcade:
    """Factory that creates mock environments and tracks aggregate stats."""

    def __init__(self) -> None:
        self._total_actions: int = 0
        self._levels_completed: int = 0
        self._envs: list[MockEnvironment] = []

    def make(
        self, game_id: str, render_mode: str | None = None
    ) -> MockEnvironment:
        """Create and return the appropriate MockEnvironment subclass."""
        if game_id not in _GAME_REGISTRY:
            raise ValueError(
                f"Unknown game_id {game_id!r}. "
                f"Available: {sorted(_GAME_REGISTRY)}"
            )
        env = _GAME_REGISTRY[game_id]()
        env._arcade = self
        self._envs.append(env)
        return env

    def get_scorecard(self) -> dict[str, Any]:
        """Return aggregate statistics across all environments."""
        return {
            "total_actions": self._total_actions,
            "levels_completed": self._levels_completed,
            "games_created": len(self._envs),
        }
