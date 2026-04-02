"""Maze navigation synthetic reality.

Forces: spatial reasoning, wall detection, pathfinding.
Colors are RANDOMIZED every reset() to force color-invariant representations.
Maze generated with Prim's algorithm -- guaranteed solvable.
"""
from typing import Dict, Tuple, Any, List, Set
import numpy as np
from collections import deque

from charith.synthetic.base_reality import SyntheticReality


class MazeReality(SyntheticReality):
    """Maze navigation environment.

    Grid: configurable size (8x8 to 32x32).
    Maze generated with Prim's algorithm (random but guaranteed solvable).
    Colors: RANDOMIZED every episode -- which color = wall, controllable,
    goal, path changes each reset().
    Controllable: single cell, moves 1 cell per d-pad action.
    Actions: 0=up, 1=down, 2=left, 3=right.
    Wall collision: action has no effect if wall in that direction.
    Goal: single differently-colored cell.
    Reward: +1.0 on reaching goal, -0.01 per step.

    Levels:
      L1 = 8x8 simple maze
      L2 = 16x16 maze
      L3 = 32x32 maze
    """

    # Available ARC-AGI-3 colors (0-12)
    AVAILABLE_COLORS = list(range(13))  # 0 through 12

    def __init__(self, level: int = 1, seed: int | None = None,
                 step_penalty: float = -0.01):
        # Level 0 = 8x8 (testing only), Level 1 = 16x16, Level 2 = 20x20, Level 3 = 32x32
        grid_size = {0: 8, 1: 16, 2: 20, 3: 32}.get(level, 16)
        max_steps = {0: 100, 1: 300, 2: 400, 3: 800}.get(level, 300)
        super().__init__(grid_size=grid_size, n_actions=4, max_steps=max_steps)

        self.level = level
        self._step_penalty = step_penalty
        self._rng = np.random.RandomState(seed)

        # Color assignments (randomized each reset)
        self._wall_color: int = 0
        self._path_color: int = 0
        self._controllable_color: int = 0
        self._goal_color: int = 0

        # Maze state: True = passage, False = wall
        self._maze: np.ndarray = np.zeros((grid_size, grid_size), dtype=bool)

        # Entity positions
        self._ctrl_row: int = 0
        self._ctrl_col: int = 0
        self._goal_row: int = 0
        self._goal_col: int = 0

    def reset(self) -> np.ndarray:
        """Reset: generate new maze, randomize colors, place entities."""
        self._step_count = 0
        self._done = False
        self._total_reward = 0.0

        # Randomize color assignments (pick 4 distinct colors from 0-12)
        colors = self._rng.choice(self.AVAILABLE_COLORS, size=4, replace=False)
        self._wall_color = int(colors[0])
        self._path_color = int(colors[1])
        self._controllable_color = int(colors[2])
        self._goal_color = int(colors[3])

        # Generate maze with Prim's algorithm
        self._maze = self._generate_maze_prims()

        # Collect all passage cells
        passages = list(zip(*np.where(self._maze)))

        # Place controllable at a random passage cell
        ctrl_idx = self._rng.randint(len(passages))
        self._ctrl_row, self._ctrl_col = passages[ctrl_idx]

        # Place goal at a passage cell far from controllable
        self._place_goal_far(passages)

        # Render grid
        self._update_grid()
        return self._render_grid()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Take action (0=up, 1=down, 2=left, 3=right).

        Returns (observation, reward, done, info).
        """
        if self._done:
            return self._render_grid(), 0.0, True, {"already_done": True}

        self._step_count += 1

        # Compute new position
        dr, dc = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}.get(action, (0, 0))
        new_row = self._ctrl_row + dr
        new_col = self._ctrl_col + dc

        # Wall collision: only move if in bounds AND target is a passage
        if (0 <= new_row < self.grid_size and
                0 <= new_col < self.grid_size and
                self._maze[new_row, new_col]):
            self._ctrl_row = new_row
            self._ctrl_col = new_col

        # Compute reward
        reward = self._step_penalty
        done = False
        info: Dict[str, Any] = {}

        # Check goal
        if self._ctrl_row == self._goal_row and self._ctrl_col == self._goal_col:
            reward = 1.0
            done = True
            info["goal_reached"] = True

        # Check max steps
        if self._step_count >= self.max_steps:
            done = True
            info["max_steps_reached"] = True

        self._done = done
        self._total_reward += reward

        self._update_grid()
        return self._render_grid(), reward, done, info

    def get_ground_truth(self) -> Dict[str, float]:
        """Return ground truth values for DESCARTES probing.

        Returns:
            controllable_relative_row: centroid row / grid_size (0-1)
            controllable_relative_col: centroid col / grid_size (0-1)
            distance_to_goal: euclidean distance / max_distance (0-1)
            wall_adjacent_up: 0.0 or 1.0
            wall_adjacent_down: 0.0 or 1.0
            wall_adjacent_left: 0.0 or 1.0
            wall_adjacent_right: 0.0 or 1.0
        """
        gs = self.grid_size
        max_dist = np.sqrt(2.0) * (gs - 1)

        # Euclidean distance to goal
        dist = np.sqrt(
            (self._ctrl_row - self._goal_row) ** 2 +
            (self._ctrl_col - self._goal_col) ** 2
        )

        return {
            "controllable_relative_row": self._ctrl_row / max(gs - 1, 1),
            "controllable_relative_col": self._ctrl_col / max(gs - 1, 1),
            "distance_to_goal": dist / max_dist if max_dist > 0 else 0.0,
            "wall_adjacent_up": self._check_wall(-1, 0),
            "wall_adjacent_down": self._check_wall(1, 0),
            "wall_adjacent_left": self._check_wall(0, -1),
            "wall_adjacent_right": self._check_wall(0, 1),
        }

    def get_color_assignment(self) -> Dict[str, int]:
        """Return current color assignment (useful for debugging)."""
        return {
            "wall": self._wall_color,
            "path": self._path_color,
            "controllable": self._controllable_color,
            "goal": self._goal_color,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_wall(self, dr: int, dc: int) -> float:
        """Return 1.0 if there is a wall (or boundary) in direction (dr, dc)."""
        r = self._ctrl_row + dr
        c = self._ctrl_col + dc
        if r < 0 or r >= self.grid_size or c < 0 or c >= self.grid_size:
            return 1.0  # boundary = wall
        if not self._maze[r, c]:
            return 1.0  # maze wall
        return 0.0

    def _update_grid(self) -> None:
        """Render the maze, controllable, and goal onto self._grid."""
        gs = self.grid_size
        self._grid = np.full((gs, gs), self._wall_color, dtype=int)

        # Draw passages
        self._grid[self._maze] = self._path_color

        # Draw goal
        self._grid[self._goal_row, self._goal_col] = self._goal_color

        # Draw controllable (on top of everything)
        self._grid[self._ctrl_row, self._ctrl_col] = self._controllable_color

    def _place_goal_far(self, passages: List[Tuple[int, int]]) -> None:
        """Place goal at a passage cell maximizing distance from controllable."""
        best_dist = -1.0
        best_idx = 0
        for i, (r, c) in enumerate(passages):
            if r == self._ctrl_row and c == self._ctrl_col:
                continue
            d = abs(r - self._ctrl_row) + abs(c - self._ctrl_col)
            if d > best_dist:
                best_dist = d
                best_idx = i
        self._goal_row, self._goal_col = passages[best_idx]

    def _generate_maze_prims(self) -> np.ndarray:
        """Generate maze using Prim's algorithm.

        Uses a grid where odd-indexed cells are potential passages and
        even-indexed cells are walls/borders. This ensures proper maze
        structure with walls between passages.

        Returns:
            Boolean array: True = passage, False = wall.
        """
        gs = self.grid_size
        maze = np.zeros((gs, gs), dtype=bool)

        # For Prim's, we work on a subgrid of "cells" at odd indices.
        # Walls between cells are at even indices.
        # This gives us (gs//2) cells per dimension.
        cell_rows = gs // 2
        cell_cols = gs // 2

        if cell_rows < 2 or cell_cols < 2:
            # Grid too small for proper maze -- make all passages
            maze[:] = True
            return maze

        # Pick a random starting cell (in cell coordinates)
        start_cr = self._rng.randint(cell_rows)
        start_cc = self._rng.randint(cell_cols)

        # Convert cell coords to grid coords (odd indices)
        def cell_to_grid(cr: int, cc: int) -> Tuple[int, int]:
            return 2 * cr + 1, 2 * cc + 1

        # Mark starting cell as passage
        sr, sc = cell_to_grid(start_cr, start_cc)
        maze[sr, sc] = True

        # Track which cells are in the maze (in cell coordinates)
        in_maze: Set[Tuple[int, int]] = {(start_cr, start_cc)}

        # Wall list: (wall_grid_row, wall_grid_col, cell_cr, cell_cc)
        # Each wall entry connects a cell in the maze to a cell not in the maze.
        # The wall is the grid position between them.
        walls: List[Tuple[int, int, int, int]] = []

        def add_walls(cr: int, cc: int) -> None:
            """Add frontier walls of cell (cr, cc) to the wall list."""
            for dcr, dcc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ncr, ncc = cr + dcr, cc + dcc
                if 0 <= ncr < cell_rows and 0 <= ncc < cell_cols:
                    if (ncr, ncc) not in in_maze:
                        # Wall grid position is between the two cells
                        wr = 2 * cr + 1 + dcr
                        wc = 2 * cc + 1 + dcc
                        walls.append((wr, wc, ncr, ncc))

        add_walls(start_cr, start_cc)

        # Prim's main loop
        while walls:
            idx = self._rng.randint(len(walls))
            wr, wc, ncr, ncc = walls[idx]

            # Remove this wall entry (swap with last for O(1))
            walls[idx] = walls[-1]
            walls.pop()

            if (ncr, ncc) in in_maze:
                continue  # Both sides already in maze

            # Carve: mark wall and new cell as passages
            maze[wr, wc] = True
            nr, nc = cell_to_grid(ncr, ncc)
            maze[nr, nc] = True

            in_maze.add((ncr, ncc))
            add_walls(ncr, ncc)

        return maze


def bfs_path_exists(maze: np.ndarray, start: Tuple[int, int],
                    goal: Tuple[int, int]) -> bool:
    """Check if a path exists from start to goal in the maze using BFS.

    Args:
        maze: Boolean array, True = passage.
        start: (row, col) start position.
        goal: (row, col) goal position.

    Returns:
        True if a path exists, False otherwise.
    """
    if not maze[start[0], start[1]] or not maze[goal[0], goal[1]]:
        return False

    rows, cols = maze.shape
    visited = np.zeros_like(maze, dtype=bool)
    queue = deque([start])
    visited[start[0], start[1]] = True

    while queue:
        r, c = queue.popleft()
        if (r, c) == goal:
            return True

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols and
                    maze[nr, nc] and not visited[nr, nc]):
                visited[nr, nc] = True
                queue.append((nr, nc))

    return False
