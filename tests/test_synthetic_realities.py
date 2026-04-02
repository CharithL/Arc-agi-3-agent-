"""Tests for synthetic reality environments."""

import numpy as np
import pytest

from charith.synthetic.base_reality import SyntheticReality
from charith.synthetic.maze_reality import MazeReality, bfs_path_exists


class TestMazeResetReturnsValidGrid:
    """test_maze_reset_returns_valid_grid: grid is correct size, int dtype,
    values in valid color range."""

    @pytest.mark.parametrize("level,expected_size", [(1, 8), (2, 16), (3, 32)])
    def test_grid_size(self, level, expected_size):
        env = MazeReality(level=level, seed=42)
        obs = env.reset()
        assert obs.shape == (expected_size, expected_size), (
            f"Level {level} grid should be {expected_size}x{expected_size}, "
            f"got {obs.shape}"
        )

    @pytest.mark.parametrize("level", [1, 2, 3])
    def test_int_dtype(self, level):
        env = MazeReality(level=level, seed=42)
        obs = env.reset()
        assert np.issubdtype(obs.dtype, np.integer), (
            f"Grid dtype must be integer, got {obs.dtype}"
        )

    @pytest.mark.parametrize("level", [1, 2, 3])
    def test_color_range(self, level):
        env = MazeReality(level=level, seed=42)
        obs = env.reset()
        assert obs.min() >= 0, f"Grid values must be >= 0, got min={obs.min()}"
        assert obs.max() <= 12, f"Grid values must be <= 12, got max={obs.max()}"


class TestMazeStepReturnsCorrectFormat:
    """test_maze_step_returns_correct_format: returns (ndarray, float, bool, dict)."""

    def test_step_tuple_format(self):
        env = MazeReality(level=1, seed=42)
        env.reset()
        result = env.step(0)

        assert isinstance(result, tuple), "step() must return a tuple"
        assert len(result) == 4, "step() must return 4-tuple"

        obs, reward, done, info = result
        assert isinstance(obs, np.ndarray), "observation must be np.ndarray"
        assert isinstance(reward, float), "reward must be float"
        assert isinstance(done, bool), "done must be bool"
        assert isinstance(info, dict), "info must be dict"

    def test_step_observation_valid(self):
        env = MazeReality(level=1, seed=42)
        env.reset()
        obs, _, _, _ = env.step(1)

        assert np.issubdtype(obs.dtype, np.integer), (
            f"Step observation dtype must be integer, got {obs.dtype}"
        )
        assert obs.min() >= 0
        assert obs.max() <= 12


class TestMazeWallCollision:
    """test_maze_wall_collision: moving into wall doesn't change position."""

    def test_wall_blocks_movement(self):
        env = MazeReality(level=1, seed=42)
        env.reset()

        # Record starting ground truth position
        gt_before = env.get_ground_truth()
        row_before = gt_before["controllable_relative_row"]
        col_before = gt_before["controllable_relative_col"]

        # Find a direction that has a wall adjacent
        for action, wall_key in [(0, "wall_adjacent_up"),
                                 (1, "wall_adjacent_down"),
                                 (2, "wall_adjacent_left"),
                                 (3, "wall_adjacent_right")]:
            gt = env.get_ground_truth()
            if gt[wall_key] == 1.0:
                # This direction has a wall -- stepping should not change pos
                env.step(action)
                gt_after = env.get_ground_truth()
                assert gt_after["controllable_relative_row"] == row_before, (
                    f"Row should not change when moving into wall (action={action})"
                )
                assert gt_after["controllable_relative_col"] == col_before, (
                    f"Col should not change when moving into wall (action={action})"
                )
                return  # Test passed -- found and tested a wall

        # If no walls adjacent at start, move until we hit one
        # Walk the controllable into a corner to guarantee walls
        for _ in range(env.grid_size):
            env.step(0)  # up
        for _ in range(env.grid_size):
            env.step(2)  # left

        gt = env.get_ground_truth()
        row_corner = gt["controllable_relative_row"]
        col_corner = gt["controllable_relative_col"]

        # Now at least boundary is a wall -- try moving up or left
        env.step(0)  # up into boundary/wall
        gt_after = env.get_ground_truth()
        # Position should be unchanged or moved to another passage --
        # but boundary walls always block
        assert gt_after["wall_adjacent_up"] == 1.0 or (
            gt_after["controllable_relative_row"] == row_corner
        ), "Wall collision should prevent movement"


class TestMazeGoalReached:
    """test_maze_goal_reached: reaching goal returns reward=+1.0 and done=True."""

    def test_goal_reward_and_done(self):
        """Use BFS to find a path to goal, walk it, verify reward."""
        env = MazeReality(level=1, seed=123)
        env.reset()

        # Use BFS to find path from controllable to goal
        path = _bfs_find_path(
            env._maze,
            (env._ctrl_row, env._ctrl_col),
            (env._goal_row, env._goal_col),
        )
        assert path is not None, "BFS must find a path (maze is solvable)"

        # Walk the path
        final_reward = 0.0
        final_done = False
        for i in range(len(path) - 1):
            cr, cc = path[i]
            nr, nc = path[i + 1]
            action = _direction_to_action(cr, cc, nr, nc)
            _, reward, done, _ = env.step(action)
            if done:
                final_reward = reward
                final_done = done
                break

        assert final_done is True, "done should be True when goal reached"
        assert final_reward == 1.0, (
            f"Reward should be +1.0 on goal, got {final_reward}"
        )


class TestMazeGroundTruthKeys:
    """test_maze_ground_truth_keys: get_ground_truth has all 7 required keys."""

    REQUIRED_KEYS = {
        "controllable_relative_row",
        "controllable_relative_col",
        "distance_to_goal",
        "wall_adjacent_up",
        "wall_adjacent_down",
        "wall_adjacent_left",
        "wall_adjacent_right",
    }

    def test_all_keys_present(self):
        env = MazeReality(level=1, seed=42)
        env.reset()
        gt = env.get_ground_truth()

        assert isinstance(gt, dict), "get_ground_truth() must return a dict"
        missing = self.REQUIRED_KEYS - set(gt.keys())
        assert not missing, f"Missing ground truth keys: {missing}"

    def test_exactly_seven_keys(self):
        env = MazeReality(level=2, seed=42)
        env.reset()
        gt = env.get_ground_truth()
        assert len(gt) == 7, f"Expected 7 keys, got {len(gt)}: {list(gt.keys())}"


class TestMazeGroundTruthRanges:
    """test_maze_ground_truth_ranges: all values are 0-1 floats."""

    def test_all_values_are_floats_in_0_1(self):
        env = MazeReality(level=2, seed=42)
        env.reset()

        # Check over several steps
        for action in [0, 1, 2, 3, 0, 1]:
            env.step(action)
            gt = env.get_ground_truth()
            for key, val in gt.items():
                assert isinstance(val, float), (
                    f"gt[{key!r}] must be float, got {type(val)}"
                )
                assert 0.0 <= val <= 1.0, (
                    f"gt[{key!r}] must be in [0, 1], got {val}"
                )


class TestMazeColorRandomization:
    """test_maze_color_randomization: two resets produce different color assignments."""

    def test_colors_change_across_resets(self):
        env = MazeReality(level=1, seed=None)

        # Collect color assignments from multiple resets
        # With random seed, colors should differ across resets
        # (probability of same 4-choose-13 twice is ~1/17160, negligible)
        assignments = []
        for _ in range(5):
            env.reset()
            assignments.append(tuple(env.get_color_assignment().values()))

        # At least two different color assignments should exist
        unique = set(assignments)
        assert len(unique) >= 2, (
            f"Color assignments should vary across resets, "
            f"but got {len(unique)} unique out of 5: {assignments}"
        )

    def test_all_four_colors_distinct(self):
        """Each reset must use 4 distinct colors for wall/path/ctrl/goal."""
        env = MazeReality(level=1, seed=42)
        for _ in range(10):
            env.reset()
            ca = env.get_color_assignment()
            colors = list(ca.values())
            assert len(set(colors)) == 4, (
                f"Must use 4 distinct colors, got {colors}"
            )


class TestMazeSolvable:
    """test_maze_solvable: there exists a path from controllable to goal (BFS)."""

    @pytest.mark.parametrize("level", [1, 2, 3])
    def test_solvable_with_multiple_seeds(self, level):
        for seed in range(20):
            env = MazeReality(level=level, seed=seed)
            env.reset()

            path_exists = bfs_path_exists(
                env._maze,
                (env._ctrl_row, env._ctrl_col),
                (env._goal_row, env._goal_col),
            )
            assert path_exists, (
                f"Maze must be solvable: level={level}, seed={seed}. "
                f"Start=({env._ctrl_row},{env._ctrl_col}), "
                f"Goal=({env._goal_row},{env._goal_col})"
            )


# ------------------------------------------------------------------
# Helper functions for tests
# ------------------------------------------------------------------

def _bfs_find_path(maze, start, goal):
    """BFS to find a path from start to goal. Returns list of (row, col) or None."""
    if not maze[start[0], start[1]] or not maze[goal[0], goal[1]]:
        return None

    rows, cols = maze.shape
    visited = np.zeros_like(maze, dtype=bool)
    parent = {}
    from collections import deque
    queue = deque([start])
    visited[start[0], start[1]] = True

    while queue:
        r, c = queue.popleft()
        if (r, c) == goal:
            # Reconstruct path
            path = []
            node = goal
            while node != start:
                path.append(node)
                node = parent[node]
            path.append(start)
            path.reverse()
            return path

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols and
                    maze[nr, nc] and not visited[nr, nc]):
                visited[nr, nc] = True
                parent[(nr, nc)] = (r, c)
                queue.append((nr, nc))

    return None


def _direction_to_action(cr, cc, nr, nc):
    """Convert a (current_row, current_col) -> (next_row, next_col) move to action."""
    dr = nr - cr
    dc = nc - cc
    if dr == -1 and dc == 0:
        return 0  # up
    elif dr == 1 and dc == 0:
        return 1  # down
    elif dr == 0 and dc == -1:
        return 2  # left
    elif dr == 0 and dc == 1:
        return 3  # right
    else:
        raise ValueError(f"Invalid move: ({cr},{cc}) -> ({nr},{nc})")
