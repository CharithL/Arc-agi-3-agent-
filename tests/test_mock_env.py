"""Tests for MockEnvironment test harness."""

import numpy as np
import pytest

from charith.mock_env import (
    MockArcade,
    MockEnvironment,
    DeterministicMovementEnv,
    HiddenGoalEnv,
    ContextDependentEnv,
    MultiLevelEnv,
)


class TestDeterministicMovement:
    """Tests for the DeterministicMovementEnv."""

    def test_deterministic_movement_action_up(self):
        """Action 0 moves block up (row decreases)."""
        env = DeterministicMovementEnv()
        obs_before = env.get_observation()
        # Block starts at (5, 5) -- find it
        assert obs_before[5, 5] == 1, "Block should start at row=5, col=5"

        env.step(0)  # up
        obs_after = env.get_observation()
        # Block should now be at (4, 5)
        assert obs_after[4, 5] == 1, "Block should move up to row=4, col=5"
        assert obs_after[5, 5] == 0, "Old position should be cleared"

    def test_deterministic_movement_action_down(self):
        """Action 1 step result has correct format."""
        env = DeterministicMovementEnv()
        result = env.step(1)  # down

        assert isinstance(result, dict), "step() must return a dict"
        assert "score" in result, "Result must have 'score' key"
        assert "level_complete" in result, "Result must have 'level_complete' key"
        assert "game_over" in result, "Result must have 'game_over' key"
        assert isinstance(result["score"], float), "score must be float"
        assert isinstance(result["level_complete"], bool), "level_complete must be bool"
        assert isinstance(result["game_over"], bool), "game_over must be bool"

    def test_deterministic_movement_wall_blocking(self):
        """Block stops at grid boundary (wall collision)."""
        env = DeterministicMovementEnv()
        # Move up 10 times from starting row 5 -- should stop at row 0
        for _ in range(10):
            env.step(0)  # up

        obs = env.get_observation()
        assert obs[0, 5] == 1, "Block should be at top row after repeated up moves"

        # One more up -- should stay at row 0
        env.step(0)
        obs = env.get_observation()
        assert obs[0, 5] == 1, "Block should stay at top row (wall collision)"


class TestHiddenGoal:
    """Tests for the HiddenGoalEnv."""

    def test_hidden_goal_no_score_until_solved(self):
        """Score is 0 initially (before reaching goal)."""
        env = HiddenGoalEnv()
        result = env.step(0)  # single step, won't reach goal at (2, 8)
        assert result["score"] == 0.0, "Score should be 0.0 before reaching goal"

    def test_hidden_goal_level_complete_on_solve(self):
        """Cycling actions eventually reaches goal, triggering level_complete."""
        env = HiddenGoalEnv()
        # Target is at (2, 8), block starts at (5, 5)
        # Move up 3 times to row 2
        for _ in range(3):
            env.step(0)  # up

        # Move right 3 times to col 8
        level_completed = False
        for _ in range(3):
            result = env.step(3)  # right
            if result["level_complete"]:
                level_completed = True
                break

        assert level_completed, "Level should complete when block reaches target"
        assert result["score"] == 1.0, "Score should be 1.0 on level completion"


class TestContextDependent:
    """Tests for the ContextDependentEnv."""

    def test_context_dependent_rules(self):
        """ContextDependentEnv runs without error for multiple steps."""
        env = ContextDependentEnv()
        obs = env.get_observation()
        assert isinstance(obs, np.ndarray), "Observation must be numpy array"
        assert obs.dtype in (np.int32, np.int64, int), "Observation dtype must be int"

        # Run 50 steps without error to cover both white and grey backgrounds
        for i in range(50):
            result = env.step(i % 4)
            assert isinstance(result, dict), f"Step {i} must return a dict"
            assert "score" in result
            assert "level_complete" in result
            assert "game_over" in result


class TestMultiLevel:
    """Tests for the MultiLevelEnv."""

    def test_multi_level_progression(self):
        """Level 2 has >= objects as level 1."""
        env = MultiLevelEnv()

        # Count non-zero cells in level 1 observation
        obs_level1 = env.get_observation()
        objects_level1 = np.count_nonzero(obs_level1)

        # Complete level 1: block at (5,5), target at (2,8)
        # Move up 3 times
        for _ in range(3):
            env.step(0)  # up
        # Move right 3 times to reach (2,8)
        for _ in range(3):
            result = env.step(3)  # right
            if result["level_complete"]:
                break

        # Now on level 2 -- count objects
        obs_level2 = env.get_observation()
        objects_level2 = np.count_nonzero(obs_level2)

        assert objects_level2 >= objects_level1, (
            f"Level 2 should have >= objects ({objects_level2}) as level 1 ({objects_level1})"
        )


class TestStepResultFormat:
    """Tests for step result format across all environments."""

    def test_step_result_format(self):
        """Result has score, level_complete, game_over with correct types."""
        envs = [
            DeterministicMovementEnv(),
            HiddenGoalEnv(),
            ContextDependentEnv(),
            MultiLevelEnv(),
        ]
        for env in envs:
            result = env.step(0)
            assert isinstance(result, dict), f"{type(env).__name__}: step() must return dict"
            assert set(result.keys()) == {"score", "level_complete", "game_over"}, (
                f"{type(env).__name__}: result keys must be exactly score, level_complete, game_over"
            )
            assert isinstance(result["score"], float), f"{type(env).__name__}: score must be float"
            assert isinstance(result["level_complete"], bool), f"{type(env).__name__}: level_complete must be bool"
            assert isinstance(result["game_over"], bool), f"{type(env).__name__}: game_over must be bool"


class TestObservationFormat:
    """Tests for observation format across all environments."""

    def test_observation_is_numpy_int_array(self):
        """All observations are np.ndarray with int dtype, values 0-9."""
        envs = [
            DeterministicMovementEnv(),
            HiddenGoalEnv(),
            ContextDependentEnv(),
            MultiLevelEnv(),
        ]
        for env in envs:
            obs = env.get_observation()
            assert isinstance(obs, np.ndarray), f"{type(env).__name__}: observation must be np.ndarray"
            assert np.issubdtype(obs.dtype, np.integer), (
                f"{type(env).__name__}: observation dtype must be integer, got {obs.dtype}"
            )
            assert obs.min() >= 0, f"{type(env).__name__}: observation values must be >= 0"
            assert obs.max() <= 9, f"{type(env).__name__}: observation values must be <= 9"


class TestScorecard:
    """Tests for MockArcade scorecard."""

    def test_scorecard(self):
        """Arcade provides scorecard with total_actions."""
        arcade = MockArcade()
        env = arcade.make("deterministic_movement")
        env.step(0)
        env.step(1)
        env.step(2)

        scorecard = arcade.get_scorecard()
        assert isinstance(scorecard, dict), "Scorecard must be a dict"
        assert "total_actions" in scorecard, "Scorecard must have total_actions"
        assert scorecard["total_actions"] == 3, "Scorecard should count 3 actions"


class TestMockArcade:
    """Tests for MockArcade factory."""

    def test_make_all_game_ids(self):
        """MockArcade.make() returns correct env subclass for each game ID."""
        arcade = MockArcade()

        env = arcade.make("deterministic_movement")
        assert isinstance(env, DeterministicMovementEnv)

        env = arcade.make("hidden_goal")
        assert isinstance(env, HiddenGoalEnv)

        env = arcade.make("context_dependent")
        assert isinstance(env, ContextDependentEnv)

        env = arcade.make("multi_level")
        assert isinstance(env, MultiLevelEnv)

    def test_make_invalid_game_id(self):
        """MockArcade.make() raises error for unknown game ID."""
        arcade = MockArcade()
        with pytest.raises((ValueError, KeyError)):
            arcade.make("nonexistent_game")
