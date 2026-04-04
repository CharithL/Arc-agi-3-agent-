"""Tests for the procedural game generator."""

import numpy as np
import pytest

from charith.gamegen.mechanics import (
    CardinalMove, IceSliding, MirroredMove,
    ColorCycle, Rotation, ColorChanger, KeyDoor,
)
from charith.gamegen.win_conditions import ReachPosition, MatchState, CollectAll
from charith.gamegen.grid_builder import GridBuilder
from charith.gamegen.generator import GameGenerator, ProceduralGame, GameSpec
from charith.gamegen.validator import validate_solvable


# ── Movement mechanics ───────────────────────────────────────────


class TestCardinalMove:

    def test_cardinal_move_respects_walls(self):
        """Wall blocks movement -- player stays in place."""
        grid = np.zeros((5, 5), dtype=int)
        walls = {(2, 1)}
        # Action 0 = right for this explicit map
        mech = CardinalMove(action_map={0: (1, 0), 1: (-1, 0), 2: (0, -1), 3: (0, 1)})
        # Player at (1, 1), wall at (2, 1) -- moving right should be blocked
        new_pos = mech.apply((1, 1), 0, grid, walls)
        assert new_pos == (1, 1), "Player should not move into a wall"

    def test_cardinal_move_normal(self):
        """Movement works when path is clear."""
        grid = np.zeros((5, 5), dtype=int)
        walls = set()
        mech = CardinalMove(action_map={0: (1, 0), 1: (-1, 0), 2: (0, -1), 3: (0, 1)})
        new_pos = mech.apply((1, 1), 0, grid, walls)
        assert new_pos == (2, 1)

    def test_cardinal_move_randomized_mapping(self):
        """Different seeds produce different action mappings."""
        m1 = CardinalMove(rng=np.random.default_rng(42))
        m2 = CardinalMove(rng=np.random.default_rng(999))
        # With different seeds, mappings should differ (with high probability)
        maps_equal = all(m1.action_map[i] == m2.action_map[i] for i in range(4))
        # It's extremely unlikely but possible they match; test 10 seeds
        any_different = False
        for seed in range(10):
            m = CardinalMove(rng=np.random.default_rng(seed))
            if any(m.action_map[i] != m1.action_map[i] for i in range(4)):
                any_different = True
                break
        assert any_different, "Randomized mappings should vary across seeds"

    def test_cardinal_move_bounds(self):
        """Movement respects grid boundaries."""
        grid = np.zeros((5, 5), dtype=int)
        walls = set()
        mech = CardinalMove(action_map={0: (-1, 0)})
        # At left edge, moving left should be blocked
        new_pos = mech.apply((0, 2), 0, grid, walls)
        assert new_pos == (0, 2)


class TestIceSliding:

    def test_ice_sliding_stops_at_wall(self):
        """Slides until hitting a wall."""
        grid = np.zeros((5, 5), dtype=int)
        walls = {(4, 1)}
        mech = IceSliding(action_map={0: (1, 0)})
        # Player at (1, 1), sliding right, wall at (4, 1)
        new_pos = mech.apply((1, 1), 0, grid, walls)
        assert new_pos == (3, 1), "Should slide to (3,1), one cell before wall at (4,1)"

    def test_ice_sliding_stops_at_edge(self):
        """Slides until hitting grid edge."""
        grid = np.zeros((5, 5), dtype=int)
        walls = set()
        mech = IceSliding(action_map={0: (1, 0)})
        new_pos = mech.apply((1, 1), 0, grid, walls)
        assert new_pos == (4, 1), "Should slide to right edge"


# ── Transformation mechanics ─────────────────────────────────────


class TestColorCycle:

    def test_color_cycle(self):
        """Cycles through palette correctly."""
        mech = ColorCycle(palette=[1, 3, 5], trigger_action=4)
        # Pressing trigger cycles through
        assert mech.apply(1, 4) == 3
        assert mech.apply(3, 4) == 5
        assert mech.apply(5, 4) == 1  # wraps around

    def test_color_cycle_wrong_action(self):
        """Non-trigger action does nothing."""
        mech = ColorCycle(palette=[1, 3, 5], trigger_action=4)
        assert mech.apply(1, 0) == 1
        assert mech.apply(3, 2) == 3


class TestRotation:

    def test_rotation(self):
        """Rotation cycles through 0, 90, 180, 270."""
        mech = Rotation(trigger_action=4)
        assert mech.apply(0, 4) == 90
        assert mech.apply(90, 4) == 180
        assert mech.apply(180, 4) == 270
        assert mech.apply(270, 4) == 0  # wraps


# ── Win conditions ───────────────────────────────────────────────


class TestWinConditions:

    def test_reach_position_win(self):
        """Reaching target returns True."""
        win = ReachPosition(target_pos=(3, 4))
        assert win.check(player_pos=(3, 4)) is True
        assert win.check(player_pos=(3, 3)) is False

    def test_match_state_win(self):
        """Matching color+rotation returns True."""
        win = MatchState(target_color=5, target_rotation=180)
        assert win.check(player_color=5, player_rotation=180) is True
        assert win.check(player_color=5, player_rotation=0) is False
        assert win.check(player_color=3, player_rotation=180) is False

    def test_collect_all_win(self):
        """Visiting all positions returns True."""
        win = CollectAll(positions={(1, 1), (2, 2), (3, 3)})
        assert win.check(visited={(1, 1), (2, 2)}) is False
        assert win.check(visited={(1, 1), (2, 2), (3, 3)}) is True
        assert win.check(visited={(1, 1), (2, 2), (3, 3), (4, 4)}) is True


# ── Grid builder ─────────────────────────────────────────────────


class TestGridBuilder:

    def test_grid_builder_connected(self):
        """Built grid has connected open spaces."""
        builder = GridBuilder(rng=np.random.default_rng(42))
        grid, walls, open_pos = builder.build(8, 8, wall_density=0.2, ensure_path=True)

        assert len(open_pos) > 0, "Should have open positions"
        assert builder._is_connected(open_pos, 8, 8), "Open positions should be connected"

    def test_grid_builder_dimensions(self):
        """Grid has correct dimensions."""
        builder = GridBuilder(rng=np.random.default_rng(0))
        grid, walls, open_pos = builder.build(6, 8)
        assert grid.shape == (8, 6), "Grid shape should be (height, width)"


# ── Generator ────────────────────────────────────────────────────


class TestGenerator:

    def test_generator_level_1(self):
        """Produces solvable Level 1 game."""
        gen = GameGenerator(seed=42)
        game = gen.generate(level=1)
        assert isinstance(game, ProceduralGame)
        obs = game.reset()
        assert isinstance(obs, np.ndarray)
        assert validate_solvable(game, max_search_depth=200)

    def test_generator_level_2(self):
        """Produces solvable Level 2 game."""
        gen = GameGenerator(seed=123)
        game = gen.generate(level=2)
        assert isinstance(game, ProceduralGame)
        game.reset()
        assert validate_solvable(game, max_search_depth=200)

    def test_generator_level_3(self):
        """Produces solvable Level 3 game."""
        gen = GameGenerator(seed=456)
        game = gen.generate(level=3)
        assert isinstance(game, ProceduralGame)
        game.reset()
        assert validate_solvable(game, max_search_depth=200)

    def test_generator_diversity(self):
        """10 games have different mechanic combinations."""
        mechanic_types = []
        for seed in range(10):
            gen = GameGenerator(seed=seed)
            game = gen.generate(level=self._random_level(seed))
            types = tuple(type(m).__name__ for m in game.spec.mechanics)
            mechanic_types.append(types)

        unique = set(mechanic_types)
        assert len(unique) > 1, (
            f"Expected diverse mechanic combos, got {unique}"
        )

    @staticmethod
    def _random_level(seed):
        """Helper to pick varied levels."""
        return (seed % 4) + 1


# ── Validator ────────────────────────────────────────────────────


class TestValidator:

    def test_validator_accepts_solvable(self):
        """Solvable game passes validation."""
        grid = np.zeros((5, 5), dtype=int)
        movement = CardinalMove(
            action_map={0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}
        )
        spec = GameSpec(
            grid_size=(5, 5),
            grid=grid,
            walls=set(),
            player_start=(0, 0),
            player_color=2,
            player_rotation=0,
            mechanics=[movement],
            win_condition=ReachPosition((4, 4)),
            max_steps=50,
            step_penalty=-0.01,
            goal_reward=1.0,
            goal_display={'type': 'position'},
            n_actions=4,
        )
        game = ProceduralGame(spec)
        assert validate_solvable(game, max_search_depth=50)

    def test_validator_rejects_unsolvable(self):
        """Impossible game fails validation."""
        grid = np.zeros((5, 5), dtype=int)
        # Surround target with walls so it's unreachable
        walls = {(3, 4), (4, 3), (4, 4)}
        target = (4, 4)
        movement = CardinalMove(
            action_map={0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}
        )
        spec = GameSpec(
            grid_size=(5, 5),
            grid=grid,
            walls=walls,
            player_start=(0, 0),
            player_color=2,
            player_rotation=0,
            mechanics=[movement],
            win_condition=ReachPosition(target),
            max_steps=50,
            step_penalty=-0.01,
            goal_reward=1.0,
            goal_display={'type': 'position'},
            n_actions=4,
        )
        game = ProceduralGame(spec)
        assert not validate_solvable(game, max_search_depth=50)


# ── Game interface ───────────────────────────────────────────────


class TestGameInterface:

    def test_game_step_format(self):
        """step returns (ndarray, float, bool, dict)."""
        gen = GameGenerator(seed=42)
        game = gen.generate(level=1)
        game.reset()
        obs, reward, done, info = game.step(0)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_game_ground_truth(self):
        """get_ground_truth returns expected keys."""
        gen = GameGenerator(seed=42)
        game = gen.generate(level=1)
        game.reset()
        gt = game.get_ground_truth()
        assert 'player_x' in gt
        assert 'player_y' in gt
        assert 'player_color' in gt
        assert 'player_rotation' in gt
        assert 'step_fraction' in gt
        assert 'done' in gt
        # Level 1 uses ReachPosition, so goal_distance should be present
        assert 'goal_distance' in gt
