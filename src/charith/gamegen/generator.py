"""Master procedural game generator.

Composes random games from mechanics, win conditions, and grid layouts.
Action effects are RANDOMIZED per game -- critical for meta-learning.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .mechanics import (
    CardinalMove, IceSliding, MirroredMove,
    ColorCycle, Rotation, ColorChanger, KeyDoor,
)
from .win_conditions import ReachPosition, MatchState, CollectAll
from .grid_builder import GridBuilder


@dataclass
class GameSpec:
    grid_size: Tuple[int, int]  # (width, height)
    grid: np.ndarray  # color values
    walls: Set[Tuple[int, int]]  # (x, y) wall positions
    player_start: Tuple[int, int]  # (x, y)
    player_color: int
    player_rotation: int  # 0, 90, 180, 270
    mechanics: List[Any]  # list of mechanic instances
    win_condition: Any  # win condition instance
    max_steps: int
    step_penalty: float
    goal_reward: float
    goal_display: Dict[str, Any]  # how to show the goal
    n_actions: int  # how many actions are available


class ProceduralGame:
    """A playable game instance created from a GameSpec.

    Follows the SyntheticReality interface:
    - reset() -> grid observation (numpy array)
    - step(action) -> (grid, reward, done, info)
    - get_ground_truth() -> dict for DESCARTES probing
    """

    def __init__(self, spec: GameSpec):
        self.spec = spec
        self._player_pos = spec.player_start
        self._player_color = spec.player_color
        self._player_rotation = spec.player_rotation
        self._walls = set(spec.walls)
        self._visited: Set[Tuple[int, int]] = set()
        self._step = 0
        self._done = False

    def reset(self) -> np.ndarray:
        """Reset and return initial grid observation."""
        self._player_pos = self.spec.player_start
        self._player_color = self.spec.player_color
        self._player_rotation = self.spec.player_rotation
        self._walls = set(self.spec.walls)
        self._visited = {self.spec.player_start}
        self._step = 0
        self._done = False

        # Reset any stateful mechanics (e.g., KeyDoor)
        for mech in self.spec.mechanics:
            if hasattr(mech, 'reset'):
                mech.reset()

        return self._render()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Returns (grid, reward, done, info)."""
        if self._done:
            return self._render(), 0.0, True, {'reason': 'already_done'}

        self._step += 1
        reward = self.spec.step_penalty

        # Apply all mechanics
        for mech in self.spec.mechanics:
            if isinstance(mech, (CardinalMove, IceSliding)):
                self._player_pos = mech.apply(
                    self._player_pos, action, self.spec.grid, self._walls
                )
            elif isinstance(mech, MirroredMove):
                # For mirrored move, we track as positions list
                # but only use the first position as player
                positions = mech.apply(
                    [self._player_pos], action, self.spec.grid, self._walls
                )
                self._player_pos = positions[0]
            elif isinstance(mech, ColorCycle):
                self._player_color = mech.apply(self._player_color, action)
            elif isinstance(mech, Rotation):
                self._player_rotation = mech.apply(self._player_rotation, action)
            elif isinstance(mech, ColorChanger):
                new_color = mech.check(self._player_pos)
                if new_color is not None:
                    self._player_color = new_color
            elif isinstance(mech, KeyDoor):
                self._walls = mech.check(self._player_pos, self._walls)

        self._visited.add(self._player_pos)

        # Check win condition
        won = self.spec.win_condition.check(
            player_pos=self._player_pos,
            player_color=self._player_color,
            player_rotation=self._player_rotation,
            visited=self._visited,
        )

        info = {
            'step': self._step,
            'player_pos': self._player_pos,
            'player_color': self._player_color,
            'player_rotation': self._player_rotation,
        }

        if won:
            reward = self.spec.goal_reward
            self._done = True
            info['reason'] = 'win'
        elif self._step >= self.spec.max_steps:
            self._done = True
            info['reason'] = 'timeout'

        return self._render(), reward, self._done, info

    def get_ground_truth(self) -> Dict[str, float]:
        """For DESCARTES probing -- position, goal distance, etc."""
        w, h = self.spec.grid_size
        px, py = self._player_pos

        truth = {
            'player_x': px / max(w - 1, 1),
            'player_y': py / max(h - 1, 1),
            'player_color': self._player_color,
            'player_rotation': self._player_rotation / 360.0,
            'step_fraction': self._step / self.spec.max_steps,
            'done': float(self._done),
        }

        # Goal distance for position-based win conditions
        if isinstance(self.spec.win_condition, ReachPosition):
            tx, ty = self.spec.win_condition.target_pos
            dist = abs(px - tx) + abs(py - ty)
            max_dist = w + h - 2
            truth['goal_distance'] = dist / max(max_dist, 1)
            truth['target_x'] = tx / max(w - 1, 1)
            truth['target_y'] = ty / max(h - 1, 1)

        if isinstance(self.spec.win_condition, CollectAll):
            total = len(self.spec.win_condition.positions)
            collected = len(self.spec.win_condition.positions & self._visited)
            truth['collect_progress'] = collected / max(total, 1)

        return truth

    def _render(self) -> np.ndarray:
        """Render current game state as a grid observation."""
        w, h = self.spec.grid_size
        grid = self.spec.grid.copy()

        # Draw walls with a distinct color
        wall_color = 10  # dark grey in ARC palette
        for wx, wy in self._walls:
            if 0 <= wy < h and 0 <= wx < w:
                grid[wy, wx] = wall_color

        # Draw player at current position with current color
        px, py = self._player_pos
        if 0 <= py < h and 0 <= px < w:
            grid[py, px] = self._player_color

        # Draw goal indicator
        if isinstance(self.spec.win_condition, ReachPosition):
            tx, ty = self.spec.win_condition.target_pos
            if 0 <= ty < h and 0 <= tx < w and (tx, ty) != self._player_pos:
                grid[ty, tx] = 4  # yellow marker for target

        if isinstance(self.spec.win_condition, CollectAll):
            for cx, cy in self.spec.win_condition.positions:
                if (cx, cy) not in self._visited and 0 <= cy < h and 0 <= cx < w:
                    grid[cy, cx] = 6  # magenta for uncollected

        return grid


class GameGenerator:
    """Generates random playable games at different complexity levels.

    Level 1: 1 movement mechanic + reach position
    Level 2: 1 movement + 1 transformation + state match
    Level 3: 1 movement + 1 transform + 1 interaction + state match
    Level 4: 2 mechanics + key-door + complex win condition
    """

    # Available colors (ARC palette 1-9, avoiding 0=background and 10=walls)
    COLORS = list(range(1, 10))

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.grid_builder = GridBuilder(rng=self.rng)

    def generate(self, level: int = 1) -> ProceduralGame:
        """Generate a random game at the given complexity level."""
        from .validator import validate_solvable

        max_attempts = 50
        for _ in range(max_attempts):
            game = self._generate_attempt(level)
            if validate_solvable(game, max_search_depth=200):
                return game

        # Fallback: return a trivially solvable game
        return self._generate_trivial()

    def _generate_attempt(self, level: int) -> ProceduralGame:
        """Single attempt at generating a game."""
        if level == 1:
            return self._gen_level_1()
        elif level == 2:
            return self._gen_level_2()
        elif level == 3:
            return self._gen_level_3()
        elif level == 4:
            return self._gen_level_4()
        else:
            raise ValueError(f"Unknown level: {level}")

    def _gen_level_1(self) -> ProceduralGame:
        """Level 1: 1 movement mechanic + reach position."""
        w = self.rng.integers(5, 10)
        h = self.rng.integers(5, 10)

        grid, walls, open_pos = self.grid_builder.build(w, h, wall_density=0.15)

        # Pick random start and target from open positions
        if len(open_pos) < 2:
            open_pos = [(0, 0), (w - 1, h - 1)]
        indices = self.rng.choice(len(open_pos), size=2, replace=False)
        start = open_pos[indices[0]]
        target = open_pos[indices[1]]

        # Randomize colors
        colors = self.rng.choice(self.COLORS, size=2, replace=False)
        player_color = int(colors[0])

        # Background color
        bg_color = int(colors[1])
        for y in range(h):
            for x in range(w):
                if (x, y) not in walls:
                    grid[y, x] = bg_color

        # Choose movement mechanic
        move_type = self.rng.choice(['cardinal', 'ice'])
        if move_type == 'cardinal':
            movement = CardinalMove(rng=self.rng)
        else:
            movement = IceSliding(rng=self.rng)

        spec = GameSpec(
            grid_size=(w, h),
            grid=grid,
            walls=walls,
            player_start=start,
            player_color=player_color,
            player_rotation=0,
            mechanics=[movement],
            win_condition=ReachPosition(target),
            max_steps=w * h * 3,
            step_penalty=-0.01,
            goal_reward=1.0,
            goal_display={'type': 'position', 'target': target},
            n_actions=4,
        )
        return ProceduralGame(spec)

    def _gen_level_2(self) -> ProceduralGame:
        """Level 2: 1 movement + 1 transformation + state match."""
        w = self.rng.integers(5, 9)
        h = self.rng.integers(5, 9)

        grid, walls, open_pos = self.grid_builder.build(w, h, wall_density=0.1)

        if len(open_pos) < 2:
            open_pos = [(0, 0), (w - 1, h - 1)]
        start = open_pos[self.rng.integers(len(open_pos))]

        # Colors for cycling
        palette = [int(c) for c in self.rng.choice(self.COLORS, size=3, replace=False)]
        player_color = palette[0]
        target_color = palette[self.rng.integers(1, len(palette))]

        bg_color = int(self.rng.choice([c for c in self.COLORS if c not in palette]))
        for y in range(h):
            for x in range(w):
                if (x, y) not in walls:
                    grid[y, x] = bg_color

        # Movement takes actions 0-3, transformation takes action 4 or 5
        movement = CardinalMove(rng=self.rng)

        # Choose transformation
        transform_type = self.rng.choice(['color', 'rotation'])
        if transform_type == 'color':
            transform = ColorCycle(palette=palette, trigger_action=4)
            target_rotation = 0
            n_actions = 5
        else:
            transform = Rotation(trigger_action=4)
            target_color = player_color
            target_rotation = int(self.rng.choice([90, 180, 270]))
            n_actions = 5

        # Win by reaching a position with the right state
        target_pos_idx = self.rng.integers(len(open_pos))
        while open_pos[target_pos_idx] == start:
            target_pos_idx = self.rng.integers(len(open_pos))
        target_pos = open_pos[target_pos_idx]

        # Combined win: reach position AND match state
        win = MatchState(target_color=target_color, target_rotation=target_rotation)

        spec = GameSpec(
            grid_size=(w, h),
            grid=grid,
            walls=walls,
            player_start=start,
            player_color=player_color,
            player_rotation=0,
            mechanics=[movement, transform],
            win_condition=win,
            max_steps=w * h * 4,
            step_penalty=-0.01,
            goal_reward=1.0,
            goal_display={'type': 'state_match', 'target_color': target_color,
                          'target_rotation': target_rotation},
            n_actions=n_actions,
        )
        return ProceduralGame(spec)

    def _gen_level_3(self) -> ProceduralGame:
        """Level 3: 1 movement + 1 transform + 1 interaction + state match."""
        w = self.rng.integers(6, 10)
        h = self.rng.integers(6, 10)

        grid, walls, open_pos = self.grid_builder.build(w, h, wall_density=0.1)

        if len(open_pos) < 4:
            grid, walls, open_pos = self.grid_builder.build(w, h, wall_density=0.0)

        indices = self.rng.choice(len(open_pos), size=min(4, len(open_pos)), replace=False)
        start = open_pos[indices[0]]

        # Colors
        palette = [int(c) for c in self.rng.choice(self.COLORS, size=4, replace=False)]
        player_color = palette[0]
        target_color = palette[1]

        bg_color = int(self.rng.choice([c for c in self.COLORS if c not in palette]))
        for y in range(h):
            for x in range(w):
                if (x, y) not in walls:
                    grid[y, x] = bg_color

        movement = CardinalMove(rng=self.rng)
        transform = ColorCycle(palette=palette[:3], trigger_action=4)

        # Place a color changer
        changer_pos = open_pos[indices[1]] if len(indices) > 1 else open_pos[0]
        interaction = ColorChanger(position=changer_pos, new_color=target_color)

        win = MatchState(target_color=target_color, target_rotation=0)

        spec = GameSpec(
            grid_size=(w, h),
            grid=grid,
            walls=walls,
            player_start=start,
            player_color=player_color,
            player_rotation=0,
            mechanics=[movement, transform, interaction],
            win_condition=win,
            max_steps=w * h * 4,
            step_penalty=-0.01,
            goal_reward=1.0,
            goal_display={'type': 'state_match', 'target_color': target_color,
                          'target_rotation': 0},
            n_actions=5,
        )
        return ProceduralGame(spec)

    def _gen_level_4(self) -> ProceduralGame:
        """Level 4: 2 mechanics + key-door + complex win condition."""
        w = self.rng.integers(7, 12)
        h = self.rng.integers(7, 12)

        grid, walls, open_pos = self.grid_builder.build(w, h, wall_density=0.15)

        if len(open_pos) < 6:
            grid, walls, open_pos = self.grid_builder.build(w, h, wall_density=0.05)

        n_needed = min(6, len(open_pos))
        indices = self.rng.choice(len(open_pos), size=n_needed, replace=False)
        start = open_pos[indices[0]]
        target = open_pos[indices[1]]

        # Key-door mechanic
        key_pos = open_pos[indices[2]] if n_needed > 2 else start
        # Place door as a wall between open areas
        door_pos = open_pos[indices[3]] if n_needed > 3 else target
        walls_set = set(walls)
        walls_set.add(door_pos)

        key_door = KeyDoor(key_pos=key_pos, door_pos=door_pos)

        # Colors
        palette = [int(c) for c in self.rng.choice(self.COLORS, size=3, replace=False)]
        player_color = palette[0]

        bg_color = int(self.rng.choice([c for c in self.COLORS if c not in palette]))
        for y in range(h):
            for x in range(w):
                if (x, y) not in walls_set:
                    grid[y, x] = bg_color

        # Two movement-related mechanics
        movement = CardinalMove(rng=self.rng)
        transform = Rotation(trigger_action=4)

        # Collect-all win condition using remaining positions
        collect_positions = set()
        for i in range(4, n_needed):
            collect_positions.add(open_pos[indices[i]])
        if not collect_positions:
            collect_positions.add(target)

        win = CollectAll(positions=collect_positions)

        spec = GameSpec(
            grid_size=(w, h),
            grid=grid,
            walls=walls_set,
            player_start=start,
            player_color=player_color,
            player_rotation=0,
            mechanics=[movement, transform, key_door],
            win_condition=win,
            max_steps=w * h * 5,
            step_penalty=-0.005,
            goal_reward=1.0,
            goal_display={'type': 'collect_all', 'positions': list(collect_positions)},
            n_actions=5,
        )
        return ProceduralGame(spec)

    def _generate_trivial(self) -> ProceduralGame:
        """Fallback: trivially solvable game."""
        w, h = 5, 5
        grid = np.ones((h, w), dtype=int)
        movement = CardinalMove(
            action_map={0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)},
            rng=self.rng
        )
        spec = GameSpec(
            grid_size=(w, h),
            grid=grid,
            walls=set(),
            player_start=(0, 0),
            player_color=2,
            player_rotation=0,
            mechanics=[movement],
            win_condition=ReachPosition((4, 4)),
            max_steps=100,
            step_penalty=-0.01,
            goal_reward=1.0,
            goal_display={'type': 'position', 'target': (4, 4)},
            n_actions=4,
        )
        return ProceduralGame(spec)
