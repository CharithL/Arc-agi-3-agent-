"""Composable game mechanics for procedural game generation.

Each mechanic is a class with an apply() method that transforms game state
based on an action. Action mappings are RANDOMIZED per game instance to force
the meta-learner to discover mechanics through in-context learning.
"""

import numpy as np
from typing import Dict, List, Optional, Set, Tuple


class CardinalMove:
    """D-pad movement (up/down/left/right) by fixed step size.

    Action mapping is randomized per game so the agent must learn
    which action corresponds to which direction each time.
    """

    def __init__(self, step_size: int = 1, action_map: Optional[Dict[int, Tuple[int, int]]] = None,
                 rng: Optional[np.random.Generator] = None):
        if action_map is not None:
            self.action_map = dict(action_map)
        else:
            # Randomize the mapping of actions to directions
            rng = rng or np.random.default_rng()
            directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # up, down, left, right
            perm = rng.permutation(4)
            self.action_map = {i: directions[perm[i]] for i in range(4)}
        self.step_size = step_size

    def apply(self, player_pos: Tuple[int, int], action: int,
              grid: np.ndarray, walls: Set[Tuple[int, int]]) -> Tuple[int, int]:
        """Returns new (x, y) position, respecting walls and grid bounds."""
        if action not in self.action_map:
            return player_pos

        dx, dy = self.action_map[action]
        nx = player_pos[0] + dx * self.step_size
        ny = player_pos[1] + dy * self.step_size

        h, w = grid.shape
        if nx < 0 or nx >= w or ny < 0 or ny >= h:
            return player_pos
        if (nx, ny) in walls:
            return player_pos

        return (nx, ny)


class IceSliding:
    """Move in direction until hitting wall or grid edge. Like Pokemon ice puzzles."""

    def __init__(self, action_map: Optional[Dict[int, Tuple[int, int]]] = None,
                 rng: Optional[np.random.Generator] = None):
        if action_map is not None:
            self.action_map = dict(action_map)
        else:
            rng = rng or np.random.default_rng()
            directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
            perm = rng.permutation(4)
            self.action_map = {i: directions[perm[i]] for i in range(4)}

    def apply(self, player_pos: Tuple[int, int], action: int,
              grid: np.ndarray, walls: Set[Tuple[int, int]]) -> Tuple[int, int]:
        """Slide in direction, stop when hitting wall or grid edge."""
        if action not in self.action_map:
            return player_pos

        dx, dy = self.action_map[action]
        h, w = grid.shape
        x, y = player_pos

        while True:
            nx, ny = x + dx, y + dy
            if nx < 0 or nx >= w or ny < 0 or ny >= h:
                break
            if (nx, ny) in walls:
                break
            x, y = nx, ny

        return (x, y)


class MirroredMove:
    """Two objects move with mirrored directions.

    One object moves normally, the other has one axis inverted.
    """

    def __init__(self, mirror_axis: str = 'x',
                 action_map: Optional[Dict[int, Tuple[int, int]]] = None,
                 rng: Optional[np.random.Generator] = None):
        self.mirror_axis = mirror_axis
        if action_map is not None:
            self.action_map = dict(action_map)
        else:
            rng = rng or np.random.default_rng()
            directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
            perm = rng.permutation(4)
            self.action_map = {i: directions[perm[i]] for i in range(4)}

    def apply(self, positions: List[Tuple[int, int]], action: int,
              grid: np.ndarray, walls: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Returns list of new positions for all paired objects."""
        if action not in self.action_map:
            return list(positions)

        dx, dy = self.action_map[action]
        h, w = grid.shape
        results = []

        for i, (x, y) in enumerate(positions):
            if i == 0:
                # Primary object moves normally
                mdx, mdy = dx, dy
            else:
                # Mirrored objects have one axis inverted
                if self.mirror_axis == 'x':
                    mdx, mdy = -dx, dy
                else:
                    mdx, mdy = dx, -dy

            nx, ny = x + mdx, y + mdy
            if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in walls:
                results.append((nx, ny))
            else:
                results.append((x, y))

        return results


class ColorCycle:
    """Cycling through a color palette on action press."""

    def __init__(self, palette: List[int], trigger_action: int):
        self.palette = list(palette)
        self.trigger_action = trigger_action

    def apply(self, current_color: int, action: int) -> int:
        """Returns new color (same if action != trigger)."""
        if action != self.trigger_action:
            return current_color

        if current_color in self.palette:
            idx = self.palette.index(current_color)
            return self.palette[(idx + 1) % len(self.palette)]
        else:
            return self.palette[0]


class Rotation:
    """Rotate player shape by 90 degrees on action press."""

    ROTATIONS = [0, 90, 180, 270]

    def __init__(self, trigger_action: int):
        self.trigger_action = trigger_action

    def apply(self, current_rotation: int, action: int) -> int:
        """Returns new rotation (0, 90, 180, 270)."""
        if action != self.trigger_action:
            return current_rotation

        idx = self.ROTATIONS.index(current_rotation) if current_rotation in self.ROTATIONS else 0
        return self.ROTATIONS[(idx + 1) % 4]


class ColorChanger:
    """Cell on grid that changes player color when touched."""

    def __init__(self, position: Tuple[int, int], new_color: int):
        self.position = position
        self.new_color = new_color

    def check(self, player_pos: Tuple[int, int]) -> Optional[int]:
        """Returns new_color if player is on this cell, else None."""
        if player_pos == self.position:
            return self.new_color
        return None


class KeyDoor:
    """Key item that unlocks a door (wall) when collected."""

    def __init__(self, key_pos: Tuple[int, int], door_pos: Tuple[int, int]):
        self.key_pos = key_pos
        self.door_pos = door_pos
        self.collected = False

    def check(self, player_pos: Tuple[int, int], walls: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
        """If player at key_pos, remove door from walls and return updated walls."""
        if player_pos == self.key_pos and not self.collected:
            self.collected = True
            walls = set(walls)
            walls.discard(self.door_pos)
            return walls
        return walls

    def reset(self):
        """Reset collected state for game reset."""
        self.collected = False
