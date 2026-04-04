"""Win condition classes for procedural games.

Each win condition has a check() method that returns True when the
condition is satisfied.
"""

from typing import Set, Tuple


class ReachPosition:
    """Win by reaching a target position."""

    def __init__(self, target_pos: Tuple[int, int]):
        self.target_pos = target_pos

    def check(self, player_pos: Tuple[int, int], **state) -> bool:
        return player_pos == self.target_pos


class MatchState:
    """Win by matching target color + rotation."""

    def __init__(self, target_color: int, target_rotation: int):
        self.target_color = target_color
        self.target_rotation = target_rotation

    def check(self, player_color: int = 0, player_rotation: int = 0, **state) -> bool:
        return player_color == self.target_color and player_rotation == self.target_rotation


class CollectAll:
    """Win by visiting all marked positions."""

    def __init__(self, positions: Set[Tuple[int, int]]):
        self.positions = frozenset(positions)

    def check(self, visited: Set[Tuple[int, int]] = None, **state) -> bool:
        if visited is None:
            return False
        return self.positions.issubset(visited)
