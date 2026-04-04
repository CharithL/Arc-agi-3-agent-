"""Solvability validator for procedural games.

Uses BFS through the game's state space to confirm a solution exists
within the step budget.
"""

from collections import deque
from typing import FrozenSet, Optional, Tuple

from .mechanics import CardinalMove, IceSliding, MirroredMove, ColorCycle, Rotation, ColorChanger, KeyDoor
from .win_conditions import ReachPosition, MatchState, CollectAll


def validate_solvable(game, max_search_depth: int = 100) -> bool:
    """BFS/DFS to confirm the game is solvable within max_steps.

    For position-based games: BFS from start to target.
    For state-match games: BFS through (position, color, rotation) state space.
    For collect-all games: BFS through (position, visited_set) state space.

    Returns True if any solution exists within budget.
    """
    spec = game.spec
    win = spec.win_condition
    max_steps = min(spec.max_steps, max_search_depth)

    if isinstance(win, ReachPosition):
        return _validate_reach(spec, max_steps)
    elif isinstance(win, MatchState):
        return _validate_match(spec, max_steps)
    elif isinstance(win, CollectAll):
        return _validate_collect(spec, max_steps)
    else:
        # Unknown win condition type -- assume solvable
        return True


def _validate_reach(spec, max_steps: int) -> bool:
    """BFS for reach-position games."""
    start = spec.player_start
    target = spec.win_condition.target_pos

    if start == target:
        return True

    visited = {start}
    queue = deque([(start, 0)])

    while queue:
        pos, steps = queue.popleft()
        if steps >= max_steps:
            continue

        for action in range(spec.n_actions):
            new_pos = _apply_movement(spec, pos, action, spec.walls)
            if new_pos == target:
                return True
            if new_pos not in visited:
                visited.add(new_pos)
                queue.append((new_pos, steps + 1))

    return False


def _validate_match(spec, max_steps: int) -> bool:
    """BFS for state-match games through (pos, color, rotation) space."""
    start_state = (
        spec.player_start,
        spec.player_color,
        spec.player_rotation,
    )

    target_color = spec.win_condition.target_color
    target_rotation = spec.win_condition.target_rotation

    if spec.player_color == target_color and spec.player_rotation == target_rotation:
        return True

    visited = {start_state}
    queue = deque([(start_state, 0)])

    while queue:
        (pos, color, rotation), steps = queue.popleft()
        if steps >= max_steps:
            continue

        for action in range(spec.n_actions):
            new_pos = _apply_movement(spec, pos, action, spec.walls)
            new_color = color
            new_rotation = rotation

            # Apply transformation mechanics
            for mech in spec.mechanics:
                if isinstance(mech, ColorCycle):
                    new_color = mech.apply(new_color, action)
                elif isinstance(mech, Rotation):
                    new_rotation = mech.apply(new_rotation, action)
                elif isinstance(mech, ColorChanger):
                    result = mech.check(new_pos)
                    if result is not None:
                        new_color = result

            if new_color == target_color and new_rotation == target_rotation:
                return True

            new_state = (new_pos, new_color, new_rotation)
            if new_state not in visited:
                visited.add(new_state)
                queue.append((new_state, steps + 1))

    return False


def _validate_collect(spec, max_steps: int) -> bool:
    """BFS for collect-all games through (pos, visited_frozenset) space."""
    targets = spec.win_condition.positions
    start_visited = frozenset({spec.player_start} & targets)

    if targets.issubset(start_visited):
        return True

    # State: (position, collected_set)
    start_state = (spec.player_start, start_visited)
    visited_states = {start_state}
    queue = deque([(start_state, 0, set(spec.walls))])

    while queue:
        (pos, collected), steps, walls = queue.popleft()
        if steps >= max_steps:
            continue

        for action in range(spec.n_actions):
            current_walls = set(walls)

            # Check key-door mechanics
            for mech in spec.mechanics:
                if isinstance(mech, KeyDoor):
                    if pos == mech.key_pos and not mech.collected:
                        current_walls.discard(mech.door_pos)

            new_pos = _apply_movement(spec, pos, action, current_walls)
            new_collected = collected | ({new_pos} & targets)

            if targets.issubset(new_collected):
                return True

            new_state = (new_pos, new_collected)
            if new_state not in visited_states:
                visited_states.add(new_state)
                queue.append((new_state, steps + 1, current_walls))

    return False


def _apply_movement(spec, pos: Tuple[int, int], action: int,
                    walls) -> Tuple[int, int]:
    """Apply movement mechanics from spec to get new position."""
    new_pos = pos
    for mech in spec.mechanics:
        if isinstance(mech, (CardinalMove, IceSliding)):
            new_pos = mech.apply(new_pos, action, spec.grid, walls)
            break  # Only apply first movement mechanic
        elif isinstance(mech, MirroredMove):
            positions = mech.apply([new_pos], action, spec.grid, walls)
            new_pos = positions[0]
            break
    return new_pos
