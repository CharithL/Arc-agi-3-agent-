"""Random grid layout generator for procedural games.

Builds grids with walls and open spaces, optionally ensuring all open
positions are connected via BFS.
"""

import numpy as np
from collections import deque
from typing import List, Optional, Set, Tuple


class GridBuilder:
    """Builds random grid layouts with walls and open spaces."""

    def __init__(self, rng: Optional[np.random.Generator] = None):
        self.rng = rng or np.random.default_rng()

    def build(self, width: int, height: int, wall_density: float = 0.2,
              ensure_path: bool = True) -> Tuple[np.ndarray, Set[Tuple[int, int]], List[Tuple[int, int]]]:
        """Returns (grid, walls_set, open_positions).

        Args:
            width: Grid width.
            height: Grid height.
            wall_density: Fraction of cells that are walls (0.0 to 1.0).
            ensure_path: If True, verify all open positions are connected.

        Returns:
            grid: numpy array (height x width) with color values.
            walls_set: set of (x, y) wall positions.
            open_positions: list of (x, y) non-wall positions.
        """
        max_attempts = 20

        for _ in range(max_attempts):
            grid = np.zeros((height, width), dtype=int)
            walls = set()

            # Place walls randomly
            n_walls = int(width * height * wall_density)
            all_positions = [(x, y) for x in range(width) for y in range(height)]
            if n_walls > 0 and n_walls < len(all_positions):
                wall_indices = self.rng.choice(len(all_positions), size=n_walls, replace=False)
                for idx in wall_indices:
                    pos = all_positions[idx]
                    walls.add(pos)

            # Mark walls on grid with a special wall color (use 0 for background)
            open_positions = [p for p in all_positions if p not in walls]

            if not open_positions:
                continue

            if ensure_path and len(open_positions) > 1:
                if self._is_connected(open_positions, width, height):
                    return grid, walls, open_positions
                else:
                    # Try to fix connectivity by removing some walls
                    grid, walls, open_positions = self._fix_connectivity(
                        grid, walls, open_positions, width, height
                    )
                    if self._is_connected(open_positions, width, height):
                        return grid, walls, open_positions
                    continue
            else:
                return grid, walls, open_positions

        # Fallback: no walls
        grid = np.zeros((height, width), dtype=int)
        open_positions = [(x, y) for x in range(width) for y in range(height)]
        return grid, set(), open_positions

    def _is_connected(self, open_positions: List[Tuple[int, int]],
                      width: int, height: int) -> bool:
        """Check if all open positions are connected using BFS."""
        if len(open_positions) <= 1:
            return True

        open_set = set(open_positions)
        start = open_positions[0]
        visited = set()
        queue = deque([start])
        visited.add(start)

        while queue:
            x, y = queue.popleft()
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (nx, ny) in open_set and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))

        return len(visited) == len(open_set)

    def _fix_connectivity(self, grid: np.ndarray, walls: Set[Tuple[int, int]],
                          open_positions: List[Tuple[int, int]],
                          width: int, height: int):
        """Try to fix connectivity by removing walls between components."""
        open_set = set(open_positions)

        # Find connected components
        components = []
        remaining = set(open_positions)

        while remaining:
            start = next(iter(remaining))
            component = set()
            queue = deque([start])
            component.add(start)

            while queue:
                x, y = queue.popleft()
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if (nx, ny) in remaining and (nx, ny) not in component:
                        component.add((nx, ny))
                        queue.append((nx, ny))

            components.append(component)
            remaining -= component

        if len(components) <= 1:
            return grid, walls, open_positions

        # Connect components by removing walls between them
        main = components[0]
        walls = set(walls)
        for comp in components[1:]:
            # Find a wall adjacent to both components
            connected = False
            for wx, wy in list(walls):
                neighbors_main = any(
                    (wx + dx, wy + dy) in main
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]
                )
                neighbors_comp = any(
                    (wx + dx, wy + dy) in comp
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]
                )
                if neighbors_main and neighbors_comp:
                    walls.discard((wx, wy))
                    open_set.add((wx, wy))
                    main.add((wx, wy))
                    main |= comp
                    connected = True
                    break

            if not connected:
                # Carve a path from comp to main using BFS through walls
                start = next(iter(comp))
                target_set = main
                visited = {start}
                parent = {start: None}
                queue = deque([start])
                found = None

                while queue and found is None:
                    x, y = queue.popleft()
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in visited:
                            visited.add((nx, ny))
                            parent[(nx, ny)] = (x, y)
                            if (nx, ny) in target_set:
                                found = (nx, ny)
                                break
                            queue.append((nx, ny))

                if found is not None:
                    # Trace path back and remove walls
                    pos = found
                    while pos is not None:
                        if pos in walls:
                            walls.discard(pos)
                            open_set.add(pos)
                        main.add(pos)
                        pos = parent[pos]
                    main |= comp

        open_positions = list(open_set)
        return grid, walls, open_positions
