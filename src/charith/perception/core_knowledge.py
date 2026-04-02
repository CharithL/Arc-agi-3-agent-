"""Core Knowledge Priors -- Spelke's 4 systems as pure functions: Grid -> StructuredPercept.

This is the foundational perception layer.  ALL other modules depend on
StructuredPercept.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class Color(IntEnum):
    """ARC colour palette (0-9)."""
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    MAGENTA = 6
    ORANGE = 7
    AZURE = 8
    MAROON = 9


@dataclass(frozen=True)
class Cell:
    """Single grid cell."""
    row: int
    col: int
    color: int


@dataclass
class Object:
    """A connected-component object extracted from the grid."""
    object_id: int
    cells: Set[Cell]
    color: int
    bbox: Tuple[int, int, int, int]   # (min_row, min_col, max_row, max_col)
    size: int
    centroid: Tuple[float, float]     # (row, col)
    shape_hash: int

    @property
    def width(self) -> int:
        return self.bbox[3] - self.bbox[1] + 1

    @property
    def height(self) -> int:
        return self.bbox[2] - self.bbox[0] + 1


@dataclass(frozen=True)
class SpatialRelation:
    """Pairwise spatial relation between two objects."""
    obj_a_id: int
    obj_b_id: int
    relation: str
    distance: float


@dataclass
class StructuredPercept:
    """The output of the full perception pipeline."""
    raw_grid: np.ndarray
    objects: List[Object]
    spatial_relations: List[SpatialRelation]
    color_counts: Dict[int, int]
    grid_dims: Tuple[int, int]
    background_color: int
    symmetry: Dict[str, bool]
    unique_colors: Set[int]
    object_count: int
    timestamp: float


# ---------------------------------------------------------------------------
# 1. ObjectnessPrior -- BFS flood-fill
# ---------------------------------------------------------------------------

class ObjectnessPrior:
    """Detect discrete objects via BFS flood-fill."""

    _NEIGHBORS_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    _NEIGHBORS_8 = [(-1, 0), (1, 0), (0, -1), (0, 1),
                    (-1, -1), (-1, 1), (1, -1), (1, 1)]

    def __init__(
        self,
        connectivity: int = 4,
        min_object_size: int = 1,
        ignore_background: bool = True,
    ) -> None:
        self.connectivity = connectivity
        self.min_object_size = min_object_size
        self.ignore_background = ignore_background

    # ------------------------------------------------------------------

    def detect(self, grid: np.ndarray, background_color: int = 0) -> List[Object]:
        """Return all connected-component objects found in *grid*."""
        rows, cols = grid.shape
        visited = np.zeros_like(grid, dtype=bool)
        neighbors = self._NEIGHBORS_8 if self.connectivity == 8 else self._NEIGHBORS_4
        objects: List[Object] = []
        obj_id = 0

        for r in range(rows):
            for c in range(cols):
                if visited[r, c]:
                    continue
                color = int(grid[r, c])
                if self.ignore_background and color == background_color:
                    visited[r, c] = True
                    continue

                # BFS flood-fill
                component_cells: Set[Cell] = set()
                queue: deque[Tuple[int, int]] = deque()
                queue.append((r, c))
                visited[r, c] = True

                while queue:
                    cr, cc = queue.popleft()
                    component_cells.add(Cell(cr, cc, color))
                    for dr, dc in neighbors:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc]:
                            if int(grid[nr, nc]) == color:
                                visited[nr, nc] = True
                                queue.append((nr, nc))

                if len(component_cells) < self.min_object_size:
                    continue

                obj = self._build_object(obj_id, component_cells, color)
                objects.append(obj)
                obj_id += 1

        return objects

    # ------------------------------------------------------------------

    @staticmethod
    def _build_object(
        object_id: int,
        cells: Set[Cell],
        color: int,
    ) -> Object:
        rows = [c.row for c in cells]
        cols = [c.col for c in cells]
        min_r, max_r = min(rows), max(rows)
        min_c, max_c = min(cols), max(cols)
        centroid = (sum(rows) / len(rows), sum(cols) / len(cols))

        # Position-invariant shape hash: normalize to origin then hash the
        # frozenset of relative (dr, dc) pairs.
        normalized: FrozenSet[Tuple[int, int]] = frozenset(
            (c.row - min_r, c.col - min_c) for c in cells
        )
        shape_hash = hash(normalized)

        return Object(
            object_id=object_id,
            cells=cells,
            color=color,
            bbox=(min_r, min_c, max_r, max_c),
            size=len(cells),
            centroid=centroid,
            shape_hash=shape_hash,
        )


# ---------------------------------------------------------------------------
# 2. SpatialPrior -- pairwise relations & symmetry
# ---------------------------------------------------------------------------

class SpatialPrior:
    """Compute pairwise spatial relations and grid symmetry."""

    _THRESHOLD = 0.5  # centroid comparison threshold

    def compute_relations(self, objects: List[Object]) -> List[SpatialRelation]:
        """Return all pairwise spatial relations between *objects*."""
        relations: List[SpatialRelation] = []
        for i, a in enumerate(objects):
            for j, b in enumerate(objects):
                if i >= j:
                    continue
                relations.extend(self._relations_between(a, b))
        return relations

    # ------------------------------------------------------------------

    def _relations_between(self, a: Object, b: Object) -> List[SpatialRelation]:
        rels: List[SpatialRelation] = []
        ar, ac = a.centroid
        br, bc = b.centroid
        dist = float(np.hypot(ar - br, ac - bc))

        # above / below (row comparison -- smaller row is higher)
        if ar < br - self._THRESHOLD:
            rels.append(SpatialRelation(a.object_id, b.object_id, "above", dist))
            rels.append(SpatialRelation(b.object_id, a.object_id, "below", dist))
        elif br < ar - self._THRESHOLD:
            rels.append(SpatialRelation(a.object_id, b.object_id, "below", dist))
            rels.append(SpatialRelation(b.object_id, a.object_id, "above", dist))

        # left / right (col comparison)
        if ac < bc - self._THRESHOLD:
            rels.append(SpatialRelation(a.object_id, b.object_id, "left", dist))
            rels.append(SpatialRelation(b.object_id, a.object_id, "right", dist))
        elif bc < ac - self._THRESHOLD:
            rels.append(SpatialRelation(a.object_id, b.object_id, "right", dist))
            rels.append(SpatialRelation(b.object_id, a.object_id, "left", dist))

        # contains -- all cells of b within bbox of a (or vice-versa)
        if self._contains(a, b):
            rels.append(SpatialRelation(a.object_id, b.object_id, "contains", dist))
        if self._contains(b, a):
            rels.append(SpatialRelation(b.object_id, a.object_id, "contains", dist))

        # adjacent -- any 4-connected neighbour pair
        if self._adjacent(a, b):
            rels.append(SpatialRelation(a.object_id, b.object_id, "adjacent", dist))

        # aligned_h / aligned_v
        if abs(ar - br) < self._THRESHOLD:
            rels.append(SpatialRelation(a.object_id, b.object_id, "aligned_h", dist))
        if abs(ac - bc) < self._THRESHOLD:
            rels.append(SpatialRelation(a.object_id, b.object_id, "aligned_v", dist))

        return rels

    # ------------------------------------------------------------------

    @staticmethod
    def _contains(outer: Object, inner: Object) -> bool:
        """All cells of *inner* lie within the bbox of *outer*."""
        r0, c0, r1, c1 = outer.bbox
        return all(r0 <= cell.row <= r1 and c0 <= cell.col <= c1 for cell in inner.cells)

    @staticmethod
    def _adjacent(a: Object, b: Object) -> bool:
        """Any cell in *a* is 4-connected to any cell in *b*."""
        coords_b = {(c.row, c.col) for c in b.cells}
        for cell in a.cells:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if (cell.row + dr, cell.col + dc) in coords_b:
                    return True
        return False

    # ------------------------------------------------------------------

    @staticmethod
    def detect_grid_symmetry(grid: np.ndarray) -> Dict[str, bool]:
        """Check horizontal, vertical, and rotational symmetry."""
        return {
            "h_symmetric": bool(np.array_equal(grid, np.flipud(grid))),
            "v_symmetric": bool(np.array_equal(grid, np.fliplr(grid))),
            "rot_90": bool(np.array_equal(grid, np.rot90(grid))),
            "rot_180": bool(np.array_equal(grid, np.rot90(grid, 2))),
        }


# ---------------------------------------------------------------------------
# 3. NumberPrior -- counting and numerical patterns
# ---------------------------------------------------------------------------

class NumberPrior:
    """Detect numerical patterns in grids and sequences."""

    def count_by_color(self, grid: np.ndarray) -> Dict[int, int]:
        """Return {color: count} using np.unique."""
        colors, counts = np.unique(grid, return_counts=True)
        return {int(c): int(n) for c, n in zip(colors, counts)}

    # ------------------------------------------------------------------

    def detect_numerical_patterns(self, counts: List[int]) -> Dict[str, bool]:
        """Analyse *counts* for constant / increasing / decreasing /
        arithmetic / periodic patterns."""
        if len(counts) < 2:
            return {
                "constant": True,
                "increasing": False,
                "decreasing": False,
                "arithmetic": True,
                "periodic": False,
            }

        diffs = [counts[i + 1] - counts[i] for i in range(len(counts) - 1)]

        constant = all(d == 0 for d in diffs)
        increasing = all(d > 0 for d in diffs)
        decreasing = all(d < 0 for d in diffs)
        arithmetic = len(set(diffs)) == 1  # all diffs equal
        periodic = self._is_periodic(counts)

        return {
            "constant": constant,
            "increasing": increasing,
            "decreasing": decreasing,
            "arithmetic": arithmetic,
            "periodic": periodic,
        }

    # ------------------------------------------------------------------

    @staticmethod
    def _is_periodic(seq: List[int], max_period: int = 5) -> bool:
        """Check if *seq* is periodic with period <= *max_period*."""
        n = len(seq)
        for p in range(1, min(max_period, n) + 1):
            if all(seq[i] == seq[i % p] for i in range(n)):
                return True
        return False

    # ------------------------------------------------------------------

    @staticmethod
    def detect_count_change(
        prev: Dict[int, int],
        curr: Dict[int, int],
    ) -> Dict[int, int]:
        """Return per-color delta between two count dicts."""
        all_colors = set(prev) | set(curr)
        return {c: curr.get(c, 0) - prev.get(c, 0) for c in all_colors}


# ---------------------------------------------------------------------------
# 4. AgencyPrior -- contingency & goal-directed motion
# ---------------------------------------------------------------------------

class AgencyPrior:
    """Detect agent-like behaviour: contingencies and goal-directed motion."""

    def __init__(self) -> None:
        self._motion_history: Dict[int, List[Tuple[float, float]]] = {}
        self._contingencies: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------

    def record_action_contingency(
        self,
        action: str,
        state_before: np.ndarray,
        state_after: np.ndarray,
    ) -> List[Tuple[int, int]]:
        """Record which cells changed between *state_before* and *state_after*.

        Returns the list of (row, col) positions that changed.
        """
        diff_mask = state_before != state_after
        changed: List[Tuple[int, int]] = list(zip(*np.where(diff_mask)))
        self._contingencies.append({
            "action": action,
            "changed_cells": changed,
        })
        return changed

    # ------------------------------------------------------------------

    def detect_controllable_objects(
        self,
        objects: List[Object],
        contingencies: Optional[List[Dict[str, Any]]] = None,
    ) -> List[int]:
        """Return object_ids affected by >30 % of recorded actions."""
        ctgs = contingencies if contingencies is not None else self._contingencies
        if not ctgs:
            return []

        total_actions = len(ctgs)
        obj_hit_count: Dict[int, int] = {o.object_id: 0 for o in objects}

        for ctg in ctgs:
            changed_set = set(ctg["changed_cells"])
            for obj in objects:
                if any((c.row, c.col) in changed_set for c in obj.cells):
                    obj_hit_count[obj.object_id] += 1

        threshold = 0.3 * total_actions
        return [
            oid for oid, count in obj_hit_count.items() if count > threshold
        ]

    # ------------------------------------------------------------------

    def record_object_displacement(
        self,
        object_id: int,
        centroid: Tuple[float, float],
    ) -> None:
        """Append *centroid* to the motion history for *object_id*."""
        self._motion_history.setdefault(object_id, []).append(centroid)

    def detect_goal_directed_motion(
        self,
        object_id: int,
        min_steps: int = 3,
    ) -> bool:
        """Return True if *object_id* has been moving consistently in one
        direction for at least *min_steps* consecutive steps."""
        history = self._motion_history.get(object_id, [])
        if len(history) < min_steps + 1:
            return False

        recent = history[-(min_steps + 1):]
        deltas = [
            (recent[i + 1][0] - recent[i][0], recent[i + 1][1] - recent[i][1])
            for i in range(len(recent) - 1)
        ]

        # Consistent direction: all deltas have the same sign pattern
        row_signs = [np.sign(d[0]) for d in deltas]
        col_signs = [np.sign(d[1]) for d in deltas]

        row_consistent = len(set(row_signs)) == 1 and row_signs[0] != 0
        col_consistent = len(set(col_signs)) == 1 and col_signs[0] != 0

        return row_consistent or col_consistent

    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._motion_history.clear()
        self._contingencies.clear()


# ---------------------------------------------------------------------------
# 5. CoreKnowledgePerception -- full pipeline
# ---------------------------------------------------------------------------

class CoreKnowledgePerception:
    """Pipeline combining all 4 core-knowledge priors.

    ``perceive(grid)`` returns a :class:`StructuredPercept`.
    """

    def __init__(self) -> None:
        self.objectness = ObjectnessPrior()
        self.spatial = SpatialPrior()
        self.number = NumberPrior()
        self.agency = AgencyPrior()
        self._tick: int = 0

    # ------------------------------------------------------------------

    def perceive(self, grid: np.ndarray) -> StructuredPercept:
        """Run the full perception pipeline on *grid*."""
        self._tick += 1

        # Background = most common colour
        color_counts = self.number.count_by_color(grid)
        background_color = max(color_counts, key=lambda c: color_counts[c])

        # 1. Objectness
        objects = self.objectness.detect(grid, background_color=background_color)

        # 2. Spatial relations + symmetry
        spatial_relations = self.spatial.compute_relations(objects)
        symmetry = self.spatial.detect_grid_symmetry(grid)

        # 3. Record centroids for agency tracking
        for obj in objects:
            self.agency.record_object_displacement(obj.object_id, obj.centroid)

        return StructuredPercept(
            raw_grid=grid,
            objects=objects,
            spatial_relations=spatial_relations,
            color_counts=color_counts,
            grid_dims=(grid.shape[0], grid.shape[1]),
            background_color=background_color,
            symmetry=symmetry,
            unique_colors=set(color_counts.keys()),
            object_count=len(objects),
            timestamp=time.time(),
        )

    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all mutable state (tick counter, agency history)."""
        self._tick = 0
        self.agency.reset()
