"""PerceptTranslator -- convert StructuredPercept to natural language for LLM consumption.

Keeps output under ~300 tokens per observation. Prioritises controllable objects
and change descriptions (the MOST IMPORTANT section).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from charith.perception.core_knowledge import Object, StructuredPercept

# ---- colour palette -------------------------------------------------------

COLOR_NAMES: Dict[int, str] = {
    0: "black",
    1: "blue",
    2: "red",
    3: "green",
    4: "yellow",
    5: "grey",
    6: "magenta",
    7: "orange",
    8: "cyan",
    9: "maroon",
    10: "dark-red",
    11: "light-green",
    12: "pink",
}


def _color_name(c: int) -> str:
    return COLOR_NAMES.get(c, f"color-{c}")


# ---- helpers ---------------------------------------------------------------

def _size_category(size: int) -> str:
    if size <= 4:
        return "tiny"
    if size <= 20:
        return "small"
    if size <= 100:
        return "medium"
    return "large"


def _position_name(centroid: Tuple[float, float], grid_dims: Tuple[int, int]) -> str:
    """Map centroid to 3x3 region label."""
    rows, cols = grid_dims
    r, c = centroid
    v = "top" if r < rows / 3 else ("bottom" if r >= 2 * rows / 3 else "center")
    h = "left" if c < cols / 3 else ("right" if c >= 2 * cols / 3 else "center")
    if v == "center" and h == "center":
        return "center"
    if v == "center":
        return f"center-{h}"
    if h == "center":
        return f"{v}-center"
    return f"{v}-{h}"


# ---- main class ------------------------------------------------------------

class PerceptTranslator:
    """Convert a StructuredPercept into a compact natural-language description."""

    def translate(
        self,
        percept: StructuredPercept,
        prev_percept: Optional[StructuredPercept] = None,
        controllable_ids: Optional[Set[int]] = None,
        tick: int = 0,
        available_actions: Optional[List[int]] = None,
    ) -> str:
        controllable_ids = controllable_ids or set()
        parts: List[str] = []

        parts.append(self._grid_overview(percept, tick))
        parts.append(self._describe_objects(percept, controllable_ids))
        parts.append(self._describe_relations(percept, controllable_ids))

        if prev_percept is not None:
            changes = self._describe_changes(prev_percept, percept, controllable_ids)
            if changes:
                parts.append(changes)

        if available_actions:
            parts.append(f"Available actions: {available_actions}")

        return "\n".join(parts)

    # ---- sections ----------------------------------------------------------

    def _grid_overview(self, percept: StructuredPercept, tick: int) -> str:
        h, w = percept.grid_dims
        bg = _color_name(percept.background_color)
        n_obj = percept.object_count
        n_col = len(percept.unique_colors)
        return f"[Tick {tick}] Grid: {w}x{h}, background={bg}, {n_obj} objects, {n_col} colors"

    def _describe_objects(
        self,
        percept: StructuredPercept,
        controllable_ids: Set[int],
    ) -> str:
        if not percept.objects:
            return "Objects: none"

        lines: List[str] = []

        # Partition: controllable first, then unique-color singletons, then large
        ctrl = [o for o in percept.objects if o.object_id in controllable_ids]
        rest = [o for o in percept.objects if o.object_id not in controllable_ids]

        # Identify unique-color singletons (only one object of that colour)
        color_counts: Dict[int, int] = {}
        for o in percept.objects:
            color_counts[o.color] = color_counts.get(o.color, 0) + 1

        singletons = [o for o in rest if color_counts.get(o.color, 0) == 1]
        others = [o for o in rest if color_counts.get(o.color, 0) > 1]

        # Sort singletons by size desc, others by size desc
        singletons.sort(key=lambda o: -o.size)
        others.sort(key=lambda o: -o.size)

        ordered = ctrl + singletons + others

        # Detect special objects for goal/template/progress hints
        # Template box: small enclosed area with a pattern (small bbox, contains multiple colors)
        # Progress bar: long thin object at edges (width >> height or vice versa, at bottom/right)
        grid_h, grid_w = percept.grid_dims

        # Cap at 8 objects to stay concise
        for obj in ordered[:8]:
            pos = _position_name(obj.centroid, percept.grid_dims)
            sc = _size_category(obj.size)
            cn = _color_name(obj.color)
            tags: List[str] = []
            if obj.object_id in controllable_ids:
                tags.append("CONTROLLABLE")
            if color_counts.get(obj.color, 0) == 1:
                tags.append("UNIQUE COLOR - POSSIBLE GOAL")
            # Detect template boxes (small objects near top with distinct inner pattern)
            if (obj.size < 100 and obj.centroid[0] < grid_h * 0.25
                    and obj.object_id not in controllable_ids
                    and obj.width > 3 and obj.height > 3):
                tags.append("POSSIBLE TEMPLATE/REFERENCE")
            # Detect progress indicators (very wide/tall thin bars at edges)
            if ((obj.width > grid_w * 0.4 and obj.height <= 4) or
                    (obj.height > grid_h * 0.4 and obj.width <= 4)):
                if (obj.centroid[0] > grid_h * 0.85 or obj.centroid[1] > grid_w * 0.85
                        or obj.centroid[0] < grid_h * 0.15):
                    tags.append("POSSIBLE PROGRESS/SCORE BAR")
            tag_str = f" [{', '.join(tags)}]" if tags else ""
            lines.append(f"  - {cn} {sc} object at {pos} (size={obj.size}){tag_str}")

        if len(percept.objects) > 8:
            lines.append(f"  ... and {len(percept.objects) - 8} more objects")

        return "Objects:\n" + "\n".join(lines)

    def _describe_relations(
        self,
        percept: StructuredPercept,
        controllable_ids: Set[int],
    ) -> str:
        if not percept.spatial_relations:
            return "Relations: none detected"

        # Prioritise relations involving controllable
        ctrl_rels = [
            r for r in percept.spatial_relations
            if r.obj_a_id in controllable_ids or r.obj_b_id in controllable_ids
        ]
        other_rels = [
            r for r in percept.spatial_relations
            if r.obj_a_id not in controllable_ids and r.obj_b_id not in controllable_ids
        ]

        # Take top 5 most informative (shortest distance first)
        selected = sorted(ctrl_rels, key=lambda r: r.distance)[:5]
        if len(selected) < 5:
            selected += sorted(other_rels, key=lambda r: r.distance)[: 5 - len(selected)]

        if not selected:
            return "Relations: none detected"

        # Build object id->label map
        obj_map: Dict[int, str] = {}
        for o in percept.objects:
            label = "Controllable" if o.object_id in controllable_ids else f"{_color_name(o.color)}-obj"
            obj_map[o.object_id] = label

        lines: List[str] = []
        for r in selected:
            a = obj_map.get(r.obj_a_id, f"obj-{r.obj_a_id}")
            b = obj_map.get(r.obj_b_id, f"obj-{r.obj_b_id}")
            lines.append(f"  - {a} is {r.relation.upper()} of {b} (distance={r.distance:.0f})")

        # Add explicit controllable-to-goal-candidate distances
        color_counts: Dict[int, int] = {}
        for o in percept.objects:
            color_counts[o.color] = color_counts.get(o.color, 0) + 1

        ctrl_objs = [o for o in percept.objects if o.object_id in controllable_ids]
        goal_candidates = [o for o in percept.objects
                          if o.object_id not in controllable_ids
                          and (color_counts.get(o.color, 0) == 1 or o.size <= 10)]

        if ctrl_objs and goal_candidates:
            ctrl = ctrl_objs[0]
            for gc in goal_candidates[:3]:
                dr = gc.centroid[0] - ctrl.centroid[0]
                dc = gc.centroid[1] - ctrl.centroid[1]
                dist = (dr**2 + dc**2)**0.5
                direction = ""
                if abs(dr) > 1:
                    direction += "DOWN " if dr > 0 else "UP "
                if abs(dc) > 1:
                    direction += "RIGHT" if dc > 0 else "LEFT"
                direction = direction.strip() or "SAME POSITION"
                cn = _color_name(gc.color)
                tag = "UNIQUE" if color_counts.get(gc.color, 0) == 1 else "small"
                lines.append(
                    f"  >>> Controllable -> {tag} {cn} object: "
                    f"distance={dist:.0f}, direction={direction} "
                    f"(need to go {direction} to reach it)"
                )

        return "Relations:\n" + "\n".join(lines)

    def _describe_changes(
        self,
        prev_percept: StructuredPercept,
        curr_percept: StructuredPercept,
        controllable_ids: Set[int],
    ) -> str:
        """Describe what changed between two consecutive percepts.

        This is the MOST IMPORTANT section -- it tells the LLM what effect
        the last action had.
        """
        lines: List[str] = []

        # 1. Controllable movement
        prev_ctrl = {o.object_id: o for o in prev_percept.objects if o.object_id in controllable_ids}
        curr_ctrl = {o.object_id: o for o in curr_percept.objects if o.object_id in controllable_ids}

        for oid in controllable_ids:
            p = prev_ctrl.get(oid)
            c = curr_ctrl.get(oid)
            if p and c:
                dr = c.centroid[0] - p.centroid[0]
                dc = c.centroid[1] - p.centroid[1]
                dist = (dr ** 2 + dc ** 2) ** 0.5
                if dist < 0.5:
                    lines.append("Controllable did NOT move (wall or no-op?)")
                else:
                    direction = self._direction_name(dr, dc)
                    lines.append(f"Controllable moved {direction} by {dist:.0f} cells")
            elif p and not c:
                lines.append("Controllable DISAPPEARED")
            elif not p and c:
                lines.append("Controllable APPEARED")

        # 2. Object count changes
        diff = curr_percept.object_count - prev_percept.object_count
        if diff > 0:
            lines.append(f"{diff} new object(s) appeared")
        elif diff < 0:
            lines.append(f"{-diff} object(s) disappeared")

        # 3. Color distribution changes
        prev_colors = prev_percept.unique_colors
        curr_colors = curr_percept.unique_colors
        new_colors = curr_colors - prev_colors
        gone_colors = prev_colors - curr_colors
        if new_colors:
            lines.append(f"New colors appeared: {', '.join(_color_name(c) for c in new_colors)}")
        if gone_colors:
            lines.append(f"Colors disappeared: {', '.join(_color_name(c) for c in gone_colors)}")

        # 4. Grid-level pixel change magnitude
        if prev_percept.raw_grid.shape == curr_percept.raw_grid.shape:
            changed_pixels = int((prev_percept.raw_grid != curr_percept.raw_grid).sum())
            total = prev_percept.raw_grid.size
            if changed_pixels == 0:
                lines.append("Grid is IDENTICAL to previous (action had NO visible effect)")
            else:
                pct = 100.0 * changed_pixels / total
                lines.append(f"Grid changed: {changed_pixels} pixels ({pct:.1f}%)")

        if not lines:
            lines.append("No significant changes detected")

        return "Changes:\n" + "\n".join(f"  - {l}" for l in lines)

    @staticmethod
    def _direction_name(dr: float, dc: float) -> str:
        """Convert row/col delta to cardinal direction name."""
        # Row increases downward, col increases rightward
        if abs(dr) > abs(dc):
            return "DOWN" if dr > 0 else "UP"
        elif abs(dc) > abs(dr):
            return "RIGHT" if dc > 0 else "LEFT"
        else:
            v = "DOWN" if dr > 0 else "UP"
            h = "RIGHT" if dc > 0 else "LEFT"
            return f"{v}-{h}"
