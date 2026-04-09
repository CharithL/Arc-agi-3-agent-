"""
Convert a StructuredPercept diff into an ActualObservation.

For Milestone 1 (ls20 focus):
- Controllable = the matched object that moved the most
- Direction = sign of the larger axis displacement
- Magnitude = L-inf distance (max of |row_delta|, |col_delta|)
- Color changes = object pairs matched with different colors
- New / removed objects = unmatched after / unmatched before
- Score = stub False for Milestone 1

Uses ObjectTracker.match() for identity persistence across frames.
"""

from typing import Optional, Tuple

from charith.perception.core_knowledge import StructuredPercept
from charith.perception.object_tracker import ObjectTracker
from charith.full_stack.hypothesis_schema import ActualObservation, Direction


_TRACKER = ObjectTracker()


def diff_to_actual_observation(
    before: StructuredPercept,
    after: StructuredPercept,
) -> ActualObservation:
    pairs = _TRACKER.match(before.objects, after.objects)

    # Build lookup tables
    before_by_id = {o.object_id: o for o in before.objects}
    after_by_id = {o.object_id: o for o in after.objects}

    # Compute per-object displacements
    max_disp: Optional[Tuple[int, int]] = None
    max_magnitude = 0
    for prev_id, curr_id in pairs:
        prev = before_by_id[prev_id]
        curr = after_by_id[curr_id]
        dr = int(round(curr.centroid[0] - prev.centroid[0]))
        dc = int(round(curr.centroid[1] - prev.centroid[1]))
        mag = max(abs(dr), abs(dc))
        if mag > max_magnitude:
            max_magnitude = mag
            max_disp = (dr, dc)

    # Derive direction from the dominant-axis displacement
    direction: Optional[Direction] = None
    if max_disp is not None and max_magnitude > 0:
        dr, dc = max_disp
        if abs(dr) >= abs(dc):
            direction = "up" if dr < 0 else "down"
        else:
            direction = "left" if dc < 0 else "right"

    # Color changes: matched pairs where color differs
    any_color_changes = []
    for prev_id, curr_id in pairs:
        prev = before_by_id[prev_id]
        curr = after_by_id[curr_id]
        if prev.color != curr.color:
            any_color_changes.append((curr, prev.color, curr.color))

    # New objects: in after but not in any matched pair
    matched_curr = {curr_id for _, curr_id in pairs}
    new_objects = [o for o in after.objects if o.object_id not in matched_curr]

    # Removed objects: in before but not in any matched pair
    matched_prev = {prev_id for prev_id, _ in pairs}
    removed_objects = [o for o in before.objects if o.object_id not in matched_prev]

    return ActualObservation(
        controllable_displacement=max_disp,
        controllable_direction=direction,
        controllable_magnitude=max_magnitude,
        any_color_changes=any_color_changes,
        new_objects=new_objects,
        removed_objects=removed_objects,
        score_changed=False,  # Milestone 1 stub
    )
