"""
Object Tracker — persistent object identity across frames.

Uses greedy matching by (color, size, proximity) to match objects
between consecutive frames. Full Hungarian matching deferred to Phase 2.
"""
from typing import List, Tuple, Set
from charith.perception.core_knowledge import Object


class ObjectTracker:
    """
    Match objects between consecutive frames.

    Strategy: greedy matching by color first, then by centroid proximity.
    Objects with unique color are matched trivially.
    Objects sharing a color are matched by nearest centroid.
    """

    def match(self, prev_objects: List[Object],
              curr_objects: List[Object]) -> List[Tuple[int, int]]:
        """
        Match objects from prev frame to curr frame.

        Returns list of (prev_object_id, curr_object_id) pairs.
        Unmatched objects are NOT in the output (appeared/disappeared).
        """
        pairs = []
        used_curr = set()

        # Group by color for efficient matching
        prev_by_color = {}
        for obj in prev_objects:
            prev_by_color.setdefault(obj.color, []).append(obj)

        curr_by_color = {}
        for obj in curr_objects:
            curr_by_color.setdefault(obj.color, []).append(obj)

        for color in prev_by_color:
            if color not in curr_by_color:
                continue

            prev_group = prev_by_color[color]
            curr_group = [o for o in curr_by_color[color]
                         if o.object_id not in used_curr]

            for prev_obj in prev_group:
                best_curr = None
                best_dist = float('inf')
                for curr_obj in curr_group:
                    if curr_obj.object_id in used_curr:
                        continue
                    dist = self._centroid_dist(prev_obj, curr_obj)
                    if dist < best_dist:
                        best_dist = dist
                        best_curr = curr_obj

                if best_curr is not None:
                    pairs.append((prev_obj.object_id, best_curr.object_id))
                    used_curr.add(best_curr.object_id)

        return pairs

    def _centroid_dist(self, a: Object, b: Object) -> float:
        return ((a.centroid[0] - b.centroid[0])**2 +
                (a.centroid[1] - b.centroid[1])**2) ** 0.5
