"""Object Tracker — persistent object identity across frames (Path 6 / S2).

Hypotheses, predictions, and history are all bound to object ids. If an object
silently changes id after moving, everything downstream corrupts *without*
erroring. This tracker makes objects stable *nouns* shared by the ledger, bank,
and translator.

Two APIs live here:

* ``match(prev, curr)`` — the Path-5 greedy matcher, kept for backward
  compatibility (existing agent loop + tests). Returns id pairs only.
* ``update(segmented, tick)`` — the Path-6 persistent tracker. Assigns stable
  oids via Hungarian matching and emits ``PerceptEvent``s (MOVED / RECOLORED /
  TRANSFORMED / APPEARED / VANISHED / REAPPEARED). This is the noun stream S3
  scores predictions against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from charith.perception.core_knowledge import Object

N_COLORS = 13  # ARC palette 0..12

# Cost-function weights and thresholds (guide S2).
_W_CENTROID = 0.5
_W_SHAPE = 0.3
_W_COLOR = 0.2
_CENTROID_SCALE = 8.0
_MATCH_REJECT = 0.55        # pairs above this cost are not matched
_TRANSFORM_IOU = 0.4        # below this (with continuity) -> TRANSFORMED, same id
_REAPPEAR_COST = 0.4        # a vanished object may reclaim its id below this
_REAPPEAR_WINDOW = 10       # ticks a vanished object can still reappear


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TrackedObject:
    oid: str                          # "obj_007" — never reused within an episode
    first_tick: int
    last_tick: int
    mask: np.ndarray                  # binary bitmap, tight bbox
    color_hist: np.ndarray            # N_COLORS bins
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[float, float]
    color: int
    status: Literal["VISIBLE", "OCCLUDED", "GONE"] = "VISIBLE"
    misses: int = 0


@dataclass(frozen=True)
class PerceptEvent:
    kind: Literal["MOVED", "RECOLORED", "TRANSFORMED",
                  "APPEARED", "VANISHED", "REAPPEARED"]
    oid: str
    detail: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Feature extraction from a raw segmented Object
# ---------------------------------------------------------------------------

def _mask_of(obj: Object) -> np.ndarray:
    r0, c0, r1, c1 = obj.bbox
    h, w = r1 - r0 + 1, c1 - c0 + 1
    mask = np.zeros((h, w), dtype=bool)
    for cell in obj.cells:
        mask[cell.row - r0, cell.col - c0] = True
    return mask


def _color_hist(obj: Object) -> np.ndarray:
    hist = np.zeros(N_COLORS, dtype=float)
    for cell in obj.cells:
        if 0 <= cell.color < N_COLORS:
            hist[cell.color] += 1.0
    return hist


def _aligned_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two binary masks aligned by their centroids (translation-invariant)."""
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    H = max(a.shape[0], b.shape[0]) * 2 + 2
    W = max(a.shape[1], b.shape[1]) * 2 + 2
    ca = _mask_centroid(a)
    cb = _mask_centroid(b)
    canvas_center = (H // 2, W // 2)
    pa = _paste(a, ca, canvas_center, (H, W))
    pb = _paste(b, cb, canvas_center, (H, W))
    inter = np.logical_and(pa, pb).sum()
    union = np.logical_or(pa, pb).sum()
    return float(inter / union) if union else 0.0


def _mask_centroid(m: np.ndarray) -> Tuple[float, float]:
    rows, cols = np.nonzero(m)
    return (rows.mean(), cols.mean())


def _paste(m: np.ndarray, m_center: Tuple[float, float],
           canvas_center: Tuple[int, int], canvas_shape: Tuple[int, int]) -> np.ndarray:
    canvas = np.zeros(canvas_shape, dtype=bool)
    off_r = int(round(canvas_center[0] - m_center[0]))
    off_c = int(round(canvas_center[1] - m_center[1]))
    rows, cols = np.nonzero(m)
    rr = rows + off_r
    cc = cols + off_c
    valid = (rr >= 0) & (rr < canvas_shape[0]) & (cc >= 0) & (cc < canvas_shape[1])
    canvas[rr[valid], cc[valid]] = True
    return canvas


def _color_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _centroid_dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class ObjectTracker:
    """Persistent object identity via Hungarian matching."""

    def __init__(self, occlusion_grace_ticks: int = 3) -> None:
        self.occlusion_grace_ticks = occlusion_grace_ticks
        self._tracked: Dict[str, TrackedObject] = {}     # live (VISIBLE/OCCLUDED)
        self._vanished: Dict[str, Tuple[TrackedObject, int]] = {}  # oid -> (obj, tick_gone)
        self._next_oid = 0

    # ---- Path-5 backward-compatible greedy matcher -------------------------

    def match(self, prev_objects: List[Object],
              curr_objects: List[Object]) -> List[Tuple[int, int]]:
        """Legacy greedy matcher by (color, centroid proximity). Returns id pairs.

        Preserved for the existing agent loop / tests. Does not maintain state.
        """
        pairs: List[Tuple[int, int]] = []
        used_curr: set = set()
        prev_by_color: Dict[int, List[Object]] = {}
        for obj in prev_objects:
            prev_by_color.setdefault(obj.color, []).append(obj)
        curr_by_color: Dict[int, List[Object]] = {}
        for obj in curr_objects:
            curr_by_color.setdefault(obj.color, []).append(obj)

        for color, prev_group in prev_by_color.items():
            if color not in curr_by_color:
                continue
            for prev_obj in prev_group:
                best_curr, best_dist = None, float("inf")
                for curr_obj in curr_by_color[color]:
                    if curr_obj.object_id in used_curr:
                        continue
                    dist = _centroid_dist(prev_obj.centroid, curr_obj.centroid)
                    if dist < best_dist:
                        best_dist, best_curr = dist, curr_obj
                if best_curr is not None:
                    pairs.append((prev_obj.object_id, best_curr.object_id))
                    used_curr.add(best_curr.object_id)
        return pairs

    # ---- Path-6 persistent tracker -----------------------------------------

    def _new_oid(self) -> str:
        oid = f"obj_{self._next_oid:03d}"
        self._next_oid += 1
        return oid

    def _to_tracked(self, obj: Object, oid: str, tick: int) -> TrackedObject:
        return TrackedObject(
            oid=oid,
            first_tick=tick,
            last_tick=tick,
            mask=_mask_of(obj),
            color_hist=_color_hist(obj),
            bbox=obj.bbox,
            centroid=obj.centroid,
            color=obj.color,
            status="VISIBLE",
            misses=0,
        )

    def update(self, segmented: List[Object], tick: int
               ) -> Tuple[List[TrackedObject], List[PerceptEvent]]:
        """Assign stable oids to this tick's segments and emit percept events."""
        events: List[PerceptEvent] = []

        prev_ids = list(self._tracked.keys())
        prev_objs = [self._tracked[i] for i in prev_ids]

        # Pre-compute current features.
        cur_masks = [_mask_of(o) for o in segmented]
        cur_hists = [_color_hist(o) for o in segmented]

        matched_prev: set = set()
        matched_cur: set = set()

        # --- Hungarian assignment between previous tracked and current segments ---
        if prev_objs and segmented:
            cost = np.ones((len(prev_objs), len(segmented)), dtype=float)
            for i, p in enumerate(prev_objs):
                for j, seg in enumerate(segmented):
                    cost[i, j] = self._pair_cost(p, seg, cur_masks[j], cur_hists[j])
            rows, cols = linear_sum_assignment(cost)
            for i, j in zip(rows, cols):
                if cost[i, j] > _MATCH_REJECT:
                    continue
                oid = prev_ids[i]
                matched_prev.add(i)
                matched_cur.add(j)
                events.extend(self._resolve_match(oid, prev_objs[i], segmented[j],
                                                   cur_masks[j], cur_hists[j], tick))

        # --- Unmatched current: try reappearance, else APPEARED ---
        for j, seg in enumerate(segmented):
            if j in matched_cur:
                continue
            reoid = self._try_reappear(seg, cur_masks[j], cur_hists[j], tick)
            if reoid is not None:
                events.append(PerceptEvent("REAPPEARED", reoid,
                                           {"centroid": tuple(round(x, 1) for x in seg.centroid)}))
            else:
                oid = self._new_oid()
                self._tracked[oid] = self._to_tracked(seg, oid, tick)
                events.append(PerceptEvent("APPEARED", oid,
                                           {"color": seg.color,
                                            "centroid": tuple(round(x, 1) for x in seg.centroid)}))

        # --- Unmatched previous: occlusion / vanishing ---
        for i, oid in enumerate(prev_ids):
            if i in matched_prev:
                continue
            t = self._tracked[oid]
            t.misses += 1
            t.status = "OCCLUDED"
            if t.misses > self.occlusion_grace_ticks:
                t.status = "GONE"
                self._vanished[oid] = (t, tick)
                del self._tracked[oid]
                events.append(PerceptEvent("VANISHED", oid, {"last_tick": t.last_tick}))

        # Expire stale vanished records.
        self._expire_vanished(tick)

        visible = [t for t in self._tracked.values() if t.status == "VISIBLE"]
        return visible, events

    # ---- matching internals ------------------------------------------------

    def _pair_cost(self, prev: TrackedObject, seg: Object,
                   seg_mask: np.ndarray, seg_hist: np.ndarray) -> float:
        d = min(1.0, _centroid_dist(prev.centroid, seg.centroid) / _CENTROID_SCALE)
        iou = _aligned_iou(prev.mask, seg_mask)
        csim = _color_sim(prev.color_hist, seg_hist)
        return _W_CENTROID * d + _W_SHAPE * (1 - iou) + _W_COLOR * (1 - csim)

    def _resolve_match(self, oid: str, prev: TrackedObject, seg: Object,
                       seg_mask: np.ndarray, seg_hist: np.ndarray, tick: int
                       ) -> List[PerceptEvent]:
        events: List[PerceptEvent] = []
        dr = seg.centroid[0] - prev.centroid[0]
        dc = seg.centroid[1] - prev.centroid[1]
        moved = (dr ** 2 + dc ** 2) ** 0.5 >= 0.5
        iou = _aligned_iou(prev.mask, seg_mask)
        recolored = seg.color != prev.color

        if recolored:
            events.append(PerceptEvent("RECOLORED", oid,
                                       {"from": prev.color, "to": seg.color}))
        # Shape changed materially but identity is continuous -> TRANSFORMED.
        if iou < _TRANSFORM_IOU:
            events.append(PerceptEvent("TRANSFORMED", oid, {"iou": round(iou, 2)}))
        if moved:
            events.append(PerceptEvent("MOVED", oid,
                                       {"dx": round(dc, 1), "dy": round(dr, 1)}))

        # Update the tracked record in place (keep identity).
        prev.last_tick = tick
        prev.misses = 0
        prev.status = "VISIBLE"
        prev.mask = seg_mask
        prev.color_hist = seg_hist
        prev.bbox = seg.bbox
        prev.centroid = seg.centroid
        prev.color = seg.color
        return events

    def _try_reappear(self, seg: Object, seg_mask: np.ndarray,
                      seg_hist: np.ndarray, tick: int) -> Optional[str]:
        best_oid, best_cost = None, _REAPPEAR_COST
        for oid, (t, _gone) in self._vanished.items():
            cost = self._pair_cost(t, seg, seg_mask, seg_hist)
            if cost < best_cost:
                best_cost, best_oid = cost, oid
        if best_oid is not None:
            t, _ = self._vanished.pop(best_oid)
            t.last_tick = tick
            t.misses = 0
            t.status = "VISIBLE"
            t.mask = seg_mask
            t.color_hist = seg_hist
            t.bbox = seg.bbox
            t.centroid = seg.centroid
            t.color = seg.color
            self._tracked[best_oid] = t
            return best_oid
        return None

    def _expire_vanished(self, tick: int) -> None:
        stale = [oid for oid, (_, gone) in self._vanished.items()
                 if tick - gone > _REAPPEAR_WINDOW]
        for oid in stale:
            del self._vanished[oid]

    def reset(self) -> None:
        self._tracked.clear()
        self._vanished.clear()
        self._next_oid = 0
