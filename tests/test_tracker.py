"""S2 tests — persistent object identity (update() API)."""
import numpy as np
import pytest

from charith.perception.core_knowledge import ObjectnessPrior
from charith.perception.object_tracker import ObjectTracker, PerceptEvent


def _detect(grid):
    return ObjectnessPrior().detect(grid, background_color=0)


def _kinds(events, oid=None):
    return {e.kind for e in events if oid is None or e.oid == oid}


def _oid_at(tracked, centroid, tol=1.5):
    """Find the tracked oid whose centroid is nearest the given point."""
    best = min(tracked, key=lambda t: (t.centroid[0] - centroid[0]) ** 2
                                       + (t.centroid[1] - centroid[1]) ** 2)
    return best.oid


# ---------------------------------------------------------------------------

def test_pure_translation_keeps_id():
    tr = ObjectTracker()
    g1 = np.zeros((12, 12), dtype=int); g1[3, 3] = 1
    g2 = np.zeros((12, 12), dtype=int); g2[3, 5] = 1  # moved right 2

    vis1, ev1 = tr.update(_detect(g1), tick=1)
    assert _kinds(ev1) == {"APPEARED"}
    oid = vis1[0].oid

    vis2, ev2 = tr.update(_detect(g2), tick=2)
    assert len(vis2) == 1
    assert vis2[0].oid == oid                    # id preserved across a move
    assert "MOVED" in _kinds(ev2, oid)
    moved = [e for e in ev2 if e.kind == "MOVED"][0]
    assert moved.detail["dx"] == 2 and moved.detail["dy"] == 0


def test_two_same_color_squares_keep_distinct_ids():
    tr = ObjectTracker()
    g1 = np.zeros((16, 16), dtype=int)
    g1[2:4, 2:4] = 1     # square A left
    g1[2:4, 10:12] = 1   # square B right
    vis1, _ = tr.update(_detect(g1), tick=1)
    assert len(vis1) == 2
    id_left = _oid_at(vis1, (2.5, 2.5))
    id_right = _oid_at(vis1, (2.5, 10.5))
    assert id_left != id_right

    # Both step toward each other by 2 cols but do not cross.
    g2 = np.zeros((16, 16), dtype=int)
    g2[2:4, 4:6] = 1
    g2[2:4, 8:10] = 1
    vis2, _ = tr.update(_detect(g2), tick=2)
    assert len(vis2) == 2
    # Position term dominates -> nearest keeps its own id.
    assert _oid_at(vis2, (2.5, 4.5)) == id_left
    assert _oid_at(vis2, (2.5, 8.5)) == id_right


def test_occlusion_two_frames_same_id_on_return():
    tr = ObjectTracker(occlusion_grace_ticks=3)
    g = np.zeros((12, 12), dtype=int); g[5, 5] = 1
    empty = np.zeros((12, 12), dtype=int)

    vis1, _ = tr.update(_detect(g), tick=1)
    oid = vis1[0].oid
    tr.update(_detect(empty), tick=2)   # miss 1 (OCCLUDED, within grace)
    tr.update(_detect(empty), tick=3)   # miss 2
    vis4, _ = tr.update(_detect(g), tick=4)
    assert len(vis4) == 1
    assert vis4[0].oid == oid


def test_vanish_then_reappear_reclaims_id():
    tr = ObjectTracker(occlusion_grace_ticks=1)  # vanish quickly
    g = np.zeros((12, 12), dtype=int); g[5, 5] = 2
    empty = np.zeros((12, 12), dtype=int)

    vis1, _ = tr.update(_detect(g), tick=1)
    oid = vis1[0].oid
    tr.update(_detect(empty), tick=2)          # miss 1 -> OCCLUDED
    _, ev3 = tr.update(_detect(empty), tick=3)  # miss 2 > grace -> VANISHED
    assert "VANISHED" in _kinds(ev3, oid)

    vis4, ev4 = tr.update(_detect(g), tick=4)   # within reappear window
    assert "REAPPEARED" in _kinds(ev4)
    assert vis4[0].oid == oid                    # same noun reclaimed


def test_inplace_rotation_emits_transformed_same_id():
    tr = ObjectTracker()
    # A thin bar rotating in place: horizontal 1x5 -> vertical 5x1, same centroid.
    # Centroid-aligned IoU collapses (~0.1) so this reads as TRANSFORMED, not two
    # separate nouns, and the shared centroid means no spurious MOVED.
    g1 = np.zeros((12, 12), dtype=int); g1[5, 3:8] = 1   # centroid (5, 5)
    g2 = np.zeros((12, 12), dtype=int); g2[3:8, 5] = 1   # centroid (5, 5)

    vis1, _ = tr.update(_detect(g1), tick=1)
    oid = vis1[0].oid
    vis2, ev2 = tr.update(_detect(g2), tick=2)
    assert vis2[0].oid == oid                    # identity survives the reshape
    assert "TRANSFORMED" in _kinds(ev2, oid)


def test_recolor_emits_recolored_same_id():
    tr = ObjectTracker()
    g1 = np.zeros((12, 12), dtype=int); g1[4:6, 4:6] = 3   # green
    g2 = np.zeros((12, 12), dtype=int); g2[4:6, 4:6] = 7   # orange, same place/shape

    vis1, _ = tr.update(_detect(g1), tick=1)
    oid = vis1[0].oid
    vis2, ev2 = tr.update(_detect(g2), tick=2)
    assert vis2[0].oid == oid
    rec = [e for e in ev2 if e.kind == "RECOLORED"]
    assert rec and rec[0].detail == {"from": 3, "to": 7}


def test_appeared_and_vanished_events():
    tr = ObjectTracker(occlusion_grace_ticks=0)
    g1 = np.zeros((12, 12), dtype=int); g1[3, 3] = 1
    g2 = np.zeros((12, 12), dtype=int); g2[9, 9] = 2  # old gone, new appears

    _, ev1 = tr.update(_detect(g1), tick=1)
    assert _kinds(ev1) == {"APPEARED"}
    _, ev2 = tr.update(_detect(g2), tick=2)
    assert "APPEARED" in _kinds(ev2)
    assert "VANISHED" in _kinds(ev2)


def test_twin_swap_does_not_crash():
    # KNOWN-LIMITATION: identical twins swapping in one tick may swap ids.
    # We only assert the tracker does not crash and keeps both nouns alive.
    tr = ObjectTracker()
    g1 = np.zeros((16, 16), dtype=int)
    g1[5, 3] = 1
    g1[5, 11] = 1
    g2 = np.zeros((16, 16), dtype=int)
    g2[5, 11] = 1   # they swap positions
    g2[5, 3] = 1

    tr.update(_detect(g1), tick=1)
    vis2, _ = tr.update(_detect(g2), tick=2)
    assert len(vis2) == 2  # no crash, both tracked


def test_legacy_match_still_works():
    tr = ObjectTracker()
    g = np.zeros((10, 10), dtype=int); g[2, 2] = 1; g[7, 7] = 2
    a = _detect(g)
    b = _detect(g)
    pairs = tr.match(a, b)
    assert len(pairs) == 2
