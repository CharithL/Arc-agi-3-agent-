"""Tests for StructuredPercept diff adapter."""
import numpy as np
from charith.perception.core_knowledge import Object, StructuredPercept
from charith.full_stack.percept_diff import diff_to_actual_observation


def _make_percept(objects):
    return StructuredPercept(
        raw_grid=np.zeros((10, 10), dtype=int),
        objects=objects,
        spatial_relations=[],
        color_counts={},
        grid_dims=(10, 10),
        background_color=0,
        symmetry={},
        unique_colors=set(),
        object_count=len(objects),
        timestamp=0.0,
    )


def _make_obj(oid, color, centroid):
    return Object(
        object_id=oid,
        cells=frozenset(),
        color=color,
        bbox=(0, 0, 0, 0),
        size=1,
        centroid=centroid,
        shape_hash=0,
    )


def test_diff_no_change_gives_empty_observation():
    obj = _make_obj(1, 5, (3.0, 3.0))
    before = _make_percept([obj])
    after = _make_percept([obj])
    actual = diff_to_actual_observation(before, after)
    assert actual.controllable_magnitude == 0
    assert actual.any_color_changes == []
    assert actual.new_objects == []
    assert actual.removed_objects == []


def test_diff_detects_upward_movement():
    before = _make_percept([_make_obj(1, 5, (8.0, 4.0))])
    after = _make_percept([_make_obj(1, 5, (3.0, 4.0))])   # row decreased → up
    actual = diff_to_actual_observation(before, after)
    assert actual.controllable_direction == "up"
    assert actual.controllable_magnitude == 5


def test_diff_detects_rightward_movement():
    before = _make_percept([_make_obj(1, 5, (4.0, 2.0))])
    after = _make_percept([_make_obj(1, 5, (4.0, 7.0))])
    actual = diff_to_actual_observation(before, after)
    assert actual.controllable_direction == "right"
    assert actual.controllable_magnitude == 5


def test_diff_detects_new_object():
    before = _make_percept([_make_obj(1, 5, (3.0, 3.0))])
    after = _make_percept([
        _make_obj(1, 5, (3.0, 3.0)),
        _make_obj(2, 7, (7.0, 7.0)),   # new object, different color
    ])
    actual = diff_to_actual_observation(before, after)
    assert len(actual.new_objects) == 1


def test_diff_detects_removed_object():
    before = _make_percept([
        _make_obj(1, 5, (3.0, 3.0)),
        _make_obj(2, 7, (7.0, 7.0)),
    ])
    after = _make_percept([_make_obj(1, 5, (3.0, 3.0))])
    actual = diff_to_actual_observation(before, after)
    assert len(actual.removed_objects) == 1
