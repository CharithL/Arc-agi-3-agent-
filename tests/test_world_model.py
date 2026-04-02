"""Tests for world model with object-level rules and relative context."""

import numpy as np
import pytest

from charith.perception.core_knowledge import (
    Cell,
    Object,
    SpatialRelation,
    StructuredPercept,
)
from charith.world_model.model import (
    ObjectEffect,
    PredictionError,
    TransitionRule,
    WorldModel,
)


# ---------------------------------------------------------------------------
# Helpers -- build minimal percepts for testing
# ---------------------------------------------------------------------------

def _make_object(object_id: int, color: int, centroid: tuple,
                 size: int = 1, shape_hash: int = 0,
                 cells: set = None, bbox: tuple = None) -> Object:
    """Build a minimal Object for testing."""
    r, c = int(centroid[0]), int(centroid[1])
    if cells is None:
        cells = {Cell(r, c, color)}
    if bbox is None:
        bbox = (r, c, r, c)
    return Object(
        object_id=object_id,
        cells=cells,
        color=color,
        bbox=bbox,
        size=size,
        centroid=centroid,
        shape_hash=shape_hash,
    )


def _make_percept(objects: list, grid_dims: tuple = (5, 5),
                  background_color: int = 0,
                  spatial_relations: list = None) -> StructuredPercept:
    """Build a minimal StructuredPercept for testing."""
    grid = np.zeros(grid_dims, dtype=int)
    if spatial_relations is None:
        spatial_relations = []
    unique_colors = {o.color for o in objects} | {background_color}
    return StructuredPercept(
        raw_grid=grid,
        objects=objects,
        spatial_relations=spatial_relations,
        color_counts={c: 1 for c in unique_colors},
        grid_dims=grid_dims,
        background_color=background_color,
        symmetry={"h_symmetric": False, "v_symmetric": False,
                  "rot_90": False, "rot_180": False},
        unique_colors=unique_colors,
        object_count=len(objects),
        timestamp=0.0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestWorldModelLearnsSimpleTransition:
    """test_world_model_learns_simple_transition: observe same action->effect
    3 times, predict returns it."""

    def test_world_model_learns_simple_transition(self):
        wm = WorldModel()
        context = {"background_color": 0, "object_count": 1,
                   "unique_colors": {0, 1}}
        effects = [ObjectEffect(
            object_color=1, displacement=(0, 1),
            shape_changed=False, size_delta=0,
            appeared=False, disappeared=False,
        )]
        # Observe the same transition 3 times
        for tick in range(3):
            wm.update(action=3, context=context,
                      observed_effects=effects, tick=tick)

        prediction = wm.predict(action=3, context=context)
        assert prediction is not None
        assert len(prediction) == 1
        assert prediction[0].displacement == (0, 1)
        assert prediction[0].object_color == 1


class TestWorldModelNoPredictionForUnseen:
    """test_world_model_no_prediction_for_unseen: unseen action returns None."""

    def test_world_model_no_prediction_for_unseen(self):
        wm = WorldModel()
        context = {"background_color": 0, "object_count": 1,
                   "unique_colors": {0, 1}}
        # Never trained on action 7
        prediction = wm.predict(action=7, context=context)
        assert prediction is None


class TestWorldModelContextMatters:
    """test_world_model_context_matters: different context -> different
    prediction for same action."""

    def test_world_model_context_matters(self):
        wm = WorldModel()

        # Context A: object near top, no wall right
        context_a = {"background_color": 0, "object_count": 1,
                     "ctrl_near_top": True, "ctrl_near_bottom": False}
        effects_a = [ObjectEffect(
            object_color=1, displacement=(0, 1),
            shape_changed=False, size_delta=0,
            appeared=False, disappeared=False,
        )]

        # Context B: object near bottom, wall right
        context_b = {"background_color": 0, "object_count": 1,
                     "ctrl_near_top": False, "ctrl_near_bottom": True}
        effects_b = [ObjectEffect(
            object_color=1, displacement=(0, 0),
            shape_changed=False, size_delta=0,
            appeared=False, disappeared=False,
        )]

        # Train both
        wm.update(action=3, context=context_a,
                  observed_effects=effects_a, tick=0)
        wm.update(action=3, context=context_b,
                  observed_effects=effects_b, tick=1)

        # Predict with context A -> displacement (0,1)
        pred_a = wm.predict(action=3, context=context_a)
        assert pred_a is not None
        assert pred_a[0].displacement == (0, 1)

        # Predict with context B -> displacement (0,0)
        pred_b = wm.predict(action=3, context=context_b)
        assert pred_b is not None
        assert pred_b[0].displacement == (0, 0)


class TestWorldModelRelativeContextOnly:
    """test_world_model_relative_context_only: extract_context has no
    forbidden keys (grid_rows, grid_cols, ctrl_row, ctrl_col,
    absolute_x, absolute_y)."""

    def test_world_model_relative_context_only(self):
        wm = WorldModel()
        obj = _make_object(object_id=0, color=1, centroid=(1.0, 2.0), size=1)
        percept = _make_percept(
            objects=[obj],
            grid_dims=(5, 5),
            spatial_relations=[
                SpatialRelation(obj_a_id=0, obj_b_id=1,
                                relation="right", distance=1.0),
            ],
        )
        controllable_ids = {0}
        context = wm.extract_context(percept, controllable_ids)

        # Must not contain any forbidden keys
        for forbidden in WorldModel.FORBIDDEN_KEYS:
            assert forbidden not in context, (
                f"Forbidden key '{forbidden}' found in context"
            )

        # Should contain relative features
        assert "background_color" in context
        assert "object_count" in context
        assert "unique_colors" in context


class TestComputeEffectsDetectsMovement:
    """test_compute_effects_detects_movement: object moved right ->
    displacement (0, 1)."""

    def test_compute_effects_detects_movement(self):
        wm = WorldModel()

        prev_obj = _make_object(object_id=0, color=1,
                                centroid=(2.0, 2.0), size=1, shape_hash=42)
        curr_obj = _make_object(object_id=1, color=1,
                                centroid=(2.0, 3.0), size=1, shape_hash=42)

        prev_percept = _make_percept([prev_obj])
        curr_percept = _make_percept([curr_obj])
        matched_pairs = [(0, 1)]  # prev_id=0 matched to curr_id=1

        effects = wm.compute_effects(prev_percept, curr_percept,
                                     matched_pairs)
        assert len(effects) == 1
        assert effects[0].displacement == (0, 1)
        assert effects[0].object_color == 1
        assert effects[0].appeared is False
        assert effects[0].disappeared is False
        assert effects[0].shape_changed is False


class TestComputeEffectsDetectsAppearance:
    """test_compute_effects_detects_appearance: new object -> appeared=True."""

    def test_compute_effects_detects_appearance(self):
        wm = WorldModel()

        prev_obj = _make_object(object_id=0, color=1,
                                centroid=(2.0, 2.0), size=1)
        curr_obj_existing = _make_object(object_id=1, color=1,
                                         centroid=(2.0, 2.0), size=1)
        curr_obj_new = _make_object(object_id=2, color=3,
                                    centroid=(4.0, 4.0), size=2)

        prev_percept = _make_percept([prev_obj])
        curr_percept = _make_percept([curr_obj_existing, curr_obj_new])
        matched_pairs = [(0, 1)]  # Only obj 0->1 matched; obj 2 is new

        effects = wm.compute_effects(prev_percept, curr_percept,
                                     matched_pairs)

        appeared_effects = [e for e in effects if e.appeared]
        assert len(appeared_effects) == 1
        assert appeared_effects[0].object_color == 3
        assert appeared_effects[0].appeared is True


class TestWorldModelConfidenceDecayOnReset:
    """test_world_model_confidence_decay_on_reset: soft reset multiplies
    confidence by 0.8."""

    def test_world_model_confidence_decay_on_reset(self):
        wm = WorldModel()
        context = {"background_color": 0, "object_count": 1}
        effects = [ObjectEffect(
            object_color=1, displacement=(0, 1),
            shape_changed=False, size_delta=0,
            appeared=False, disappeared=False,
        )]
        wm.update(action=3, context=context,
                  observed_effects=effects, tick=0)

        # Confidence should be 1.0 initially
        rules = wm._rules[3]
        assert len(rules) == 1
        assert rules[0].confidence == 1.0

        # Soft reset -> confidence * 0.8
        wm.reset()
        assert rules[0].confidence == pytest.approx(0.8)

        # Two resets -> 0.8 * 0.8 = 0.64
        wm.reset()
        assert rules[0].confidence == pytest.approx(0.64)


class TestWorldModelErrorHistory:
    """test_world_model_error_history: record_error populates history."""

    def test_world_model_error_history(self):
        wm = WorldModel()
        assert wm.get_recent_errors() == []

        err1 = PredictionError(
            predicted_grid=None, observed_grid=None,
            error_magnitude=0.5, error_cells=[(0, 0)],
            precision=0.9, weighted_error=0.45,
            is_novel=True,
        )
        err2 = PredictionError(
            predicted_grid=None, observed_grid=None,
            error_magnitude=0.2, error_cells=[],
            precision=0.95, weighted_error=0.19,
            is_novel=False,
        )

        wm.record_error(err1)
        wm.record_error(err2)

        errors = wm.get_recent_errors()
        assert len(errors) == 2
        assert errors[0] == pytest.approx(0.45)
        assert errors[1] == pytest.approx(0.19)
