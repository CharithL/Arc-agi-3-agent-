"""Tests for core knowledge priors perception module."""

import numpy as np
import pytest

from charith.perception.core_knowledge import (
    AgencyPrior,
    Cell,
    Color,
    CoreKnowledgePerception,
    NumberPrior,
    Object,
    ObjectnessPrior,
    SpatialPrior,
    SpatialRelation,
    StructuredPercept,
)


# ---------------------------------------------------------------------------
# ObjectnessPrior tests
# ---------------------------------------------------------------------------

class TestObjectnessPrior:
    def test_objectness_single_object(self):
        """2x2 blue block on black background -> 1 object, color=1, size=4."""
        grid = np.array([
            [1, 1],
            [1, 1],
        ], dtype=int)
        prior = ObjectnessPrior()
        objects = prior.detect(grid, background_color=0)
        assert len(objects) == 1
        obj = objects[0]
        assert obj.color == Color.BLUE
        assert obj.size == 4

    def test_objectness_multiple_colors(self):
        """Blue pixel + red pixel on black background -> 2 objects."""
        grid = np.array([
            [1, 0],
            [0, 2],
        ], dtype=int)
        prior = ObjectnessPrior()
        objects = prior.detect(grid, background_color=0)
        assert len(objects) == 2
        colors = {obj.color for obj in objects}
        assert Color.BLUE in colors
        assert Color.RED in colors

    def test_objectness_diagonal_not_connected(self):
        """Diagonal same-color pixels -> 2 objects with 4-connectivity."""
        grid = np.array([
            [1, 0],
            [0, 1],
        ], dtype=int)
        prior = ObjectnessPrior(connectivity=4)
        objects = prior.detect(grid, background_color=0)
        assert len(objects) == 2

    def test_objectness_8_connectivity(self):
        """Diagonal same-color pixels with connectivity=8 -> 1 object."""
        grid = np.array([
            [1, 0],
            [0, 1],
        ], dtype=int)
        prior = ObjectnessPrior(connectivity=8)
        objects = prior.detect(grid, background_color=0)
        assert len(objects) == 1

    def test_objectness_shape_hash_position_invariant(self):
        """Same shape at different positions -> same shape_hash."""
        grid1 = np.array([
            [1, 1, 0],
            [1, 0, 0],
            [0, 0, 0],
        ], dtype=int)
        grid2 = np.array([
            [0, 0, 0],
            [0, 1, 1],
            [0, 1, 0],
        ], dtype=int)
        prior = ObjectnessPrior()
        objs1 = prior.detect(grid1, background_color=0)
        objs2 = prior.detect(grid2, background_color=0)
        assert len(objs1) == 1
        assert len(objs2) == 1
        assert objs1[0].shape_hash == objs2[0].shape_hash


# ---------------------------------------------------------------------------
# SpatialPrior tests
# ---------------------------------------------------------------------------

class TestSpatialPrior:
    def test_spatial_above_below(self):
        """Top vs bottom objects -> 'above' relation present."""
        grid = np.array([
            [1, 0],
            [0, 0],
            [0, 0],
            [0, 2],
        ], dtype=int)
        obj_prior = ObjectnessPrior()
        objects = obj_prior.detect(grid, background_color=0)
        spatial = SpatialPrior()
        relations = spatial.compute_relations(objects)
        relation_types = {r.relation for r in relations}
        assert "above" in relation_types
        assert "below" in relation_types

    def test_spatial_adjacency(self):
        """Touching objects -> 'adjacent' relation present."""
        grid = np.array([
            [1, 2],
            [0, 0],
        ], dtype=int)
        obj_prior = ObjectnessPrior()
        objects = obj_prior.detect(grid, background_color=0)
        spatial = SpatialPrior()
        relations = spatial.compute_relations(objects)
        relation_types = {r.relation for r in relations}
        assert "adjacent" in relation_types

    def test_spatial_symmetry_detection(self):
        """Symmetric grid -> h_symmetric=True, v_symmetric=True."""
        grid = np.array([
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1],
        ], dtype=int)
        spatial = SpatialPrior()
        sym = spatial.detect_grid_symmetry(grid)
        assert sym["h_symmetric"] is True
        assert sym["v_symmetric"] is True


# ---------------------------------------------------------------------------
# NumberPrior tests
# ---------------------------------------------------------------------------

class TestNumberPrior:
    def test_number_count_by_color(self):
        """Correct cell counts per color."""
        grid = np.array([
            [0, 1, 1],
            [2, 0, 1],
        ], dtype=int)
        number = NumberPrior()
        counts = number.count_by_color(grid)
        assert counts[0] == 2
        assert counts[1] == 3
        assert counts[2] == 1

    def test_number_periodic_detection(self):
        """[1,2,1,2,1,2] -> periodic=True."""
        number = NumberPrior()
        patterns = number.detect_numerical_patterns([1, 2, 1, 2, 1, 2])
        assert patterns["periodic"] is True

    def test_number_arithmetic_detection(self):
        """[2,4,6,8] -> arithmetic=True."""
        number = NumberPrior()
        patterns = number.detect_numerical_patterns([2, 4, 6, 8])
        assert patterns["arithmetic"] is True


# ---------------------------------------------------------------------------
# AgencyPrior tests
# ---------------------------------------------------------------------------

class TestAgencyPrior:
    def test_agency_record_contingency(self):
        """Recording contingency tracks changed cells."""
        agency = AgencyPrior()
        before = np.array([
            [0, 0],
            [0, 0],
        ], dtype=int)
        after = np.array([
            [0, 1],
            [0, 0],
        ], dtype=int)
        result = agency.record_action_contingency("move_right", before, after)
        assert len(result) > 0  # at least one cell changed


# ---------------------------------------------------------------------------
# Full pipeline test
# ---------------------------------------------------------------------------

class TestCoreKnowledgePerception:
    def test_full_perception_pipeline(self):
        """Grid -> StructuredPercept with correct fields."""
        grid = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 2, 0],
            [0, 0, 0, 0, 0],
        ], dtype=int)
        ckp = CoreKnowledgePerception()
        percept = ckp.perceive(grid)
        assert isinstance(percept, StructuredPercept)
        assert percept.grid_dims == (5, 5)
        assert percept.background_color == Color.BLACK
        assert percept.object_count >= 2
        assert len(percept.objects) >= 2
        assert isinstance(percept.raw_grid, np.ndarray)
        assert percept.unique_colors is not None
        assert percept.color_counts is not None
        assert percept.symmetry is not None

    def test_perception_reset(self):
        """Reset clears tick counter."""
        ckp = CoreKnowledgePerception()
        grid = np.array([[1, 0], [0, 2]], dtype=int)
        ckp.perceive(grid)
        assert ckp._tick > 0
        ckp.reset()
        assert ckp._tick == 0
