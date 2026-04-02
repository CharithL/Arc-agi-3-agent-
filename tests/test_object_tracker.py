"""Tests for ObjectTracker — persistent identity across frames."""
import numpy as np
import pytest
from charith.perception.core_knowledge import ObjectnessPrior
from charith.perception.object_tracker import ObjectTracker


class TestObjectTracker:
    def test_match_identical_objects(self):
        """Same objects in same position -> matched pairs."""
        grid = np.zeros((10, 10), dtype=int)
        grid[2:4, 2:4] = 1
        grid[7:9, 7:9] = 2

        prior = ObjectnessPrior()
        objects_a = prior.detect(grid)
        prior._next_id = 0
        objects_b = prior.detect(grid)

        tracker = ObjectTracker()
        pairs = tracker.match(objects_a, objects_b)
        assert len(pairs) == 2

    def test_match_moved_object(self):
        """Object that moved one cell -> still matched by color+proximity."""
        grid1 = np.zeros((10, 10), dtype=int)
        grid1[3, 3] = 1

        grid2 = np.zeros((10, 10), dtype=int)
        grid2[3, 4] = 1

        prior = ObjectnessPrior()
        objects1 = prior.detect(grid1)
        prior._next_id = 0
        objects2 = prior.detect(grid2)

        tracker = ObjectTracker()
        pairs = tracker.match(objects1, objects2)
        assert len(pairs) == 1

    def test_match_detects_appeared_disappeared(self):
        """Object appears -> not in matched pairs."""
        grid1 = np.zeros((10, 10), dtype=int)
        grid1[3, 3] = 1

        grid2 = np.zeros((10, 10), dtype=int)
        grid2[3, 3] = 1
        grid2[7, 7] = 2  # new object

        prior = ObjectnessPrior()
        objects1 = prior.detect(grid1)
        prior._next_id = 0
        objects2 = prior.detect(grid2)

        tracker = ObjectTracker()
        pairs = tracker.match(objects1, objects2)
        assert len(pairs) == 1  # Only blue matched
