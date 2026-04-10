"""Tests for the shared has_valid_grid helper used across ALFA phases."""
import numpy as np

from charith.alfa_loop._frame_utils import has_valid_grid


class _Obs:
    """Minimal obs-like object with a .frame attribute."""
    def __init__(self, frame):
        self.frame = frame


def test_none_obs_is_invalid():
    assert has_valid_grid(None) is False


def test_obs_without_frame_attribute_is_invalid():
    class _NoFrame:
        pass
    assert has_valid_grid(_NoFrame()) is False


def test_obs_with_empty_frame_list_is_invalid():
    assert has_valid_grid(_Obs([])) is False


def test_obs_with_none_first_element_is_invalid():
    assert has_valid_grid(_Obs([None])) is False


def test_obs_with_valid_numpy_grid_is_valid():
    grid = np.zeros((10, 10), dtype=int)
    assert has_valid_grid(_Obs([grid])) is True


def test_obs_with_multiple_frames_only_checks_first():
    grid = np.zeros((10, 10), dtype=int)
    # Second frame is None but first is valid — still valid
    assert has_valid_grid(_Obs([grid, None])) is True
