"""
Golden tests for compute_match_score.
Locks in the semantics from the design doc §5.3–5.4.
"""
import pytest
from charith.full_stack.hypothesis_schema import ExpectedOutcome, ActualObservation
from charith.full_stack.match_score import compute_match_score


def _empty_actual() -> ActualObservation:
    return ActualObservation(
        controllable_displacement=None,
        controllable_direction=None,
        controllable_magnitude=0,
        any_color_changes=[],
        new_objects=[],
        removed_objects=[],
        score_changed=False,
    )


def test_no_effect_perfect_match():
    expected = ExpectedOutcome(no_effect=True)
    actual = _empty_actual()
    assert compute_match_score(expected, actual) == 1.0


def test_no_effect_violated():
    expected = ExpectedOutcome(no_effect=True)
    actual = _empty_actual()
    actual.controllable_displacement = (-5, 0)
    actual.controllable_direction = "up"
    actual.controllable_magnitude = 5
    assert compute_match_score(expected, actual) == 0.0


def test_direction_match_magnitude_off_by_one_tolerated():
    expected = ExpectedOutcome(direction="up", magnitude_cells=5)
    actual = _empty_actual()
    actual.controllable_direction = "up"
    actual.controllable_magnitude = 4  # off by 1
    assert compute_match_score(expected, actual) == 1.0


def test_direction_match_magnitude_off_by_two_fails_magnitude():
    expected = ExpectedOutcome(direction="up", magnitude_cells=5)
    actual = _empty_actual()
    actual.controllable_direction = "up"
    actual.controllable_magnitude = 3  # off by 2
    # direction: 1.0, magnitude: 0.0, mean: 0.5
    assert compute_match_score(expected, actual) == 0.5


def test_mean_aggregation_three_fields():
    expected = ExpectedOutcome(direction="up", magnitude_cells=5, score_change=True)
    actual = _empty_actual()
    actual.controllable_direction = "up"        # match
    actual.controllable_magnitude = 5           # match
    actual.score_changed = False                # miss
    # 1 + 1 + 0 = 2/3 ≈ 0.667
    assert abs(compute_match_score(expected, actual) - 2/3) < 0.01


def test_empty_expected_returns_neutral():
    expected = ExpectedOutcome()  # all None, no_effect=False
    actual = _empty_actual()
    assert compute_match_score(expected, actual) == 0.5


def test_color_match_case_insensitive():
    expected = ExpectedOutcome(color_change_to="Green")
    actual = _empty_actual()
    actual.any_color_changes = [(None, 0, "green")]
    assert compute_match_score(expected, actual) == 1.0
