"""Tests for hypothesis_schema dataclasses."""
import pytest
from charith.full_stack.hypothesis_schema import (
    ExpectedOutcome, Hypothesis, ActualObservation
)


def test_expected_outcome_defaults_all_none():
    e = ExpectedOutcome()
    assert e.direction is None
    assert e.magnitude_cells is None
    assert e.no_effect is False


def test_hypothesis_initial_status_untested():
    h = Hypothesis(
        rule="test rule", confidence=0.5, test_action=1,
        expected=ExpectedOutcome(direction="up")
    )
    assert h.status == "untested"
    assert h.match_score is None


def test_actual_observation_construction():
    a = ActualObservation(
        controllable_displacement=None,
        controllable_direction=None,
        controllable_magnitude=0,
        any_color_changes=[],
        new_objects=[],
        removed_objects=[],
        score_changed=False,
    )
    assert a.controllable_magnitude == 0
    assert a.score_changed is False
