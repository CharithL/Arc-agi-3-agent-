"""Tests for ArcTableModel — ported from c1c2-hybrid."""
import pytest
from charith.causal_engine.table_model import ArcTableModel


def test_record_and_predict_single_action():
    t = ArcTableModel(num_actions=8)
    t.record(action=1, changes={"red_moved_up"})
    t.record(action=1, changes={"red_moved_up"})
    t.record(action=1, changes={"red_moved_up"})

    pred = t.predict(action=1, target="red_moved_up")
    assert pred[0] is True
    assert pred[1] == 1.0
    assert pred[2] == "single"


def test_predict_unseen_returns_unseen():
    t = ArcTableModel(num_actions=8)
    pred = t.predict(action=5, target="anything")
    assert pred == (False, 0.0, "unseen")


def test_enable_expansion_is_one_way():
    t = ArcTableModel(num_actions=8)
    assert t.enable_expansion("sequential", "test") is True
    assert t.sequence_enabled is True
    # Re-enabling returns False (already enabled)
    assert t.enable_expansion("sequential", "test2") is False


def test_get_active_expansions_reports_enabled():
    t = ArcTableModel(num_actions=8)
    assert t.get_active_expansions() == ["single"]
    t.enable_expansion("sequential", "test")
    assert "sequential" in t.get_active_expansions()
    t.enable_expansion("context", "test")
    assert "context" in t.get_active_expansions()


def test_sequence_prediction_when_enabled():
    t = ArcTableModel(num_actions=8)
    t.enable_expansion("sequential", "test")

    # Record sequence: action 1 -> action 2 -> target changes
    t.record(action=1, changes=set())
    t.record(action=2, changes={"effect_fired"})
    t.record(action=1, changes=set())
    t.record(action=2, changes={"effect_fired"})

    pred = t.predict(action=2, target="effect_fired", prev_action=1)
    assert pred[0] is True
    assert pred[2] == "sequence"
