"""S3 tests — comparator scoring table + ledger + gated expansion."""
import pytest

from charith.config import PROFILE_PATH6
from charith.llm_agent.schema import (
    Decision, PredictedEffect, ActionChoice,
)
from charith.perception.object_tracker import PerceptEvent
from charith.hum.comparator import score_prediction
from charith.hum.ledger import Ledger, ExpansionGate, gate_expansion


def _pe(kind, oid=None, detail=""):
    return PredictedEffect(kind=kind, object_id=oid, detail=detail)


def _ev(kind, oid="obj_1", **detail):
    return PerceptEvent(kind=kind, oid=oid, detail=detail)


# ---------------------------------------------------------------------------
# Comparator — one assertion per scoring-table row
# ---------------------------------------------------------------------------

def test_move_confirmed_same_direction():
    assert score_prediction(_pe("MOVE", "obj_1", "RIGHT by 1"),
                            [_ev("MOVED", "obj_1", dx=1.0, dy=0.0)], 0) == "CONFIRMED"


def test_move_refuted_other_direction():
    assert score_prediction(_pe("MOVE", "obj_1", "RIGHT by 1"),
                            [_ev("MOVED", "obj_1", dx=-1.0, dy=0.0)], 0) == "REFUTED"


def test_move_refuted_no_move():
    assert score_prediction(_pe("MOVE", "obj_1", "RIGHT"), [], 0) == "REFUTED"


def test_move_untestable_unknown_oid():
    assert score_prediction(_pe("MOVE", "obj_9", "RIGHT"),
                            [_ev("MOVED", "obj_1", dx=1.0, dy=0.0)], 0,
                            live_oids={"obj_1"}) == "UNTESTABLE"


def test_no_effect_confirmed():
    assert score_prediction(_pe("NO_EFFECT"), [], 0) == "CONFIRMED"


def test_no_effect_refuted_by_event():
    assert score_prediction(_pe("NO_EFFECT"), [_ev("MOVED", "obj_1", dx=1.0)], 0) == "REFUTED"


def test_no_effect_refuted_by_score():
    assert score_prediction(_pe("NO_EFFECT"), [], 5) == "REFUTED"


def test_color_change_confirmed_detail_matches():
    assert score_prediction(_pe("COLOR_CHANGE", "obj_1", "color 3 -> 7"),
                            [_ev("RECOLORED", "obj_1", **{"from": 3, "to": 7})], 0) == "CONFIRMED"


def test_color_change_partial_detail_differs():
    assert score_prediction(_pe("COLOR_CHANGE", "obj_1", "color 3 -> 9"),
                            [_ev("RECOLORED", "obj_1", **{"from": 3, "to": 7})], 0) == "PARTIAL"


def test_color_change_refuted_no_recolor():
    assert score_prediction(_pe("COLOR_CHANGE", "obj_1", "to 7"), [], 0) == "REFUTED"


def test_appear_confirmed():
    assert score_prediction(_pe("APPEAR"), [_ev("APPEARED", "obj_2")], 0) == "CONFIRMED"


def test_appear_refuted():
    assert score_prediction(_pe("APPEAR"), [], 0) == "REFUTED"


def test_disappear_confirmed():
    assert score_prediction(_pe("DISAPPEAR"), [_ev("VANISHED", "obj_2")], 0) == "CONFIRMED"


def test_score_change_confirmed():
    assert score_prediction(_pe("SCORE_CHANGE"), [], 3) == "CONFIRMED"


def test_score_change_refuted():
    assert score_prediction(_pe("SCORE_CHANGE"), [], 0) == "REFUTED"


def test_transform_confirmed():
    assert score_prediction(_pe("TRANSFORM", "obj_1"),
                            [_ev("TRANSFORMED", "obj_1", iou=0.2)], 0) == "CONFIRMED"


def test_transform_refuted():
    assert score_prediction(_pe("TRANSFORM", "obj_1"), [], 0) == "REFUTED"


def test_unknown_untestable():
    assert score_prediction(_pe("UNKNOWN"), [_ev("MOVED", "obj_1", dx=1.0)], 0) == "UNTESTABLE"


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------

def _decision(kind="MOVE", oid="obj_1", detail="RIGHT", action="3", hid="h1", tick=1):
    return Decision(
        tick=tick, hypothesis_id=hid,
        predicted_effect=_pe(kind, oid, detail),
        action=ActionChoice(action_id=action), confidence=0.5,
    )


def test_ledger_records_and_resolves():
    led = Ledger()
    led.record(1, _decision(detail="RIGHT"))
    rec = led.resolve(2, [_ev("MOVED", "obj_1", dx=1.0, dy=0.0)], 0)
    assert rec.outcome == "CONFIRMED"


def test_enforce_predictions_blocks_bypass():
    led = Ledger(enforce=True)
    with pytest.raises(RuntimeError):
        led.resolve(1, [_ev("MOVED", "obj_1", dx=1.0)], 0)


def test_no_enforce_returns_none():
    led = Ledger(enforce=False)
    assert led.resolve(1, [], 0) is None


def test_rolling_accuracy():
    led = Ledger()
    # 2 confirmed, 1 refuted -> 2/3
    led.record(1, _decision(detail="RIGHT"))
    led.resolve(1, [_ev("MOVED", "obj_1", dx=1.0)], 0)
    led.record(2, _decision(detail="RIGHT"))
    led.resolve(2, [_ev("MOVED", "obj_1", dx=1.0)], 0)
    led.record(3, _decision(detail="RIGHT"))
    led.resolve(3, [_ev("MOVED", "obj_1", dx=-1.0)], 0)  # refuted
    assert abs(led.rolling_accuracy() - 2 / 3) < 1e-9


def test_orphans_are_unexplained_events():
    led = Ledger()
    # Predict a MOVE; an *extra* TRANSFORMED event is unexplained -> orphan.
    led.record(1, _decision(kind="MOVE", oid="obj_1", detail="RIGHT"))
    led.resolve(1, [
        _ev("MOVED", "obj_1", dx=1.0, dy=0.0),      # explained by the MOVE
        _ev("TRANSFORMED", "obj_4", iou=0.3),       # orphan
    ], 0)
    orphans = led.orphans(window=6)
    kinds = {e.kind for e in orphans}
    assert kinds == {"TRANSFORMED"}


def test_orphan_window_expires():
    led = Ledger()
    led.record(1, _decision())
    led.resolve(1, [_ev("APPEARED", "obj_9")], 0)   # orphan at tick 1
    # advance many ticks with explained-only events
    led.record(10, _decision(detail="RIGHT"))
    led.resolve(10, [_ev("MOVED", "obj_1", dx=1.0)], 0)
    assert led.orphans(window=3) == []  # tick-1 orphan is outside the window


# ---------------------------------------------------------------------------
# Gated expansion
# ---------------------------------------------------------------------------

def test_expansion_denied_without_orphans():
    led = Ledger()
    led.record(1, _decision(detail="RIGHT"))
    led.resolve(1, [_ev("MOVED", "obj_1", dx=1.0)], 0)  # fully explained, no orphan
    granted, msg = gate_expansion("GRAVITY", led, PROFILE_PATH6, ticks_since_last_expansion=100)
    assert granted is False
    assert "no unexplained events" in msg


def test_expansion_granted_after_orphan():
    led = Ledger()
    led.record(1, _decision(kind="MOVE", oid="obj_1", detail="RIGHT"))
    led.resolve(1, [_ev("MOVED", "obj_1", dx=1.0), _ev("TRANSFORMED", "obj_4", iou=0.2)], 0)
    granted, msg = gate_expansion("ROTATION", led, PROFILE_PATH6, ticks_since_last_expansion=100)
    assert granted is True
    assert msg == ""


def test_expansion_respects_cooldown():
    cfg = PROFILE_PATH6
    led = Ledger()
    led.record(1, _decision(kind="MOVE", detail="RIGHT"))
    led.resolve(1, [_ev("MOVED", "obj_1", dx=1.0), _ev("TRANSFORMED", "obj_4", iou=0.2)], 0)
    granted, msg = gate_expansion("ROTATION", led, cfg,
                                  ticks_since_last_expansion=cfg.expansion_cooldown_ticks - 1)
    assert granted is False
    assert "cooldown" in msg


def test_expansion_gate_tracks_last_tick():
    gate = ExpansionGate(config=PROFILE_PATH6)
    led = Ledger()
    led.record(1, _decision(kind="MOVE", detail="RIGHT"))
    led.resolve(1, [_ev("MOVED", "obj_1", dx=1.0), _ev("TRANSFORMED", "obj_4", iou=0.2)], 0)
    # First expansion at tick 20 (cooldown satisfied vs default -10000).
    granted, _ = gate.try_expand("ROTATION", led, tick=20)
    assert granted is True
    # Immediate second attempt is inside cooldown.
    granted2, msg2 = gate.try_expand("MIRROR", led, tick=21)
    assert granted2 is False and "cooldown" in msg2


def test_expansion_denied_for_alias():
    led = Ledger()
    led.record(1, _decision(kind="MOVE", detail="RIGHT"))
    led.resolve(1, [_ev("MOVED", "obj_1", dx=1.0), _ev("TRANSFORMED", "obj_4", iou=0.2)], 0)
    granted, msg = gate_expansion("WALL2", led, PROFILE_PATH6, 100,
                                  is_alias=lambda c: c == "WALL2")
    assert granted is False and "alias" in msg
