"""S1 tests — structured Decision schema, coherence validator, retry protocol."""
from dataclasses import dataclass
from typing import Optional

import pytest

from charith.config import PROFILE_PATH6
from charith.llm_agent.schema import (
    Decision,
    PredictedEffect,
    ActionChoice,
    validate_decision,
    sanitize_requests,
)
from charith.llm_agent.decision import (
    parse_decision,
    run_decision,
    IncidentLog,
    least_tried_fallback,
)


# ---- minimal bank stubs (S4 not built yet) --------------------------------

@dataclass
class _Binding:
    action_id: Optional[str]


@dataclass
class _Hyp:
    hid: str
    statement: str
    htype: str
    binding: Optional[_Binding]


def _bank_with_mapping(hid="h_003", action="3"):
    return {hid: _Hyp(hid, "ACTION 3 moves the agent RIGHT", "ACTION_MAPPING", _Binding(action))}


def _valid_decision(action_id="3", hid="h_003"):
    return Decision(
        tick=5,
        hypothesis_id=hid,
        predicted_effect=PredictedEffect(kind="MOVE", object_id="obj_1", detail="RIGHT by 1"),
        action=ActionChoice(action_id=action_id),
        confidence=0.6,
        notes="test the right-move mapping",
    )


# ---- roundtrip -------------------------------------------------------------

def test_roundtrip():
    d = _valid_decision()
    js = d.model_dump_json()
    d2 = Decision.model_validate_json(js)
    assert d2 == d


def test_parse_decision_from_prose_wrapped_json():
    raw = 'Sure! Here is my choice:\n{"tick": 1, "hypothesis_id": "EXPLORE", ' \
          '"predicted_effect": {"kind": "MOVE", "object_id": null, "detail": "up"}, ' \
          '"action": {"action_id": "1", "target": null}, "confidence": 0.3, ' \
          '"notes": "", "requests": []}\nHope that helps.'
    d = parse_decision(raw)
    assert d.action.action_id == "1"
    assert d.hypothesis_id == "EXPLORE"


# ---- validator rules -------------------------------------------------------

def test_illegal_action_rejected():
    d = _valid_decision(action_id="9")
    errors = validate_decision(d, bank=_bank_with_mapping(), legal_actions=["1", "2", "3", "4"])
    assert any(e.startswith("E-ILLEGAL-ACTION") for e in errors)


def test_valid_decision_passes():
    d = _valid_decision()
    errors = validate_decision(d, bank=_bank_with_mapping(), legal_actions=["1", "2", "3", "4"])
    assert errors == []


def test_action_mismatch_detected():
    # Cite h_003 (about action 3) but choose action 1.
    d = _valid_decision(action_id="1", hid="h_003")
    errors = validate_decision(d, bank=_bank_with_mapping(), legal_actions=["1", "2", "3", "4"])
    assert any(e.startswith("E-ACTION-MISMATCH") for e in errors)


def test_explore_bypasses_mapping_check():
    d = _valid_decision(action_id="1", hid="EXPLORE")
    errors = validate_decision(d, bank=_bank_with_mapping(), legal_actions=["1", "2", "3", "4"])
    assert errors == []


def test_unknown_hyp_rejected_when_bank_present():
    d = _valid_decision(hid="h_999")
    errors = validate_decision(d, bank=_bank_with_mapping(), legal_actions=["1", "2", "3", "4"])
    assert any(e.startswith("E-UNKNOWN-HYP") for e in errors)


def test_free_form_hyp_provisional_without_bank():
    d = _valid_decision(hid="whatever")
    errors = validate_decision(d, bank=None, legal_actions=["1", "2", "3", "4"])
    assert errors == []  # provisional pre-S4


def test_dead_object_rejected():
    d = _valid_decision()
    errors = validate_decision(
        d, bank=_bank_with_mapping(), legal_actions=["1", "2", "3", "4"],
        live_oids={"obj_2", "obj_5"},  # obj_1 not live
    )
    assert any(e.startswith("E-DEAD-OBJECT") for e in errors)


def test_unknown_target_rejected():
    d = Decision(
        tick=1, hypothesis_id="EXPLORE",
        predicted_effect=PredictedEffect(kind="APPEAR"),
        action=ActionChoice(action_id="CLICK", target="aff_99"),
        confidence=0.2,
    )
    errors = validate_decision(d, affordances=["aff_1", "aff_2"])
    assert any(e.startswith("E-UNKNOWN-TARGET") for e in errors)


def test_requests_are_advisory_never_error():
    d = _valid_decision()
    d.requests = ["garbage", "PROBE:rotation obj_1", "not a request"]
    errors = validate_decision(d, bank=_bank_with_mapping(), legal_actions=["1", "2", "3", "4"])
    assert errors == []
    kept, dropped = sanitize_requests(d.requests)
    assert kept == ["PROBE:rotation obj_1"]
    assert set(dropped) == {"garbage", "not a request"}


# ---- retry protocol --------------------------------------------------------

def test_action_mismatch_bounced_then_fixed():
    bank = _bank_with_mapping()
    responses = iter([
        # 1st: cites h_003 (action 3) but chooses 1 -> E-ACTION-MISMATCH
        '{"tick":5,"hypothesis_id":"h_003","predicted_effect":{"kind":"MOVE","object_id":null,'
        '"detail":"right"},"action":{"action_id":"1","target":null},"confidence":0.6,"notes":"","requests":[]}',
        # 2nd: corrected -> chooses 3
        '{"tick":5,"hypothesis_id":"h_003","predicted_effect":{"kind":"MOVE","object_id":null,'
        '"detail":"right"},"action":{"action_id":"3","target":null},"confidence":0.6,"notes":"","requests":[]}',
    ])
    seen_prompts = []

    def query(prompt):
        seen_prompts.append(prompt)
        return next(responses)

    result = run_decision(
        query, "BASE PROMPT", config=PROFILE_PATH6, bank=bank,
        legal_actions=["1", "2", "3", "4"],
        fallback=lambda: "1",
    )
    assert result.decision is not None
    assert result.decision.action.action_id == "3"
    assert result.attempts == 2
    # The retry prompt must surface the error to the model.
    assert "E-ACTION-MISMATCH" in seen_prompts[1]


def test_fallback_after_two_failures(tmp_path):
    bank = _bank_with_mapping()
    # Always mismatched -> never valid.
    bad = ('{"tick":5,"hypothesis_id":"h_003","predicted_effect":{"kind":"MOVE","object_id":null,'
           '"detail":"right"},"action":{"action_id":"1","target":null},"confidence":0.6,"notes":"","requests":[]}')
    incidents = IncidentLog(str(tmp_path / "run" / "incidents.log"))

    result = run_decision(
        lambda prompt: bad, "BASE", config=PROFILE_PATH6, bank=bank,
        legal_actions=["1", "2", "3", "4"],
        fallback=lambda: "2", incidents=incidents,
    )
    assert result.decision is None
    assert result.fallback_action == "2"
    assert result.attempts == 3  # max_decision_retries(2) + 1
    assert len(incidents.records) == 1
    assert incidents.records[0]["kind"] == "DECISION_FALLBACK"
    # Raw text was persisted to disk.
    content = (tmp_path / "run" / "incidents.log").read_text(encoding="utf-8")
    assert "DECISION_FALLBACK" in content


def test_parse_failure_triggers_retry_then_fallback():
    incidents = IncidentLog()
    result = run_decision(
        lambda prompt: "I refuse to output JSON.", "BASE", config=PROFILE_PATH6,
        legal_actions=["1", "2"], fallback=lambda: "1", incidents=incidents,
    )
    assert result.decision is None
    assert result.fallback_action == "1"
    assert any(e.startswith("E-PARSE") for e in result.errors)


def test_least_tried_fallback():
    assert least_tried_fallback(["1", "2", "3"], {"1": 5, "2": 0, "3": 2}) == "2"
    assert least_tried_fallback(["1", "2"], {}) == "1"
