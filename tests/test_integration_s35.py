"""S3.5 Integration Checkpoint — the falsification engine, live in act().

Offline/deterministic: the mock backend speaks Decision (guide §2.3).
"""
import json

import numpy as np
import pytest

from charith.config import PROFILE_PATH6, PROFILE_LEGACY, AgentConfig, config_from_cli
from charith.llm_agent.agent import LLMAgent
from charith.llm_agent.llm_client import LLMClient, estimate_tokens
from charith.llm_agent.decision import IncidentLog
from charith.eval.episode import EpisodeRecorder, load_episode


def _agent(config=PROFILE_PATH6):
    a = LLMAgent(config=config)
    a.llm._backend = "mock"
    return a


def _grid(r, c):
    g = np.zeros((12, 12), dtype=int)
    g[r, c] = 1
    return g


# ---------------------------------------------------------------------------
# Core wiring
# ---------------------------------------------------------------------------

def test_path6_act_returns_legal_action():
    a = _agent()
    action = a.act(_grid(3, 3), available_actions=[1, 2, 3, 4])
    assert action in [1, 2, 3, 4]


def test_ledger_records_every_tick():
    a = _agent()
    a.act(_grid(3, 3))
    a.act(_grid(3, 5))
    a.act(_grid(3, 7))
    assert len(a.ledger.records) == 3


def test_record_before_execute_no_divergence():
    """The executed action equals the ledger's recorded prediction action."""
    a = _agent()
    action = a.act(_grid(3, 3))
    last = a.ledger.records[-1]
    assert int(last.action_id) == action


def test_stable_oid_preserved_across_move():
    a = _agent()
    a.act(_grid(3, 3))
    oid1 = set(a._last_tracked_oids)
    a.act(_grid(3, 5))  # same object moved right
    oid2 = set(a._last_tracked_oids)
    assert oid1 == oid2 and len(oid1) == 1


def test_previous_prediction_resolved():
    a = _agent()
    a.act(_grid(3, 3))          # tick1: records UNKNOWN prediction
    a.act(_grid(3, 5))          # tick2: resolves tick1
    outcomes = [r.outcome for r in a.ledger.records]
    # Mock predicts UNKNOWN -> UNTESTABLE (excluded from rolling_accuracy).
    assert outcomes[0] == "UNTESTABLE"
    assert a.ledger.rolling_accuracy() == 0.0  # no CONFIRMED/REFUTED yet


def test_no_round_robin_forced_exploration_under_path6():
    """Legacy forced round-robin is disabled: action is the model's choice, not a cycle."""
    a = _agent()
    a1 = a.act(_grid(3, 3), available_actions=[1, 2, 3, 4])
    a2 = a.act(_grid(3, 5), available_actions=[1, 2, 3, 4])
    a3 = a.act(_grid(3, 7), available_actions=[1, 2, 3, 4])
    # Mock always chooses the first legal action; a round-robin would cycle 1,2,3.
    assert a1 == a2 == a3 == 1


def test_context_history_preserved():
    a = _agent()
    a.act(_grid(3, 3))
    a.act(_grid(3, 5))
    assert len(a.context.full_history) == 2


# ---------------------------------------------------------------------------
# Legacy path untouched
# ---------------------------------------------------------------------------

def test_legacy_path_does_not_touch_ledger():
    a = _agent(PROFILE_LEGACY)
    action = a.act(_grid(3, 3))
    assert action in [1, 2, 3, 4]
    assert len(a.ledger.records) == 0  # legacy loop never records


# ---------------------------------------------------------------------------
# Fallback + incident logging
# ---------------------------------------------------------------------------

def test_invalid_llm_triggers_fallback_and_records():
    a = _agent()
    a.incidents = IncidentLog(None)
    # LLM returns unparseable text every time -> fallback fires.
    a.llm.query_structured = lambda *args, **kwargs: "I refuse to output JSON."
    action = a.act(_grid(3, 3), available_actions=[1, 2, 3, 4])
    assert action in [1, 2, 3, 4]
    # Fallback still records a prediction (record-before-execute invariant).
    assert len(a.ledger.records) == 1
    assert int(a.ledger.records[-1].action_id) == action
    assert any(r["kind"] == "DECISION_FALLBACK" for r in a.incidents.records)


# ---------------------------------------------------------------------------
# Gated expansion wired into the loop
# ---------------------------------------------------------------------------

def test_expand_request_denied_without_orphan():
    a = _agent()
    a.incidents = IncidentLog(None)

    def fake_query(*args, **kwargs):
        return json.dumps({
            "tick": 1, "hypothesis_id": "EXPLORE",
            "predicted_effect": {"kind": "UNKNOWN", "object_id": None, "detail": ""},
            "action": {"action_id": "1", "target": None},
            "confidence": 0.2, "notes": "",
            "requests": ["EXPAND:GRAVITY because things fell"],
        })

    a.llm.query_structured = fake_query
    a.act(_grid(3, 3), available_actions=[1, 2, 3, 4])

    names = {c.name.upper() for c in a.c1c2.c1_vocabulary}
    assert "GRAVITY" not in names  # denied — no orphan events
    assert any(r["kind"] == "EXPANSION_DENIED" for r in a.incidents.records)


# ---------------------------------------------------------------------------
# num_ctx + context guard + mock-speaks-Decision
# ---------------------------------------------------------------------------

def test_num_ctx_config_exists():
    assert PROFILE_PATH6.num_ctx == 16384
    cfg = config_from_cli("path6", ["num_ctx=8192"])
    assert cfg.num_ctx == 8192


def test_estimate_tokens():
    assert estimate_tokens("a" * 400) == 100


def test_context_guard_logs_overflow_incident():
    client = LLMClient.__new__(LLMClient)  # bypass backend probing
    client._backend = "mock"
    client._call_count = 0
    inc = IncidentLog(None)
    cfg = AgentConfig(num_ctx=100)  # tiny window
    big_user = "x" * 4000          # ~1000 tokens >> 0.75*100
    out = client.query_structured("sys", big_user, cfg, [1, 2, 3], incidents=inc)
    assert any(r["kind"] == "CONTEXT_OVERFLOW" for r in inc.records)
    # mock still returns a valid Decision JSON
    from charith.llm_agent.decision import parse_decision
    d = parse_decision(out)
    assert d.hypothesis_id == "EXPLORE" and d.predicted_effect.kind == "UNKNOWN"


def test_mock_decision_picks_legal_action():
    client = LLMClient.__new__(LLMClient)
    client._backend = "mock"
    client._call_count = 0
    from charith.llm_agent.decision import parse_decision
    out = client.query_structured("s", "u", PROFILE_PATH6, [5, 6, 7])
    d = parse_decision(out)
    assert d.action.action_id == "5"


# ---------------------------------------------------------------------------
# Episode recorder (§2.6)
# ---------------------------------------------------------------------------

def test_episode_recorder_roundtrip(tmp_path):
    rec = EpisodeRecorder("ls20", str(tmp_path / "ep.json"))
    rec.add(1, _grid(3, 3), 2, [1, 2, 3, 4], 0)
    rec.add(2, _grid(3, 4), 1, [1, 2, 3, 4], 5)
    path = rec.save()
    data = load_episode(path)
    assert data["game_id"] == "ls20"
    assert len(data["ticks"]) == 2
    assert data["ticks"][1]["action_taken"] == 1
    assert data["ticks"][1]["score"] == 5
    assert isinstance(data["ticks"][0]["grid"], np.ndarray)
