"""Tests for Phase 2 — Hypothesizer."""
from charith.alfa_loop.explorer import Evidence
from charith.alfa_loop.hypothesizer import Hypothesizer
from charith.full_stack.hypothesis_schema import ExpectedOutcome
from tests.fixtures.mock_llm import MockOllamaReasoner


def _fake_evidence() -> list:
    return [
        Evidence(
            action=i,
            percept_before=None,
            percept_after=None,
            changes=[],
            description=f"action {i}: no change",
            reward=0.0,
            done=False,
        )
        for i in range(1, 9)
    ]


def test_hypothesize_calls_llm_once():
    llm = MockOllamaReasoner()
    h = Hypothesizer(llm)
    hypotheses, goal = h.generate(_fake_evidence(), active_expansions=["single"])
    assert llm.call_count == 1
    assert len(hypotheses) >= 1
    assert isinstance(goal, str)


def test_hypothesize_parses_structured_expected_outcome():
    llm = MockOllamaReasoner()
    h = Hypothesizer(llm)
    hypotheses, _ = h.generate(_fake_evidence(), active_expansions=["single"])
    hyp = hypotheses[0]
    assert isinstance(hyp.expected, ExpectedOutcome)
    assert hyp.expected.direction == "up"
    assert hyp.expected.magnitude_cells == 5


def test_hypothesize_empty_llm_response_returns_empty_list():
    llm = MockOllamaReasoner(hypothesize_response={"hypotheses": []})
    h = Hypothesizer(llm)
    hypotheses, _ = h.generate(_fake_evidence(), active_expansions=["single"])
    assert hypotheses == []


def test_hypothesize_invalid_test_action_marked_untestable():
    llm = MockOllamaReasoner(
        hypothesize_response={
            "hypotheses": [
                {"rule": "bad", "confidence": 0.5, "test_action": 99,
                 "expected": {"direction": "up"}}
            ],
            "goal_guess": "x",
        }
    )
    h = Hypothesizer(llm, num_actions=8)
    hypotheses, _ = h.generate(_fake_evidence(), active_expansions=["single"])
    assert hypotheses[0].status == "untestable"
