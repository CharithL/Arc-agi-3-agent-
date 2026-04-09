"""Smoke test for MockOllamaReasoner."""
from tests.fixtures.mock_llm import MockOllamaReasoner, DEFAULT_HYPOTHESIZE_RESPONSE


def test_mock_llm_returns_fixed_hypothesize_response():
    llm = MockOllamaReasoner()
    result = llm.reason_json("You are discovering the rules...", "evidence here")
    assert "hypotheses" in result
    assert llm.call_count == 1


def test_mock_llm_routes_by_keyword():
    llm = MockOllamaReasoner(
        expansion_response={"type": "sequential", "reason": "test"},
        plan_response={"plan": [1, 2, 3], "reasoning": "r"},
    )
    exp = llm.reason_json("Available expansions: sequential...", "user")
    assert exp["type"] == "sequential"
    plan = llm.reason_json("You are planning actions...", "user")
    assert plan["plan"] == [1, 2, 3]
    assert llm.call_count == 2


def test_mock_llm_records_call_log():
    llm = MockOllamaReasoner()
    llm.reason_json("sys1", "user1")
    llm.reason_json("sys2", "user2")
    assert len(llm.calls) == 2
    assert llm.calls[0] == ("sys1", "user1")
