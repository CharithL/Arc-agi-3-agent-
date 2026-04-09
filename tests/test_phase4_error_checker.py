"""Tests for Phase 4 — ErrorChecker."""
from charith.causal_engine.table_model import ArcTableModel
from charith.causal_engine.error_analyzer import ArcErrorAnalyzer
from charith.alfa_loop.error_checker import ErrorChecker
from tests.fixtures.mock_llm import MockOllamaReasoner


def test_no_structure_returns_no_expansion():
    table = ArcTableModel(num_actions=8)
    analyzer = ArcErrorAnalyzer()
    for i in range(30):
        analyzer.record(step=i, action=(i % 4) + 1, predicted_right=True, prev_action=None)
    llm = MockOllamaReasoner()
    checker = ErrorChecker(table, analyzer, llm)
    result = checker.check()
    assert result["expanded"] is False
    assert llm.call_count == 0


def test_kruskal_fires_triggers_llm_and_expansion():
    table = ArcTableModel(num_actions=8)
    analyzer = ArcErrorAnalyzer()
    for i in range(40):
        prev = 1 if i % 2 == 0 else 2
        correct = (prev == 2)
        analyzer.record(step=i, action=3, predicted_right=correct, prev_action=prev)
    llm = MockOllamaReasoner(expansion_response={"type": "sequential", "reason": "Kruskal fired"})
    checker = ErrorChecker(table, analyzer, llm)
    result = checker.check()
    assert result["expanded"] is True
    assert result["expansion_type"] == "sequential"
    assert table.sequence_enabled is True
    assert llm.call_count == 1


def test_llm_suggests_none_no_expansion():
    table = ArcTableModel(num_actions=8)
    analyzer = ArcErrorAnalyzer()
    for i in range(40):
        prev = 1 if i % 2 == 0 else 2
        correct = (prev == 2)
        analyzer.record(step=i, action=3, predicted_right=correct, prev_action=prev)
    llm = MockOllamaReasoner(expansion_response={"type": "none", "reason": "noise"})
    checker = ErrorChecker(table, analyzer, llm)
    result = checker.check()
    assert result["expanded"] is False
