"""Tests for ArcErrorAnalyzer — ported from c1c2-hybrid."""
import pytest
from charith.causal_engine.error_analyzer import ArcErrorAnalyzer


def test_insufficient_data_under_20():
    a = ArcErrorAnalyzer()
    for i in range(10):
        a.record(step=i, action=1, predicted_right=True, prev_action=None)
    result = a.analyze()
    assert result["sufficient_data"] is False


def test_no_structure_when_all_correct():
    a = ArcErrorAnalyzer()
    for i in range(30):
        a.record(step=i, action=(i % 4) + 1, predicted_right=True,
                 prev_action=(i - 1) % 4 + 1 if i > 0 else None)
    result = a.analyze()
    assert result["sufficient_data"] is True
    assert result["error_rate"] == 0.0
    assert result["any_structure"] is False


def test_kruskal_fires_on_prev_clustered_errors():
    """When errors cluster by previous action, Kruskal-by-prev should fire."""
    a = ArcErrorAnalyzer()
    # 40 observations: errors only when prev_action == 1
    for i in range(40):
        prev = 1 if i % 2 == 0 else 2
        # Errors concentrated where prev=1, never where prev=2
        correct = (prev == 2)
        a.record(step=i, action=3, predicted_right=correct, prev_action=prev)
    result = a.analyze()
    assert result["sufficient_data"] is True
    # Kruskal should detect the cluster
    assert result["kruskal_fires"] is True
    assert result["any_structure"] is True


def test_summary_always_shows_all_four_tests():
    """The LLM-facing summary must contain all 4 test names even when none fire."""
    a = ArcErrorAnalyzer()
    for i in range(25):
        a.record(step=i, action=1, predicted_right=True, prev_action=None)
    result = a.analyze()
    summary = result["summary"]
    assert "Ljung-Box" in summary
    assert "Kruskal by PREVIOUS action" in summary
    assert "Kruskal by CURRENT cell" in summary  # ported verbatim
    assert "Variance ratio" in summary
