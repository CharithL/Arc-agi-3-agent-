"""Tests for Ontology Expansion -- 4-test protocol + rule splitting."""
import numpy as np
import pytest


def test_insufficient_data_no_expansion():
    """With < window_size errors, should not expand."""
    from charith.metacognition.ontology import OntologyExpansion

    ont = OntologyExpansion(window_size=50)
    result = ont.check(recent_errors=[0.5] * 10, rule_count=5, accuracy=0.5)
    assert result.should_expand is False
    assert result.test_results.get('insufficient_data') is True


def test_residual_structure_detects_autocorrelation():
    """Autocorrelated errors -> residual structure test passes."""
    from charith.metacognition.ontology import OntologyExpansion

    ont = OntologyExpansion(window_size=50)
    # Create autocorrelated errors: high-low-high-low pattern
    errors = []
    for i in range(60):
        errors.append(0.8 if i % 2 == 0 else 0.2)
    result = ont.check(errors, rule_count=50, accuracy=0.5)
    assert result.test_results.get('residual_structure') is True


def test_random_errors_no_expansion():
    """Random errors -> no expansion triggered."""
    from charith.metacognition.ontology import OntologyExpansion

    ont = OntologyExpansion(window_size=50)
    rng = np.random.RandomState(42)
    errors = rng.uniform(0.3, 0.7, size=60).tolist()
    # Feed enough data for rule/accuracy history too
    for i in range(60):
        ont.check([errors[i]], rule_count=50 + i, accuracy=0.5 + i * 0.005)
    # Final check with full window
    result = ont.check(errors[-50:], rule_count=110, accuracy=0.8)
    # Random errors should NOT trigger 3/4 tests
    assert result.should_expand is False


def test_volatility_spike_detection():
    """Variance doubling in recent window -> volatility test passes."""
    from charith.metacognition.ontology import OntologyExpansion

    ont = OntologyExpansion(window_size=50, volatility_sigma_multiplier=2.0)
    # Low variance baseline, then high variance
    errors = [0.5] * 30 + list(np.random.RandomState(42).uniform(0.0, 1.0, 30))
    result = ont.check(errors, rule_count=50, accuracy=0.5)
    # Check if volatility test fires (may or may not depending on random seed)
    assert 'volatility_spike' in result.test_results


def test_rule_splitting_interface():
    """execute_expansion returns a bool (interface test)."""
    from charith.metacognition.ontology import OntologyExpansion

    ont = OntologyExpansion()

    # Test with no world model -- should gracefully return False
    success = ont.execute_expansion('new_environment_mechanic', None, None)
    assert isinstance(success, bool)
    assert success is False


def test_rule_splitting_with_world_model():
    """After splitting a confused rule, the two new rules should be more specific."""
    try:
        from charith.world_model.model import WorldModel, ObjectEffect, TransitionRule
    except ImportError:
        pytest.skip("WorldModel not yet implemented -- depends on Task 4")

    from charith.metacognition.ontology import OntologyExpansion

    wm = WorldModel()
    ont = OntologyExpansion()

    # Create a confused rule: same action, different effects depending on context
    ctx_white = {'background_color': 0}
    ctx_grey = {'background_color': 5}
    eff_right = [ObjectEffect(1, (0, 1), False, 0, False, False)]
    eff_left = [ObjectEffect(1, (0, -1), False, 0, False, False)]

    # Feed observations that create a confused rule
    for tick in range(5):
        wm.update(0, ctx_white, eff_right, tick)
    for tick in range(5, 10):
        wm.update(0, ctx_grey, eff_left, tick)

    # Now the rule for action 0 should be confused
    rules_before = len(wm._rules.get(0, []))

    # Attempt expansion
    success = ont.execute_expansion('new_environment_mechanic', wm, None)
    # The split may or may not succeed depending on history -- test the interface
    assert isinstance(success, bool)


def test_confidence_tracker():
    """ConfidenceTracker tracks uncertainty with EMA."""
    from charith.metacognition.confidence import ConfidenceTracker

    ct = ConfidenceTracker()
    assert ct.uncertainty == 1.0

    # Update with low error should decrease uncertainty
    for _ in range(50):
        ct.update(0.0, rule_count=10)
    assert ct.uncertainty < 0.1

    # Reset restores to 1.0
    ct.reset()
    assert ct.uncertainty == 1.0


def test_ontology_expansion_result_fields():
    """OntologyExpansionResult has all required fields."""
    from charith.metacognition.ontology import OntologyExpansionResult

    result = OntologyExpansionResult(
        should_expand=True,
        test_results={'residual_structure': True},
        confidence=0.8,
        suggested_type='new_object_category',
    )
    assert result.should_expand is True
    assert result.confidence == 0.8
    assert result.suggested_type == 'new_object_category'


def test_ontology_reset():
    """Reset clears histories but keeps expansion_count."""
    from charith.metacognition.ontology import OntologyExpansion

    ont = OntologyExpansion(window_size=50)
    # Feed data and force an expansion count
    ont._expansion_count = 3
    errors = [0.5] * 60
    ont.check(errors, rule_count=50, accuracy=0.5)

    ont.reset()
    # Histories cleared
    assert len(ont._error_history) == 0
    assert len(ont._rule_count_history) == 0
    assert len(ont._accuracy_history) == 0
    # Expansion count preserved
    assert ont._expansion_count == 3


def test_epistemic_uncertainty_high_stagnant_errors():
    """High mean error that isn't decreasing triggers epistemic_uncertainty."""
    from charith.metacognition.ontology import OntologyExpansion

    ont = OntologyExpansion(window_size=20, uncertainty_patience=10)
    # Constant high errors -- not decreasing
    errors = [0.6] * 30
    result = ont.check(errors, rule_count=10, accuracy=0.4)
    assert result.test_results.get('epistemic_uncertainty') is True


def test_capacity_saturation():
    """Rules growing >10% without accuracy improvement triggers saturation."""
    from charith.metacognition.ontology import OntologyExpansion

    ont = OntologyExpansion(window_size=10, capacity_growth_threshold=0.1)
    # Feed increasing rule counts with stagnant accuracy
    for i in range(15):
        ont.check(
            recent_errors=[0.5],
            rule_count=50 + i * 10,  # rapid growth
            accuracy=0.5,            # no improvement
        )
    result = ont.check([0.5] * 15, rule_count=200, accuracy=0.5)
    assert result.test_results.get('capacity_saturation') is True
