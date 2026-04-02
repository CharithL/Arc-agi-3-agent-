"""
Ontology Expansion — detecting when the agent needs new concepts.

Derived from HIMARI's 4-test detection protocol:
1. Residual Structure: Can we predict our own errors?
2. Volatility Spike: Are errors getting wilder?
3. Epistemic Uncertainty: Is model uncertainty high and not decreasing?
4. Capacity Saturation: Is the model's rule count growing without accuracy gains?

Trigger: 3/4 tests positive -> expand representational vocabulary.

Amendment 2: Implements context-conditioned rule splitting — the core
ontology expansion mechanism that finds the context feature best
separating a confused rule's successes from failures.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from collections import deque, Counter
import numpy as np


@dataclass
class OntologyExpansionResult:
    """Result of the 4-test ontology gap detection."""
    should_expand: bool
    test_results: dict
    confidence: float
    suggested_type: str  # 'new_object_category', 'new_action_effect', 'new_environment_mechanic'


class OntologyExpansion:
    """
    Detects when the agent's representational vocabulary is insufficient.

    HIMARI principle: if prediction errors have learnable structure,
    the model is missing concepts, not just making noise.
    """

    def __init__(self,
                 window_size: int = 50,
                 residual_r2_threshold: float = 0.1,
                 volatility_sigma_multiplier: float = 2.0,
                 uncertainty_patience: int = 20,
                 capacity_growth_threshold: float = 0.1):
        self._window = window_size
        self._r2_thresh = residual_r2_threshold
        self._vol_sigma = volatility_sigma_multiplier
        self._uncertainty_patience = uncertainty_patience
        self._cap_thresh = capacity_growth_threshold

        self._error_history: deque = deque(maxlen=window_size * 2)
        self._rule_count_history: deque = deque(maxlen=window_size)
        self._accuracy_history: deque = deque(maxlen=window_size)
        self._expansion_count: int = 0

    def check(self, recent_errors: List[float],
              rule_count: int,
              accuracy: float) -> OntologyExpansionResult:
        """
        Run the 4-test protocol.

        Args:
            recent_errors: last N weighted prediction errors
            rule_count: current number of transition rules
            accuracy: current prediction accuracy

        Returns:
            OntologyExpansionResult with decision and details
        """
        self._error_history.extend(recent_errors)
        self._rule_count_history.append(rule_count)
        self._accuracy_history.append(accuracy)

        if len(self._error_history) < self._window:
            return OntologyExpansionResult(
                should_expand=False,
                test_results={'insufficient_data': True},
                confidence=0.0,
                suggested_type='none'
            )

        errors = np.array(list(self._error_history)[-self._window:])

        # Run 4 tests
        t1 = self._test_residual_structure(errors)
        t2 = self._test_volatility_spike(errors)
        t3 = self._test_epistemic_uncertainty(errors)
        t4 = self._test_capacity_saturation()

        tests = {
            'residual_structure': bool(t1),
            'volatility_spike': bool(t2),
            'epistemic_uncertainty': bool(t3),
            'capacity_saturation': bool(t4),
        }
        positive = sum(tests.values())

        should_expand = bool(positive >= 3)
        if should_expand:
            self._expansion_count += 1

        return OntologyExpansionResult(
            should_expand=should_expand,
            test_results=tests,
            confidence=float(positive / 4.0),
            suggested_type=self._suggest_expansion_type(errors, tests)
        )

    def _test_residual_structure(self, errors: np.ndarray) -> bool:
        """
        Test 1: Can we predict our own prediction errors?

        If a simple linear model on error_t predicts error_{t+1},
        the errors have learnable structure -> missing concepts.
        """
        if len(errors) < 10:
            return False

        X = errors[:-1]
        y = errors[1:]

        X_mean = X.mean()
        y_mean = y.mean()
        X_centered = X - X_mean
        y_centered = y - y_mean

        numerator = np.sum(X_centered * y_centered)
        denominator = np.sum(X_centered ** 2) + 1e-8

        slope = numerator / denominator
        intercept = y_mean - slope * X_mean

        predictions = slope * X + intercept
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2) + 1e-8
        r2 = 1 - ss_res / ss_tot

        return r2 > self._r2_thresh

    def _test_volatility_spike(self, errors: np.ndarray) -> bool:
        """
        Test 2: Are prediction errors getting increasingly variable?

        Compare variance of recent errors to baseline.
        """
        if len(errors) < self._window:
            return False

        half = len(errors) // 2
        baseline_var = np.var(errors[:half]) + 1e-8
        recent_var = np.var(errors[half:])

        return recent_var > self._vol_sigma * baseline_var

    def _test_epistemic_uncertainty(self, errors: np.ndarray) -> bool:
        """
        Test 3: Is mean error high and NOT decreasing?

        If the model has been learning, errors should decrease.
        Stagnant high errors = stuck, need new concepts.
        """
        if len(errors) < self._uncertainty_patience:
            return False

        recent = errors[-self._uncertainty_patience:]
        first_half = np.mean(recent[:len(recent) // 2])
        second_half = np.mean(recent[len(recent) // 2:])

        # High error AND not improving
        return second_half > 0.2 and second_half >= first_half * 0.95

    def _test_capacity_saturation(self) -> bool:
        """
        Test 4: Is rule count growing without accuracy improvement?

        More rules without better predictions = representational
        vocabulary growing in the wrong direction.
        """
        if len(self._rule_count_history) < 10 or len(self._accuracy_history) < 10:
            return False

        rc = list(self._rule_count_history)
        ac = list(self._accuracy_history)

        # Rules growing?
        rule_growth = (rc[-1] - rc[0]) / max(rc[0], 1)
        # Accuracy stagnant?
        acc_change = ac[-1] - ac[0]

        return rule_growth > self._cap_thresh and acc_change < 0.05

    def _suggest_expansion_type(self, errors: np.ndarray,
                                 tests: dict) -> str:
        """Suggest what kind of concept is missing."""
        if tests.get('residual_structure') and tests.get('volatility_spike'):
            return 'new_environment_mechanic'
        elif tests.get('capacity_saturation'):
            return 'new_object_category'
        elif tests.get('epistemic_uncertainty'):
            return 'new_action_effect'
        return 'unknown'

    def execute_expansion(self, expansion_type: str,
                          world_model, perception) -> bool:
        """
        Execute ontology expansion by splitting confused rules.

        Amendment 2: Find the rule that is most often wrong,
        then split it by the context feature that best separates
        successes from failures.
        """
        if world_model is None:
            return False

        # Find the most confused rule (high total, low confidence)
        most_confused = None
        worst_confidence = 1.0

        rules_dict = getattr(world_model, '_rules', {})
        for action, rules in rules_dict.items():
            for rule in rules:
                if rule.total >= 5 and rule.confidence < worst_confidence:
                    worst_confidence = rule.confidence
                    most_confused = (action, rule)

        if most_confused is None or worst_confidence > 0.7:
            return False

        action, confused_rule = most_confused
        return self._split_rule_by_context(world_model, action, confused_rule)

    def _split_rule_by_context(self, world_model, action: int, rule) -> bool:
        """
        Split a confused rule into two context-dependent rules.

        Look at the rule's history of successes and failures.
        Find the context feature that best separates them.
        """
        history = getattr(rule, '_observation_history', [])
        if len(history) < 5:
            return False

        successes = [obs for obs in history if obs['matched']]
        failures = [obs for obs in history if not obs['matched']]

        if not successes or not failures:
            return False

        # Find the context feature that best separates successes from failures
        best_feature = None
        best_separation = 0.0

        all_keys = set()
        for obs in history:
            all_keys.update(obs['context'].keys())

        for key in all_keys:
            success_values = [obs['context'].get(key) for obs in successes
                             if key in obs['context']]
            failure_values = [obs['context'].get(key) for obs in failures
                             if key in obs['context']]

            if not success_values or not failure_values:
                continue

            # For boolean features
            if all(isinstance(v, bool) for v in success_values + failure_values):
                success_true_rate = sum(success_values) / len(success_values)
                failure_true_rate = sum(failure_values) / len(failure_values)
                separation = abs(success_true_rate - failure_true_rate)
            # For categorical features
            elif all(isinstance(v, (int, str)) for v in success_values + failure_values):
                success_set = set(success_values)
                failure_set = set(failure_values)
                overlap = len(success_set & failure_set)
                total = len(success_set | failure_set)
                separation = 1.0 - (overlap / max(total, 1))
            else:
                continue

            if separation > best_separation:
                best_separation = separation
                best_feature = key

        if best_feature is None or best_separation < 0.3:
            return False

        # Get most common feature values for each group
        success_feature_values = [obs['context'].get(best_feature) for obs in successes
                                  if best_feature in obs['context']]
        failure_feature_values = [obs['context'].get(best_feature) for obs in failures
                                  if best_feature in obs['context']]

        most_common_success = Counter(success_feature_values).most_common(1)[0][0]
        most_common_failure = Counter(failure_feature_values).most_common(1)[0][0]

        # Build split contexts
        success_context = rule.context_features.copy()
        success_context[best_feature] = most_common_success

        failure_context = rule.context_features.copy()
        failure_context[best_feature] = most_common_failure

        # Get failure effect template
        failure_effects_list = [obs['actual_effects'] for obs in failures]
        failure_effect_template = failure_effects_list[0] if failure_effects_list else rule.effects

        # Import TransitionRule type from the rule itself
        RuleType = type(rule)

        rule_a = RuleType(
            action=action,
            context_features=success_context,
            effects=rule.effects,
            confidence=0.8,
            successes=len(successes),
            total=len(successes),
            last_used=rule.last_used,
        )

        rule_b = RuleType(
            action=action,
            context_features=failure_context,
            effects=failure_effect_template,
            confidence=0.8,
            successes=len(failures),
            total=len(failures),
            last_used=rule.last_used,
        )

        # Remove old confused rule, add two new specific rules
        world_model._rules[action].remove(rule)
        world_model._rules[action].append(rule_a)
        world_model._rules[action].append(rule_b)

        self._expansion_count += 1
        return True

    def reset(self):
        """Reset for new game."""
        self._error_history.clear()
        self._rule_count_history.clear()
        self._accuracy_history.clear()
