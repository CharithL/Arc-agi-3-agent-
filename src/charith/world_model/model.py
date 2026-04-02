"""WorldModel -- predictive processing engine with object-level rules.

Amendment 1: learns rules about OBJECTS (color, displacement, shape changes),
NOT grid hashes.

Amendment 4: uses only RELATIVE context features for cross-level transfer
(no absolute positions, grid dimensions, or controller coordinates).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from charith.perception.core_knowledge import StructuredPercept


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ObjectEffect:
    """What happened to a single object after an action."""
    object_color: int
    displacement: Tuple[int, int]  # (delta_row, delta_col)
    shape_changed: bool
    size_delta: int  # positive=grew, negative=shrank, 0=same
    appeared: bool
    disappeared: bool


@dataclass
class TransitionRule:
    """A learned rule: action + context -> effects on objects."""
    action: int
    context_features: Dict[str, Any]
    effects: List[ObjectEffect]
    confidence: float
    successes: int = 1
    total: int = 1
    last_used: int = 0
    _observation_history: List[Dict] = field(default_factory=list)

    def matches_context(self, current_context: Dict[str, Any]) -> bool:
        """Return True if every feature in this rule's context matches
        the corresponding feature in *current_context*.

        Features present in the rule but absent in the query are ignored
        (open-world assumption). Features present in both must agree.
        """
        for key, value in self.context_features.items():
            if key in current_context and current_context[key] != value:
                return False
        return True

    def update(self, observed_effects: List[ObjectEffect], tick: int,
               context: Dict[str, Any] = None) -> None:
        """Update confidence after observing a new transition."""
        self.total += 1
        self.last_used = tick
        matched = self._effects_match(observed_effects)
        if matched:
            self.successes += 1
        self.confidence = self.successes / self.total

        if context is not None:
            self._observation_history.append({
                'tick': tick,
                'context': context.copy(),
                'matched': matched,
                'actual_effects': observed_effects,
            })
            # Cap history to avoid unbounded memory growth
            if len(self._observation_history) > 100:
                self._observation_history = self._observation_history[-50:]

    def _effects_match(self, observed: List[ObjectEffect]) -> bool:
        """Check whether the observed effects match the expected effects."""
        if len(observed) != len(self.effects):
            return False
        for exp, obs in zip(
            sorted(self.effects, key=lambda e: e.object_color),
            sorted(observed, key=lambda e: e.object_color),
        ):
            if exp.displacement != obs.displacement:
                return False
            if exp.shape_changed != obs.shape_changed:
                return False
        return True


@dataclass
class PredictionError:
    """Difference between predicted and observed state."""
    predicted_grid: Optional[np.ndarray]
    observed_grid: Optional[np.ndarray]
    error_magnitude: float
    error_cells: List[Tuple[int, int]]
    precision: float
    weighted_error: float
    is_novel: bool


# ---------------------------------------------------------------------------
# WorldModel
# ---------------------------------------------------------------------------

class WorldModel:
    """Predictive processing engine that learns object-level transition rules.

    Key design decisions (Amendments 1 + 4):
    - Rules describe what happens to OBJECTS (displacement, shape change,
      appearance, disappearance), NOT grid-hash transitions.
    - Context features are RELATIVE only -- no absolute positions,
      no grid dimensions.  This enables cross-level transfer.
    """

    FORBIDDEN_KEYS = frozenset({
        'ctrl_row', 'ctrl_col', 'grid_rows', 'grid_cols',
        'absolute_x', 'absolute_y',
    })

    def __init__(self, max_rules: int = 10000,
                 rule_decay_ticks: int = 500) -> None:
        self._rules: Dict[int, List[TransitionRule]] = defaultdict(list)
        self._max_rules = max_rules
        self._rule_decay = rule_decay_ticks
        self._tick = 0
        self._error_history: List[float] = []
        self._total_predictions = 0
        self._correct_predictions = 0

    # ------------------------------------------------------------------
    # Context extraction  (Amendment 4: relative only)
    # ------------------------------------------------------------------

    def extract_context(self, percept: StructuredPercept,
                        controllable_ids: Set[int]) -> Dict[str, Any]:
        """Extract RELATIVE context features from a percept.

        Amendment 4: no absolute positions -- only relational and
        proportional features that transfer across grid sizes.
        """
        context: Dict[str, Any] = {
            'background_color': percept.background_color,
            'object_count': percept.object_count,
            'unique_colors': percept.unique_colors,
        }

        for obj in percept.objects:
            if obj.object_id in controllable_ids:
                # Spatial relations involving this controllable object
                for rel in percept.spatial_relations:
                    if rel.obj_a_id == obj.object_id:
                        context[f'adjacent_{rel.relation}'] = rel.obj_b_id

                # Proportional position features (relative to grid size)
                context['ctrl_near_top'] = (
                    obj.centroid[0] < percept.grid_dims[0] * 0.3
                )
                context['ctrl_near_bottom'] = (
                    obj.centroid[0] > percept.grid_dims[0] * 0.7
                )
                context['ctrl_near_left'] = (
                    obj.centroid[1] < percept.grid_dims[1] * 0.3
                )
                context['ctrl_near_right'] = (
                    obj.centroid[1] > percept.grid_dims[1] * 0.7
                )
                break  # Only process the first controllable object

        # Strip any forbidden absolute keys that may have leaked in
        for key in self.FORBIDDEN_KEYS:
            context.pop(key, None)

        return context

    # ------------------------------------------------------------------
    # Effect computation
    # ------------------------------------------------------------------

    def compute_effects(self, prev_percept: StructuredPercept,
                        curr_percept: StructuredPercept,
                        matched_pairs: List[Tuple[int, int]],
                        ) -> List[ObjectEffect]:
        """Compute object-level effects between two consecutive percepts.

        *matched_pairs* is a list of (prev_object_id, curr_object_id) from
        the ObjectTracker.
        """
        effects: List[ObjectEffect] = []
        prev_map = {o.object_id: o for o in prev_percept.objects}
        curr_map = {o.object_id: o for o in curr_percept.objects}

        # 1. Matched pairs -- displacement / shape / size changes
        for prev_id, curr_id in matched_pairs:
            prev_obj = prev_map[prev_id]
            curr_obj = curr_map[curr_id]
            displacement = (
                round(curr_obj.centroid[0] - prev_obj.centroid[0]),
                round(curr_obj.centroid[1] - prev_obj.centroid[1]),
            )
            effects.append(ObjectEffect(
                object_color=prev_obj.color,
                displacement=displacement,
                shape_changed=(prev_obj.shape_hash != curr_obj.shape_hash),
                size_delta=curr_obj.size - prev_obj.size,
                appeared=False,
                disappeared=False,
            ))

        # 2. Appeared objects -- in curr but not matched
        matched_curr_ids = {cid for _, cid in matched_pairs}
        for obj in curr_percept.objects:
            if obj.object_id not in matched_curr_ids:
                effects.append(ObjectEffect(
                    object_color=obj.color,
                    displacement=(0, 0),
                    shape_changed=False,
                    size_delta=obj.size,
                    appeared=True,
                    disappeared=False,
                ))

        # 3. Disappeared objects -- in prev but not matched
        matched_prev_ids = {pid for pid, _ in matched_pairs}
        for obj in prev_percept.objects:
            if obj.object_id not in matched_prev_ids:
                effects.append(ObjectEffect(
                    object_color=obj.color,
                    displacement=(0, 0),
                    shape_changed=False,
                    size_delta=-obj.size,
                    appeared=False,
                    disappeared=True,
                ))

        return effects

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, action: int,
                context: Dict[str, Any]) -> Optional[List[ObjectEffect]]:
        """Predict the effects of *action* in the given *context*.

        Returns the effects from the highest-confidence matching rule,
        or None if no rule matches.
        """
        rules = self._rules.get(action, [])
        matching = [r for r in rules if r.matches_context(context)]
        if not matching:
            return None
        best = max(matching, key=lambda r: r.confidence)
        return best.effects

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def update(self, action: int, context: Dict[str, Any],
               observed_effects: List[ObjectEffect], tick: int) -> None:
        """Update rules after observing a transition.

        If a matching rule exists, update its confidence.
        Otherwise create a new rule.
        """
        self._tick = tick
        rules = self._rules.get(action, [])

        for rule in rules:
            if rule.matches_context(context):
                rule.update(observed_effects, tick, context)
                return

        # No matching rule -- create a new one
        new_rule = TransitionRule(
            action=action,
            context_features=context.copy(),
            effects=observed_effects,
            confidence=1.0,
            successes=1,
            total=1,
            last_used=tick,
        )
        self._rules[action].append(new_rule)

    # ------------------------------------------------------------------
    # Error tracking
    # ------------------------------------------------------------------

    def record_error(self, error: PredictionError) -> None:
        """Record a prediction error for monitoring."""
        self._error_history.append(error.weighted_error)

    def get_recent_errors(self, n: int = 50) -> List[float]:
        """Return the last *n* recorded weighted errors."""
        return self._error_history[-n:]

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_accuracy(self) -> float:
        """Return the overall prediction accuracy."""
        if self._total_predictions == 0:
            return 0.0
        return self._correct_predictions / self._total_predictions

    def get_rule_count(self) -> int:
        """Return the total number of learned rules across all actions."""
        return sum(len(v) for v in self._rules.values())

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Soft reset -- decay rule confidence (Amendment 4).

        Rules are retained so knowledge can transfer across levels,
        but their confidence is reduced to account for the new context.
        """
        self._error_history.clear()
        self._tick = 0
        for action, rules in self._rules.items():
            for rule in rules:
                rule.confidence *= 0.8
                rule._observation_history.clear()

    def hard_reset(self) -> None:
        """Full reset -- discard all learned rules and history."""
        self._rules.clear()
        self._error_history.clear()
        self._tick = 0
        self._total_predictions = 0
        self._correct_predictions = 0
