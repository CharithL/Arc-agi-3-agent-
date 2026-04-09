"""
compute_match_score — locks in Section 5.3–5.4 of the design doc.

Compares an LLM-emitted ExpectedOutcome to a mechanistically-derived
ActualObservation and returns a score in [0, 1].

Design calls (user-locked):
  - Mean aggregation (partial credit)
  - ±1 cell magnitude tolerance
  - object_ref deliberately NOT checked (liberal for ls20)
"""

from charith.full_stack.hypothesis_schema import ExpectedOutcome, ActualObservation


def compute_match_score(expected: ExpectedOutcome, actual: ActualObservation) -> float:
    """
    Return a match score in [0, 1].

    See design doc §5.3 for exact rules.
    """
    # Special case: no_effect claim
    if expected.no_effect:
        is_empty = (
            actual.controllable_magnitude == 0
            and not actual.any_color_changes
            and not actual.new_objects
            and not actual.removed_objects
            and not actual.score_changed
        )
        return 1.0 if is_empty else 0.0

    scores = []

    # Direction: strict match
    if expected.direction is not None:
        if expected.direction == "none":
            scores.append(1.0 if actual.controllable_magnitude == 0 else 0.0)
        else:
            scores.append(
                1.0 if actual.controllable_direction == expected.direction else 0.0
            )

    # Magnitude: ±1 tolerance
    if expected.magnitude_cells is not None and actual.controllable_magnitude > 0:
        diff = abs(actual.controllable_magnitude - expected.magnitude_cells)
        scores.append(1.0 if diff <= 1 else 0.0)

    # Color change: substring, case-insensitive
    if expected.color_change_to is not None:
        matched = any(
            expected.color_change_to.lower() in str(new_c).lower()
            for _, _, new_c in actual.any_color_changes
        )
        scores.append(1.0 if matched else 0.0)

    # Object appears / disappears
    if expected.object_appears is not None:
        scores.append(1.0 if expected.object_appears == bool(actual.new_objects) else 0.0)
    if expected.object_disappears is not None:
        scores.append(
            1.0 if expected.object_disappears == bool(actual.removed_objects) else 0.0
        )

    # Score change
    if expected.score_change is not None:
        scores.append(1.0 if expected.score_change == actual.score_changed else 0.0)

    # Aggregation: MEAN
    if not scores:
        return 0.5  # LLM claimed nothing specific — ambiguous
    return sum(scores) / len(scores)
