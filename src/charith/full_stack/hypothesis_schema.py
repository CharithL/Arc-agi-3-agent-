"""
Hypothesis data contracts for the CHARITH full stack agent.

The LLM (Phase 2) emits Hypothesis objects with structured ExpectedOutcome.
The Verifier (Phase 3) converts StructuredPercept diffs into ActualObservation
and compares using compute_match_score (see match_score.py).
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple, Any

Direction = Literal["up", "down", "left", "right", "none"]
HypothesisStatus = Literal[
    "untested", "confirmed", "refuted", "untestable", "ambiguous"
]


@dataclass
class ExpectedOutcome:
    """
    Structured prediction. Every field is optional — None means
    "no claim made" (not penalized, not rewarded).
    """
    direction: Optional[Direction] = None
    magnitude_cells: Optional[int] = None
    object_ref: Optional[str] = None
    color_change_to: Optional[str] = None
    object_appears: Optional[bool] = None
    object_disappears: Optional[bool] = None
    score_change: Optional[bool] = None
    no_effect: bool = False


@dataclass
class Hypothesis:
    rule: str
    confidence: float
    test_action: int
    expected: ExpectedOutcome
    status: HypothesisStatus = "untested"
    actual_summary: Optional[str] = None
    match_score: Optional[float] = None


@dataclass
class ActualObservation:
    """
    Distilled from a StructuredPercept diff, ready for structured matching.

    Any 'Object' here refers to charith.perception.core_knowledge.Object —
    typed as Any to avoid import cycles during scaffolding.
    """
    controllable_displacement: Optional[Tuple[int, int]]
    controllable_direction: Optional[Direction]
    controllable_magnitude: int
    any_color_changes: List[Tuple[Any, int, int]]  # (obj, old_color, new_color)
    new_objects: List[Any]
    removed_objects: List[Any]
    score_changed: bool
