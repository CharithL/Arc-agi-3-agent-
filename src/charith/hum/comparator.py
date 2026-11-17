"""Effect Comparator — machine-score a PredictedEffect against observed events.

Pure function, one unit test per scoring-table row (guide S3). Scores a
prediction made *before* acting against the ``PerceptEvent``s and score delta
observed *after* acting. The LLM never gets to grade its own homework.
"""

from __future__ import annotations

import re
from typing import Iterable, Literal, Optional

from charith.llm_agent.schema import PredictedEffect
from charith.perception.object_tracker import PerceptEvent

Outcome = Literal["CONFIRMED", "REFUTED", "PARTIAL", "UNTESTABLE"]


def _dirs(detail: str) -> dict[str, bool]:
    d = detail.upper()
    return {
        "RIGHT": "RIGHT" in d,
        "LEFT": "LEFT" in d,
        "UP": "UP" in d,
        "DOWN": "DOWN" in d,
    }


def _move_matches(predicted_detail: str, ev_detail: dict) -> bool:
    """True if the observed dx/dy is consistent with the predicted direction.

    dx = column delta (RIGHT>0, LEFT<0); dy = row delta (DOWN>0, UP<0).
    A prediction with no named direction matches any movement.
    """
    dx = ev_detail.get("dx", 0.0)
    dy = ev_detail.get("dy", 0.0)
    dirs = _dirs(predicted_detail)
    if not any(dirs.values()):
        return True  # direction unspecified -> any move confirms
    if dirs["RIGHT"] and dx <= 0:
        return False
    if dirs["LEFT"] and dx >= 0:
        return False
    if dirs["DOWN"] and dy <= 0:
        return False
    if dirs["UP"] and dy >= 0:
        return False
    return True


def _color_detail_matches(predicted_detail: str, ev_detail: dict) -> bool:
    """True if the predicted target color matches the observed 'to' color.

    Lenient: if the prediction names no specific color, it counts as matching
    (the recolor itself was predicted). If it names a number, that number must
    equal the observed 'to' color.
    """
    nums = [int(x) for x in re.findall(r"\d+", predicted_detail)]
    if not nums:
        return True
    to_color = ev_detail.get("to")
    return to_color in nums


def score_prediction(
    predicted: PredictedEffect,
    events: Iterable[PerceptEvent],
    score_delta: int,
    live_oids: Optional[set[str]] = None,
) -> Outcome:
    events = list(events)
    kind = predicted.kind
    oid = predicted.object_id

    if kind == "UNKNOWN":
        return "UNTESTABLE"

    if kind == "MOVE":
        if oid is not None and live_oids is not None and oid not in live_oids:
            return "UNTESTABLE"
        moved = [e for e in events if e.kind == "MOVED" and (oid is None or e.oid == oid)]
        if not moved:
            return "REFUTED"
        if any(_move_matches(predicted.detail, e.detail) for e in moved):
            return "CONFIRMED"
        return "REFUTED"

    if kind == "NO_EFFECT":
        if len(events) == 0 and score_delta == 0:
            return "CONFIRMED"
        return "REFUTED"

    if kind == "COLOR_CHANGE":
        rec = [e for e in events if e.kind == "RECOLORED" and (oid is None or e.oid == oid)]
        if not rec:
            return "REFUTED"
        if any(_color_detail_matches(predicted.detail, e.detail) for e in rec):
            return "CONFIRMED"
        return "PARTIAL"

    if kind == "APPEAR":
        return "CONFIRMED" if any(e.kind in ("APPEARED", "REAPPEARED") for e in events) else "REFUTED"

    if kind == "DISAPPEAR":
        return "CONFIRMED" if any(e.kind == "VANISHED" for e in events) else "REFUTED"

    if kind == "SCORE_CHANGE":
        return "CONFIRMED" if score_delta != 0 else "REFUTED"

    if kind == "TRANSFORM":
        trans = [e for e in events if e.kind == "TRANSFORMED" and (oid is None or e.oid == oid)]
        return "CONFIRMED" if trans else "REFUTED"

    return "UNTESTABLE"
