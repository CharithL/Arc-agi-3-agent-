"""S1 — Structured Decision interface (fixes P4: the say-do gap).

The verbal plan and the action token are not causally bound inside the LLM.
A prompt rule is aspiration; a schema + validator is mechanism. Here coherence
becomes *rejectable*, not merely requested.

``Decision`` is the single structured payload the LLM must emit each turn.
``validate_decision`` returns a list of human-readable error strings (empty =
valid) which are fed back to the LLM on retry.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

EffectKind = Literal[
    "MOVE", "COLOR_CHANGE", "APPEAR", "DISAPPEAR",
    "TRANSFORM", "SCORE_CHANGE", "NO_EFFECT", "UNKNOWN",
]


class PredictedEffect(BaseModel):
    kind: EffectKind
    object_id: Optional[str] = None   # must be a live tracked oid (post-S2) or None
    detail: str = ""                  # "RIGHT by 1", "color 3 -> 7", etc.


class ActionChoice(BaseModel):
    action_id: str
    target: Optional[str] = None      # affordance id for click games (post-S6)


class Decision(BaseModel):
    tick: int
    hypothesis_id: str                # existing hid | "EXPLORE" | "PLAN:<plan_id>"
    predicted_effect: PredictedEffect
    action: ActionChoice
    confidence: float = Field(ge=0.0, le=1.0)
    notes: str = Field(default="", max_length=300)
    requests: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Response-format prompt block (appended verbatim to the system prompt when
# structured_decisions is on). Kept here so schema and prompt cannot drift.
# ---------------------------------------------------------------------------

RESPONSE_FORMAT_BLOCK = """\
RESPONSE FORMAT -- reply with ONLY one JSON object:
{"tick": <int>, "hypothesis_id": "<id or EXPLORE>",
 "predicted_effect": {"kind": "...", "object_id": "... or null", "detail": "..."},
 "action": {"action_id": "...", "target": null},
 "confidence": 0.0-1.0, "notes": "<= 40 words", "requests": []}
Rules: your action must TEST the hypothesis you cite. Your prediction must be
checkable from the next observation. No hypothesis worth testing? Use "EXPLORE"
and predict what the exploratory action will do anyway."""


# ---------------------------------------------------------------------------
# Coherence validation
# ---------------------------------------------------------------------------

_REQUEST_PREFIXES = ("PROBE:", "EXPAND:", "HYP:")


def _bank_lookup(bank: Any, hid: str) -> Optional[Any]:
    """Return the hypothesis object for ``hid`` from a duck-typed bank, or None.

    Supports ``bank`` being None, a mapping, or an object exposing ``.get``.
    """
    if bank is None:
        return None
    getter = getattr(bank, "get", None)
    if callable(getter):
        return getter(hid)
    try:
        return bank[hid]  # mapping-like
    except (KeyError, TypeError):
        return None


def _bank_contains(bank: Any, hid: str) -> bool:
    if bank is None:
        return False
    try:
        return hid in bank
    except TypeError:
        return _bank_lookup(bank, hid) is not None


def validate_decision(
    d: Decision,
    bank: Any = None,
    legal_actions: Optional[list[str]] = None,
    affordances: Optional[list[str]] = None,
    *,
    live_oids: Optional[set[str]] = None,
) -> list[str]:
    """Return coherence errors for a Decision (empty list == valid).

    Parameters
    ----------
    d : Decision
    bank : mapping | HypothesisBank | None
        Hypothesis store. If None, hypothesis ids are accepted as provisional
        (pre-S4 behavior per guide rule 2).
    legal_actions : list[str] | None
        Lawful-filtered action set (post-S6); raw set before that. If None, the
        legality check is skipped.
    affordances : list[str] | None
        Valid affordance ids for click games (post-S6). If None, target checks
        are skipped.
    live_oids : set[str] | None
        Live tracked object ids (post-S2). If None, the object-liveness check
        is skipped.
    """
    errors: list[str] = []

    # Rule 1 — action must be legal.
    if legal_actions is not None and d.action.action_id not in {str(a) for a in legal_actions}:
        errors.append(
            f"E-ILLEGAL-ACTION: {d.action.action_id} is not in the legal action "
            f"set {list(legal_actions)}."
        )

    # Rule 1b — if this is a click game (affordances provided) and a target is
    # given, it must name a real affordance.
    if affordances is not None and d.action.target is not None:
        if d.action.target not in set(affordances):
            errors.append(
                f"E-UNKNOWN-TARGET: target {d.action.target} is not a proposed "
                f"affordance {list(affordances)}."
            )

    # Rule 2 — hypothesis id must resolve, EXPLORE, or PLAN:*  (provisional if no bank).
    hid = d.hypothesis_id
    is_special = hid == "EXPLORE" or hid.startswith("PLAN:")
    if bank is not None and not is_special and not _bank_contains(bank, hid):
        errors.append(
            f"E-UNKNOWN-HYP: hypothesis_id {hid} is not in the bank and is not "
            f'"EXPLORE" or "PLAN:<id>".'
        )

    # Rule 3 — the anti-say-do rule.
    hyp = _bank_lookup(bank, hid) if not is_special else None
    if hyp is not None:
        htype = getattr(hyp, "htype", None)
        binding = getattr(hyp, "binding", None)
        bound_action = getattr(binding, "action_id", None) if binding is not None else None
        if htype == "ACTION_MAPPING" and bound_action is not None:
            if d.action.action_id != str(bound_action):
                statement = getattr(hyp, "statement", "")
                errors.append(
                    f'E-ACTION-MISMATCH: You cited {hid} ("{statement}"), which is '
                    f"about action {bound_action}, but chose {d.action.action_id}. "
                    f"Either choose {bound_action}, or cite EXPLORE, or cite a "
                    f"hypothesis about {d.action.action_id}."
                )

    # Rule 4 — predicted object_id, if set, must be a live tracked id.
    oid = d.predicted_effect.object_id
    if live_oids is not None and oid is not None and oid not in live_oids:
        errors.append(
            f"E-DEAD-OBJECT: predicted_effect.object_id {oid} is not a live "
            f"tracked object."
        )

    # Rule 5 — malformed requests are advisory: dropped with a note, never bounced.
    return errors


def sanitize_requests(requests: list[str]) -> tuple[list[str], list[str]]:
    """Split requests into (kept, dropped) by prefix.

    Requests are advisory (guide rule 5): malformed entries are dropped and a
    note is surfaced in the next prompt rather than causing a validation error.
    """
    kept: list[str] = []
    dropped: list[str] = []
    for r in requests:
        stripped = r.strip()
        if any(stripped.upper().startswith(p) for p in _REQUEST_PREFIXES):
            kept.append(stripped)
        else:
            dropped.append(r)
    return kept, dropped
