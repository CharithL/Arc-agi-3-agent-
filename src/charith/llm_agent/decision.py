"""S1 — structured-decision retry protocol and incident logging.

The LLM is asked for a JSON ``Decision``. If it is malformed or fails coherence
validation, the error list is fed back and the model retries up to
``max_decision_retries`` times. After the final failure a fallback action is
taken and the raw LLM text is written to ``incidents.log`` (guide ground rule 9).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from charith.config import AgentConfig
from charith.llm_agent.schema import Decision, validate_decision


# ---------------------------------------------------------------------------
# JSON extraction / parsing
# ---------------------------------------------------------------------------

def parse_decision(text: str) -> Decision:
    """Parse a Decision out of raw LLM text.

    Tolerates surrounding prose by extracting the first balanced ``{...}`` span.
    Raises ValueError on failure (used as an E-PARSE error upstream).
    """
    obj = _extract_json(text)
    if obj is None:
        raise ValueError("no JSON object found in response")
    try:
        return Decision.model_validate(obj)
    except Exception as exc:  # pydantic ValidationError -> ValueError for callers
        raise ValueError(f"schema validation failed: {exc}") from exc


def _extract_json(text: str) -> Optional[dict]:
    text = text.strip()
    # Fast path: whole thing is JSON.
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    # Slow path: scan for the first balanced brace span.
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


# ---------------------------------------------------------------------------
# Incident logging
# ---------------------------------------------------------------------------

class IncidentLog:
    """Append-only incident log. No-op when path is None."""

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path
        self.records: list[dict] = []
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)

    def log(self, kind: str, detail: str, raw: str = "") -> None:
        rec = {"kind": kind, "detail": detail, "raw": raw}
        self.records.append(rec)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(f"[{kind}] {detail}\n")
                if raw:
                    fh.write(f"    RAW: {raw!r}\n")


# ---------------------------------------------------------------------------
# Retry loop
# ---------------------------------------------------------------------------

@dataclass
class DecisionResult:
    decision: Optional[Decision]     # None if all attempts failed
    fallback_action: Optional[str]   # set when a fallback was taken
    attempts: int
    errors: list[str] = field(default_factory=list)  # errors from the last attempt


def run_decision(
    query: Callable[[str], str],
    base_prompt: str,
    *,
    config: AgentConfig,
    bank: Any = None,
    legal_actions: Optional[list[str]] = None,
    affordances: Optional[list[str]] = None,
    live_oids: Optional[set[str]] = None,
    fallback: Callable[[], str],
    incidents: Optional[IncidentLog] = None,
) -> DecisionResult:
    """Drive the structured-decision retry protocol.

    ``query`` takes the (possibly error-augmented) user prompt and returns raw
    LLM text. ``fallback`` returns an action_id to use if all attempts fail.
    """
    max_attempts = config.max_decision_retries + 1
    prompt = base_prompt
    last_errors: list[str] = []

    for attempt in range(1, max_attempts + 1):
        raw = query(prompt)
        try:
            decision = parse_decision(raw)
            errors = validate_decision(
                decision, bank=bank, legal_actions=legal_actions,
                affordances=affordances, live_oids=live_oids,
            )
        except ValueError as exc:
            decision = None
            errors = [f"E-PARSE: {exc}"]

        if decision is not None and not errors:
            return DecisionResult(decision=decision, fallback_action=None,
                                  attempts=attempt, errors=[])

        last_errors = errors
        if attempt < max_attempts:
            prompt = (
                f"{base_prompt}\n\nYour previous response was invalid:\n"
                + "\n".join(f"  - {e}" for e in errors)
                + "\nReturn a corrected JSON object."
            )
        else:
            # Final failure: fallback + incident.
            action = fallback()
            if incidents is not None:
                incidents.log(
                    "DECISION_FALLBACK",
                    f"invalid after {attempt} attempts: {errors}",
                    raw=raw,
                )
            return DecisionResult(decision=None, fallback_action=action,
                                  attempts=attempt, errors=last_errors)

    # Unreachable, but keeps type-checkers happy.
    return DecisionResult(decision=None, fallback_action=fallback(),
                          attempts=max_attempts, errors=last_errors)


def least_tried_fallback(legal_actions: list[str], tried_counts: dict[str, int]) -> str:
    """Pre-S4 fallback: the least-tried legal action (ties -> first)."""
    if not legal_actions:
        raise ValueError("no legal actions available for fallback")
    return min(legal_actions, key=lambda a: tried_counts.get(str(a), 0))
