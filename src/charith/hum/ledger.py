"""Prediction Ledger + gated balloon expansion (S3).

Predictions are recorded *before* acting and machine-scored *after*. Balloon
expansion of the C1 ontology becomes a privilege unlocked only by machine-
verified surprise (orphan events), not by the LLM declaring confusion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Literal, Optional

from charith.config import AgentConfig
from charith.llm_agent.schema import Decision, PredictedEffect
from charith.hum.comparator import Outcome, score_prediction
from charith.perception.object_tracker import PerceptEvent


@dataclass
class PredictionRecord:
    tick: int
    hypothesis_id: str
    action_id: str
    predicted: PredictedEffect
    outcome: Literal["PENDING", "CONFIRMED", "REFUTED", "PARTIAL", "UNTESTABLE"] = "PENDING"


# Which observed event kinds a predicted effect "explains" (entails).
_ENTAILED_KINDS = {
    "MOVE": {"MOVED"},
    "COLOR_CHANGE": {"RECOLORED"},
    "TRANSFORM": {"TRANSFORMED"},
    "APPEAR": {"APPEARED", "REAPPEARED"},
    "DISAPPEAR": {"VANISHED"},
}


class Ledger:
    """Record predictions, resolve them against observations, track orphans."""

    def __init__(self, enforce: bool = True) -> None:
        self.enforce = enforce
        self.records: List[PredictionRecord] = []
        self._pending: Optional[PredictionRecord] = None
        # rolling event buffer: [tick, event, explained]
        self._buffer: List[list] = []
        self._last_tick: int = 0

    # ---- prediction lifecycle ---------------------------------------------

    def record(self, tick: int, decision: Decision) -> PredictionRecord:
        rec = PredictionRecord(
            tick=tick,
            hypothesis_id=decision.hypothesis_id,
            action_id=decision.action.action_id,
            predicted=decision.predicted_effect,
        )
        self.records.append(rec)
        self._pending = rec
        return rec

    def resolve(
        self,
        tick: int,
        events: List[PerceptEvent],
        score_delta: int,
        live_oids: Optional[set[str]] = None,
    ) -> Optional[PredictionRecord]:
        """Score the pending prediction against observations.

        With ``enforce=True`` (S3 flag), resolving with no recorded prediction
        raises — the loop cannot silently bypass the ledger.
        """
        self._last_tick = tick
        for e in events:
            self._buffer.append([tick, e, False])

        if self._pending is None:
            if self.enforce:
                raise RuntimeError(
                    "Ledger.resolve called with no recorded prediction "
                    "(enforce_predictions is on)."
                )
            return None

        rec = self._pending
        self._pending = None
        rec.outcome = score_prediction(rec.predicted, events, score_delta, live_oids=live_oids)
        if rec.outcome in ("CONFIRMED", "PARTIAL"):
            self._mark_explained(rec.predicted, tick)
        return rec

    def has_pending(self) -> bool:
        return self._pending is not None

    # ---- metrics -----------------------------------------------------------

    def rolling_accuracy(self, n: int = 12) -> float:
        """CONFIRMED / (CONFIRMED + REFUTED + PARTIAL) over the last n testable records."""
        testable = [r for r in self.records
                    if r.outcome in ("CONFIRMED", "REFUTED", "PARTIAL")][-n:]
        if not testable:
            return 0.0
        return sum(r.outcome == "CONFIRMED" for r in testable) / len(testable)

    def untestable_rate(self, n: int = 12) -> float:
        resolved = [r for r in self.records
                    if r.outcome != "PENDING"][-n:]
        if not resolved:
            return 0.0
        return sum(r.outcome == "UNTESTABLE" for r in resolved) / len(resolved)

    # ---- orphans (machine-verified surprise) ------------------------------

    def _mark_explained(self, predicted: PredictedEffect, tick: int) -> None:
        kinds = _ENTAILED_KINDS.get(predicted.kind, set())
        oid = predicted.object_id
        for entry in self._buffer:
            etick, event, explained = entry
            if explained or etick != tick:
                continue
            if event.kind in kinds and (oid is None or event.oid == oid):
                entry[2] = True

    def orphans(self, window: int) -> List[PerceptEvent]:
        """Events in the last `window` ticks not entailed by a resolved prediction."""
        cutoff = self._last_tick - window + 1
        return [event for etick, event, explained in self._buffer
                if not explained and etick >= cutoff]

    def mark_entailed(self, is_entailed: Callable[[PerceptEvent], bool]) -> None:
        """Let an external source (e.g. the S4 bank) mark events as explained."""
        for entry in self._buffer:
            if not entry[2] and is_entailed(entry[1]):
                entry[2] = True


# ---------------------------------------------------------------------------
# Gated balloon expansion
# ---------------------------------------------------------------------------

_DENY_NO_ORPHANS = (
    "EXPANSION DENIED: no unexplained events in the last {n} ticks. Existing "
    "concepts have not been shown insufficient. Make a prediction that fails, "
    "then ask again."
)
_DENY_COOLDOWN = (
    "EXPANSION DENIED: expansion cooldown active ({elapsed}/{cooldown} ticks). "
    "Gather more evidence before expanding again."
)
_DENY_ALIAS = (
    "EXPANSION DENIED: {concept} is an alias of an existing concept. Reuse it "
    "instead of expanding."
)


def gate_expansion(
    concept: str,
    ledger: Ledger,
    config: AgentConfig,
    ticks_since_last_expansion: int,
    is_alias: Callable[[str], bool] = lambda c: False,
) -> tuple[bool, str]:
    """Return (granted, message). Message is a denial template when not granted.

    Honored iff (a) orphan buffer non-empty within orphan_window, (b) cooldown
    elapsed, and (c) concept is not an alias of an existing C1 concept.
    """
    if not ledger.orphans(config.orphan_window):
        return False, _DENY_NO_ORPHANS.format(n=config.orphan_window)
    if ticks_since_last_expansion < config.expansion_cooldown_ticks:
        return False, _DENY_COOLDOWN.format(
            elapsed=ticks_since_last_expansion, cooldown=config.expansion_cooldown_ticks
        )
    if is_alias(concept):
        return False, _DENY_ALIAS.format(concept=concept)
    return True, ""


@dataclass
class ExpansionGate:
    """Stateful wrapper tracking the last-expansion tick for cooldown checks."""

    config: AgentConfig
    last_expansion_tick: int = field(default=-10_000)

    def try_expand(
        self,
        concept: str,
        ledger: Ledger,
        tick: int,
        is_alias: Callable[[str], bool] = lambda c: False,
    ) -> tuple[bool, str]:
        elapsed = tick - self.last_expansion_tick
        granted, msg = gate_expansion(concept, ledger, self.config, elapsed, is_alias)
        if granted:
            self.last_expansion_tick = tick
        return granted, msg
