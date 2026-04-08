"""
Expandable table world model for ARC-AGI-3.

Port of ExpandableTableModel from c1c2-hybrid, adapted for ARC:
  - Keys: action_id (1..num_actions) instead of cell indices
  - Values: frozenset of stringified change descriptions

Starts minimal (single-action table only) and expands when triggered by
error structure analysis. Expansion is one-way.

Levels of representation:
  Level 0 (always active): action -> effects
  Level 1 (sequential):    (prev_action, action) -> effects
  Level 2 (context):       (action, context_hash) -> effects
  Level 3 (spatial):       (action, neighbor_hash) -> effects
"""

from collections import defaultdict
from typing import Iterable, Optional, Tuple


class ArcTableModel:
    def __init__(self, num_actions: int = 8):
        self.num_actions = num_actions

        # Level 0: always active
        self.single_table = defaultdict(list)   # action -> [frozenset(change_str)]

        # Level 1: sequential
        self.sequence_table = defaultdict(list)  # (prev_action, action) -> [frozenset]
        self.sequence_enabled = False

        # Level 2: context
        self.context_table = defaultdict(list)   # (action, ctx_hash) -> [frozenset]
        self.context_enabled = False

        # Level 3: spatial (reserved slot, not wired to predict yet)
        self.spatial_table = defaultdict(list)
        self.spatial_enabled = False

        self.prev_action: Optional[int] = None
        self.expansion_history = []
        self.total_observations = 0

    def record(
        self,
        action: int,
        changes: Iterable,
        context: Optional[dict] = None,
    ) -> None:
        """Record an observation. Writes to all currently active tables."""
        # Changes can be a set of strings, a set of arbitrary objects,
        # or a frozenset. Stringify everything for hashability.
        frozen = frozenset(str(c) for c in changes)
        self.total_observations += 1

        self.single_table[action].append(frozen)

        if self.sequence_enabled and self.prev_action is not None:
            self.sequence_table[(self.prev_action, action)].append(frozen)

        if self.context_enabled and context:
            ctx_key = (action, self._hash_context(context))
            self.context_table[ctx_key].append(frozen)

        self.prev_action = action

    def predict(
        self,
        action: int,
        target: Optional[str] = None,
        prev_action: Optional[int] = None,
        context: Optional[dict] = None,
    ) -> Tuple[bool, float, str]:
        """
        Predict whether activating `action` causes `target` (a change
        description string) to appear in the observed changes.

        Checks most-specific table first and falls back:
            sequence -> context -> single -> unseen

        Returns: (predicts_effect, confidence, source).
        If target is None, returns a no-op default.
        """
        if target is None:
            return False, 0.0, "no_target"

        # Level 1: sequence
        if self.sequence_enabled and prev_action is not None:
            key = (prev_action, action)
            obs = self.sequence_table.get(key)
            if obs and len(obs) >= 2:
                times = sum(1 for o in obs if target in o)
                freq = times / len(obs)
                return freq > 0.5, freq, "sequence"

        # Level 2: context
        if self.context_enabled and context:
            ctx_key = (action, self._hash_context(context))
            obs = self.context_table.get(ctx_key)
            if obs and len(obs) >= 2:
                times = sum(1 for o in obs if target in o)
                freq = times / len(obs)
                return freq > 0.5, freq, "context"

        # Level 0: single
        obs = self.single_table.get(action)
        if obs:
            times = sum(1 for o in obs if target in o)
            freq = times / len(obs)
            return freq > 0.5, freq, "single"

        return False, 0.0, "unseen"

    def enable_expansion(self, expansion_type: str, reason: str) -> bool:
        """One-way enable of an expansion slot. Returns True if newly enabled."""
        if expansion_type == "sequential" and not self.sequence_enabled:
            self.sequence_enabled = True
            self.expansion_history.append(
                {"type": "sequential", "reason": reason, "step": self.total_observations}
            )
            return True
        if expansion_type == "context" and not self.context_enabled:
            self.context_enabled = True
            self.expansion_history.append(
                {"type": "context", "reason": reason, "step": self.total_observations}
            )
            return True
        if expansion_type == "spatial" and not self.spatial_enabled:
            self.spatial_enabled = True
            self.expansion_history.append(
                {"type": "spatial", "reason": reason, "step": self.total_observations}
            )
            return True
        return False

    def get_active_expansions(self) -> list:
        active = ["single"]
        if self.sequence_enabled:
            active.append("sequential")
        if self.context_enabled:
            active.append("context")
        if self.spatial_enabled:
            active.append("spatial")
        return active

    def _hash_context(self, context: dict) -> int:
        return hash(frozenset(context.items()))
