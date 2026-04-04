"""ContextManager -- compressed interaction history for LLM context window.

Maintains full history but compresses it for the prompt:
- First tick always included (initial state)
- Important ticks always included (discoveries, goal events)
- Last 3 ticks always included in full
- Everything else summarised as a one-liner
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TickRecord:
    """One tick of interaction history."""
    tick: int
    observation: str
    action: int
    result_description: str
    hypothesis: str = ""
    c1_expansion: Optional[str] = None
    is_important: bool = False


class ContextManager:
    """Maintain and compress interaction history for LLM prompts."""

    def __init__(self, max_tokens: int = 4000) -> None:
        self.max_tokens = max_tokens
        self.full_history: List[TickRecord] = []
        self.important_ticks: List[int] = []
        self.action_effect_map: Dict[int, str] = {}  # action_int -> effect description

    def add_tick(
        self,
        tick: int,
        observation: str,
        action: int,
        result_description: str,
        hypothesis: str = "",
        c1_expansion: Optional[str] = None,
        is_important: bool = False,
    ) -> None:
        record = TickRecord(
            tick=tick,
            observation=observation,
            action=action,
            result_description=result_description,
            hypothesis=hypothesis,
            c1_expansion=c1_expansion,
            is_important=is_important,
        )
        self.full_history.append(record)
        if is_important:
            self.important_ticks.append(tick)

    def get_history_text(self) -> str:
        """Compressed history: first tick + important ticks + last 3 in full, rest summarised."""
        if not self.full_history:
            return "History: no previous observations"

        n = len(self.full_history)
        # Determine which ticks get full rendering
        full_indices: set = set()
        full_indices.add(0)  # first tick
        for t in self.important_ticks:
            for i, rec in enumerate(self.full_history):
                if rec.tick == t:
                    full_indices.add(i)
        # Last 3
        for i in range(max(0, n - 3), n):
            full_indices.add(i)

        lines: List[str] = ["Interaction history:"]
        summarised_count = 0

        for i, rec in enumerate(self.full_history):
            if i in full_indices:
                if summarised_count > 0:
                    lines.append(f"  ... ({summarised_count} ticks summarised)")
                    summarised_count = 0
                tag = " [IMPORTANT]" if rec.is_important else ""
                lines.append(
                    f"  Tick {rec.tick}{tag}: action={rec.action} -> {rec.result_description}"
                )
                if rec.hypothesis:
                    lines.append(f"    Hypothesis: {rec.hypothesis}")
                if rec.c1_expansion:
                    lines.append(f"    C1 expansion: {rec.c1_expansion}")
            else:
                summarised_count += 1

        if summarised_count > 0:
            lines.append(f"  ... ({summarised_count} ticks summarised)")

        return "\n".join(lines)

    def get_discovered_effects(self) -> str:
        """Format known action effects for the prompt."""
        if not self.action_effect_map:
            return "Known action effects: none discovered yet"
        lines = ["Known action effects:"]
        for action, effect in sorted(self.action_effect_map.items()):
            lines.append(f"  ACTION {action}: {effect}")
        return "\n".join(lines)

    def record_action_effect(self, action: int, effect: str) -> None:
        """Store a discovered action->effect mapping."""
        self.action_effect_map[action] = effect

    def reset(self) -> None:
        """Clear all history for a new game."""
        self.full_history.clear()
        self.important_ticks.clear()
        self.action_effect_map.clear()
