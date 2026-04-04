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
        self.action_effect_map: Dict[int, List[str]] = {}  # action_int -> list of ALL observed effects

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
        """Format known action effects for the prompt.

        Shows the MOST SIGNIFICANT effect first, then notes if the action
        is context-dependent (different effects observed at different times).
        """
        if not self.action_effect_map:
            return "Known action effects: none discovered yet"
        lines = ["Known action effects:"]
        for action, effects in sorted(self.action_effect_map.items()):
            # Deduplicate and find unique effect types
            unique = list(dict.fromkeys(effects))  # preserves order, removes dupes
            # Sort by significance: real movement first
            significant = [e for e in unique if 'moved' in e.lower() and 'by 0' not in e.lower()]
            minor = [e for e in unique if e not in significant]

            if significant:
                primary = significant[0]
                if minor:
                    lines.append(f"  ACTION {action}: {primary} (sometimes: {minor[0][:40]})")
                else:
                    lines.append(f"  ACTION {action}: {primary}")
            elif unique:
                lines.append(f"  ACTION {action}: {unique[0]}")

        return "\n".join(lines)

    def record_action_effect(self, action: int, effect: str) -> None:
        """Store an observed action effect. Keeps ALL effects per action
        to detect context-dependent mechanics."""
        if not effect:
            return

        if action not in self.action_effect_map:
            self.action_effect_map[action] = []

        # Cap at 10 effects per action to avoid unbounded growth
        effects = self.action_effect_map[action]
        if len(effects) >= 10:
            # Remove the least significant one
            def _sig(t: str) -> int:
                tl = t.lower()
                if 'no movement' in tl or 'wall' in tl or 'by 0' in tl:
                    return 0
                if 'moved' in tl:
                    return 2
                if 'pixel' in tl or 'changed' in tl:
                    return 1
                return 0
            effects.sort(key=_sig, reverse=True)
            effects.pop()  # Remove lowest significance

        effects.append(effect)

    def reset(self) -> None:
        """Clear all history for a new game."""
        self.full_history.clear()
        self.important_ticks.clear()
        self.action_effect_map.clear()
