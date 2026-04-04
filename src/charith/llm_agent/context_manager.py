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
        self.action_effect_map: Dict[int, Dict[str, int]] = {}  # action_int -> {effect_text: count}

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
        """Format known action effects with counts.

        Output like: A2: moves RIGHT (seen 3x), no movement (seen 8x)
        This tells the LLM frequency — USUALLY doesn't work but SOMETIMES
        moves right = context-dependent mechanic.
        """
        if not self.action_effect_map:
            return "Known action effects: none discovered yet"
        lines = ["Known action effects:"]
        for action, counter in sorted(self.action_effect_map.items()):
            # Sort by significance (real movement first), then by count
            def _sig(text: str) -> int:
                t = text.lower()
                if 'moved' in t and 'by 0' not in t:
                    return 2
                if 'pixel' in t or 'changed' in t:
                    return 1
                return 0

            entries = sorted(counter.items(), key=lambda kv: (-_sig(kv[0]), -kv[1]))
            parts = [f"{effect} (seen {count}x)" for effect, count in entries]
            lines.append(f"  ACTION {action}: {', '.join(parts)}")

        return "\n".join(lines)

    def record_action_effect(self, action: int, effect: str) -> None:
        """Store an observed action effect as a counted, deduplicated map.

        self.action_effect_map[action] is a Dict[str, int] counting
        how many times each unique effect was observed. Capped at 3
        unique effects per action.
        """
        if not effect:
            return

        if action not in self.action_effect_map:
            self.action_effect_map[action] = {}

        counter = self.action_effect_map[action]

        # Normalize effect text for deduplication
        key = effect.strip()

        if key in counter:
            counter[key] += 1
            return

        # Cap at 3 unique effects per action
        if len(counter) >= 3:
            # Drop the least significant entry
            def _sig(text: str) -> int:
                t = text.lower()
                if 'moved' in t and 'by 0' not in t:
                    return 2
                if 'pixel' in t or 'changed' in t:
                    return 1
                return 0
            worst = min(counter.keys(), key=lambda k: (_sig(k), counter[k]))
            if _sig(key) > _sig(worst) or (counter[worst] == 1 and _sig(key) >= _sig(worst)):
                del counter[worst]
            else:
                return  # New effect is less significant, don't add

        counter[key] = 1

    def reset(self) -> None:
        """Clear all history for a new game."""
        self.full_history.clear()
        self.important_ticks.clear()
        self.action_effect_map.clear()
