"""C1/C2 Framework -- conceptual vocabulary and hypothesis management.

C1 = conceptual vocabulary (Spelke core + discovered concepts).
C2 = current hypotheses about game mechanics.

Inspired by Global Workspace Theory: C1 is the repertoire of concepts
the agent can reason about; C2 is the "broadcast" of current beliefs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Concept:
    """A named concept in the C1 vocabulary."""
    name: str
    description: str
    source: str = "core"   # "core" or "discovered"
    tick_added: int = 0


@dataclass
class Hypothesis:
    """A C2 hypothesis about the game."""
    text: str
    confidence: str = "low"  # "low", "med", "high"
    tick_updated: int = 0


# --- initial Spelke concepts ------------------------------------------------

_INITIAL_C1: List[Concept] = [
    Concept("OBJECT", "Discrete bounded entities that persist and move as units"),
    Concept("MOVEMENT", "Objects can move in cardinal directions; motion is contingent on actions"),
    Concept("WALL", "Some cells/objects block movement; collisions prevent displacement"),
    Concept("GOAL", "A target location or state the agent should reach or achieve"),
    Concept("SPATIAL_RELATION", "Objects have relative positions: above, below, left, right, adjacent"),
    Concept("COUNTING", "The number of objects/colors matters; quantities can change"),
    Concept("SYMMETRY", "Patterns may have horizontal, vertical, or rotational symmetry"),
]


class C1C2Framework:
    """Manage the conceptual vocabulary (C1) and working hypotheses (C2)."""

    def __init__(self) -> None:
        self.c1_vocabulary: List[Concept] = [
            Concept(c.name, c.description, c.source, c.tick_added)
            for c in _INITIAL_C1
        ]
        self.c1_expansions: List[Concept] = []
        self.hypotheses: List[Hypothesis] = []

    # ---- C1 methods --------------------------------------------------------

    def get_c1_text(self) -> str:
        """Format C1 vocabulary for inclusion in LLM prompt."""
        lines = ["Conceptual vocabulary (C1):"]
        for c in self.c1_vocabulary:
            tag = " [discovered]" if c.source == "discovered" else ""
            lines.append(f"  - {c.name}: {c.description}{tag}")
        return "\n".join(lines)

    def expand_c1(self, name: str, description: str, tick: int = 0) -> None:
        """Add a new discovered concept (balloon expansion)."""
        # Avoid duplicates
        existing_names = {c.name.upper() for c in self.c1_vocabulary}
        if name.upper() in existing_names:
            return
        concept = Concept(name=name, description=description, source="discovered", tick_added=tick)
        self.c1_vocabulary.append(concept)
        self.c1_expansions.append(concept)

    # ---- C2 methods --------------------------------------------------------

    def get_hypotheses_text(self) -> str:
        """Format current hypotheses for LLM prompt. Keep last 5."""
        recent = self.hypotheses[-5:] if len(self.hypotheses) > 5 else self.hypotheses
        if not recent:
            return "Current hypotheses: none yet"
        lines = ["Current hypotheses:"]
        for h in recent:
            lines.append(f"  - [{h.confidence}] {h.text}")
        return "\n".join(lines)

    def update_hypothesis(self, text: str, confidence: str = "low") -> None:
        """Add or update a hypothesis. Caps total at 10."""
        # Check if hypothesis already exists (fuzzy: same first 40 chars)
        prefix = text[:40].lower().strip()
        for h in self.hypotheses:
            if h.text[:40].lower().strip() == prefix:
                h.confidence = confidence
                return
        self.hypotheses.append(Hypothesis(text=text, confidence=confidence))
        # Cap at 10 total
        if len(self.hypotheses) > 10:
            self.hypotheses = self.hypotheses[-10:]

    # ---- lifecycle ---------------------------------------------------------

    def reset(self) -> None:
        """Reset for a new game. Keep discovered concepts but clear hypotheses."""
        self.hypotheses.clear()
