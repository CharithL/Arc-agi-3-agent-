"""If-then rule store -- stub for Phase 2 consolidation."""
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class ExtractedRule:
    condition: Dict[str, Any]
    action: int
    effect: str
    confidence: float
    source_episodes: int

class RuleStore:
    def __init__(self):
        self._rules: List[ExtractedRule] = []

    def add(self, rule: ExtractedRule):
        self._rules.append(rule)

    @property
    def count(self) -> int:
        return len(self._rules)

    def clear(self):
        self._rules.clear()
