"""Result dataclasses for the full-stack agent."""

from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class AttemptResult:
    completed: bool
    actions_taken: int
    llm_calls: int
    reason: str
    phase_reached: int
    hypotheses_generated: int
    hypotheses_confirmed: int
    hypotheses_refuted: int
    expansions_triggered: List[str]
    final_error_summary: str


@dataclass
class LevelResult:
    completed: bool
    attempts: List[AttemptResult]
    total_actions: int
    total_llm_calls: int
    final_table_stats: Dict
