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


@dataclass
class GameResult:
    """Result of playing multiple levels of a single game."""
    game_id: str
    levels_completed: int
    levels_attempted: int
    level_results: List[LevelResult] = field(default_factory=list)
    total_actions: int = 0
    total_llm_calls: int = 0
    wall_time_sec: float = 0.0
    stopped_reason: str = ""
