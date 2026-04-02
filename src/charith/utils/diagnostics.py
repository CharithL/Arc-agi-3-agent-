"""
Diagnostic logging for CHARITH agent.

Produces per-tick structured logs and per-game summary reports.
Used for debugging, ablation studies, and paper figures.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
import json
import time
from pathlib import Path


@dataclass
class TickLog:
    """One tick of agent operation."""
    tick: int
    action_taken: int
    available_actions: List[int]
    prediction_correct: bool
    prediction_confidence: float
    error_magnitude: float
    num_objects: int
    num_controllable: int
    controllable_positions: List[Tuple[float, float]]
    rule_count: int
    goal_hypothesis: Optional[str]
    goal_confidence: float
    ontology_expansion_triggered: bool
    score: Optional[float]
    level_complete: bool
    state_hash: int


@dataclass
class LevelLog:
    """Summary of one level attempt."""
    level_index: int
    ticks_taken: int
    completed: bool
    rules_at_start: int
    rules_at_end: int
    accuracy: float
    ontology_expansions: int
    unique_states_visited: int


@dataclass
class GameLog:
    """Complete log of one game."""
    game_id: str
    total_ticks: int
    levels_completed: int
    total_levels: int
    final_accuracy: float
    total_rules: int
    total_ontology_expansions: int
    available_actions: List[int]
    grid_size: Tuple[int, int]
    ticks: List[TickLog] = field(default_factory=list)
    levels: List[LevelLog] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def duration_seconds(self) -> float:
        return self.end_time - self.start_time

    def save(self, path: str):
        """Save log to JSON for analysis."""
        data = {
            'game_id': self.game_id,
            'total_ticks': self.total_ticks,
            'levels_completed': self.levels_completed,
            'total_levels': self.total_levels,
            'final_accuracy': self.final_accuracy,
            'total_rules': self.total_rules,
            'grid_size': self.grid_size,
            'available_actions': self.available_actions,
            'duration_seconds': self.duration_seconds,
            'ticks': [
                {
                    'tick': t.tick,
                    'action': t.action_taken,
                    'pred_correct': t.prediction_correct,
                    'error': t.error_magnitude,
                    'objects': t.num_objects,
                    'controllable': t.num_controllable,
                    'ctrl_positions': t.controllable_positions,
                    'goal': t.goal_hypothesis,
                    'goal_conf': t.goal_confidence,
                    'ontology_triggered': t.ontology_expansion_triggered,
                    'level_complete': t.level_complete,
                }
                for t in self.ticks
            ],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)


class DiagnosticLogger:
    """
    Hooks into CHARITHAgent to produce GameLog.

    Usage:
        logger = DiagnosticLogger()
        agent = CHARITHAgent(diagnostic_logger=logger)
        agent.play_game("ls20", max_actions=200)
        logger.game_log.save("logs/ls20.json")
    """

    def __init__(self):
        self.game_log: Optional[GameLog] = None
        self._current_level_start_tick = 0
        self._current_level_index = 0
        self._level_rules_start = 0
        self._level_expansions = 0
        self._level_states: set = set()

    def on_game_start(self, game_id: str, grid_size: Tuple[int, int],
                      available_actions: List[int], total_levels: int = 0):
        self.game_log = GameLog(
            game_id=game_id,
            total_ticks=0,
            levels_completed=0,
            total_levels=total_levels,
            final_accuracy=0.0,
            total_rules=0,
            total_ontology_expansions=0,
            available_actions=available_actions,
            grid_size=grid_size,
            start_time=time.time(),
        )
        self._current_level_start_tick = 0
        self._current_level_index = 0
        self._level_states = set()

    def on_tick(self, tick_log: TickLog):
        if self.game_log:
            self.game_log.ticks.append(tick_log)
            self._level_states.add(tick_log.state_hash)

    def on_level_complete(self, tick: int, rules: int, accuracy: float,
                          expansions: int):
        if self.game_log:
            self.game_log.levels.append(LevelLog(
                level_index=self._current_level_index,
                ticks_taken=tick - self._current_level_start_tick,
                completed=True,
                rules_at_start=self._level_rules_start,
                rules_at_end=rules,
                accuracy=accuracy,
                ontology_expansions=expansions - self._level_expansions,
                unique_states_visited=len(self._level_states),
            ))
            self._current_level_index += 1
            self._current_level_start_tick = tick
            self._level_rules_start = rules
            self._level_expansions = expansions
            self._level_states = set()
            self.game_log.levels_completed += 1

    def on_game_end(self, total_ticks: int, accuracy: float,
                    rules: int, expansions: int):
        if self.game_log:
            self.game_log.total_ticks = total_ticks
            self.game_log.final_accuracy = accuracy
            self.game_log.total_rules = rules
            self.game_log.total_ontology_expansions = expansions
            self.game_log.end_time = time.time()
