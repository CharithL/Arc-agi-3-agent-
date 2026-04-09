"""Agent budget caps. Frozen dataclass to prevent accidental mutation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentBudgets:
    max_actions_per_level: int = 150
    max_llm_calls_per_level: int = 8
    max_attempts_per_level: int = 3
    max_expansion_cycles_per_attempt: int = 2
    explore_num_actions: int = 8
    max_hypotheses_to_verify: int = 8
    max_plan_length: int = 20
    consecutive_surprises_to_halt: int = 3
    # After a plan exhausts without reaching the target, re-observe the
    # controllable's position and plan again this many times before giving
    # up the attempt. Total actions stay under max_actions_per_level.
    max_replans_per_attempt: int = 3
