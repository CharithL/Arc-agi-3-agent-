"""
Phase 1: Explore.

Try each action (1..num_actions) once. Record what happens in the causal
table. No LLM calls. No error recording (no predictions yet).

If the environment signals done=True mid-exploration, return immediately
with whatever evidence was gathered.
"""

from dataclasses import dataclass
from typing import List

from charith.perception.core_knowledge import CoreKnowledgePerception, StructuredPercept
from charith.causal_engine.table_model import ArcTableModel


@dataclass
class Evidence:
    action: int
    percept_before: StructuredPercept
    percept_after: StructuredPercept
    changes: list        # raw diff summary (strings for Milestone 1)
    description: str     # short text for LLM prompt
    reward: float
    done: bool


class Explorer:
    def __init__(self, env, perception: CoreKnowledgePerception, table: ArcTableModel):
        self.env = env
        self.perception = perception
        self.table = table

    def explore(self, num_actions: int = 8) -> List[Evidence]:
        evidence: List[Evidence] = []

        for action_id in range(1, num_actions + 1):
            obs_before = self.env.get_observation()
            percept_before = self.perception.perceive(obs_before.frame[0])

            obs_after, reward, done, _info = self.env.step(action_id)
            percept_after = self.perception.perceive(obs_after.frame[0])

            changes = self._summarise_changes(percept_before, percept_after)
            description = self._describe(action_id, changes)

            self.table.record(action=action_id, changes=[description])

            evidence.append(
                Evidence(
                    action=action_id,
                    percept_before=percept_before,
                    percept_after=percept_after,
                    changes=changes,
                    description=description,
                    reward=float(reward),
                    done=bool(done),
                )
            )

            if done:
                break

        return evidence

    def _summarise_changes(self, before: StructuredPercept, after: StructuredPercept) -> list:
        diffs = []
        if len(before.objects) != len(after.objects):
            diffs.append(f"object count {len(before.objects)}->{len(after.objects)}")
        return diffs

    def _describe(self, action_id: int, changes: list) -> str:
        if not changes:
            return f"action {action_id}: no change"
        return f"action {action_id}: {'; '.join(changes)}"
