"""
Phase 1: Explore.

Try each action (1..num_actions) once. Record what happens in the causal
table. No LLM calls. No error recording (no predictions yet).

If the environment signals done=True mid-exploration, return immediately
with whatever evidence was gathered.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from charith.perception.core_knowledge import CoreKnowledgePerception, StructuredPercept
from charith.causal_engine.table_model import ArcTableModel
from charith.full_stack.percept_diff import diff_to_actual_observation


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
            grid_before = obs_before.frame[0] if obs_before is not None else None
            percept_before = self.perception.perceive(grid_before)

            obs_after, reward, done, _info = self.env.step(action_id)
            grid_after = obs_after.frame[0] if obs_after is not None else None
            percept_after = self.perception.perceive(grid_after)

            changes = self._summarise_changes(
                percept_before, percept_after, grid_before, grid_after
            )
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

    def _summarise_changes(
        self,
        before: StructuredPercept,
        after: StructuredPercept,
        grid_before: Optional[np.ndarray] = None,
        grid_after: Optional[np.ndarray] = None,
    ) -> list:
        diffs = []
        if len(before.objects) != len(after.objects):
            diffs.append(f"object count {len(before.objects)}->{len(after.objects)}")

        # Structured displacement: reuse the same diff adapter Phase 3 uses so
        # Phase 1 evidence describes motion in the same vocabulary ("moved up
        # by 5") that the LLM must match in Phase 2 hypotheses. Aligns the
        # Explorer's telemetry with the Verifier's scoring rubric.
        try:
            actual = diff_to_actual_observation(before, after)
            if actual.controllable_magnitude > 0:
                diffs.append(
                    f"moved {actual.controllable_direction} "
                    f"by {actual.controllable_magnitude} cells"
                )
            if actual.any_color_changes:
                diffs.append(f"{len(actual.any_color_changes)} color change(s)")
            if actual.new_objects:
                diffs.append(f"{len(actual.new_objects)} new object(s)")
            if actual.removed_objects:
                diffs.append(f"{len(actual.removed_objects)} removed object(s)")
        except Exception:
            pass

        # Raw pixel-diff fallback: when structured perception sees nothing but
        # the underlying grid still changed, report the raw count so the LLM
        # at least knows SOMETHING happened.
        if not diffs and grid_before is not None and grid_after is not None:
            try:
                gb = np.asarray(grid_before)
                ga = np.asarray(grid_after)
                if gb.shape == ga.shape:
                    changed = int(np.sum(gb != ga))
                    if changed > 0:
                        diffs.append(f"{changed} cells changed")
            except Exception:
                pass

        return diffs

    def _describe(self, action_id: int, changes: list) -> str:
        if not changes:
            return f"action {action_id}: no change"
        return f"action {action_id}: {'; '.join(changes)}"
