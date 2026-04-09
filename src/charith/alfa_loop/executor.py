"""
Phase 6: Execute + monitor δ.

Walk the plan. At each step:
  - predict via table
  - execute
  - compare prediction to actual
  - record
  - halt if consecutive_surprises hits threshold
  - success if env returns done=True
"""

from typing import Dict, List

import numpy as np

from charith.perception.core_knowledge import CoreKnowledgePerception
from charith.causal_engine.table_model import ArcTableModel
from charith.causal_engine.error_analyzer import ArcErrorAnalyzer
from charith.full_stack.percept_diff import diff_to_actual_observation


# When >= this fraction of cells change after one action, it's a level
# transition or global reset, not a normal controllable effect.
_LEVEL_TRANSITION_CHANGE_FRACTION = 0.5


def _has_valid_grid(obs) -> bool:
    """
    True when `obs` has a non-empty .frame list whose first element is a
    usable grid (not None). Guards every Executor loop iteration against
    the real arc_agi SDK quirk of returning frame=[] mid-episode.
    """
    if obs is None:
        return False
    frame_list = getattr(obs, "frame", None)
    if not frame_list:
        return False
    return frame_list[0] is not None


class Executor:
    def __init__(
        self,
        env,
        perception: CoreKnowledgePerception,
        table: ArcTableModel,
        error_analyzer: ArcErrorAnalyzer,
        halt_threshold: int = 3,
    ):
        self.env = env
        self.perception = perception
        self.table = table
        self.error_analyzer = error_analyzer
        self.halt_threshold = halt_threshold
        self._step_counter = 0

    def execute(self, plan: List[int]) -> Dict:
        consecutive_surprises = 0
        actions_taken = 0
        prev_action = self.table.prev_action

        for action_id in plan:
            obs_before = self.env.get_observation()
            # Defensive: env may hand us a frame with no usable grid (real
            # arc_agi SDK has been observed returning frame=[] mid-episode
            # on ls20). Stop cleanly rather than crashing with IndexError.
            if not _has_valid_grid(obs_before):
                return {
                    "completed": False,
                    "actions_taken": actions_taken,
                    "reason": "env_empty_frame_before",
                }
            grid_before = obs_before.frame[0]
            percept_before = self.perception.perceive(grid_before)

            prediction = self.table.predict(
                action=action_id,
                target=None,
                prev_action=prev_action,
            )

            obs_after, _reward, done, _info = self.env.step(action_id)
            if not _has_valid_grid(obs_after):
                # The step was dispatched but the env couldn't produce a
                # usable post-state. Count the action as taken (it hit the
                # env) and return with the done flag the env gave us.
                return {
                    "completed": bool(done),
                    "actions_taken": actions_taken + 1,
                    "reason": "env_empty_frame_after",
                }
            grid_after = obs_after.frame[0]
            percept_after = self.perception.perceive(grid_after)
            actual = diff_to_actual_observation(percept_before, percept_after)

            # Detect level transition: when nearly every cell changes at
            # once, the step wasn't a normal movement — it's a scene reset.
            # Don't penalize the table or count it as a surprise.
            is_level_transition = self._is_level_transition(grid_before, grid_after)

            change_desc = (
                "level_transition" if is_level_transition
                else f"direction={actual.controllable_direction},mag={actual.controllable_magnitude}"
            )
            self.table.record(action=action_id, changes=[change_desc])
            actions_taken += 1

            if is_level_transition:
                # A full-grid redraw after our action is the game signalling
                # we reached a goal and the level advanced. Treat it as
                # completion — the plan worked, even if the env's done flag
                # didn't propagate (some SDKs only set done on game-end, not
                # level-end).
                self.error_analyzer.record(
                    step=self._step_counter,
                    action=action_id,
                    predicted_right=True,
                    prev_action=prev_action,
                )
                self._step_counter += 1
                return {
                    "completed": True,
                    "actions_taken": actions_taken,
                    "reason": "level_transition_detected",
                }

            matched = self._matches(prediction, actual)
            if matched:
                consecutive_surprises = 0
            else:
                consecutive_surprises += 1

            self.error_analyzer.record(
                step=self._step_counter,
                action=action_id,
                predicted_right=matched,
                prev_action=prev_action,
            )
            self._step_counter += 1
            prev_action = action_id

            if consecutive_surprises >= self.halt_threshold:
                return {
                    "completed": False,
                    "actions_taken": actions_taken,
                    "reason": "delta_spike",
                }

            if done:
                return {
                    "completed": True,
                    "actions_taken": actions_taken,
                    "reason": "success",
                }

        return {
            "completed": False,
            "actions_taken": actions_taken,
            "reason": "plan_exhausted",
        }

    def _is_level_transition(self, grid_before, grid_after) -> bool:
        """Return True if a majority of cells changed — likely a scene reset."""
        try:
            gb = np.asarray(grid_before)
            ga = np.asarray(grid_after)
            if gb.shape != ga.shape or gb.size == 0:
                return False
            frac_changed = float(np.sum(gb != ga)) / float(gb.size)
            return frac_changed >= _LEVEL_TRANSITION_CHANGE_FRACTION
        except Exception:
            return False

    def _matches(self, prediction, actual) -> bool:
        """
        Coarse match: if confidence is low, accept anything.
        Otherwise: require some change if the table predicted effects.
        """
        _predicts_effect, confidence, _source = prediction
        if confidence < 0.3:
            return True
        return (
            actual.controllable_magnitude > 0
            or bool(actual.any_color_changes)
            or bool(actual.new_objects)
            or bool(actual.removed_objects)
        )
