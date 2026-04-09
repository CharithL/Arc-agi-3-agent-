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

from charith.perception.core_knowledge import CoreKnowledgePerception
from charith.causal_engine.table_model import ArcTableModel
from charith.causal_engine.error_analyzer import ArcErrorAnalyzer
from charith.full_stack.percept_diff import diff_to_actual_observation


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
            percept_before = self.perception.perceive(obs_before.frame[0])

            prediction = self.table.predict(
                action=action_id,
                target=None,
                prev_action=prev_action,
            )

            obs_after, _reward, done, _info = self.env.step(action_id)
            percept_after = self.perception.perceive(obs_after.frame[0])
            actual = diff_to_actual_observation(percept_before, percept_after)

            change_desc = (
                f"direction={actual.controllable_direction},mag={actual.controllable_magnitude}"
            )
            self.table.record(action=action_id, changes=[change_desc])
            actions_taken += 1

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
