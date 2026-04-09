"""
Phase 3: Verify. The novel phase.

For each LLM hypothesis, execute its test_action (an intervention do(X)),
compute the actual observation via structured diff, score via
compute_match_score, and assign status:

    match_score >= 0.70 → 'confirmed'
    match_score <  0.30 → 'refuted'
    otherwise           → 'ambiguous'

No LLM calls here. Pure mechanistic testing.
"""

from typing import List, Optional

from charith.perception.core_knowledge import CoreKnowledgePerception
from charith.causal_engine.table_model import ArcTableModel
from charith.causal_engine.error_analyzer import ArcErrorAnalyzer
from charith.full_stack.hypothesis_schema import Hypothesis
from charith.full_stack.match_score import compute_match_score
from charith.full_stack.percept_diff import diff_to_actual_observation


CONFIRM_THRESHOLD = 0.70
REFUTE_THRESHOLD = 0.30


class Verifier:
    def __init__(
        self,
        env,
        perception: CoreKnowledgePerception,
        table: ArcTableModel,
        error_analyzer: ArcErrorAnalyzer,
    ):
        self.env = env
        self.perception = perception
        self.table = table
        self.error_analyzer = error_analyzer
        self._step_counter = 0

    def verify(self, hypotheses: List[Hypothesis]) -> List[Hypothesis]:
        """
        Test each hypothesis in order. State drift between tests is
        accepted per design §2.4 (user-approved).
        """
        prev_action: Optional[int] = self.table.prev_action

        for h in hypotheses:
            if h.status == "untestable":
                continue

            obs_before = self.env.get_observation()
            percept_before = self.perception.perceive(obs_before.frame[0])

            obs_after, _reward, done, _info = self.env.step(h.test_action)
            percept_after = self.perception.perceive(obs_after.frame[0])

            actual = diff_to_actual_observation(percept_before, percept_after)
            score = compute_match_score(h.expected, actual)

            h.match_score = score
            h.actual_summary = self._summarise(actual)
            if score >= CONFIRM_THRESHOLD:
                h.status = "confirmed"
                predicted_right = True
            elif score < REFUTE_THRESHOLD:
                h.status = "refuted"
                predicted_right = False
            else:
                h.status = "ambiguous"
                predicted_right = False

            change_desc = f"direction={actual.controllable_direction},mag={actual.controllable_magnitude}"
            self.table.record(action=h.test_action, changes=[change_desc])

            self.error_analyzer.record(
                step=self._step_counter,
                action=h.test_action,
                predicted_right=predicted_right,
                prev_action=prev_action,
            )
            self._step_counter += 1
            prev_action = h.test_action

            if done:
                break

        return hypotheses

    def _summarise(self, actual) -> str:
        parts = []
        if actual.controllable_magnitude > 0:
            parts.append(f"moved {actual.controllable_direction} by {actual.controllable_magnitude}")
        if actual.any_color_changes:
            parts.append(f"{len(actual.any_color_changes)} color change(s)")
        if actual.new_objects:
            parts.append(f"{len(actual.new_objects)} new object(s)")
        if actual.removed_objects:
            parts.append(f"{len(actual.removed_objects)} removed")
        return ", ".join(parts) if parts else "no change"
