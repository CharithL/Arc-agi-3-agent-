"""
Phase 5: Plan.

Given verified hypotheses and a goal, ask the LLM for an action sequence.
Zero-confirmed edge case → emergency fallback using table predictions,
per design §6.4 (Option C: when LLM fails, fall back to mechanistic
component).
"""

from typing import List
import math
import random

from charith.causal_engine.table_model import ArcTableModel
from charith.full_stack.hypothesis_schema import Hypothesis


SYSTEM_PROMPT = """You are planning actions in a grid game.
You receive causal rules that have been tested through intervention.

Rules come in three tiers:
  CONFIRMED: match score >= 0.70 (high-confidence ground truth)
  PARTIALLY VERIFIED: match score in [0.30, 0.70) (some prediction matched
    actual outcome, use with caution)
  REFUTED: match score < 0.30 (do NOT use these actions for this effect)

You also receive a spatial description of the current scene. The orchestrator
has already identified the CONTROLLABLE object (the one that moves when you
act) and a best-guess TARGET object. A DISTANCE line gives you delta_row and
delta_col from controllable to target.

REQUIRED PROCEDURE:
  1. Read the DISTANCE line: delta_row is the VERTICAL gap (positive = target
     is below the controllable; negative = target is above). delta_col is
     the HORIZONTAL gap (positive = target is to the right; negative = to
     the left).
  2. Translate each component into a count of actions:
        row_moves  = abs(delta_row)  / magnitude_of_up_or_down_action
        col_moves  = abs(delta_col)  / magnitude_of_left_or_right_action
     Round UP when the remainder is non-trivial.
  3. INTERLEAVE the row and column actions into a single plan. Do NOT only
     use one direction — a 2D target requires both vertical and horizontal
     movement. Example: if you need 3 down and 2 right with "2=down, 4=right",
     output [2, 4, 2, 4, 2] (or any interleaving).

Plan a MULTI-ACTION sequence (typically 4-15 actions). Prefer CONFIRMED rules;
fall back to PARTIALLY VERIFIED when needed. Never use REFUTED rules. If
multiple confirmed rules describe the same direction, pick one and stick with it.

Respond ONLY with JSON: {"plan": [action_id, action_id, ...], "reasoning": "..."}
"""


class Planner:
    def __init__(self, llm, table: ArcTableModel, max_plan_length: int = 20):
        self.llm = llm
        self.table = table
        self.max_plan_length = max_plan_length

    def plan(
        self,
        verified: List[Hypothesis],
        goal: str,
        state_desc: str,
        num_actions: int = 8,
    ) -> List[int]:
        confirmed = [h for h in verified if h.status == "confirmed"]
        ambiguous = [h for h in verified if h.status == "ambiguous"]
        refuted = [h for h in verified if h.status == "refuted"]

        # Emergency fallback only when there is no usable tier at all
        if not confirmed and not ambiguous:
            return self._emergency_fallback(num_actions)

        confirmed_text = "\n".join(
            f"  CONFIRMED: action {h.test_action} -> {h.actual_summary or h.rule}"
            for h in confirmed
        ) or "  (none)"
        ambiguous_text = "\n".join(
            f"  PARTIALLY VERIFIED: action {h.test_action} -> "
            f"{h.actual_summary or h.rule} (score={h.match_score:.2f})"
            for h in ambiguous
        ) or "  (none)"
        refuted_text = "\n".join(f"  REFUTED: {h.rule}" for h in refuted) or "  (none)"

        user = (
            f"Current state: {state_desc}\n"
            f"Goal: {goal}\n\n"
            f"CONFIRMED rules (high-confidence ground truth):\n{confirmed_text}\n\n"
            f"PARTIALLY VERIFIED rules (use with caution):\n{ambiguous_text}\n\n"
            f"REFUTED hypotheses (do NOT use):\n{refuted_text}\n\n"
            f"Plan the shortest action sequence to reach the goal."
        )

        result = self.llm.reason_json(SYSTEM_PROMPT, user)

        raw_plan = result.get("plan", []) if isinstance(result, dict) else []
        valid_plan: List[int] = []
        for a in raw_plan:
            try:
                ia = int(a)
                if 1 <= ia <= num_actions:
                    valid_plan.append(ia)
            except (TypeError, ValueError):
                continue

        if not valid_plan:
            return self._emergency_fallback(num_actions)

        return valid_plan[: self.max_plan_length]

    def _emergency_fallback(self, num_actions: int) -> List[int]:
        """
        When no confirmed hypothesis exists, fall back to table-weighted
        random walk. Per design §6.4: when the LLM fails, use the
        mechanistic component (the table), not the LLM.
        """
        weights = []
        for action_id in range(1, num_actions + 1):
            pred = self.table.predict(
                action=action_id,
                target=None,
                prev_action=self.table.prev_action,
            )
            confidence = pred[1]
            obs = self.table.single_table.get(action_id, [])
            n = len(obs)
            weight = confidence * math.log1p(n)
            weights.append(max(weight, 0.05))

        total = sum(weights)
        probs = [w / total for w in weights]

        rng = random.Random(0)
        plan: List[int] = []
        for _ in range(10):
            r = rng.random()
            cum = 0.0
            for idx, p in enumerate(probs):
                cum += p
                if r <= cum:
                    plan.append(idx + 1)
                    break
        return plan
