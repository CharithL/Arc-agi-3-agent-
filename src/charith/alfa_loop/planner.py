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

You also receive a spatial description of the current scene including grid
size and the (row, col) position of every object. You must:
  1. Identify which object is the CONTROLLABLE (the one the rules describe
     moving) and which is the TARGET (a goal object to reach).
  2. Compute the row/col distance from controllable to target.
  3. Translate that distance into a sequence of actions using the confirmed
     movement rules. For example, if the controllable must move up 14 rows
     and each "up" action moves 5 cells, that requires 3 "up" actions.

Plan a MULTI-ACTION sequence (typically 4-15 actions) — single-action plans
are rarely enough to reach a target. Prefer CONFIRMED rules; fall back to
PARTIALLY VERIFIED when needed. Never use REFUTED rules.

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
