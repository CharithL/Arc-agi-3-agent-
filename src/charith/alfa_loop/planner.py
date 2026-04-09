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
You receive VERIFIED causal rules (tested through intervention — ground truth).
You also have a goal description and a description of the current state.

Plan an efficient sequence of actions to achieve the goal.
Use only actions that appear in CONFIRMED rules. Avoid REFUTED rules.

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

        if not confirmed:
            return self._emergency_fallback(num_actions)

        confirmed_text = "\n".join(
            f"  CONFIRMED: action {h.test_action} → {h.actual_summary or h.rule}"
            for h in confirmed
        )
        refuted = [h for h in verified if h.status == "refuted"]
        refuted_text = "\n".join(f"  REFUTED: {h.rule}" for h in refuted)

        user = (
            f"Current state: {state_desc}\n"
            f"Goal: {goal}\n\n"
            f"Verified rules (ground truth):\n{confirmed_text}\n\n"
            f"Refuted hypotheses (do NOT use):\n{refuted_text or '  (none)'}\n\n"
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
