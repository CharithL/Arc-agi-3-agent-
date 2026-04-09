"""
Phase 2: Hypothesize.

One LLM call. Build prompt from evidence + current vocabulary, get
structured hypotheses back. Validate and normalize into Hypothesis objects.
"""

from typing import List, Tuple

from charith.alfa_loop.explorer import Evidence
from charith.full_stack.hypothesis_schema import (
    ExpectedOutcome, Hypothesis,
)


SYSTEM_PROMPT = """You are discovering the rules of an unknown grid game by hypothesis.
You just observed 8 actions being tried once each.

Generate one hypothesis for EACH action that produced a change. Do not group
multiple actions into one hypothesis even if they look similar — every action
needs its own test so the verifier can run each one independently.

IMPORTANT: Available actions are 1 through 8 only. Your test_action MUST be an
integer between 1 and 8 inclusive. Any other value is invalid and will be discarded.

CRITICAL: Each hypothesis must be TESTABLE. Your expected_outcome MUST be STRUCTURED
(schema below). Do not describe expected outcomes in prose. Only populate fields you
are actually predicting; leave others null.

ExpectedOutcome schema:
{
  "direction": "up|down|left|right|none",
  "magnitude_cells": 5,
  "object_ref": "controllable|red|blue|...",
  "color_change_to": "green",
  "object_appears": false,
  "object_disappears": false,
  "score_change": false,
  "no_effect": false
}

Respond ONLY with JSON:
{
  "hypotheses": [
    {
      "rule": "Action 1 moves the controllable up by 5 cells",
      "confidence": 0.8,
      "test_action": 1,
      "expected": {"direction": "up", "magnitude_cells": 5, "object_ref": "controllable"}
    }
  ],
  "goal_guess": "Move controllable to the target"
}
"""


class Hypothesizer:
    def __init__(self, llm, num_actions: int = 8):
        self.llm = llm
        self.num_actions = num_actions

    def generate(
        self,
        evidence: List[Evidence],
        active_expansions: List[str],
    ) -> Tuple[List[Hypothesis], str]:
        evidence_text = "\n".join(f"  {e.description}" for e in evidence)
        user_prompt = (
            f"Exploration results (each action tried once):\n{evidence_text}\n\n"
            f"Current vocabulary: {active_expansions}\n"
            f"What are the game's rules? What is the goal?"
        )

        result = self.llm.reason_json(SYSTEM_PROMPT, user_prompt)

        raw_hyps = result.get("hypotheses", []) if isinstance(result, dict) else []
        goal = result.get("goal_guess", "") if isinstance(result, dict) else ""

        parsed: List[Hypothesis] = []
        for rh in raw_hyps:
            try:
                test_action = int(rh.get("test_action", -1))
            except (TypeError, ValueError):
                test_action = -1

            expected_dict = rh.get("expected", {}) or {}
            expected = ExpectedOutcome(
                direction=expected_dict.get("direction"),
                magnitude_cells=expected_dict.get("magnitude_cells"),
                object_ref=expected_dict.get("object_ref"),
                color_change_to=expected_dict.get("color_change_to"),
                object_appears=expected_dict.get("object_appears"),
                object_disappears=expected_dict.get("object_disappears"),
                score_change=expected_dict.get("score_change"),
                no_effect=bool(expected_dict.get("no_effect", False)),
            )

            h = Hypothesis(
                rule=str(rh.get("rule", "")),
                confidence=float(rh.get("confidence", 0.5)),
                test_action=test_action,
                expected=expected,
            )

            if not (1 <= h.test_action <= self.num_actions):
                h.status = "untestable"

            parsed.append(h)

        return parsed, goal
