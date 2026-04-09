"""
Canned-response LLM for testing.

Routes responses by keyword in the system prompt:
  - "discover" or "hypothesis" → hypothesize_response
  - "expansion"                → expansion_response
  - "planning" or "plan"       → plan_response

All responses are dicts matching the real reason_json output shape.
"""

from typing import Any, Dict, List, Optional, Tuple


DEFAULT_HYPOTHESIZE_RESPONSE = {
    "hypotheses": [
        {
            "rule": "Action 1 moves the controllable up by 5 cells",
            "confidence": 0.8,
            "test_action": 1,
            "expected": {
                "direction": "up",
                "magnitude_cells": 5,
                "object_ref": "controllable",
                "no_effect": False,
            },
        }
    ],
    "goal_guess": "Move controllable to target",
}

DEFAULT_EXPANSION_RESPONSE = {"type": "none", "reason": "errors are random"}

DEFAULT_PLAN_RESPONSE = {"plan": [1, 1, 1, 1], "reasoning": "repeat action 1"}


class MockOllamaReasoner:
    """Returns fixed JSON responses routed by keyword in system prompt."""

    def __init__(
        self,
        hypothesize_response: Optional[Dict] = None,
        expansion_response: Optional[Dict] = None,
        plan_response: Optional[Dict] = None,
    ):
        self.hypothesize_response = hypothesize_response or DEFAULT_HYPOTHESIZE_RESPONSE
        self.expansion_response = expansion_response or DEFAULT_EXPANSION_RESPONSE
        self.plan_response = plan_response or DEFAULT_PLAN_RESPONSE
        self.call_count: int = 0
        self.calls: List[Tuple[str, str]] = []

    def reason_json(self, system: str, user: str) -> Dict[str, Any]:
        self.call_count += 1
        self.calls.append((system, user))

        lower = system.lower()
        if "discover" in lower or "hypothesis" in lower or "hypotheses" in lower:
            return self.hypothesize_response
        if "expansion" in lower:
            return self.expansion_response
        if "planning" in lower or "plan " in lower or "action sequence" in lower:
            return self.plan_response
        return {"raw": "unknown", "parse_error": True}

    def reason(self, system: str, user: str) -> str:
        """Fallback plain-text method for counterfactual queries (unused in Milestone 1)."""
        self.call_count += 1
        return "mock response"
