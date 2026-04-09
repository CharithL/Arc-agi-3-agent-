"""
Phase 4: Error Check.

If error_analyzer detects structured errors, ask the LLM what expansion
is needed and apply it to the table. 0 or 1 LLM calls.
"""

from typing import Dict

from charith.causal_engine.table_model import ArcTableModel
from charith.causal_engine.error_analyzer import ArcErrorAnalyzer


SYSTEM_PROMPT = """You analyze error patterns in a game-learning system.
The system currently has a limited vocabulary and its errors are showing structure.

Available expansions:
- "sequential": Remember previous action. Use when errors depend on action order.
- "context": Remember game state. Use when same action has different effects in different states.
- "none": No expansion needed.

Respond ONLY with JSON: {"type": "sequential|context|none", "reason": "one sentence"}
"""


class ErrorChecker:
    def __init__(self, table: ArcTableModel, analyzer: ArcErrorAnalyzer, llm):
        self.table = table
        self.analyzer = analyzer
        self.llm = llm

    def check(self) -> Dict:
        analysis = self.analyzer.analyze()

        if not analysis.get("sufficient_data"):
            return {"expanded": False, "reason": "insufficient_data"}

        if not analysis.get("any_structure"):
            return {"expanded": False, "reason": "random_errors"}

        user_prompt = f"Error analysis report:\n\n{analysis.get('summary', '')}"
        result = self.llm.reason_json(SYSTEM_PROMPT, user_prompt)

        if not isinstance(result, dict):
            return {"expanded": False, "reason": "llm_parse_error"}

        exp_type = result.get("type", "none")
        reason = result.get("reason", "")

        if exp_type == "none":
            return {"expanded": False, "reason": f"llm_said_none: {reason}"}

        applied = self.table.enable_expansion(exp_type, reason)
        return {
            "expanded": applied,
            "expansion_type": exp_type,
            "reason": reason,
        }
