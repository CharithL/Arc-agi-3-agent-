"""ResponseParser -- extract structured action + metadata from LLM text output.

Robust to formatting variations: the LLM may produce extra whitespace,
different casing, missing fields, etc.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional


class ResponseParser:
    """Parse LLM output into a structured dict."""

    _RE_ACTION = re.compile(r"ACTION\s*:\s*(\d+)", re.IGNORECASE)
    _RE_HYPOTHESIS = re.compile(
        r"HYPOTHESIS\s*:\s*(.+?)(?=\n\s*(?:CONFIDENCE|C1_EXPANSION|REASONING|ACTION)|$)",
        re.IGNORECASE | re.DOTALL,
    )
    _RE_CONFIDENCE = re.compile(r"CONFIDENCE\s*:\s*(low|med|medium|high)", re.IGNORECASE)
    _RE_C1_EXPANSION = re.compile(
        r"C1_EXPANSION\s*:\s*(.+?)(?=\n\s*(?:REASONING|HYPOTHESIS|CONFIDENCE|ACTION)|$)",
        re.IGNORECASE | re.DOTALL,
    )
    _RE_REASONING = re.compile(
        r"REASONING\s*:\s*(.+?)$",
        re.IGNORECASE | re.DOTALL,
    )

    def parse(self, response_text: str, available_actions: List[int]) -> Dict:
        """Extract ACTION, HYPOTHESIS, CONFIDENCE, C1_EXPANSION, REASONING.

        Returns a dict with keys: action (int), hypothesis (str),
        confidence (str), c1_expansion (str|None), reasoning (str).
        Falls back to the first available action if parsing fails.
        """
        result: Dict = {
            "action": available_actions[0] if available_actions else 0,
            "hypothesis": "",
            "confidence": "low",
            "c1_expansion": None,
            "reasoning": "",
        }

        # ACTION
        m = self._RE_ACTION.search(response_text)
        if m:
            action = int(m.group(1))
            if action in available_actions:
                result["action"] = action
            # If not available, keep the fallback default

        # HYPOTHESIS
        m = self._RE_HYPOTHESIS.search(response_text)
        if m:
            result["hypothesis"] = m.group(1).strip()

        # CONFIDENCE
        m = self._RE_CONFIDENCE.search(response_text)
        if m:
            conf = m.group(1).lower()
            if conf == "medium":
                conf = "med"
            result["confidence"] = conf

        # C1_EXPANSION
        m = self._RE_C1_EXPANSION.search(response_text)
        if m:
            val = m.group(1).strip()
            if val.upper() != "NONE" and val:
                result["c1_expansion"] = val
            # else stays None

        # REASONING
        m = self._RE_REASONING.search(response_text)
        if m:
            result["reasoning"] = m.group(1).strip()

        return result
