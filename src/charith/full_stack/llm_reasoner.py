"""
Thin wrapper around existing OllamaClient.

Adds:
  - reason_json(): parses markdown fences and returns dict (or parse_error)
  - Duck-typed: accepts anything with a `generate(system, user, ...)` method,
    which is what the mock and real client both implement.
"""

import json
import re
from typing import Any, Dict


_FENCE_RE = re.compile(r"```(?:json)?\s*\n(.*?)\n```", re.DOTALL)


class LLMReasoner:
    def __init__(self, backend):
        """
        backend: any object with `generate(system_prompt, user_prompt, ...) -> str`.
        Real: charith.llm_agent.ollama_client.OllamaClient
        """
        self.backend = backend
        self.call_count = 0

    def reason_json(self, system: str, user: str) -> Dict[str, Any]:
        self.call_count += 1
        raw = self.backend.generate(system, user + "\n\nRespond ONLY with valid JSON.")
        return self._parse(raw)

    def reason(self, system: str, user: str) -> str:
        self.call_count += 1
        return self.backend.generate(system, user)

    @staticmethod
    def _parse(raw) -> Dict[str, Any]:
        if not isinstance(raw, str):
            return {"raw": raw, "parse_error": True}

        stripped = raw.strip()

        m = _FENCE_RE.search(stripped)
        if m:
            stripped = m.group(1)

        try:
            return json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            return {"raw": raw, "parse_error": True}
