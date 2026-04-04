"""OllamaClient -- simple interface to local Ollama LLM.

Gracefully degrades to deterministic mock responses when Ollama is not
installed, enabling all tests to run without the dependency.
"""

from __future__ import annotations


class OllamaClient:
    """Query a local Ollama model. Falls back to mock if unavailable."""

    def __init__(
        self,
        model: str = "gemma3:12b",
        temperature: float = 0.3,
        max_tokens: int = 300,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._available = self._check_ollama()

    def _check_ollama(self) -> bool:
        """Check if the ollama Python package is importable."""
        try:
            import ollama  # noqa: F401
            return True
        except ImportError:
            return False

    @property
    def available(self) -> bool:
        return self._available

    def query(self, system_prompt: str, user_prompt: str) -> str:
        """Send a prompt to the LLM and return its text response.

        Falls back to a deterministic mock when Ollama is not installed.
        """
        if not self._available:
            return self._mock_response()

        import ollama
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        )
        return response["message"]["content"]

    def _mock_response(self) -> str:
        """Deterministic mock for testing without Ollama."""
        return (
            "ACTION: 1\n"
            "HYPOTHESIS: Exploring to discover game mechanics\n"
            "CONFIDENCE: low\n"
            "C1_EXPANSION: NONE\n"
            "REASONING: Taking first available action to observe what happens."
        )
