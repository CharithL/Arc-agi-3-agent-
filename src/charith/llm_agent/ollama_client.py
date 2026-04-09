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
        # One-time log so callers can see what got through
        self._last_retry_mode: str = ""

    def _check_ollama(self) -> bool:
        """Check if the ollama Python package is importable and server reachable."""
        try:
            from ollama import Client
            client = Client(host='http://localhost:11434')
            client.list()  # Quick ping to verify server is running
            return True
        except Exception:
            return False

    @property
    def available(self) -> bool:
        return self._available

    def query(self, system_prompt: str, user_prompt: str) -> str:
        """Send a prompt to the LLM and return its text response.

        Falls back to a deterministic mock when Ollama is not installed.

        Resilience: if the chat-API returns an empty string (observed with some
        Gemma variants that ignore the system field), retry once with the
        generate API passing system+user concatenated into a single prompt.
        """
        if not self._available:
            return self._mock_response()

        from ollama import Client
        client = Client(host='http://localhost:11434')
        self._last_retry_mode = "chat"
        response = client.chat(
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
        text = response["message"]["content"]

        if not text or not text.strip():
            # Empty response -> retry with generate API and concat prompt
            self._last_retry_mode = "generate_concat"
            combined = f"{system_prompt}\n\n{user_prompt}"
            gen_response = client.generate(
                model=self.model,
                prompt=combined,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
                stream=False,
            )
            text = gen_response.get("response", "") if isinstance(gen_response, dict) \
                else getattr(gen_response, "response", "")

        # Strip non-ASCII chars that break Windows console encoding
        return text.encode('ascii', errors='replace').decode('ascii')

    def _mock_response(self) -> str:
        """Deterministic mock for testing without Ollama."""
        return (
            "ACTION: 1\n"
            "HYPOTHESIS: Exploring to discover game mechanics\n"
            "CONFIDENCE: low\n"
            "C1_EXPANSION: NONE\n"
            "REASONING: Taking first available action to observe what happens."
        )
