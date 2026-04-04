"""LLM Client — supports Anthropic Claude API, Ollama, or mock.

Priority: Anthropic (if ANTHROPIC_API_KEY set) > Ollama (if running) > Mock
"""
from __future__ import annotations
import os


class LLMClient:
    """Unified LLM client. Uses best available backend."""

    def __init__(self, model: str = "auto", temperature: float = 0.3,
                 max_tokens: int = 500):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._backend = None
        self._model = model
        self._call_count = 0

        # Try Anthropic first
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if api_key and len(api_key) > 10:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=api_key)
                self._backend = "anthropic"
                self._model = "claude-sonnet-4-20250514" if model == "auto" else model
                print(f"[LLM] Using Anthropic API ({self._model})")
                return
            except Exception as e:
                print(f"[LLM] Anthropic failed: {e}")

        # Try Ollama
        try:
            from ollama import Client
            client = Client(host='http://localhost:11434')
            client.list()
            self._backend = "ollama"
            self._model = "gemma4" if model == "auto" else model
            print(f"[LLM] Using Ollama ({self._model})")
            return
        except Exception:
            pass

        # Mock fallback
        self._backend = "mock"
        print("[LLM] Using mock (no LLM available)")

    @property
    def backend(self) -> str:
        return self._backend

    def query(self, system_prompt: str, user_prompt: str) -> str:
        self._call_count += 1

        if self._backend == "anthropic":
            return self._query_anthropic(system_prompt, user_prompt)
        elif self._backend == "ollama":
            return self._query_ollama(system_prompt, user_prompt)
        else:
            return self._mock_response()

    def _query_anthropic(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=self.temperature,
        )
        text = response.content[0].text
        return text.encode('ascii', errors='replace').decode('ascii')

    def _query_ollama(self, system_prompt: str, user_prompt: str) -> str:
        from ollama import Client
        client = Client(host='http://localhost:11434')
        response = client.chat(
            model=self._model,
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
        return text.encode('ascii', errors='replace').decode('ascii')

    def _mock_response(self) -> str:
        return (
            "ACTION: 1\n"
            "HYPOTHESIS: Exploring to discover game mechanics\n"
            "CONFIDENCE: low\n"
            "C1_EXPANSION: NONE\n"
            "REASONING: Taking first available action to observe what happens."
        )
