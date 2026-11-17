"""LLM Client — supports Anthropic Claude API, Ollama, or mock.

Priority: Anthropic (if ANTHROPIC_API_KEY set) > Ollama (if running) > Mock

Two query modes:
* ``query(system, user)``        — free-text (Path-5 legacy path).
* ``query_structured(...)``      — schema-constrained decoding returning a JSON
                                   string that parses as a ``Decision`` (S3.5).
"""
from __future__ import annotations

import json
import os
from typing import Optional

from charith.config import AgentConfig
from charith.llm_agent.schema import Decision


def estimate_tokens(s: str) -> int:
    """Cheap token estimate (guide §2.1): ~4 chars per token."""
    return len(s) // 4


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
                MODEL_MAP = {
                    "auto": "claude-haiku-4-5-20251001",
                    "haiku": "claude-haiku-4-5-20251001",
                    "sonnet": "claude-sonnet-4-20250514",
                    "opus": "claude-opus-4-6",
                }
                self._model = MODEL_MAP.get(model, model)
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
            self._model = "qwen3:8b" if model == "auto" else model
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

    @property
    def available(self) -> bool:
        return self._backend != "mock"

    # ---- free-text (legacy) -----------------------------------------------

    def query(self, system_prompt: str, user_prompt: str) -> str:
        self._call_count += 1
        try:
            if self._backend == "anthropic":
                return self._query_anthropic(system_prompt, user_prompt)
            elif self._backend == "ollama":
                return self._query_ollama(system_prompt, user_prompt)
            else:
                return self._mock_response()
        except Exception:
            return self._mock_response()

    def _query_anthropic(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self.max_tokens,
            system=[{
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }],
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

    # ---- schema-constrained decoding (S3.5) -------------------------------

    def query_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        config: AgentConfig,
        legal_actions: list,
        incidents=None,
        first_call: bool = False,
    ) -> str:
        """Return a JSON string that parses as a Decision.

        Uses constrained decoding (Anthropic forced tool / Ollama ``format``).
        The mock returns a valid EXPLORE/UNKNOWN Decision so offline plumbing
        tests pass.
        """
        self._call_count += 1
        self._context_guard(system_prompt, user_prompt, config, incidents, first_call)
        try:
            if self._backend == "anthropic":
                return self._structured_anthropic(system_prompt, user_prompt, config)
            elif self._backend == "ollama":
                return self._structured_ollama(system_prompt, user_prompt, config)
            else:
                return self._mock_decision(legal_actions)
        except Exception as exc:
            if incidents is not None:
                incidents.log("LLM_ERROR", f"structured query failed: {exc}")
            # Degrade to a valid mock Decision so the retry loop can proceed.
            return self._mock_decision(legal_actions)

    def _context_guard(self, system_prompt, user_prompt, config, incidents, first_call):
        sys_t = estimate_tokens(system_prompt)
        usr_t = estimate_tokens(user_prompt)
        total = sys_t + usr_t
        if first_call:
            print(f"[LLM] context estimate: system={sys_t} user={usr_t} "
                  f"total={total} / num_ctx={config.num_ctx}")
        if total > 0.75 * config.num_ctx:
            msg = (f"context {total} tok exceeds 75% of num_ctx {config.num_ctx} "
                   f"— Ollama truncates the FRONT (system/C1) silently")
            print(f"[LLM][WARN] {msg}")
            if incidents is not None:
                incidents.log("CONTEXT_OVERFLOW", msg)

    def _structured_anthropic(self, system_prompt: str, user_prompt: str,
                              config: AgentConfig) -> str:
        tools = [{
            "name": "submit_decision",
            "description": "Submit your decision for this tick.",
            "input_schema": Decision.model_json_schema(),
        }]
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self.max_tokens,
            system=[{"type": "text", "text": system_prompt,
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user_prompt}],
            tools=tools,
            tool_choice={"type": "tool", "name": "submit_decision"},
            temperature=config.decision_temperature,
        )
        for block in response.content:
            if getattr(block, "type", None) == "tool_use":
                return json.dumps(block.input)
        raise ValueError("Anthropic returned no tool_use block")

    def _structured_ollama(self, system_prompt: str, user_prompt: str,
                           config: AgentConfig) -> str:
        from ollama import Client
        client = Client(host='http://localhost:11434')
        kwargs = dict(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            format=Decision.model_json_schema(),
            options={
                "temperature": config.decision_temperature,
                "num_ctx": config.num_ctx,
            },
            stream=False,
            keep_alive="30m",
        )
        # Disable thinking mode for per-tick calls if supported (qwen3/deepseek-r1).
        try:
            response = client.chat(think=False, **kwargs)
        except TypeError:
            response = client.chat(**kwargs)
        return response["message"]["content"]

    def _mock_decision(self, legal_actions: list) -> str:
        action_id = str(legal_actions[0]) if legal_actions else "1"
        decision = {
            "tick": 0,
            "hypothesis_id": "EXPLORE",
            "predicted_effect": {"kind": "UNKNOWN", "object_id": None, "detail": ""},
            "action": {"action_id": action_id, "target": None},
            "confidence": 0.2,
            "notes": "",
            "requests": [],
        }
        return json.dumps(decision)
