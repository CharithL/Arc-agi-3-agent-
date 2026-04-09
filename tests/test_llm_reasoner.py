"""Tests for LLMReasoner adapter."""
from charith.full_stack.llm_reasoner import LLMReasoner


class _FakeOllama:
    def __init__(self, response_text):
        self.response_text = response_text

    def generate(self, system_prompt, user_prompt, **kwargs):
        return self.response_text


def test_reason_json_parses_clean_json():
    ollama = _FakeOllama('{"type": "sequential", "reason": "x"}')
    llm = LLMReasoner(ollama)
    result = llm.reason_json("sys", "user")
    assert result["type"] == "sequential"


def test_reason_json_handles_markdown_fence():
    ollama = _FakeOllama('```json\n{"type": "none"}\n```')
    llm = LLMReasoner(ollama)
    result = llm.reason_json("sys", "user")
    assert result["type"] == "none"


def test_reason_json_returns_parse_error_dict_on_invalid():
    ollama = _FakeOllama("not json at all")
    llm = LLMReasoner(ollama)
    result = llm.reason_json("sys", "user")
    assert result.get("parse_error") is True
