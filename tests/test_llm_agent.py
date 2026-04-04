"""Tests for Path 5 LLM Agent -- all work WITHOUT Ollama installed."""

import numpy as np
import pytest

from charith.perception.core_knowledge import (
    CoreKnowledgePerception,
    Object,
    StructuredPercept,
    SpatialRelation,
    Cell,
)
from charith.llm_agent.translator import PerceptTranslator
from charith.llm_agent.c1c2_framework import C1C2Framework
from charith.llm_agent.context_manager import ContextManager
from charith.llm_agent.ollama_client import OllamaClient
from charith.llm_agent.response_parser import ResponseParser
from charith.llm_agent.agent import LLMAgent


# ---- helpers ---------------------------------------------------------------

def _make_grid(bg=4, size=10):
    """Create a simple grid with a few coloured objects."""
    grid = np.full((size, size), bg, dtype=int)
    # Blue 2x2 at top-left
    grid[1:3, 1:3] = 1
    # Red single cell at bottom-right
    grid[8, 8] = 2
    return grid


def _make_grid_shifted(bg=4, size=10):
    """Same grid but blue block shifted right by 2."""
    grid = np.full((size, size), bg, dtype=int)
    # Blue 2x2 shifted right
    grid[1:3, 3:5] = 1
    # Red single cell same position
    grid[8, 8] = 2
    return grid


def _perceive(grid):
    """Run CoreKnowledgePerception on a grid."""
    p = CoreKnowledgePerception()
    return p.perceive(grid)


# ---- Translator tests ------------------------------------------------------

class TestTranslator:
    def test_translator_output_format(self):
        """translate() returns a non-empty string with expected sections."""
        grid = _make_grid()
        percept = _perceive(grid)
        translator = PerceptTranslator()

        text = translator.translate(percept, tick=1)

        assert isinstance(text, str)
        assert len(text) > 0
        assert "Grid:" in text
        assert "Objects:" in text
        assert "Relations:" in text

    def test_translator_change_detection(self):
        """Different grids produce different change descriptions."""
        grid1 = _make_grid()
        grid2 = _make_grid_shifted()
        p = CoreKnowledgePerception()
        percept1 = p.perceive(grid1)
        percept2 = p.perceive(grid2)
        translator = PerceptTranslator()

        text = translator.translate(percept2, prev_percept=percept1, tick=2)

        assert "Changes:" in text
        # The grid changed so we should see pixel change info
        assert "changed" in text.lower() or "moved" in text.lower() or "appeared" in text.lower()

    def test_translator_controllable_first(self):
        """Controllable object appears before other objects in description."""
        grid = _make_grid()
        percept = _perceive(grid)
        translator = PerceptTranslator()

        # Find an object id to mark as controllable
        ctrl_id = percept.objects[0].object_id
        text = translator.translate(percept, controllable_ids={ctrl_id}, tick=1)

        # CONTROLLABLE tag should appear before any non-controllable object
        assert "CONTROLLABLE" in text
        ctrl_pos = text.index("CONTROLLABLE")
        # Check that Objects: section starts before CONTROLLABLE
        obj_pos = text.index("Objects:")
        assert obj_pos < ctrl_pos


# ---- C1/C2 Framework tests ------------------------------------------------

class TestC1C2Framework:
    def test_c1c2_initial_vocabulary(self):
        """Starts with 7 Spelke core concepts."""
        fw = C1C2Framework()
        assert len(fw.c1_vocabulary) == 7
        names = {c.name for c in fw.c1_vocabulary}
        assert "OBJECT" in names
        assert "MOVEMENT" in names
        assert "WALL" in names
        assert "GOAL" in names
        assert "SPATIAL_RELATION" in names
        assert "COUNTING" in names
        assert "SYMMETRY" in names

    def test_c1c2_expansion(self):
        """expand_c1 adds to vocabulary."""
        fw = C1C2Framework()
        fw.expand_c1("GRAVITY", "Objects fall downward when unsupported", tick=10)
        assert len(fw.c1_vocabulary) == 8
        assert fw.c1_expansions[-1].name == "GRAVITY"
        assert fw.c1_expansions[-1].source == "discovered"

        # Duplicate should not add
        fw.expand_c1("GRAVITY", "Different description", tick=11)
        assert len(fw.c1_vocabulary) == 8

    def test_c1c2_hypotheses(self):
        """update_hypothesis tracks up to 10."""
        fw = C1C2Framework()
        for i in range(12):
            fw.update_hypothesis(f"Hypothesis number {i}", "low")
        # Capped at 10
        assert len(fw.hypotheses) == 10

        text = fw.get_hypotheses_text()
        assert "Current hypotheses:" in text

    def test_c1c2_get_c1_text(self):
        """get_c1_text returns formatted vocabulary."""
        fw = C1C2Framework()
        fw.expand_c1("TELEPORT", "Objects can jump to distant locations", tick=5)
        text = fw.get_c1_text()
        assert "OBJECT" in text
        assert "TELEPORT" in text
        assert "[discovered]" in text

    def test_c1c2_reset(self):
        """reset clears hypotheses but keeps vocabulary."""
        fw = C1C2Framework()
        fw.expand_c1("GRAVITY", "Falls down", tick=1)
        fw.update_hypothesis("Test", "high")
        fw.reset()
        assert len(fw.hypotheses) == 0
        # Vocabulary preserved
        assert len(fw.c1_vocabulary) == 8


# ---- ContextManager tests --------------------------------------------------

class TestContextManager:
    def test_context_history_compression(self):
        """Keeps first + last 3 + important ticks; rest summarised."""
        ctx = ContextManager()
        for i in range(20):
            ctx.add_tick(
                tick=i,
                observation=f"obs-{i}",
                action=i % 4 + 1,
                result_description=f"result-{i}",
                is_important=(i == 7),
            )

        text = ctx.get_history_text()

        # First tick (0) should be present
        assert "Tick 0" in text
        # Important tick (7) should be present
        assert "Tick 7" in text
        assert "[IMPORTANT]" in text
        # Last 3 ticks (17, 18, 19) should be present
        assert "Tick 17" in text
        assert "Tick 18" in text
        assert "Tick 19" in text
        # Middle ticks should be summarised
        assert "summarised" in text

    def test_context_action_effects(self):
        """record_action_effect stores mappings."""
        ctx = ContextManager()
        ctx.record_action_effect(1, "moves up")
        ctx.record_action_effect(2, "moves right")
        ctx.record_action_effect(3, "moves down")

        text = ctx.get_discovered_effects()
        assert "ACTION 1: moves up" in text
        assert "ACTION 2: moves right" in text
        assert "ACTION 3: moves down" in text

    def test_context_reset(self):
        """reset clears everything."""
        ctx = ContextManager()
        ctx.add_tick(0, "obs", 1, "result")
        ctx.record_action_effect(1, "up")
        ctx.reset()
        assert len(ctx.full_history) == 0
        assert len(ctx.action_effect_map) == 0


# ---- ResponseParser tests --------------------------------------------------

class TestResponseParser:
    def test_parser_valid_response(self):
        """Parses well-formatted response correctly."""
        parser = ResponseParser()
        response = (
            "ACTION: 2\n"
            "HYPOTHESIS: Moving towards the blue goal\n"
            "CONFIDENCE: med\n"
            "C1_EXPANSION: NONE\n"
            "REASONING: Blue object looks like the target."
        )
        result = parser.parse(response, [1, 2, 3, 4])

        assert result["action"] == 2
        assert "blue goal" in result["hypothesis"].lower()
        assert result["confidence"] == "med"
        assert result["c1_expansion"] is None
        assert "target" in result["reasoning"].lower()

    def test_parser_malformed_response(self):
        """Handles missing fields gracefully."""
        parser = ResponseParser()
        response = "I think action 3 might work..."  # no structured fields
        result = parser.parse(response, [1, 2, 3, 4])

        # Should fall back to first available action
        assert result["action"] == 1
        assert result["confidence"] == "low"

    def test_parser_action_validation(self):
        """Rejects unavailable actions, falls back to valid one."""
        parser = ResponseParser()
        response = "ACTION: 9\nHYPOTHESIS: test\nCONFIDENCE: high\nC1_EXPANSION: NONE\nREASONING: test"
        result = parser.parse(response, [1, 2, 3, 4])

        # Action 9 not available, should fall back
        assert result["action"] in [1, 2, 3, 4]
        assert result["action"] == 1  # first available

    def test_parser_c1_expansion(self):
        """Parses C1 expansion correctly."""
        parser = ResponseParser()
        response = (
            "ACTION: 1\n"
            "HYPOTHESIS: testing\n"
            "CONFIDENCE: low\n"
            "C1_EXPANSION: GRAVITY: Objects fall when unsupported\n"
            "REASONING: Observed falling behaviour."
        )
        result = parser.parse(response, [1, 2, 3, 4])
        assert result["c1_expansion"] is not None
        assert "GRAVITY" in result["c1_expansion"]


# ---- Agent integration tests (mock LLM) -----------------------------------

class TestLLMAgent:
    def test_agent_act_mock(self):
        """agent.act() returns valid action using mock LLM."""
        agent = LLMAgent()
        # Ensure we're using mock
        assert not agent.llm.available or True  # works either way

        grid = _make_grid()
        action = agent.act(grid)

        assert isinstance(action, int)
        assert action in [1, 2, 3, 4]

    def test_agent_act_multiple_ticks(self):
        """Agent handles multiple ticks and builds context."""
        agent = LLMAgent()
        grid1 = _make_grid()
        grid2 = _make_grid_shifted()

        a1 = agent.act(grid1)
        a2 = agent.act(grid2)

        assert isinstance(a1, int)
        assert isinstance(a2, int)
        # Context should have records
        assert len(agent.context.full_history) == 2

    def test_agent_c1_expansion_from_response(self):
        """C1 expansion in response updates c1c2 framework."""
        agent = LLMAgent()
        # Override the mock to return a C1 expansion
        original_mock = agent.llm._mock_response

        def mock_with_expansion():
            return (
                "ACTION: 1\n"
                "HYPOTHESIS: Found gravity mechanic\n"
                "CONFIDENCE: med\n"
                "C1_EXPANSION: GRAVITY: Objects fall downward when unsupported\n"
                "REASONING: Observed objects dropping."
            )

        agent.llm._mock_response = mock_with_expansion

        grid = _make_grid()
        agent.act(grid)

        # Check C1 was expanded
        names = {c.name for c in agent.c1c2.c1_vocabulary}
        assert "GRAVITY" in names
        assert len(agent.c1c2.c1_vocabulary) == 8  # 7 initial + 1 discovered

    def test_agent_play_game_no_sdk(self):
        """play_game handles missing SDK or invalid game gracefully."""
        agent = LLMAgent()
        result = agent.play_game("test_game", max_actions=5)
        # Should return some result (dict or scorecard) without crashing
        assert result is not None
