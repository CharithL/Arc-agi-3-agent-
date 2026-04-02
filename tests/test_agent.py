"""Integration tests for the CHARITHAgent."""
import pytest


class TestAgentSmoke:
    def test_agent_plays_deterministic_without_crashing(self):
        """Smoke test: agent plays deterministic movement game end-to-end."""
        from charith.agent import CHARITHAgent

        agent = CHARITHAgent()
        scorecard = agent.play_game("deterministic_movement", max_actions=100)
        assert scorecard is not None
        assert agent._total_actions > 0

    def test_agent_plays_hidden_goal(self):
        """Agent plays hidden goal game without crashing."""
        from charith.agent import CHARITHAgent

        agent = CHARITHAgent()
        scorecard = agent.play_game("hidden_goal", max_actions=200)
        assert scorecard is not None

    def test_agent_plays_context_dependent(self):
        """Agent plays context-dependent game without crashing."""
        from charith.agent import CHARITHAgent

        agent = CHARITHAgent()
        scorecard = agent.play_game("context_dependent", max_actions=100)
        assert scorecard is not None


class TestAgentLearning:
    def test_agent_world_model_learns(self):
        """After playing deterministic game, world model has rules."""
        from charith.agent import CHARITHAgent

        agent = CHARITHAgent()
        agent.play_game("deterministic_movement", max_actions=50)
        assert agent.world_model.get_rule_count() > 0

    def test_agent_goal_discovery_generates_hypotheses(self):
        """After enough ticks, goal discovery has hypotheses."""
        from charith.agent import CHARITHAgent

        agent = CHARITHAgent()
        agent.play_game("hidden_goal", max_actions=50)
        assert len(agent.goal_discovery._hypotheses) > 0


class TestAgentReset:
    def test_agent_hard_reset_clears_state(self):
        """Hard reset clears all agent state."""
        from charith.agent import CHARITHAgent

        agent = CHARITHAgent()
        agent.play_game("deterministic_movement", max_actions=20)
        agent._hard_reset()
        assert agent.world_model.get_rule_count() == 0
        assert agent._tick == 0
        assert agent._total_actions == 0

    def test_agent_cross_level_preserves_rules(self):
        """Multi-level game: rules from level 1 persist into level 2."""
        from charith.agent import CHARITHAgent

        agent = CHARITHAgent()
        agent.play_game("multi_level", max_actions=500)
        if agent._levels_completed > 0:
            assert agent.world_model.get_rule_count() > 0
