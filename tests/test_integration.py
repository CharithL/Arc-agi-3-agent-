"""Full integration tests — validate the architecture actually works."""
import pytest


class TestWorldModelLearning:
    def test_world_model_learns_deterministic_movement(self):
        """WorldModel learns rules after playing deterministic mock."""
        from charith.agent import CHARITHAgent

        agent = CHARITHAgent()
        agent.play_game("deterministic_movement", max_actions=50)
        assert agent.world_model.get_rule_count() > 0

    def test_thompson_converges_on_deterministic(self):
        """Thompson Sampling should favor some actions over others."""
        from charith.agent import CHARITHAgent

        agent = CHARITHAgent()
        agent.play_game("deterministic_movement", max_actions=100)
        summary = agent.explorer.get_action_summary()
        max_taken = max(s['times_taken'] for s in summary.values())
        assert max_taken > 5


class TestAllGamesComplete:
    @pytest.mark.parametrize("game_id", [
        "deterministic_movement",
        "hidden_goal",
        "context_dependent",
        "multi_level",
    ])
    def test_game_completes_without_error(self, game_id):
        """Smoke test: every mock scenario runs to completion."""
        from charith.agent import CHARITHAgent

        agent = CHARITHAgent()
        scorecard = agent.play_game(game_id, max_actions=100)
        assert scorecard is not None
