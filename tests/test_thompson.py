"""Tests for Thompson Sampling with sequence memory boost."""
import numpy as np
import pytest

from charith.action.thompson import ThompsonSampler, ActionStats
from charith.action.action_space import Action, N_ACTIONS
from charith.memory.sequences import ActionSequenceMemory


class TestActionStats:
    def test_default_prior(self):
        s = ActionStats()
        assert s.alpha == 1.0
        assert s.beta == 1.0
        assert s.times_taken == 0
        assert s.total_reward == 0.0

    def test_mean(self):
        s = ActionStats(alpha=3.0, beta=1.0)
        assert s.mean == pytest.approx(0.75)

    def test_sample_in_range(self):
        s = ActionStats()
        for _ in range(100):
            val = s.sample()
            assert 0.0 <= val <= 1.0

    def test_update_positive(self):
        s = ActionStats()
        s.update(1.0)
        assert s.alpha == 2.0
        assert s.beta == 1.0
        assert s.times_taken == 1
        assert s.total_reward == 1.0

    def test_update_negative(self):
        s = ActionStats()
        s.update(0.0)
        assert s.alpha == 1.0
        assert s.beta == 2.0


class TestThompsonSampler:
    def test_thompson_selects_from_available_actions(self):
        """Selected action should be in the available set."""
        sampler = ThompsonSampler(n_actions=N_ACTIONS)
        available = [Action.UP, Action.DOWN, Action.LEFT]
        for _ in range(50):
            action = sampler.select_action(
                context_hash=0, available_actions=available
            )
            assert action in available

    def test_thompson_converges_to_best_action(self):
        """After 100 updates where action 2 always rewards, action 2 should
        be selected >50% of 100 samples."""
        sampler = ThompsonSampler(n_actions=N_ACTIONS)
        ctx = 42
        best = 2
        # Train: action 2 always gets reward 1.0, others get 0.0
        for _ in range(100):
            for a in range(N_ACTIONS):
                reward = 1.0 if a == best else 0.0
                sampler.update(ctx, a, reward)

        # Sample 100 times
        counts = {a: 0 for a in range(N_ACTIONS)}
        for _ in range(100):
            a = sampler.select_action(ctx)
            counts[a] += 1
        assert counts[best] > 50

    def test_thompson_exploration_mode(self):
        """With no data, is_exploring() should be True."""
        sampler = ThompsonSampler(n_actions=N_ACTIONS)
        assert sampler.is_exploring() is True

    def test_thompson_goal_directed_bias(self):
        """With goal_directed=True, goal_action should be chosen ~50% of time
        (between 60 and 140 out of 200 trials)."""
        sampler = ThompsonSampler(n_actions=N_ACTIONS)
        goal = Action.CONFIRM
        count = 0
        for _ in range(200):
            a = sampler.select_action(
                context_hash=0,
                goal_directed=True,
                goal_action=goal,
            )
            if a == goal:
                count += 1
        assert 60 <= count <= 140, f"Goal action chosen {count}/200 times"

    def test_get_average_uncertainty(self):
        sampler = ThompsonSampler(n_actions=N_ACTIONS)
        # With uniform priors, uncertainty should be positive
        u = sampler.get_average_uncertainty()
        assert u > 0

    def test_get_action_summary(self):
        sampler = ThompsonSampler(n_actions=N_ACTIONS)
        sampler.update(0, 1, 1.0)
        summary = sampler.get_action_summary()
        assert isinstance(summary, dict)
        assert 1 in summary

    def test_reset_context_keeps_global(self):
        sampler = ThompsonSampler(n_actions=N_ACTIONS)
        sampler.update(99, 0, 1.0)
        sampler.reset_context()
        # Context stats cleared
        assert 99 not in sampler._stats
        # Global stats preserved
        assert sampler._global_stats[0].times_taken > 0

    def test_hard_reset_clears_everything(self):
        sampler = ThompsonSampler(n_actions=N_ACTIONS)
        sampler.update(99, 0, 1.0)
        sampler.hard_reset()
        assert len(sampler._stats) == 0
        assert len(sampler._global_stats) == 0


class TestActionSequenceMemory:
    def test_sequence_memory_learns_good_pairs(self):
        """After rewarding (1,2) 20 times, suggest_action(1) should return 2."""
        mem = ActionSequenceMemory(n_actions=N_ACTIONS)
        for _ in range(20):
            mem.update(prev_action=1, curr_action=2, reward=1.0)
        assert mem.suggest_action(1) == 2

    def test_sequence_boost_positive(self):
        """A good pair should give a boost > 0."""
        mem = ActionSequenceMemory(n_actions=N_ACTIONS)
        for _ in range(20):
            mem.update(prev_action=1, curr_action=2, reward=1.0)
        boost = mem.get_sequence_boost(prev_action=1, candidate_action=2)
        assert boost > 0

    def test_sequence_boost_zero_for_unknown(self):
        """An unknown pair should give boost of 0."""
        mem = ActionSequenceMemory(n_actions=N_ACTIONS)
        boost = mem.get_sequence_boost(prev_action=0, candidate_action=1)
        assert boost == 0.0

    def test_reset_keeps_bigrams(self):
        mem = ActionSequenceMemory(n_actions=N_ACTIONS)
        mem.update(1, 2, 1.0)
        mem.reset()
        # Bigrams should persist for cross-level transfer
        boost = mem.get_sequence_boost(1, 2)
        assert boost > 0

    def test_hard_reset_clears_bigrams(self):
        mem = ActionSequenceMemory(n_actions=N_ACTIONS)
        mem.update(1, 2, 1.0)
        mem.hard_reset()
        boost = mem.get_sequence_boost(1, 2)
        assert boost == 0.0


class TestThompsonWithSequenceMemory:
    def test_thompson_with_sequence_boost(self):
        """Action 3 should be preferred after action 1 when sequence memory
        favors (1, 3)."""
        sampler = ThompsonSampler(n_actions=N_ACTIONS)
        mem = ActionSequenceMemory(n_actions=N_ACTIONS)

        # Train sequence memory to strongly favor (1, 3)
        for _ in range(50):
            mem.update(prev_action=1, curr_action=3, reward=1.0)

        # Sample many times with prev_action=1 and sequence memory
        counts = {a: 0 for a in range(N_ACTIONS)}
        for _ in range(200):
            a = sampler.select_action(
                context_hash=0,
                prev_action=1,
                sequence_memory=mem,
            )
            counts[a] += 1

        # Action 3 should appear more than average (200/N_ACTIONS = 25)
        assert counts[3] > 200 // N_ACTIONS
