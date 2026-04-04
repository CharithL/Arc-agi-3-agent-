"""Tests for tokenizers and context manager."""
import torch
import numpy as np
import pytest


class TestPerceptTokenizer:
    def test_output_shape(self):
        """Percept token is [D_model] = [256]."""
        from charith.transformer.percept_tokenizer import PerceptTokenizer
        from charith.perception.core_knowledge import CoreKnowledgePerception

        grid = np.zeros((16, 16), dtype=int)
        grid[5, 5] = 1
        perception = CoreKnowledgePerception()
        percept = perception.perceive(grid)

        tokenizer = PerceptTokenizer()
        token = tokenizer(percept, grid_dims=(16, 16))
        assert token.shape == (256,)

    def test_change_features_first_tick(self):
        """First tick has zero change features (no previous)."""
        from charith.transformer.percept_tokenizer import PerceptTokenizer
        from charith.perception.core_knowledge import CoreKnowledgePerception

        grid = np.zeros((16, 16), dtype=int)
        perception = CoreKnowledgePerception()
        percept = perception.perceive(grid)
        tokenizer = PerceptTokenizer()
        token = tokenizer(percept, grid_dims=(16, 16))
        assert token.shape == (256,)  # Should work without prev_percept

    def test_change_features_detect_movement(self):
        """Change features should differ when objects move."""
        from charith.transformer.percept_tokenizer import PerceptTokenizer
        from charith.perception.core_knowledge import CoreKnowledgePerception

        perception = CoreKnowledgePerception()
        grid1 = np.zeros((16, 16), dtype=int)
        grid1[5, 5] = 1
        percept1 = perception.perceive(grid1)

        grid2 = np.zeros((16, 16), dtype=int)
        grid2[5, 7] = 1  # object moved
        percept2 = perception.perceive(grid2)

        tokenizer = PerceptTokenizer()
        token_no_change = tokenizer(percept1, (16, 16))
        token_with_change = tokenizer(percept2, (16, 16), prev_percept=percept1)

        # Tokens should be different when there's a change
        assert not torch.allclose(token_no_change, token_with_change)


class TestActionTokenizer:
    def test_output_shape(self):
        from charith.transformer.action_tokenizer import ActionTokenizer
        tokenizer = ActionTokenizer(n_actions=4)
        token = tokenizer(action=2, reward=0.5, done=False)
        assert token.shape == (256,)

    def test_different_actions_produce_different_tokens(self):
        from charith.transformer.action_tokenizer import ActionTokenizer
        tokenizer = ActionTokenizer(n_actions=4)
        t1 = tokenizer(0, 0.0, False)
        t2 = tokenizer(1, 0.0, False)
        assert not torch.allclose(t1, t2)


class TestContextManager:
    def test_add_and_get(self):
        from charith.transformer.context_manager import ContextManager
        cm = ContextManager(d_model=256, max_length=200)
        cm.add_percept(torch.randn(256))
        cm.add_action(torch.randn(256))
        cm.add_percept(torch.randn(256))
        seq = cm.get_sequence()
        assert seq.shape == (3, 256)

    def test_sliding_window(self):
        from charith.transformer.context_manager import ContextManager
        cm = ContextManager(d_model=256, max_length=20)
        for _ in range(30):
            cm.add_percept(torch.randn(256))
            cm.add_action(torch.randn(256))
        assert cm.get_length() <= 20

    def test_reset(self):
        from charith.transformer.context_manager import ContextManager
        cm = ContextManager()
        cm.add_percept(torch.randn(256))
        cm.reset()
        assert cm.is_empty

    def test_empty_returns_none(self):
        from charith.transformer.context_manager import ContextManager
        cm = ContextManager()
        assert cm.get_sequence() is None
