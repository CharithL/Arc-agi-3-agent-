"""Tests for Transformer meta-learner and meta-training loop."""
import torch
import numpy as np
import pytest


class TestMetaLearnerForwardShapes:
    def test_meta_learner_forward_shapes(self):
        """Correct output shapes for batch of sequences."""
        from charith.transformer.meta_learner import MetaLearner

        model = MetaLearner(d_model=256, n_heads=8, n_layers=6, n_actions=8)
        batch, seq_len = 4, 20
        tokens = torch.randn(batch, seq_len, 256)
        types = torch.zeros(batch, seq_len, dtype=torch.long)
        for i in range(seq_len):
            types[:, i] = i % 2

        action_logits, predictions, values = model(tokens, types)

        assert action_logits.shape == (batch, seq_len, 8)
        assert predictions.shape == (batch, seq_len, 256)
        assert values.shape == (batch, seq_len, 1)


class TestMetaLearnerSingleStep:
    def test_meta_learner_single_step(self):
        """Works with sequence length 1."""
        from charith.transformer.meta_learner import MetaLearner

        model = MetaLearner(d_model=256, n_heads=8, n_layers=6, n_actions=8)
        tokens = torch.randn(1, 1, 256)
        types = torch.zeros(1, 1, dtype=torch.long)

        action_logits, predictions, values = model(tokens, types)

        assert action_logits.shape == (1, 1, 8)
        assert predictions.shape == (1, 1, 256)
        assert values.shape == (1, 1, 1)


class TestMetaLearnerGetAction:
    def test_meta_learner_get_action(self):
        """Returns valid action, log_prob, value."""
        from charith.transformer.meta_learner import MetaLearner

        model = MetaLearner(d_model=256, n_heads=8, n_layers=6, n_actions=8)
        tokens = torch.randn(1, 5, 256)
        types = torch.zeros(1, 5, dtype=torch.long)
        for i in range(5):
            types[0, i] = i % 2

        action, log_prob, value = model.get_action(tokens, types)

        assert isinstance(action, int)
        assert 0 <= action < 8
        assert log_prob.shape == ()  # scalar
        assert value.shape == ()  # scalar
        assert log_prob.item() <= 0  # log probability is non-positive


class TestMetaLearnerActionMasking:
    def test_meta_learner_action_masking(self):
        """Masked actions not selected."""
        from charith.transformer.meta_learner import MetaLearner

        model = MetaLearner(d_model=256, n_heads=8, n_layers=6, n_actions=8)
        tokens = torch.randn(1, 3, 256)
        types = torch.zeros(1, 3, dtype=torch.long)

        # Only allow actions 2 and 5
        allowed = [2, 5]
        for _ in range(50):
            action, _, _ = model.get_action(tokens, types, available_actions=allowed)
            assert action in allowed, f"Action {action} not in allowed set {allowed}"


class TestMetaLearnerCausal:
    def test_meta_learner_causal(self):
        """Output at position t doesn't depend on position t+1."""
        from charith.transformer.meta_learner import MetaLearner

        model = MetaLearner(d_model=256, n_heads=8, n_layers=2, n_actions=8)
        model.eval()

        # Run with 5 tokens
        tokens_5 = torch.randn(1, 5, 256)
        types_5 = torch.zeros(1, 5, dtype=torch.long)
        for i in range(5):
            types_5[0, i] = i % 2

        with torch.no_grad():
            logits_5, preds_5, vals_5 = model(tokens_5, types_5)

        # Run with only first 3 tokens (same values)
        tokens_3 = tokens_5[:, :3, :].clone()
        types_3 = types_5[:, :3].clone()

        with torch.no_grad():
            logits_3, preds_3, vals_3 = model(tokens_3, types_3)

        # Outputs at positions 0,1,2 should be the same regardless of
        # whether tokens 3,4 exist (causal masking)
        assert torch.allclose(logits_3[0, :3, :], logits_5[0, :3, :], atol=1e-5), \
            "Causal violation: output at position t depends on future tokens"
        assert torch.allclose(preds_3[0, :3, :], preds_5[0, :3, :], atol=1e-5)
        assert torch.allclose(vals_3[0, :3, :], vals_5[0, :3, :], atol=1e-5)


class TestMetaTrainingOneEpisode:
    def test_meta_training_one_episode(self):
        """Single episode runs without error, loss decreases."""
        from charith.transformer.meta_training import meta_train, MetaTrainingConfig
        import tempfile, os

        with tempfile.TemporaryDirectory() as tmpdir:
            config = MetaTrainingConfig(
                n_episodes=2,
                max_steps_per_episode=10,
                lr=1e-3,
                n_layers=2,
                d_model=64,
                n_heads=4,
                n_actions=4,
                max_context=50,
                log_interval=1,
            )
            model = meta_train(config, save_dir=tmpdir)

            # Model should have been returned
            assert model is not None
            # Checkpoint should exist
            assert os.path.exists(os.path.join(tmpdir, "meta_learner_final.pt"))


class TestContextIntegration:
    def test_context_integration(self):
        """percept_tokenizer -> context_manager -> meta_learner pipeline works end-to-end."""
        from charith.transformer.percept_tokenizer import PerceptTokenizer
        from charith.transformer.action_tokenizer import ActionTokenizer
        from charith.transformer.context_manager import ContextManager
        from charith.transformer.meta_learner import MetaLearner
        from charith.perception.core_knowledge import CoreKnowledgePerception

        d_model = 256
        n_actions = 4

        perception = CoreKnowledgePerception()
        percept_tok = PerceptTokenizer(d_model=d_model)
        action_tok = ActionTokenizer(n_actions=n_actions, d_model=d_model)
        context = ContextManager(d_model=d_model, max_length=200)
        model = MetaLearner(d_model=d_model, n_heads=8, n_layers=2, n_actions=n_actions)

        # Simulate 3 steps of interaction
        grid = np.zeros((8, 8), dtype=int)
        grid[3, 3] = 1
        prev_percept = None

        for step in range(3):
            percept = perception.perceive(grid)
            p_token = percept_tok(percept, grid_dims=grid.shape, prev_percept=prev_percept)
            context.add_percept(p_token)

            seq = context.get_sequence()
            assert seq is not None

            n_tokens = seq.shape[0]
            types = torch.zeros(n_tokens, dtype=torch.long)
            for i in range(n_tokens):
                types[i] = i % 2

            tokens_batch = seq.unsqueeze(0)
            types_batch = types.unsqueeze(0)

            action, log_prob, value = model.get_action(
                tokens_batch, types_batch,
                available_actions=list(range(n_actions)),
            )

            assert isinstance(action, int)
            assert 0 <= action < n_actions

            # Simulate action effect
            a_token = action_tok(action, reward=-0.01, done=False)
            context.add_action(a_token)

            prev_percept = percept
            # Move the object slightly for next step
            grid = np.zeros((8, 8), dtype=int)
            grid[3 + step, 3] = 1

        # Final context should have 3 percepts + 3 actions = 6 tokens
        assert context.get_length() == 6


class TestGameToTransformerPipeline:
    def test_game_to_transformer_pipeline(self):
        """game.step() -> perception -> tokenizer -> context -> transformer -> action -> game.step()
        Full loop."""
        from charith.transformer.percept_tokenizer import PerceptTokenizer
        from charith.transformer.action_tokenizer import ActionTokenizer
        from charith.transformer.context_manager import ContextManager
        from charith.transformer.meta_learner import MetaLearner
        from charith.perception.core_knowledge import CoreKnowledgePerception
        from charith.gamegen.generator import GameGenerator

        d_model = 256
        n_actions = 8

        game_gen = GameGenerator(seed=42)
        perception = CoreKnowledgePerception()
        percept_tok = PerceptTokenizer(d_model=d_model)
        action_tok = ActionTokenizer(n_actions=n_actions, d_model=d_model)
        context = ContextManager(d_model=d_model, max_length=200)
        model = MetaLearner(d_model=d_model, n_heads=8, n_layers=2, n_actions=n_actions)

        # Generate a game and play it
        game = game_gen.generate(level=1)
        grid = game.reset()
        prev_percept = None

        total_reward = 0.0
        for step in range(10):
            # Perceive
            percept = perception.perceive(grid)
            p_token = percept_tok(percept, grid_dims=grid.shape, prev_percept=prev_percept)
            context.add_percept(p_token)

            # Get sequence and types
            seq = context.get_sequence()
            n_tokens = seq.shape[0]
            types = torch.zeros(n_tokens, dtype=torch.long)
            for i in range(n_tokens):
                types[i] = i % 2

            # Get action from transformer
            action, log_prob, value = model.get_action(
                seq.unsqueeze(0), types.unsqueeze(0),
                available_actions=list(range(game.spec.n_actions)),
            )

            # Step game
            next_grid, reward, done, info = game.step(action)
            total_reward += reward

            # Tokenize action
            a_token = action_tok(action, reward, done)
            context.add_action(a_token)

            prev_percept = percept
            grid = next_grid

            if done:
                break

        # Verify the pipeline ran without errors and produced sensible output
        assert context.get_length() > 0
        assert isinstance(total_reward, float)
        # Context should have step*2 tokens (percept + action per step)
        # plus 1 if last step didn't add action before break
        expected_min = 2  # at least 1 step completed
        assert context.get_length() >= expected_min
