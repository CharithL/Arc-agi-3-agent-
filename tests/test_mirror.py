"""Tests for Mirror model, decision logic, and DualAgent diagnostics."""

import torch
import torch.nn as nn
import numpy as np
import pytest

from charith.neural.mirror_model import MirrorModel
from charith.neural.world_model_net import WorldModelNet
from charith.neural.dual_agent import DualAgent, DualAgentDiagnostics


# --- Fixtures ---

@pytest.fixture
def mirror():
    """Create a MirrorModel with default params."""
    return MirrorModel(
        player_hidden_total=512,
        n_decoded_features=10,
        n_actions=4,
        mirror_hidden=128,
    )


@pytest.fixture
def player():
    """Create a WorldModelNet with default params."""
    return WorldModelNet(
        input_size=365,
        hidden_size=256,
        num_layers=2,
        n_actions=4,
    )


# --- Tests ---

class TestMirrorForward:
    """Test MirrorModel forward pass output shapes."""

    def test_mirror_forward_shapes(self, mirror):
        """All outputs have correct shapes for batch=1 and batch=8."""
        for batch_size in [1, 8]:
            h = torch.randn(batch_size, 512)
            decoded, confidence, strategy, override = mirror(h)

            assert decoded.shape == (batch_size, 10), \
                f"decoded_features shape wrong: {decoded.shape}"
            assert confidence.shape == (batch_size, 1), \
                f"confidence shape wrong: {confidence.shape}"
            assert strategy.shape == (batch_size, 3), \
                f"strategy_logits shape wrong: {strategy.shape}"
            assert override.shape == (batch_size, 4), \
                f"override_logits shape wrong: {override.shape}"

    def test_confidence_bounded(self, mirror):
        """Confidence output is in [0, 1] due to Sigmoid."""
        h = torch.randn(16, 512)
        _, confidence, _, _ = mirror(h)
        assert (confidence >= 0.0).all() and (confidence <= 1.0).all(), \
            "Confidence must be in [0, 1]"


class TestMirrorDecision:
    """Test the get_decision arbiter logic."""

    def test_mirror_decision_trust(self, mirror):
        """When confidence is high (>0.7), Mirror should trust Player."""
        # Force confidence to be high by manipulating the confidence head
        # Set bias to a large positive value -> sigmoid -> ~1.0
        with torch.no_grad():
            # Access the last linear layer of confidence_head
            # confidence_head: Linear(128,32) -> ReLU -> Linear(32,1) -> Sigmoid
            mirror.confidence_head[2].bias.fill_(5.0)  # large positive -> sigmoid ~1.0
            mirror.confidence_head[2].weight.fill_(0.0)  # zero out weights

        h = torch.randn(1, 512)
        player_logits = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # Player prefers action 0

        action, meta = mirror.get_decision(h, player_logits)

        assert meta["decision"] == "trust", \
            f"Expected trust with high confidence, got {meta['decision']}"
        assert meta["confidence"] > 0.7, \
            f"Confidence should be >0.7, got {meta['confidence']}"
        assert action == 0, \
            f"Should trust Player's action (0), got {action}"

    def test_mirror_decision_explore(self, mirror):
        """When confidence is low (<0.3), Mirror should explore."""
        # Force confidence to be low
        with torch.no_grad():
            mirror.confidence_head[2].bias.fill_(-5.0)  # large negative -> sigmoid ~0.0
            mirror.confidence_head[2].weight.fill_(0.0)

            # Also force strategy to NOT be 0 (trust) -- set strategy logits
            # so strategy 1 (explore) wins
            mirror.strategy_head.bias.data = torch.tensor([-10.0, 10.0, -10.0])
            mirror.strategy_head.weight.fill_(0.0)

        h = torch.randn(1, 512)
        player_logits = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

        action, meta = mirror.get_decision(h, player_logits)

        assert meta["decision"] == "explore", \
            f"Expected explore with low confidence, got {meta['decision']}"
        assert meta["confidence"] < 0.3, \
            f"Confidence should be <0.3, got {meta['confidence']}"


class TestMirrorTraining:
    """Test that gradients flow through the backbone."""

    def test_mirror_backbone_grad(self, mirror):
        """Backbone receives gradients during training."""
        h = torch.randn(4, 512)
        target = torch.randn(4, 10)

        decoded, _, _, _ = mirror(h)
        loss = nn.functional.mse_loss(decoded, target)
        loss.backward()

        # Check that backbone parameters have non-zero gradients
        backbone_grads = []
        for param in mirror.backbone.parameters():
            if param.grad is not None:
                backbone_grads.append(param.grad.abs().sum().item())

        assert len(backbone_grads) > 0, "No backbone parameters received gradients"
        assert any(g > 0 for g in backbone_grads), \
            "All backbone gradients are zero -- no gradient flow"


class TestDualAgentDiagnostics:
    """Test DualAgentDiagnostics tracking."""

    def test_dual_agent_diagnostic_tracking(self):
        """Diagnostics correctly count trust/explore/override decisions."""
        diag = DualAgentDiagnostics()

        # Simulate some decisions
        diag.trust_count = 30
        diag.explore_count = 10
        diag.override_count = 10
        diag.confidence_history = [0.8, 0.7, 0.9, 0.2, 0.5]
        diag.prediction_errors = [0.1, 0.2, 0.15, 0.3]
        diag.total_steps = 50

        summary = diag.summary()

        assert summary["trust_count"] == 30
        assert summary["explore_count"] == 10
        assert summary["override_count"] == 10
        assert summary["total_steps"] == 50

        # Trust rate = 30 / 50 = 0.6
        assert abs(summary["trust_rate"] - 0.6) < 1e-6, \
            f"Trust rate should be 0.6, got {summary['trust_rate']}"

        # Mean confidence
        expected_conf = np.mean([0.8, 0.7, 0.9, 0.2, 0.5])
        assert abs(summary["mean_confidence"] - expected_conf) < 1e-6

        # Mean prediction error
        expected_err = np.mean([0.1, 0.2, 0.15, 0.3])
        assert abs(summary["mean_prediction_error"] - expected_err) < 1e-6

    def test_empty_diagnostics(self):
        """Empty diagnostics return sensible defaults."""
        diag = DualAgentDiagnostics()
        assert diag.mean_confidence == 0.0
        assert diag.mean_prediction_error == float("inf")
        assert diag.trust_rate == 0.0
