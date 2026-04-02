"""Tests for the neural world model: state encoder, GRU network, action head."""

import time

import numpy as np
import pytest
import torch

from charith.perception.core_knowledge import (
    Cell,
    CoreKnowledgePerception,
    Object,
    StructuredPercept,
)
from charith.neural.state_encoder import (
    D_INPUT,
    MAX_OBJECTS,
    N_ACTIONS,
    N_COLORS,
    OBJECT_FEATURE_SIZE,
    GLOBAL_FEATURE_SIZE,
    encode,
)
from charith.neural.world_model_net import WorldModelNet
from charith.neural.action_head import select_action


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_percept(n_objects: int = 3, grid_h: int = 10, grid_w: int = 10) -> StructuredPercept:
    """Build a minimal StructuredPercept with *n_objects* for testing."""
    # Ensure grid is large enough for all objects (each needs row i*2, col i*2+1)
    min_h = max(grid_h, n_objects * 2) if n_objects > 0 else grid_h
    min_w = max(grid_w, n_objects * 2) if n_objects > 0 else grid_w
    grid_h, grid_w = min_h, min_w
    grid = np.zeros((grid_h, grid_w), dtype=int)
    objects = []
    for i in range(n_objects):
        color = (i + 1) % N_COLORS
        row, col = i * 2, i * 2
        cells = {Cell(row, col, color), Cell(row, col + 1, color)}
        grid[row, col] = color
        grid[row, col + 1] = color
        obj = Object(
            object_id=i,
            cells=cells,
            color=color,
            bbox=(row, col, row, col + 1),
            size=2,
            centroid=(float(row), float(col) + 0.5),
            shape_hash=hash(frozenset({(0, 0), (0, 1)})),
        )
        objects.append(obj)

    return StructuredPercept(
        raw_grid=grid,
        objects=objects,
        spatial_relations=[],
        color_counts={0: grid_h * grid_w - n_objects * 2, 1: 2},
        grid_dims=(grid_h, grid_w),
        background_color=0,
        symmetry={"h_symmetric": False, "v_symmetric": True, "rot_90": False, "rot_180": False},
        unique_colors={0, 1, 2, 3} if n_objects >= 3 else {0, 1},
        object_count=n_objects,
        timestamp=time.time(),
    )


# ---------------------------------------------------------------------------
# StateEncoder tests
# ---------------------------------------------------------------------------

class TestStateEncoder:
    def test_state_encoder_output_shape(self):
        """encode() returns tensor of shape [D_INPUT] = [365]."""
        percept = _make_percept(n_objects=3)
        tensor = encode(percept, action=0)
        assert tensor.shape == (D_INPUT,)
        assert D_INPUT == 365

    def test_state_encoder_normalized(self):
        """All values are between 0 and 1 (one-hots are 0 or 1)."""
        percept = _make_percept(n_objects=5)
        tensor = encode(percept, action=2, controllable_ids={0, 1})
        assert tensor.min() >= 0.0, f"Min value {tensor.min()} < 0"
        assert tensor.max() <= 1.0, f"Max value {tensor.max()} > 1"

    def test_state_encoder_deterministic(self):
        """Same percept + action = same encoding."""
        percept = _make_percept(n_objects=4)
        t1 = encode(percept, action=1, controllable_ids={0})
        t2 = encode(percept, action=1, controllable_ids={0})
        assert torch.equal(t1, t2)

    def test_state_encoder_zero_objects(self):
        """Encoding with zero objects should still produce correct shape."""
        percept = _make_percept(n_objects=0)
        tensor = encode(percept, action=0)
        assert tensor.shape == (D_INPUT,)
        # Object portion should be all zeros
        obj_portion = tensor[:MAX_OBJECTS * OBJECT_FEATURE_SIZE]
        assert torch.all(obj_portion == 0.0)

    def test_state_encoder_max_objects_exceeded(self):
        """More than MAX_OBJECTS should still produce correct shape (extras skipped)."""
        percept = _make_percept(n_objects=MAX_OBJECTS + 5)
        tensor = encode(percept, action=0)
        assert tensor.shape == (D_INPUT,)

    def test_state_encoder_action_onehot(self):
        """Action one-hot is set correctly in the global features."""
        percept = _make_percept(n_objects=1)
        for action_idx in range(N_ACTIONS):
            tensor = encode(percept, action=action_idx)
            # Global features start after object features
            global_start = MAX_OBJECTS * OBJECT_FEATURE_SIZE
            # Action one-hot starts at global offset N_COLORS + 4
            action_start = global_start + N_COLORS + 4
            action_vec = tensor[action_start:action_start + N_ACTIONS]
            assert action_vec[action_idx] == 1.0
            assert action_vec.sum() == 1.0

    def test_state_encoder_controllable_flag(self):
        """Controllable objects should have is_controllable=1.0."""
        percept = _make_percept(n_objects=3)
        # Sort objects like encoder does (largest first -- all same size=2 here,
        # so order is stable from the sort)
        sorted_objs = sorted(percept.objects, key=lambda o: o.size, reverse=True)
        controllable_ids = {sorted_objs[0].object_id}

        tensor = encode(percept, action=0, controllable_ids=controllable_ids)
        # First object's is_controllable flag
        flag_offset = 0 * OBJECT_FEATURE_SIZE + N_COLORS + 3
        assert tensor[flag_offset] == 1.0
        # Second object should not be controllable
        flag_offset2 = 1 * OBJECT_FEATURE_SIZE + N_COLORS + 3
        assert tensor[flag_offset2] == 0.0


# ---------------------------------------------------------------------------
# WorldModelNet tests
# ---------------------------------------------------------------------------

class TestWorldModelNet:
    def test_world_model_forward_shapes(self):
        """forward returns correct shapes for prediction, policy, value, hidden."""
        model = WorldModelNet(input_size=D_INPUT, hidden_size=128, num_layers=2, n_actions=4)
        batch_size = 4
        x = torch.randn(batch_size, D_INPUT)
        hidden = model.init_hidden(batch_size)

        pred, policy, value, new_hidden = model(x, hidden)

        assert pred.shape == (batch_size, D_INPUT)
        assert policy.shape == (batch_size, 4)
        assert value.shape == (batch_size, 1)
        assert new_hidden.shape == (2, batch_size, 128)

    def test_world_model_forward_sequence(self):
        """forward with sequence dimension returns correct shapes."""
        model = WorldModelNet(input_size=D_INPUT, hidden_size=128, num_layers=2, n_actions=4)
        batch_size = 2
        seq_len = 5
        x = torch.randn(batch_size, seq_len, D_INPUT)
        hidden = model.init_hidden(batch_size)

        pred, policy, value, new_hidden = model(x, hidden)

        assert pred.shape == (batch_size, seq_len, D_INPUT)
        assert policy.shape == (batch_size, seq_len, 4)
        assert value.shape == (batch_size, seq_len, 1)
        assert new_hidden.shape == (2, batch_size, 128)

    def test_world_model_init_hidden(self):
        """init_hidden returns correct shape of zeros."""
        model = WorldModelNet(hidden_size=256, num_layers=2)
        hidden = model.init_hidden(batch_size=8)
        assert hidden.shape == (2, 8, 256)
        assert torch.all(hidden == 0.0)

    def test_world_model_sequential(self):
        """Can process multiple steps sequentially updating hidden state."""
        model = WorldModelNet(input_size=D_INPUT, hidden_size=128, num_layers=2, n_actions=4)
        hidden = model.init_hidden(batch_size=1)

        # Step through 3 time steps
        for step in range(3):
            x = torch.randn(1, D_INPUT)
            with torch.no_grad():
                pred, policy, value, hidden = model(x, hidden)

            assert pred.shape == (1, D_INPUT)
            assert policy.shape == (1, 4)
            assert value.shape == (1, 1)
            assert hidden.shape == (2, 1, 128)

        # Hidden should have been updated (not all zeros anymore)
        assert not torch.all(hidden == 0.0)

    def test_world_model_get_hidden_state(self):
        """get_hidden_state extracts last layer for DESCARTES probing."""
        model = WorldModelNet(hidden_size=128, num_layers=2)
        hidden = model.init_hidden(batch_size=3)
        # Set last layer to non-zero for verification
        hidden[-1] = torch.ones(3, 128)

        extracted = model.get_hidden_state(hidden)
        assert extracted.shape == (3, 128)
        assert torch.all(extracted == 1.0)

    def test_world_model_no_hidden(self):
        """forward works without providing an initial hidden state."""
        model = WorldModelNet(input_size=D_INPUT, hidden_size=128, num_layers=2, n_actions=4)
        x = torch.randn(2, D_INPUT)
        with torch.no_grad():
            pred, policy, value, hidden = model(x)  # no hidden arg
        assert pred.shape == (2, D_INPUT)
        assert hidden.shape == (2, 2, 128)


# ---------------------------------------------------------------------------
# ActionHead tests
# ---------------------------------------------------------------------------

class TestActionHead:
    def test_action_selection_valid(self):
        """Selected action is in available_actions."""
        logits = torch.randn(8)
        available = [1, 3, 5]
        for _ in range(20):  # run multiple times since it is stochastic
            action, log_prob = select_action(logits, available_actions=available)
            assert action in available, f"Action {action} not in {available}"
            assert log_prob.dim() == 0  # scalar

    def test_action_selection_greedy(self):
        """Greedy mode returns highest logit action (among available)."""
        logits = torch.tensor([0.1, 0.5, 0.9, 0.2, 0.8, 0.3, 0.1, 0.0])
        available = [0, 2, 4]
        action, log_prob = select_action(logits, available_actions=available, greedy=True)
        # Among available [0,2,4], action 2 has highest logit (0.9)
        assert action == 2

    def test_action_selection_no_mask(self):
        """Without available_actions, any action can be selected."""
        logits = torch.randn(4)
        action, log_prob = select_action(logits)
        assert 0 <= action < 4
        assert log_prob.dim() == 0

    def test_action_selection_temperature(self):
        """Very low temperature should approximate greedy behavior."""
        logits = torch.tensor([0.0, 0.0, 10.0, 0.0])
        # With very low temperature, action 2 should be selected almost always
        actions = []
        for _ in range(50):
            a, _ = select_action(logits, temperature=0.01)
            actions.append(a)
        assert all(a == 2 for a in actions), "Low temperature should be near-greedy"

    def test_action_selection_single_available(self):
        """With only one available action, that action must be selected."""
        logits = torch.randn(8)
        action, log_prob = select_action(logits, available_actions=[5])
        assert action == 5
