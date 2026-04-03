"""GRU World Model with prediction head and action/value head."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WorldModelNet(nn.Module):
    """GRU-based world model.

    h_t = GRU(x_t, h_{t-1})
    prediction: y_hat = PredictionHead(h_t)  -- predicts next state encoding
    policy: pi(a) = ActionHead(h_t)  -- action probabilities
    value: V(s) = ValueHead(h_t)  -- state value for PPO
    """

    def __init__(
        self,
        input_size: int = 365,
        hidden_size: int = 256,
        num_layers: int = 2,
        n_actions: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Prediction head: predict next state encoding
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
        )

        # Action head: policy (action probabilities)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, n_actions),
        )

        # Value head: state value
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

        # Auxiliary spatial prediction head (Fix 3 for zombie solver):
        # Predicts controllable position (row, col) after K steps of
        # repeating the same action. Forces hidden state to encode
        # spatial information because the loss REQUIRES it.
        # Input: hidden_state (256) + action_onehot (n_actions)
        # Output: predicted (relative_row, relative_col) after K steps
        self.spatial_head = nn.Sequential(
            nn.Linear(hidden_size + n_actions, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 2),  # (row, col) in [0, 1]
            nn.Sigmoid(),  # Positions are normalized 0-1
        )

    def forward(self, x, hidden=None):
        """Forward pass through the world model.

        Args:
            x: input tensor [batch, seq_len, input_size] or [batch, input_size]
            hidden: GRU hidden state [num_layers, batch, hidden_size]

        Returns:
            prediction: predicted next state [batch, (seq_len), input_size]
            policy_logits: action logits [batch, (seq_len), n_actions]
            value: state value [batch, (seq_len), 1]
            hidden: updated hidden state
        """
        # Handle single step (no seq dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, 1, input_size]
            squeeze = True
        else:
            squeeze = False

        gru_out, hidden = self.gru(x, hidden)

        prediction = self.pred_head(gru_out)
        policy_logits = self.action_head(gru_out)
        value = self.value_head(gru_out)

        if squeeze:
            prediction = prediction.squeeze(1)
            policy_logits = policy_logits.squeeze(1)
            value = value.squeeze(1)

        return prediction, policy_logits, value, hidden

    def predict_future_position(self, hidden: torch.Tensor,
                                   action_onehot: torch.Tensor) -> torch.Tensor:
        """Predict controllable position after K steps of taking action.

        Args:
            hidden: last-layer hidden state [batch, hidden_size]
            action_onehot: one-hot action [batch, n_actions]
        Returns:
            predicted position [batch, 2] — (relative_row, relative_col) in [0,1]
        """
        h = self.get_hidden_state(hidden) if hidden.dim() == 3 else hidden
        x = torch.cat([h, action_onehot], dim=-1)
        return self.spatial_head(x)

    def init_hidden(self, batch_size: int = 1) -> torch.Tensor:
        """Initialize hidden state to zeros."""
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)

    def get_hidden_state(self, hidden: torch.Tensor) -> torch.Tensor:
        """Extract the last layer's hidden state for DESCARTES probing."""
        return hidden[-1]  # [batch, hidden_size] -- last layer

    def get_hidden_flat(self, hidden: torch.Tensor) -> torch.Tensor:
        """Flatten ALL layers' hidden states for Mirror model input.

        Args:
            hidden: [num_layers, batch, hidden_size]
        Returns:
            [batch, num_layers * hidden_size] — all layers concatenated
        """
        # hidden shape: [num_layers, batch, hidden_size]
        return hidden.permute(1, 0, 2).reshape(
            hidden.shape[1], self.num_layers * self.hidden_size
        )
