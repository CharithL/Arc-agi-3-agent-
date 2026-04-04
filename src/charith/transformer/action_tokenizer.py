"""Action Tokenizer: (action, reward, done) -> D_model vector."""
import torch
import torch.nn as nn

D_MODEL = 256
MAX_ACTIONS = 8


class ActionTokenizer(nn.Module):
    """Converts (action, reward, done) into a D_model dimensional token."""

    def __init__(self, n_actions=8, d_model=256):
        super().__init__()
        # action one-hot (8) + reward (1) + done (1) = 10
        self.projection = nn.Sequential(
            nn.Linear(n_actions + 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.n_actions = n_actions

    def forward(self, action: int, reward: float, done: bool) -> torch.Tensor:
        """Convert action+reward+done to token vector.
        Returns: [D_model] tensor
        """
        dev = next(self.parameters()).device
        features = torch.zeros(self.n_actions + 2, device=dev)
        features[action] = 1.0  # one-hot
        features[self.n_actions] = reward
        features[self.n_actions + 1] = float(done)
        return self.projection(features)
