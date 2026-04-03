"""Mirror Model: reads Player's hidden states and decides whether to trust, explore, or override.

The Mirror is a lightweight network that takes the Player's flattened hidden states
(all GRU layers concatenated) and produces:
  - decoded_features: what does the Player "know" about the world?
  - confidence: how structured/reliable are the hidden states?
  - strategy_logits: trust Player / explore / override
  - override_logits: which action to take if overriding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MirrorModel(nn.Module):
    """Mirror that reads Player hidden states and arbitrates action selection.

    Architecture:
        Shared backbone -> 4 heads (feature decoder, confidence, strategy, override)
    """

    def __init__(
        self,
        player_hidden_total: int = 512,
        n_decoded_features: int = 10,
        n_actions: int = 4,
        mirror_hidden: int = 128,
    ):
        super().__init__()
        self.player_hidden_total = player_hidden_total
        self.n_decoded_features = n_decoded_features
        self.n_actions = n_actions
        self.mirror_hidden = mirror_hidden

        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(player_hidden_total, mirror_hidden),
            nn.ReLU(),
            nn.Linear(mirror_hidden, mirror_hidden),
            nn.ReLU(),
        )

        # Head 1: Feature decoder -- what does the Player "know"?
        self.feature_decoder = nn.Linear(mirror_hidden, n_decoded_features)

        # Head 2: Confidence -- how structured are the hidden states? (0-1)
        self.confidence_head = nn.Sequential(
            nn.Linear(mirror_hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # Head 3: Strategy -- trust(0) / explore(1) / override(2)
        self.strategy_head = nn.Linear(mirror_hidden, 3)

        # Head 4: Override action selection
        self.override_head = nn.Linear(mirror_hidden, n_actions)

    def forward(self, h_t_flat: torch.Tensor):
        """Forward pass.

        Args:
            h_t_flat: [batch, player_hidden_total] -- flattened Player hidden states

        Returns:
            decoded_features: [batch, n_decoded_features]
            confidence: [batch, 1]
            strategy_logits: [batch, 3]
            override_logits: [batch, n_actions]
        """
        z = self.backbone(h_t_flat)
        decoded_features = self.feature_decoder(z)
        confidence = self.confidence_head(z)
        strategy_logits = self.strategy_head(z)
        override_logits = self.override_head(z)
        return decoded_features, confidence, strategy_logits, override_logits

    def get_decision(self, h_t_flat: torch.Tensor, player_action_logits: torch.Tensor):
        """Decide final action using Mirror's arbitration logic.

        Args:
            h_t_flat: [batch, player_hidden_total]
            player_action_logits: [batch, n_actions] -- Player's proposed action logits

        Returns:
            final_action: int -- selected action index (for batch=1)
            meta_info: dict with confidence, strategy, override_action, etc.
        """
        with torch.no_grad():
            decoded, confidence, strategy_logits, override_logits = self.forward(h_t_flat)

        conf_val = confidence.item()
        strategy = strategy_logits.argmax(dim=-1).item()

        # Arbiter logic
        if strategy == 0 or conf_val > 0.7:
            # Trust Player
            final_action = player_action_logits.argmax(dim=-1).item()
            decision = "trust"
        elif strategy == 1 or conf_val < 0.3:
            # Explore: random action
            final_action = torch.randint(0, self.n_actions, (1,)).item()
            decision = "explore"
        else:
            # Override: use Mirror's own action
            final_action = override_logits.argmax(dim=-1).item()
            decision = "override"

        meta_info = {
            "decision": decision,
            "confidence": conf_val,
            "strategy": strategy,
            "strategy_logits": strategy_logits.squeeze(0).tolist(),
            "override_action": override_logits.argmax(dim=-1).item(),
            "decoded_features": decoded.squeeze(0).tolist(),
        }

        return final_action, meta_info
