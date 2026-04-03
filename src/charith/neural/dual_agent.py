"""DualAgent: Player + Mirror playing real ARC-AGI-3 games.

The Player proposes actions, the Mirror reads Player's hidden states,
and the Arbiter (Mirror's get_decision) decides the final action.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from charith.neural.world_model_net import WorldModelNet
from charith.neural.mirror_model import MirrorModel
from charith.neural.state_encoder import encode, D_INPUT
from charith.perception.core_knowledge import CoreKnowledgePerception


@dataclass
class DualAgentDiagnostics:
    """Diagnostics from a DualAgent run."""
    trust_count: int = 0
    explore_count: int = 0
    override_count: int = 0
    confidence_history: List[float] = field(default_factory=list)
    prediction_errors: List[float] = field(default_factory=list)
    decisions: List[str] = field(default_factory=list)
    total_steps: int = 0

    @property
    def mean_confidence(self) -> float:
        if self.confidence_history:
            return float(np.mean(self.confidence_history))
        return 0.0

    @property
    def mean_prediction_error(self) -> float:
        if self.prediction_errors:
            return float(np.mean(self.prediction_errors))
        return float("inf")

    @property
    def trust_rate(self) -> float:
        total = self.trust_count + self.explore_count + self.override_count
        return self.trust_count / max(total, 1)

    def summary(self) -> Dict:
        return {
            "trust_count": self.trust_count,
            "explore_count": self.explore_count,
            "override_count": self.override_count,
            "trust_rate": self.trust_rate,
            "mean_confidence": self.mean_confidence,
            "mean_prediction_error": self.mean_prediction_error,
            "total_steps": self.total_steps,
        }


class DualAgent:
    """Agent combining Player (WorldModelNet) and Mirror for real ARC-AGI-3 games."""

    def __init__(
        self,
        player: WorldModelNet,
        mirror: MirrorModel,
    ):
        self.player = player
        self.mirror = mirror
        self.player.eval()
        self.mirror.eval()

    def evaluate_on_game(
        self, game_id: str, n_steps: int = 50
    ) -> Tuple[float, DualAgentDiagnostics]:
        """Play a real ARC-AGI-3 game and measure prediction error + diagnostics.

        Args:
            game_id: ARC-AGI-3 game identifier
            n_steps: number of steps to play

        Returns:
            (mean_prediction_error, diagnostics)
        """
        import arc_agi
        from arcengine import GameAction

        arcade = arc_agi.Arcade()
        env = arcade.make(game_id)
        frame = env.reset()
        grid = np.array(frame.frame[0])
        avail = frame.available_actions

        perception = CoreKnowledgePerception()
        hidden = self.player.init_hidden(batch_size=1)
        diagnostics = DualAgentDiagnostics()
        prev_encoding = None

        for step in range(n_steps):
            percept = perception.perceive(grid)

            # Pick an available action index for encoding
            avail_action = avail[step % len(avail)]
            x = encode(percept, action=avail_action - 1,
                       controllable_ids=set(), grid_dims=grid.shape)

            with torch.no_grad():
                pred, player_logits, value, hidden = self.player(
                    x.unsqueeze(0), hidden
                )

            # Prediction error
            if prev_encoding is not None:
                error = float(torch.nn.functional.mse_loss(
                    pred.squeeze(0), x
                ).item())
                diagnostics.prediction_errors.append(error)

            # Mirror arbitration
            h_flat = self.player.get_hidden_flat(hidden)  # [1, 512]
            final_action, meta = self.mirror.get_decision(h_flat, player_logits)

            # Track diagnostics
            decision = meta["decision"]
            diagnostics.decisions.append(decision)
            diagnostics.confidence_history.append(meta["confidence"])

            if decision == "trust":
                diagnostics.trust_count += 1
            elif decision == "explore":
                diagnostics.explore_count += 1
            else:
                diagnostics.override_count += 1

            diagnostics.total_steps += 1
            prev_encoding = x

            # Step the real game
            # Map final_action (0-3) to available game actions
            action_idx = final_action % len(avail)
            game_action = avail[action_idx]

            try:
                result = env.step(GameAction[f"ACTION{game_action}"])
                grid = np.array(result.frame[0])
                avail = result.available_actions
            except Exception:
                break

        return diagnostics.mean_prediction_error, diagnostics

    def evaluate_player_only(self, game_id: str, n_steps: int = 50) -> float:
        """Play a real game using ONLY the Player (no Mirror), for comparison.

        Returns mean prediction error.
        """
        import arc_agi
        from arcengine import GameAction

        arcade = arc_agi.Arcade()
        env = arcade.make(game_id)
        frame = env.reset()
        grid = np.array(frame.frame[0])
        avail = frame.available_actions

        perception = CoreKnowledgePerception()
        hidden = self.player.init_hidden(batch_size=1)
        pred_errors = []
        prev_encoding = None

        for step in range(n_steps):
            percept = perception.perceive(grid)
            avail_action = avail[step % len(avail)]
            x = encode(percept, action=avail_action - 1,
                       controllable_ids=set(), grid_dims=grid.shape)

            with torch.no_grad():
                pred, logits, value, hidden = self.player(x.unsqueeze(0), hidden)

            if prev_encoding is not None:
                error = float(torch.nn.functional.mse_loss(
                    pred.squeeze(0), x
                ).item())
                pred_errors.append(error)

            prev_encoding = x

            # Use Player's own action
            action = logits.squeeze(0).argmax().item()
            action_idx = action % len(avail)
            game_action = avail[action_idx]

            try:
                result = env.step(GameAction[f"ACTION{game_action}"])
                grid = np.array(result.frame[0])
                avail = result.available_actions
            except Exception:
                break

        if pred_errors:
            return float(np.mean(pred_errors))
        return float("inf")
