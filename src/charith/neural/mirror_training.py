"""Mirror Training: 3-phase training pipeline for the Mirror model.

Phase A (supervised): Train feature_decoder to predict ground truth from Player hidden states.
Phase B (self-supervised): Train confidence_head -- high when decoding is accurate, low otherwise.
Phase C (RL): Train strategy_head and override_head with REINFORCE.

The Player model is FROZEN during all phases.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field

from charith.neural.mirror_model import MirrorModel
from charith.neural.world_model_net import WorldModelNet
from charith.neural.state_encoder import encode, D_INPUT
from charith.neural.action_head import select_action
from charith.perception.core_knowledge import CoreKnowledgePerception
from charith.synthetic.maze_reality import MazeReality


@dataclass
class MirrorTrainingStats:
    """Statistics from Mirror training."""
    phase_a_loss: float = 0.0
    phase_b_loss: float = 0.0
    phase_c_reward: float = 0.0
    total_episodes: int = 0


def _collect_episodes(
    player: WorldModelNet,
    reality: MazeReality,
    n_episodes: int,
    max_steps: int = 200,
    verbose: bool = False,
) -> List[Dict]:
    """Collect episodes using the Player, recording hidden states and ground truths.

    Returns list of episode dicts, each with keys:
        h_flat_list: list of [1, 512] tensors (flattened Player hidden)
        gt_list: list of ground truth dicts from reality
        reward_list: list of float rewards
        action_list: list of int actions
        total_reward: float
    """
    perception = CoreKnowledgePerception()
    episodes = []

    for ep in range(n_episodes):
        perception.reset()
        grid = reality.reset()
        hidden = player.init_hidden(batch_size=1)

        ep_data = {
            "h_flat_list": [],
            "gt_list": [],
            "reward_list": [],
            "action_list": [],
            "total_reward": 0.0,
        }

        ctrl_ids = set()

        for step in range(max_steps):
            percept = perception.perceive(grid)
            action_for_enc = 0 if step == 0 else ep_data["action_list"][-1]
            x = encode(percept, action=action_for_enc,
                       controllable_ids=ctrl_ids, grid_dims=grid.shape)

            with torch.no_grad():
                pred, logits, value, hidden = player(x.unsqueeze(0), hidden)

            # Record flattened hidden state
            h_flat = player.get_hidden_flat(hidden)  # [1, 512]
            ep_data["h_flat_list"].append(h_flat.clone())
            ep_data["gt_list"].append(reality.get_ground_truth())

            # Select action using Player's policy
            action, _ = select_action(logits.squeeze(0),
                                      available_actions=list(range(reality.n_actions)),
                                      temperature=1.0)
            ep_data["action_list"].append(action)

            # Step
            next_grid, reward, done, info = reality.step(action)
            ep_data["reward_list"].append(reward)
            ep_data["total_reward"] += reward

            # Update controllable tracking
            if step > 0:
                perception.agency.record_action_contingency(action, grid, next_grid)
                ctrl_ids = set(perception.agency.detect_controllable_objects(
                    percept.objects, perception.agency._contingencies
                ))

            grid = next_grid
            if done:
                break

        episodes.append(ep_data)

        if verbose and (ep + 1) % max(1, n_episodes // 5) == 0:
            print(f"    Collected {ep + 1}/{n_episodes} episodes, "
                  f"reward={ep_data['total_reward']:.2f}")

    return episodes


def _ground_truth_to_tensor(gt: Dict[str, float]) -> torch.Tensor:
    """Convert ground truth dict to 10-element tensor.

    Order:
        0: distance_to_goal
        1: wall_adjacent_up
        2: wall_adjacent_down
        3: wall_adjacent_left
        4: wall_adjacent_right
        5: goal_direction_row (0.0=up, 1.0=down, 0.5=same)
        6: goal_direction_col (0.0=left, 1.0=right, 0.5=same)
        7: action_had_effect (default 0.0)
        8: is_controllable_nearby (default 0.0)
        9: hidden_state_surprise (default 0.0)
    """
    # Compute goal direction from positions if available
    ctrl_row = gt.get("controllable_relative_row", 0.5)
    ctrl_col = gt.get("controllable_relative_col", 0.5)
    # We don't have absolute goal position, but distance_to_goal is available
    # Use a heuristic: if distance > 0, we don't know exact direction without
    # goal position. We set 0.5 (unknown) as default.
    goal_dir_row = 0.5
    goal_dir_col = 0.5

    return torch.tensor([
        gt.get("distance_to_goal", 0.0),
        gt.get("wall_adjacent_up", 0.0),
        gt.get("wall_adjacent_down", 0.0),
        gt.get("wall_adjacent_left", 0.0),
        gt.get("wall_adjacent_right", 0.0),
        goal_dir_row,
        goal_dir_col,
        gt.get("action_had_effect", 0.0),
        gt.get("is_controllable_nearby", 0.0),
        gt.get("hidden_state_surprise", 0.0),
    ], dtype=torch.float32)


def _encode_obs(grid: np.ndarray, action: int, ctrl_ids=None) -> torch.Tensor:
    """Encode observation using the real StateEncoder."""
    perception = CoreKnowledgePerception()
    percept = perception.perceive(grid)
    if ctrl_ids is None:
        ctrl_ids = set()
    return encode(percept, action=action, controllable_ids=ctrl_ids,
                  grid_dims=grid.shape)


class MirrorTrainer:
    """Three-phase training pipeline for the Mirror model."""

    def __init__(
        self,
        player: WorldModelNet,
        mirror: MirrorModel,
        reality: MazeReality,
        lr: float = 1e-3,
        verbose: bool = True,
    ):
        self.player = player
        self.mirror = mirror
        self.reality = reality
        self.lr = lr
        self.verbose = verbose

        # Freeze Player
        for param in self.player.parameters():
            param.requires_grad = False
        self.player.eval()

    def train_phase_a(self, n_episodes: int = 200, n_epochs: int = 50) -> float:
        """Phase A: Supervised feature decoding.

        Collect (h_t_flat, ground_truth) pairs and train feature_decoder with MSE.
        """
        if self.verbose:
            print("\n--- Phase A: Supervised feature decoding ---")

        # Collect data
        episodes = _collect_episodes(
            self.player, self.reality, n_episodes,
            max_steps=200, verbose=self.verbose,
        )

        # Build dataset: (h_flat, gt_tensor) pairs
        all_h = []
        all_gt = []
        for ep in episodes:
            for h_flat, gt in zip(ep["h_flat_list"], ep["gt_list"]):
                all_h.append(h_flat.squeeze(0))  # [512]
                all_gt.append(_ground_truth_to_tensor(gt))  # [10]

        if not all_h:
            return float("inf")

        H = torch.stack(all_h)   # [N, 512]
        GT = torch.stack(all_gt)  # [N, 10]

        if self.verbose:
            print(f"    Dataset: {H.shape[0]} samples")

        # Train feature_decoder (backbone + feature_decoder only)
        optimizer = torch.optim.Adam(
            list(self.mirror.backbone.parameters()) +
            list(self.mirror.feature_decoder.parameters()),
            lr=self.lr,
        )

        self.mirror.train()
        best_loss = float("inf")

        for epoch in range(n_epochs):
            # Shuffle
            perm = torch.randperm(H.shape[0])
            H_shuf = H[perm]
            GT_shuf = GT[perm]

            # Mini-batch training
            batch_size = 256
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, H.shape[0], batch_size):
                h_batch = H_shuf[i:i + batch_size]
                gt_batch = GT_shuf[i:i + batch_size]

                decoded, _, _, _ = self.mirror(h_batch)
                loss = F.mse_loss(decoded, gt_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            best_loss = min(best_loss, avg_loss)

            if self.verbose and (epoch + 1) % max(1, n_epochs // 5) == 0:
                print(f"    Epoch {epoch + 1}/{n_epochs}: loss={avg_loss:.6f}")

        if self.verbose:
            print(f"    Phase A done. Best loss: {best_loss:.6f}")

        return best_loss

    def train_phase_b(self, n_episodes: int = 200, n_epochs: int = 30) -> float:
        """Phase B: Self-supervised confidence calibration.

        Confidence should be HIGH when feature decoding is accurate,
        LOW when inaccurate. Freeze feature_decoder, train confidence_head only.
        """
        if self.verbose:
            print("\n--- Phase B: Self-supervised confidence calibration ---")

        # Freeze feature_decoder
        for param in self.mirror.feature_decoder.parameters():
            param.requires_grad = False

        # Collect fresh data
        episodes = _collect_episodes(
            self.player, self.reality, n_episodes,
            max_steps=200, verbose=self.verbose,
        )

        # Build dataset with accuracy labels
        all_h = []
        all_targets = []  # 1.0 if decoding is accurate, 0.0 if not

        self.mirror.eval()
        with torch.no_grad():
            for ep in episodes:
                for h_flat, gt in zip(ep["h_flat_list"], ep["gt_list"]):
                    h = h_flat.squeeze(0)
                    gt_tensor = _ground_truth_to_tensor(gt)

                    # Compute decoding error
                    decoded, _, _, _ = self.mirror(h.unsqueeze(0))
                    error = F.mse_loss(decoded.squeeze(0), gt_tensor).item()

                    # Binary target: accurate if error < median threshold
                    all_h.append(h)
                    all_targets.append(error)

        if not all_h:
            return float("inf")

        H = torch.stack(all_h)
        errors = np.array(all_targets)

        # Use median error as threshold: below median -> confident (1.0),
        # above median -> not confident (0.0)
        median_error = np.median(errors)
        confidence_targets = torch.tensor(
            [1.0 if e < median_error else 0.0 for e in errors],
            dtype=torch.float32,
        ).unsqueeze(1)  # [N, 1]

        if self.verbose:
            print(f"    Dataset: {H.shape[0]} samples, median_error={median_error:.6f}")

        # Train confidence_head only (backbone is shared but also trains here)
        optimizer = torch.optim.Adam(
            list(self.mirror.backbone.parameters()) +
            list(self.mirror.confidence_head.parameters()),
            lr=self.lr * 0.5,
        )

        self.mirror.train()
        best_loss = float("inf")

        for epoch in range(n_epochs):
            perm = torch.randperm(H.shape[0])
            H_shuf = H[perm]
            CT_shuf = confidence_targets[perm]

            batch_size = 256
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, H.shape[0], batch_size):
                h_batch = H_shuf[i:i + batch_size]
                ct_batch = CT_shuf[i:i + batch_size]

                _, confidence, _, _ = self.mirror(h_batch)
                loss = F.binary_cross_entropy(confidence, ct_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            best_loss = min(best_loss, avg_loss)

            if self.verbose and (epoch + 1) % max(1, n_epochs // 5) == 0:
                print(f"    Epoch {epoch + 1}/{n_epochs}: loss={avg_loss:.6f}")

        # Unfreeze feature_decoder for future use
        for param in self.mirror.feature_decoder.parameters():
            param.requires_grad = True

        if self.verbose:
            print(f"    Phase B done. Best loss: {best_loss:.6f}")

        return best_loss

    def train_phase_c(self, n_episodes: int = 500) -> float:
        """Phase C: RL training for strategy and override heads.

        Train strategy_head and override_head with REINFORCE.
        Override rewards are 2x amplified.
        """
        if self.verbose:
            print("\n--- Phase C: RL training for strategy + override ---")

        # Freeze backbone and other heads for this phase
        for param in self.mirror.feature_decoder.parameters():
            param.requires_grad = False
        for param in self.mirror.confidence_head.parameters():
            param.requires_grad = False

        optimizer = torch.optim.Adam(
            list(self.mirror.strategy_head.parameters()) +
            list(self.mirror.override_head.parameters()) +
            list(self.mirror.backbone.parameters()),
            lr=self.lr * 0.1,
        )

        perception = CoreKnowledgePerception()
        gamma = 0.99
        all_rewards = []

        for ep in range(n_episodes):
            perception.reset()
            grid = self.reality.reset()
            hidden = self.player.init_hidden(batch_size=1)

            log_probs = []
            rewards = []
            ctrl_ids = set()
            prev_action = 0

            self.mirror.train()

            for step in range(200):
                percept = perception.perceive(grid)
                action_for_enc = 0 if step == 0 else prev_action
                x = encode(percept, action=action_for_enc,
                           controllable_ids=ctrl_ids, grid_dims=grid.shape)

                with torch.no_grad():
                    pred, player_logits, value, hidden = self.player(
                        x.unsqueeze(0), hidden
                    )

                h_flat = self.player.get_hidden_flat(hidden)  # [1, 512]

                # Mirror forward (with gradients for strategy + override)
                _, confidence, strategy_logits, override_logits = self.mirror(h_flat)

                # Sample strategy
                strategy_probs = F.softmax(strategy_logits, dim=-1)
                strategy_dist = torch.distributions.Categorical(strategy_probs)
                strategy_sample = strategy_dist.sample()
                strategy_log_prob = strategy_dist.log_prob(strategy_sample)

                strategy_val = strategy_sample.item()
                conf_val = confidence.item()

                # Determine action based on strategy
                if strategy_val == 0 or conf_val > 0.7:
                    # Trust Player
                    action = player_logits.squeeze(0).argmax().item()
                    action_log_prob = strategy_log_prob  # Only strategy contributes
                elif strategy_val == 1 or conf_val < 0.3:
                    # Explore
                    action = torch.randint(0, self.reality.n_actions, (1,)).item()
                    action_log_prob = strategy_log_prob
                else:
                    # Override -- use Mirror's action head
                    override_probs = F.softmax(override_logits, dim=-1)
                    override_dist = torch.distributions.Categorical(override_probs)
                    override_sample = override_dist.sample()
                    override_log_prob = override_dist.log_prob(override_sample)
                    action = override_sample.item()
                    # Both strategy and override contribute to gradient
                    action_log_prob = strategy_log_prob + override_log_prob

                log_probs.append(action_log_prob)
                prev_action = action

                # Step environment
                next_grid, reward, done, info = self.reality.step(action)

                # Amplify override rewards 2x
                if strategy_val == 2:
                    reward *= 2.0

                rewards.append(reward)

                # Update controllable
                if step > 0:
                    perception.agency.record_action_contingency(action, grid, next_grid)
                    ctrl_ids = set(perception.agency.detect_controllable_objects(
                        percept.objects, perception.agency._contingencies
                    ))

                grid = next_grid
                if done:
                    break

            # REINFORCE update
            if log_probs:
                returns = []
                G = 0.0
                for r in reversed(rewards):
                    G = r + gamma * G
                    returns.insert(0, G)
                returns = torch.tensor(returns, dtype=torch.float32)

                if len(returns) > 1:
                    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

                policy_loss = 0.0
                for lp, G in zip(log_probs, returns):
                    policy_loss -= lp * G
                policy_loss = policy_loss / len(log_probs)

                optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.mirror.parameters(), max_norm=1.0)
                optimizer.step()

            ep_reward = sum(rewards)
            all_rewards.append(ep_reward)

            if self.verbose and (ep + 1) % max(1, n_episodes // 5) == 0:
                recent = all_rewards[-50:]
                print(f"    Ep {ep + 1}/{n_episodes}: "
                      f"reward={np.mean(recent):.3f}")

        # Unfreeze all
        for param in self.mirror.parameters():
            param.requires_grad = True

        avg_reward = float(np.mean(all_rewards[-100:])) if all_rewards else 0.0
        if self.verbose:
            print(f"    Phase C done. Avg reward (last 100): {avg_reward:.3f}")

        return avg_reward

    def train_all(
        self,
        phase_a_episodes: int = 200,
        phase_a_epochs: int = 50,
        phase_b_episodes: int = 200,
        phase_b_epochs: int = 30,
        phase_c_episodes: int = 500,
    ) -> MirrorTrainingStats:
        """Run all three training phases sequentially."""
        loss_a = self.train_phase_a(phase_a_episodes, phase_a_epochs)
        loss_b = self.train_phase_b(phase_b_episodes, phase_b_epochs)
        reward_c = self.train_phase_c(phase_c_episodes)

        total_eps = phase_a_episodes + phase_b_episodes + phase_c_episodes

        return MirrorTrainingStats(
            phase_a_loss=loss_a,
            phase_b_loss=loss_b,
            phase_c_reward=reward_c,
            total_episodes=total_eps,
        )
