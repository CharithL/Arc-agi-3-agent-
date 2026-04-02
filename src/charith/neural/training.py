"""
Training loop for Path 3: GRU World Model on Synthetic Realities.

Trains the GRU with two losses:
1. Prediction loss (MSE): predict next state encoding from current hidden state
2. RL loss (REINFORCE with baseline): learn policy to maximize reward

Collects (hidden_state, ground_truth) pairs for DESCARTES probing after training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path

from charith.synthetic.base_reality import SyntheticReality
from charith.perception.core_knowledge import CoreKnowledgePerception
from charith.neural.state_encoder import encode, D_INPUT
from charith.neural.world_model_net import WorldModelNet
from charith.neural.action_head import select_action


@dataclass
class EpisodeData:
    """Collected data from one episode for probing."""
    hidden_states: List[np.ndarray] = field(default_factory=list)
    ground_truths: List[Dict[str, float]] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    total_reward: float = 0.0
    steps: int = 0
    solved: bool = False


@dataclass
class TrainingStats:
    """Statistics from a training run."""
    episodes: int = 0
    total_steps: int = 0
    mean_reward: float = 0.0
    solve_rate: float = 0.0
    mean_pred_loss: float = 0.0
    mean_policy_loss: float = 0.0


def train_on_reality(
    model: WorldModelNet,
    reality: SyntheticReality,
    n_episodes: int = 500,
    lr: float = 3e-4,
    gamma: float = 0.99,
    pred_loss_weight: float = 1.0,
    rl_loss_weight: float = 0.5,
    spatial_loss_weight: float = 1.0,
    spatial_lookahead_k: int = 5,
    max_steps_per_episode: int = 200,
    collect_probing_data: bool = True,
    verbose: bool = True,
) -> Tuple[TrainingStats, List[EpisodeData]]:
    """
    Train the GRU world model on a synthetic reality.

    Args:
        model: the WorldModelNet to train
        reality: the synthetic reality environment
        n_episodes: number of episodes to train
        lr: learning rate
        gamma: reward discount factor
        pred_loss_weight: weight for prediction MSE loss
        rl_loss_weight: weight for REINFORCE policy loss
        max_steps_per_episode: max steps before truncation
        collect_probing_data: whether to collect h_t and ground truth for probes
        verbose: print progress

    Returns:
        (stats, probe_data): training statistics and collected probing data
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    perception = CoreKnowledgePerception()

    all_episode_data: List[EpisodeData] = []
    total_rewards = []
    total_solved = 0
    all_pred_losses = []
    all_policy_losses = []

    for ep in range(n_episodes):
        model.train()
        perception.reset()

        # Reset environment
        grid = reality.reset()
        hidden = model.init_hidden(batch_size=1)

        # Episode storage
        ep_data = EpisodeData()
        log_probs = []
        values = []
        rewards = []
        pred_losses = []
        spatial_losses = []

        # Spatial prediction buffer: store (hidden, action_onehot) K steps ago
        # so we can compute loss when we know the actual position K steps later
        spatial_buffer = []  # list of (hidden_state, action_onehot, step_idx)
        position_history = []  # list of (relative_row, relative_col) per step

        # Get initial percept
        percept = perception.perceive(grid)
        gt = reality.get_ground_truth()

        # Identify controllable (for the encoder)
        ctrl_ids = set()

        prev_encoding = None

        for step in range(max_steps_per_episode):
            # Encode current state
            action_for_encoding = 0 if step == 0 else action
            x = encode(percept, action=action_for_encoding,
                       controllable_ids=ctrl_ids, grid_dims=grid.shape)

            # GRU forward
            with torch.set_grad_enabled(True):
                pred, policy_logits, value, hidden = model(
                    x.unsqueeze(0), hidden
                )

            # Prediction loss (predict current encoding from previous hidden)
            if prev_encoding is not None:
                pred_loss = F.mse_loss(pred.squeeze(0), x.detach())
                pred_losses.append(pred_loss)

            # Select action
            action, log_prob = select_action(
                policy_logits.squeeze(0),
                available_actions=list(range(reality.n_actions)),
                temperature=1.0,
            )

            # Store spatial prediction: predict where we'll be in K steps
            action_oh = torch.zeros(reality.n_actions)
            action_oh[action] = 1.0
            spatial_buffer.append((
                hidden.detach().clone(), action_oh, step
            ))

            # Step environment
            next_grid, reward, done, info = reality.step(action)
            next_percept = perception.perceive(next_grid)
            next_gt = reality.get_ground_truth()

            # Record current position for spatial loss targets
            position_history.append((
                next_gt.get('controllable_relative_row', 0.0),
                next_gt.get('controllable_relative_col', 0.0),
            ))

            # Compute spatial loss for predictions made K steps ago
            if len(spatial_buffer) > spatial_lookahead_k:
                old_hidden, old_action_oh, old_step = spatial_buffer[
                    len(spatial_buffer) - 1 - spatial_lookahead_k
                ]
                # Actual position now (K steps after the prediction)
                actual_pos = torch.tensor(
                    position_history[-1], dtype=torch.float32
                ).unsqueeze(0)
                # Predict from old hidden state
                predicted_pos = model.predict_future_position(
                    old_hidden, old_action_oh.unsqueeze(0)
                )
                sp_loss = F.mse_loss(predicted_pos, actual_pos)
                spatial_losses.append(sp_loss)

            # Record action contingency for controllable detection
            if step > 0:
                perception.agency.record_action_contingency(action, grid, next_grid)
                ctrl_ids = set(perception.agency.detect_controllable_objects(
                    percept.objects, perception.agency._contingencies
                ))

            # Store for REINFORCE
            log_probs.append(log_prob)
            values.append(value.squeeze())
            rewards.append(reward)

            # Store for probing
            if collect_probing_data:
                h_np = model.get_hidden_state(hidden).detach().squeeze(0).numpy()
                ep_data.hidden_states.append(h_np)
                ep_data.ground_truths.append(next_gt.copy())
                ep_data.rewards.append(reward)

            prev_encoding = x.detach()
            grid = next_grid
            percept = next_percept
            gt = next_gt

            if done:
                ep_data.solved = True
                total_solved += 1
                break

        ep_data.steps = step + 1
        ep_data.total_reward = sum(rewards)

        # ---- Compute losses and update ----
        if log_probs:
            # Compute discounted returns
            returns = []
            G = 0.0
            for r in reversed(rewards):
                G = r + gamma * G
                returns.insert(0, G)
            returns = torch.tensor(returns, dtype=torch.float32)

            # Normalize returns
            if len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            # Policy loss (REINFORCE with value baseline)
            policy_loss = 0.0
            value_loss = 0.0
            for lp, val, G in zip(log_probs, values, returns):
                advantage = G - val.detach()
                policy_loss -= lp * advantage
                value_loss += F.mse_loss(val, G)

            policy_loss = policy_loss / len(log_probs)
            value_loss = value_loss / len(values)

            # Prediction loss (average over episode)
            if pred_losses:
                mean_pred_loss = torch.stack(pred_losses).mean()
            else:
                mean_pred_loss = torch.tensor(0.0)

            # Spatial prediction loss (auxiliary head)
            if spatial_losses:
                mean_spatial_loss = torch.stack(spatial_losses).mean()
            else:
                mean_spatial_loss = torch.tensor(0.0)

            # Total loss: prediction + RL + spatial auxiliary
            total_loss = (pred_loss_weight * mean_pred_loss +
                         rl_loss_weight * (policy_loss + 0.5 * value_loss) +
                         spatial_loss_weight * mean_spatial_loss)

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            all_pred_losses.append(mean_pred_loss.item())
            all_policy_losses.append(policy_loss.item() if isinstance(policy_loss, torch.Tensor) else policy_loss)

        total_rewards.append(ep_data.total_reward)

        # Detach hidden state for next episode (prevent graph accumulation)
        hidden = hidden.detach()

        if collect_probing_data:
            all_episode_data.append(ep_data)

        # Progress logging (every 50 episodes or 10 intervals, whichever is more frequent)
        log_interval = min(50, max(1, n_episodes // 10))
        if verbose and (ep + 1) % log_interval == 0:
            recent = total_rewards[-50:]
            recent_solved = sum(1 for d in all_episode_data[-50:] if d.solved)
            recent_pred = all_pred_losses[-50:] if all_pred_losses else [0]
            print(f"  Ep {ep+1:4d}/{n_episodes}: "
                  f"reward={np.mean(recent):+.3f}, "
                  f"solved={recent_solved}/50, "
                  f"pred_loss={np.mean(recent_pred):.4f}, "
                  f"steps={ep_data.steps}")

    stats = TrainingStats(
        episodes=n_episodes,
        total_steps=sum(d.steps for d in all_episode_data),
        mean_reward=float(np.mean(total_rewards)),
        solve_rate=total_solved / n_episodes,
        mean_pred_loss=float(np.mean(all_pred_losses)) if all_pred_losses else 0.0,
        mean_policy_loss=float(np.mean(all_policy_losses)) if all_policy_losses else 0.0,
    )

    return stats, all_episode_data


def collect_probing_dataset(
    episode_data: List[EpisodeData],
    feature_name: str,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Extract (hidden_states, targets, episode_boundaries) for DESCARTES probing.

    Args:
        episode_data: list of EpisodeData from training
        feature_name: which ground truth feature to extract (e.g., 'distance_to_goal')

    Returns:
        hidden_states: [N, hidden_size] numpy array
        targets: [N] numpy array
        episode_boundaries: list of start indices for each episode
    """
    all_h = []
    all_targets = []
    episode_boundaries = []

    for ep_data in episode_data:
        if not ep_data.hidden_states:
            continue
        episode_boundaries.append(len(all_h))
        for h, gt in zip(ep_data.hidden_states, ep_data.ground_truths):
            all_h.append(h)
            all_targets.append(gt.get(feature_name, 0.0))

    if not all_h:
        return np.array([]), np.array([]), []

    return np.array(all_h), np.array(all_targets), episode_boundaries
