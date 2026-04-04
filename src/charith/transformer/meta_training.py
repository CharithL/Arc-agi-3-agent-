"""
Meta-Training Loop: Train the Transformer across 50K diverse procedural games.

Each episode:
1. Generate a new random game
2. Play it for max_steps ticks (context grows each tick)
3. Compute prediction loss + policy gradient loss
4. Update Transformer weights

After meta-training, the weights encode "how to learn any game fast."
The context window provides game-specific knowledge at deployment.
"""
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

from charith.gamegen.generator import GameGenerator, ProceduralGame
from charith.perception.core_knowledge import CoreKnowledgePerception
from charith.transformer.percept_tokenizer import PerceptTokenizer
from charith.transformer.action_tokenizer import ActionTokenizer
from charith.transformer.context_manager import ContextManager
from charith.transformer.meta_learner import MetaLearner


@dataclass
class MetaTrainingConfig:
    n_episodes: int = 50000
    max_steps_per_episode: int = 50
    lr: float = 3e-4
    gamma: float = 0.99
    pred_loss_weight: float = 1.0
    policy_loss_weight: float = 0.5
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    max_context: int = 200
    n_actions: int = 8
    discovery_bonus: float = 0.05  # bonus for first new effect per action
    log_interval: int = 100


@dataclass
class EpisodeResult:
    total_reward: float
    steps: int
    solved: bool
    mean_pred_loss: float
    game_level: int


def meta_train(config: MetaTrainingConfig = None,
               checkpoint_path: Optional[str] = None,
               save_dir: str = "checkpoints") -> MetaLearner:
    """Run full meta-training.

    Args:
        config: training hyperparameters
        checkpoint_path: resume from checkpoint if provided
        save_dir: where to save checkpoints
    Returns:
        trained MetaLearner
    """
    if config is None:
        config = MetaTrainingConfig()

    # Device detection — use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == 'cuda' else ""))

    # Initialize
    model = MetaLearner(
        d_model=config.d_model, n_heads=config.n_heads,
        n_layers=config.n_layers, n_actions=config.n_actions,
        max_context=config.max_context,
    ).to(device)
    if checkpoint_path and Path(checkpoint_path).exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        print(f"Resumed from {checkpoint_path}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    game_gen = GameGenerator()
    perception = CoreKnowledgePerception()
    percept_tok = PerceptTokenizer(d_model=config.d_model).to(device)
    action_tok = ActionTokenizer(n_actions=config.n_actions, d_model=config.d_model).to(device)
    context = ContextManager(d_model=config.d_model, max_length=config.max_context)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Meta-Learner: {param_count:,} parameters")

    results: List[EpisodeResult] = []

    for ep in range(config.n_episodes):
        model.train()
        perception.reset()
        context.reset()

        # Determine game level based on curriculum
        if ep < config.n_episodes * 0.2:
            level = 1
        elif ep < config.n_episodes * 0.5:
            level = 2
        elif ep < config.n_episodes * 0.8:
            level = 3
        else:
            level = np.random.choice([1, 2, 3, 4])

        game = game_gen.generate(level=level)
        grid = game.reset()

        # Episode storage
        log_probs = []
        values = []
        rewards = []
        pred_losses = []
        seen_effects = {}  # action -> set of observed effects (for discovery bonus)

        prev_percept = None
        step = 0
        done = False
        info = {}

        for step in range(config.max_steps_per_episode):
            # Perceive
            percept = perception.perceive(grid)

            # Tokenize percept and add to context
            p_token = percept_tok(percept, grid_dims=grid.shape,
                                  prev_percept=prev_percept)
            context.add_percept(p_token)

            # Get context sequence
            seq = context.get_sequence()
            if seq is None:
                break

            # Build token types: alternating 0 (percept) and 1 (action)
            n_tokens = seq.shape[0]
            types = torch.zeros(n_tokens, dtype=torch.long)
            for i in range(n_tokens):
                types[i] = i % 2  # 0, 1, 0, 1, ...

            # Forward pass — move to device
            tokens_batch = seq.unsqueeze(0).to(device)  # [1, seq_len, d_model]
            types_batch = types.unsqueeze(0).to(device)  # [1, seq_len]

            action, log_prob, value = model.get_action(
                tokens_batch, types_batch,
                available_actions=list(range(game.spec.n_actions)),
            )

            # Prediction loss (predict current percept from context before it was added)
            if n_tokens >= 3:  # Need at least P, A, P to predict
                action_logits, predictions, _ = model(tokens_batch, types_batch)
                # The prediction at position -2 (last action token) should predict current percept
                predicted_percept = predictions[0, -2, :]  # prediction after last action
                actual_percept = p_token.detach().to(device)
                pred_loss = F.mse_loss(predicted_percept, actual_percept)
                pred_losses.append(pred_loss)

            # Step game
            next_grid, reward, done, info = game.step(action)

            # Discovery bonus: first time this action produces a specific change
            change_signature = int(np.sum(grid != next_grid))
            if action not in seen_effects:
                seen_effects[action] = set()
            if change_signature not in seen_effects[action] and change_signature > 0:
                reward += config.discovery_bonus
                seen_effects[action].add(change_signature)

            # Tokenize action+reward and add to context
            a_token = action_tok(action, reward, done)
            context.add_action(a_token)

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)

            prev_percept = percept
            grid = next_grid

            if done:
                break

        # Compute losses and update
        if log_probs:
            # Discounted returns
            returns = []
            G = 0.0
            for r in reversed(rewards):
                G = r + config.gamma * G
                returns.insert(0, G)
            returns = torch.tensor(returns, dtype=torch.float32, device=device)
            if len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            # Policy loss (REINFORCE with baseline)
            policy_loss = torch.tensor(0.0, device=device)
            value_loss = torch.tensor(0.0, device=device)
            for lp, val, G in zip(log_probs, values, returns):
                advantage = G - val.detach()
                policy_loss = policy_loss - lp * advantage
                value_loss = value_loss + F.mse_loss(val, G)
            policy_loss = policy_loss / len(log_probs)
            value_loss = value_loss / len(values)

            # Prediction loss
            mean_pred = torch.stack(pred_losses).mean() if pred_losses else torch.tensor(0.0, device=device)

            # Total loss
            total_loss = (config.pred_loss_weight * mean_pred +
                         config.policy_loss_weight * (policy_loss + 0.5 * value_loss))

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Record
        ep_result = EpisodeResult(
            total_reward=sum(rewards) if rewards else 0.0,
            steps=step + 1,
            solved=info.get('reason', '') == 'win' if isinstance(info, dict) else False,
            mean_pred_loss=mean_pred.item() if pred_losses else 0.0,
            game_level=level,
        )
        results.append(ep_result)

        # Log
        if (ep + 1) % config.log_interval == 0:
            recent = results[-config.log_interval:]
            mean_r = np.mean([r.total_reward for r in recent])
            solve_rate = np.mean([r.solved for r in recent])
            mean_pl = np.mean([r.mean_pred_loss for r in recent])
            print(f"  Ep {ep+1:6d}/{config.n_episodes}: "
                  f"reward={mean_r:+.3f} solve={solve_rate:.1%} "
                  f"pred_loss={mean_pl:.4f} level={level}")

        # Checkpoint every 5000 episodes
        if (ep + 1) % 5000 == 0:
            Path(save_dir).mkdir(exist_ok=True)
            torch.save(model.state_dict(), f"{save_dir}/meta_learner_ep{ep+1}.pt")

    # Final save
    Path(save_dir).mkdir(exist_ok=True)
    torch.save(model.state_dict(), f"{save_dir}/meta_learner_final.pt")
    print(f"\nSaved to {save_dir}/meta_learner_final.pt")

    return model
