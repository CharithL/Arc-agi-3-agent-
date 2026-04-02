"""Action selection from policy logits."""

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


def select_action(
    policy_logits: torch.Tensor,
    available_actions: Optional[List[int]] = None,
    temperature: float = 1.0,
    greedy: bool = False,
) -> Tuple[int, torch.Tensor]:
    """Select action from policy logits.

    Args:
        policy_logits: [n_actions] tensor of raw logits.
        available_actions: list of valid action indices (mask others to -inf).
        temperature: softmax temperature (lower = more greedy).
        greedy: if True, return argmax.

    Returns:
        action: int -- selected action index.
        log_prob: log probability of selected action.
    """
    if available_actions is not None:
        mask = torch.full_like(policy_logits, float("-inf"))
        for a in available_actions:
            mask[a] = 0.0
        policy_logits = policy_logits + mask

    if greedy:
        action = policy_logits.argmax().item()
        log_prob = F.log_softmax(policy_logits / temperature, dim=-1)[action]
    else:
        probs = F.softmax(policy_logits / temperature, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action_tensor = dist.sample()
        action = action_tensor.item()
        log_prob = dist.log_prob(action_tensor)

    return action, log_prob
