"""Context Manager: maintains the growing sequence of tokens for the Transformer."""
import torch
from typing import Optional


class ContextManager:
    """Manages the growing interaction history for the Transformer.

    The context is a sequence: [P1, A1, P2, A2, ..., P_T]
    At tick T, the sequence has 2T-1 tokens.

    Uses a sliding window when context exceeds max_length:
    keeps first 10 tokens (initial discoveries) + last (max-10) tokens.
    """

    def __init__(self, d_model=256, max_length=200):
        self.d_model = d_model
        self.max_length = max_length
        self._tokens = []  # list of [d_model] tensors
        self._token_types = []  # 'percept' or 'action' for each token

    def add_percept(self, token: torch.Tensor):
        """Add a percept token to the context."""
        self._tokens.append(token.detach())
        self._token_types.append('percept')
        self._truncate_if_needed()

    def add_action(self, token: torch.Tensor):
        """Add an action token to the context."""
        self._tokens.append(token.detach())
        self._token_types.append('action')
        self._truncate_if_needed()

    def get_sequence(self) -> Optional[torch.Tensor]:
        """Get the full context sequence as a tensor.
        Returns: [seq_len, d_model] tensor, or None if empty
        """
        if not self._tokens:
            return None
        return torch.stack(self._tokens)  # [seq_len, d_model]

    def get_length(self) -> int:
        return len(self._tokens)

    def _truncate_if_needed(self):
        """Sliding window: keep first 10 + last (max-10) tokens."""
        if len(self._tokens) > self.max_length:
            anchor = 10  # Keep first 10 tokens (initial discoveries)
            keep_recent = self.max_length - anchor
            first = self._tokens[:anchor]
            first_types = self._token_types[:anchor]
            recent = self._tokens[-keep_recent:]
            recent_types = self._token_types[-keep_recent:]
            self._tokens = first + recent
            self._token_types = first_types + recent_types

    def reset(self):
        """Clear context for new episode."""
        self._tokens.clear()
        self._token_types.clear()

    @property
    def is_empty(self) -> bool:
        return len(self._tokens) == 0
