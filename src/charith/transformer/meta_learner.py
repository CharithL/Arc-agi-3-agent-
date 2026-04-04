"""
Causal Transformer Meta-Learner for in-context game learning.

The Transformer receives a sequence of alternating percept and action tokens:
[P_1, A_1, P_2, A_2, ..., P_T]

Through attention over this history, it identifies patterns like
"every time I did ACTION3, the red object moved left" -- without
any weight updates. The weights encode HOW TO LEARN. The context
encodes WHAT THIS GAME DOES.

~3.6M parameters. Trainable on single GPU in hours.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MetaLearner(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_layers=6, d_ff=512,
                 max_context=200, n_actions=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_actions = n_actions
        self.max_context = max_context

        # Positional encoding (learned)
        self.pos_embedding = nn.Embedding(max_context, d_model)

        # Token type embedding: 0=percept, 1=action
        self.type_embedding = nn.Embedding(2, d_model)

        # Transformer decoder layers (causal attention)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation='gelu', batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Output heads
        # Action head: given context ending with P_t, predict which action to take
        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, n_actions),
        )

        # Prediction head: given context ending with A_t, predict next P_{t+1}
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),  # outputs predicted percept token
        )

        # Value head: for REINFORCE baseline
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tokens, token_types):
        """
        Forward pass on a sequence of tokens.

        Args:
            tokens: [batch, seq_len, d_model] -- the context sequence
            token_types: [batch, seq_len] -- 0 for percept, 1 for action (LongTensor)

        Returns:
            action_logits: [batch, seq_len, n_actions] -- action prediction at each position
            predictions: [batch, seq_len, d_model] -- next-percept prediction at each position
            values: [batch, seq_len, 1] -- value estimate at each position
        """
        batch, seq_len, _ = tokens.shape
        device = tokens.device

        # Add positional and type embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)
        pos_emb = self.pos_embedding(positions.clamp(max=self.max_context - 1))
        type_emb = self.type_embedding(token_types)

        x = tokens + pos_emb + type_emb

        # Causal mask: each position can only attend to itself and earlier positions
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)

        # TransformerDecoder needs a memory tensor -- use zeros (self-attention only)
        memory = torch.zeros(batch, 1, self.d_model, device=device)

        # Forward through transformer
        out = self.transformer(x, memory, tgt_mask=causal_mask)

        # Compute outputs at every position
        action_logits = self.action_head(out)
        predictions = self.prediction_head(out)
        values = self.value_head(out)

        return action_logits, predictions, values

    def get_action(self, tokens, token_types, available_actions=None, temperature=1.0):
        """Get action from the LAST position in the context (deployment mode).

        Args:
            tokens: [1, seq_len, d_model]
            token_types: [1, seq_len]
            available_actions: list of valid action indices (mask others)
            temperature: softmax temperature
        Returns:
            action: int
            log_prob: tensor
            value: tensor
        """
        action_logits, _, values = self.forward(tokens, token_types)

        # Use logits from the last position
        last_logits = action_logits[0, -1, :]  # [n_actions]
        last_value = values[0, -1, 0]  # scalar

        # Mask unavailable actions
        if available_actions is not None:
            mask = torch.full_like(last_logits, float('-inf'))
            for a in available_actions:
                if a < len(mask):
                    mask[a] = 0.0
            last_logits = last_logits + mask

        # Sample action
        probs = F.softmax(last_logits / temperature, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob, last_value

    def get_attention_weights(self, tokens, token_types):
        """Extract attention weights for DESCARTES probing.
        Returns list of attention matrices, one per layer.
        Each matrix is [n_heads, seq_len, seq_len].
        """
        # This requires hooks -- implement as needed for probing
        # For now, return None
        return None
