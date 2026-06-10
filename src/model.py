"""
Nine Men's Morris -- Relational ActorCritic.

Token-based architecture:
  - 24 board tokens, one per board point, embed `node_feats[:, p, :]` +
    learned positional embedding for point p.
  - 1 global token, embedding `global_feats` (phase, piece counts, etc.).
  - n_layers pre-LN transformer encoder blocks over the 25 tokens. Each
    attention head adds learned per-head scalars on adjacency and mill
    cohabitation **between board tokens only**; the global token's row
    and column carry no structural bias. The bias matrices are sigma-
    invariant (graph automorphism), so the same matrices are correct in
    every augmented frame.
  - Policy head: either a pointer-style per-node + from->to scoring head
    (the final design) or a flat pooled linear (Milestone A, kept behind
    config.policy_head == 'flat' for debugging).
  - Value head: MLP on the global token (optionally concat mean-pool over
    board tokens), clipped to +/- config.value_clip.

Forward signature:
  forward(node_feats[B,24,F], global_feats[B,G]) -> (logits[B,600], value[B])
The legal-action mask is applied OUTSIDE forward, as before.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from utils import ADJ_MATRIX, MILL_MATRIX


class _AttentionBlock(nn.Module):
    """Multi-head self-attention with per-head structural biases between board
    tokens; the global token (last) sees no structural bias."""

    def __init__(self, d_model: int, n_heads: int, d_k: int,
                 dropout: float, n_positions: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.scale = 1.0 / math.sqrt(d_k)
        self.n_positions = n_positions

        inner = n_heads * d_k
        self.ln = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, inner, bias=False)
        self.k_proj = nn.Linear(d_model, inner, bias=False)
        self.v_proj = nn.Linear(d_model, inner, bias=False)
        self.out_proj = nn.Linear(inner, d_model)
        self.dropout = nn.Dropout(dropout)

        # Per-head structural biases between board tokens (independent so the
        # head can weight adjacency and mill-cohabitation separately).
        self.b_adj = nn.Parameter(torch.zeros(n_heads))
        self.b_mill = nn.Parameter(torch.zeros(n_heads))

        # Cache the [24, 24] bool matrices as buffers so .to(device) moves them.
        adj = torch.from_numpy(np.ascontiguousarray(ADJ_MATRIX.astype(np.float32)))
        mill = torch.from_numpy(np.ascontiguousarray(MILL_MATRIX.astype(np.float32)))
        self.register_buffer("adj_board", adj, persistent=False)
        self.register_buffer("mill_board", mill, persistent=False)

        nn.init.xavier_uniform_(self.q_proj.weight, gain=0.1)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=0.1)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=0.1)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.1)
        nn.init.zeros_(self.out_proj.bias)

    def _structural_bias(self, seq_len: int, device, dtype) -> torch.Tensor:
        """Return [n_heads, seq_len, seq_len] bias to add to attention logits.

        Non-zero only on the board-board sub-block (positions 0..23). The global
        token (last index) sees and is seen with zero structural bias.
        """
        n_pos = self.n_positions
        # [H, n_pos, n_pos]
        sub = (self.b_adj.view(-1, 1, 1) * self.adj_board
               + self.b_mill.view(-1, 1, 1) * self.mill_board)
        # Pad to [H, seq_len, seq_len].
        out = sub.new_zeros((self.n_heads, seq_len, seq_len))
        out[:, :n_pos, :n_pos] = sub
        return out.to(dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, D]
        B, S, _ = x.shape
        h = self.ln(x)
        q = self.q_proj(h).view(B, S, self.n_heads, self.d_k).transpose(1, 2)  # [B,H,S,d_k]
        k = self.k_proj(h).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(h).view(B, S, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B,H,S,S]
        scores = scores + self._structural_bias(S, x.device, scores.dtype).unsqueeze(0)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        ctx = torch.matmul(attn, v)  # [B,H,S,d_k]
        ctx = ctx.transpose(1, 2).contiguous().view(B, S, self.n_heads * self.d_k)
        return x + self.dropout(self.out_proj(ctx))


class _FeedForward(nn.Module):
    def __init__(self, d_model: int, ff_mult: int, dropout: float):
        super().__init__()
        inner = ff_mult * d_model
        self.ln = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, inner)
        self.fc2 = nn.Linear(inner, d_model)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.1)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        h = F.gelu(self.fc1(h))
        h = self.dropout(h)
        h = self.fc2(h)
        return x + self.dropout(h)


class ActorCritic(nn.Module):
    """Relational token-based ActorCritic.

    Constructor signature kept positional-compatible with the rest of the
    codebase (obs_size, num_actions, config) so model_loader / trainer
    instantiation paths don't need to change.
    """

    def __init__(self, obs_size: int, num_actions: int, config: Config):
        super().__init__()
        self.num_actions = num_actions
        self.value_clip = config.value_clip

        self.d_model = config.d_model
        self.n_layers = config.n_layers
        self.n_heads = config.n_heads
        self.d_k = config.d_k
        self.ff_mult = config.ff_mult
        self.n_positions = config.num_positions
        self.node_feat_dim = config.node_feat_dim
        self.global_feat_dim = config.global_feat_dim
        self.policy_head_kind = config.policy_head
        self.value_pool = config.value_pool
        dropout = float(config.dropout)

        # Per-board-point token embedding.
        self.node_embed = nn.Linear(self.node_feat_dim, self.d_model)
        self.position_embed = nn.Embedding(self.n_positions, self.d_model)
        # Global-token embedding.
        self.global_embed = nn.Linear(self.global_feat_dim, self.d_model)

        nn.init.xavier_uniform_(self.node_embed.weight, gain=0.5)
        nn.init.zeros_(self.node_embed.bias)
        nn.init.normal_(self.position_embed.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.global_embed.weight, gain=0.5)
        nn.init.zeros_(self.global_embed.bias)

        # Trunk: n_layers x (attention + feed-forward).
        self.attn_blocks = nn.ModuleList([
            _AttentionBlock(self.d_model, self.n_heads, self.d_k,
                            dropout, self.n_positions)
            for _ in range(self.n_layers)
        ])
        self.ff_blocks = nn.ModuleList([
            _FeedForward(self.d_model, self.ff_mult, dropout)
            for _ in range(self.n_layers)
        ])
        self.trunk_ln = nn.LayerNorm(self.d_model)

        # Policy head.
        if self.policy_head_kind == 'pointer':
            self.node_score = nn.Linear(self.d_model, 1)
            self.move_q = nn.Linear(self.d_model, self.d_k, bias=False)
            self.move_k = nn.Linear(self.d_model, self.d_k, bias=False)
            self.move_scale = 1.0 / math.sqrt(self.d_k)
            nn.init.orthogonal_(self.node_score.weight, gain=0.01)
            nn.init.zeros_(self.node_score.bias)
            nn.init.orthogonal_(self.move_q.weight, gain=0.01)
            nn.init.orthogonal_(self.move_k.weight, gain=0.01)
        elif self.policy_head_kind == 'flat':
            pool_dim = self.d_model * (2 if self.value_pool == 'global+mean' else 1)
            self.flat_policy = nn.Linear(pool_dim, self.num_actions)
            nn.init.orthogonal_(self.flat_policy.weight, gain=0.01)
            nn.init.zeros_(self.flat_policy.bias)
        else:
            raise ValueError(f"unknown policy_head: {self.policy_head_kind!r}")

        # Value head.
        v_in = self.d_model * (2 if self.value_pool == 'global+mean' else 1)
        self.value_net = nn.Sequential(
            nn.LayerNorm(v_in),
            nn.Linear(v_in, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
        )
        self.value_head = nn.Linear(self.d_model // 2, 1)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

        # Cached position indices [0..23].
        self.register_buffer("_pos_ids", torch.arange(self.n_positions, dtype=torch.long),
                             persistent=False)

    def _embed_tokens(self, node_feats: torch.Tensor, global_feats: torch.Tensor) -> torch.Tensor:
        """node_feats: [B,24,F], global_feats: [B,G] -> [B,25,D]."""
        B = node_feats.shape[0]
        node_tok = self.node_embed(node_feats) + self.position_embed(self._pos_ids).unsqueeze(0)
        glob_tok = self.global_embed(global_feats).unsqueeze(1)  # [B,1,D]
        return torch.cat([node_tok, glob_tok], dim=1)

    def _pool(self, tokens: torch.Tensor) -> torch.Tensor:
        """Pool the 25 trunk outputs into a single vector per batch element."""
        global_tok = tokens[:, -1, :]
        if self.value_pool == 'global+mean':
            board_mean = tokens[:, :self.n_positions, :].mean(dim=1)
            return torch.cat([global_tok, board_mean], dim=-1)
        return global_tok

    def _policy_logits(self, board_tokens: torch.Tensor, pooled: torch.Tensor) -> torch.Tensor:
        """Compute the [B, 600] action logits.

        board_tokens: [B, 24, D]; pooled: [B, pool_dim].
        Action layout matches the engine: 0..23 = place/capture, 24+f*24+t = move.
        """
        if self.policy_head_kind == 'pointer':
            node_logits = self.node_score(board_tokens).squeeze(-1)        # [B, 24]
            q = self.move_q(board_tokens)                                  # [B, 24, d_k]
            k = self.move_k(board_tokens)                                  # [B, 24, d_k]
            move_scores = torch.matmul(q, k.transpose(-2, -1)) * self.move_scale  # [B, 24, 24]
            move_logits = move_scores.reshape(move_scores.shape[0], -1)    # [B, 576] (from-major)
            return torch.cat([node_logits, move_logits], dim=-1)           # [B, 600]
        return self.flat_policy(pooled)

    def forward(self, node_feats: torch.Tensor, global_feats: torch.Tensor):
        """Returns (logits[B, num_actions], value[B])."""
        x = self._embed_tokens(node_feats, global_feats)
        for attn, ff in zip(self.attn_blocks, self.ff_blocks):
            x = attn(x)
            x = ff(x)
        x = self.trunk_ln(x)

        board_tokens = x[:, :self.n_positions, :]
        pooled = self._pool(x)

        logits = self._policy_logits(board_tokens, pooled)

        v = self.value_net(pooled)
        value = self.value_head(v).squeeze(-1)
        value = torch.clamp(value, -self.value_clip, self.value_clip)
        return logits, value

    def get_action_and_value(self, node_feats, global_feats, mask,
                             deterministic: bool = False, temperature: float = 1.0):
        logits, value = self.forward(node_feats, global_feats)
        masked = logits.float()
        masked[mask == 0] = -1e9

        if deterministic:
            action = masked.argmax(dim=-1)
            probs = F.softmax(masked, dim=-1)
            dist = torch.distributions.Categorical(probs)
        else:
            scaled = masked / max(temperature, 0.01)
            probs = F.softmax(scaled, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value

    def get_action_for_play(self, node_feats, global_feats, mask, temperature: float = 0.4):
        with torch.no_grad():
            logits, _ = self.forward(node_feats, global_feats)
            masked = logits.float()
            masked[mask == 0] = -1e9
            scaled = masked / max(temperature, 0.01)
            probs = F.softmax(scaled, dim=-1)
            dist = torch.distributions.Categorical(probs)
            return int(dist.sample().item())
