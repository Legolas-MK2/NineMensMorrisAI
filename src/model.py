"""
Nine Men's Morris - Neural Network Model
Actor-Critic with residual blocks and attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


class ResidualBlock(nn.Module):
    """Residual block with layer norm and GELU activation."""
    
    def __init__(self, dim: int, dropout: float = 0.05):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize with small weights for stable training
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.1)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        
    def forward(self, x):
        h = self.ln1(x)
        h = F.gelu(self.fc1(h))
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.dropout(h)
        return x + h * 0.1  # Scale residual


class MultiHeadAttention(nn.Module):
    """Simplified multi-head attention for board state."""
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.05):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.ln = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        nn.init.xavier_uniform_(self.qkv.weight, gain=0.1)
        nn.init.xavier_uniform_(self.proj.weight, gain=0.1)
        nn.init.zeros_(self.proj.bias)
        
    def forward(self, x):
        B = x.shape[0]
        h = self.ln(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # each (B, num_heads, head_dim)

        # Standard scaled dot-product attention
        # Each "head" attends to all other heads: (B, num_heads, num_heads)
        attn = torch.bmm(q, k.transpose(1, 2)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, v)   # (B, num_heads, head_dim)
        out = out.reshape(B, -1)   # (B, dim)

        out = self.dropout(self.proj(out))
        return x + out * 0.1


class ActorCritic(nn.Module):
    """
    Actor-Critic network for Nine Men's Morris.
    Separate value and policy heads with shared backbone.
    """
    
    def __init__(self, obs_size: int, num_actions: int, config: Config):
        super().__init__()
        dim = config.hidden_dim
        self.value_clip = config.value_clip
        self.raw_obs_size = obs_size

        # Determine encoding from obs_shape set in config at runtime.
        # pyspiel nine_mens_morris: shape = [5, 7, 7] = 245 dims
        #   Channel 0 (cells 0..n_cells-1)    : player 0 piece presence on the grid
        #   Channel 1 (cells n_cells..2*n_cells): player 1 piece presence on the grid
        #   Channels 2+ (rest)                 : game state (phase, remaining pieces, etc.)
        # We one-hot encode channels 0+1 as [empty, p0, p1] per cell,
        # and pass channels 2+ through unchanged.
        obs_shape = getattr(config, 'obs_shape', None)
        if obs_shape and len(obs_shape) >= 3:
            self._n_total_channels = obs_shape[0]          # e.g. 5
            self._n_cells = obs_size // obs_shape[0]       # e.g. 49 (7×7)
            # 3 one-hot dims for the 2 piece channels + remaining channels unchanged
            encoded_obs_size = self._n_cells * 3 + (self._n_total_channels - 2) * self._n_cells
        else:
            self._n_total_channels = None
            self._n_cells = None
            encoded_obs_size = obs_size  # fallback: raw obs passthrough

        # Input embedding
        self.input_embed = nn.Sequential(
            nn.Linear(encoded_obs_size, dim),
            nn.LayerNorm(dim),
            nn.GELU()
        )
        
        # Shared backbone: residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(dim, config.dropout)
            for _ in range(config.num_res_blocks)
        ])
        
        # Attention layer
        self.attention = MultiHeadAttention(
            dim, config.num_attention_heads, config.dropout
        )
        
        # Policy head (actor)
        self.policy_net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
        )
        self.policy_head = nn.Linear(dim // 2, num_actions)
        
        # Value head (critic) - separate from policy
        self.value_net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
        )
        self.value_head = nn.Linear(dim // 2, 1)
        
        # Initialize heads carefully
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)
        
    def _encode_obs(self, obs):
        """One-hot encode the two piece-presence channels of the observation.

        pyspiel nine_mens_morris flat obs layout (e.g. 245 dims = 5 × 49 cells):
          Channel 0 [0 : n]     — player 0 piece presence on the board grid
          Channel 1 [n : 2n]    — player 1 piece presence on the board grid
          Channels 2+ [2n : end] — game state info (phase, remaining pieces, etc.)

        The first two channels are encoded as [empty, p0, p1] per cell (one-hot),
        making each of the three board states an explicit independent feature.
        The remaining channels are passed through unchanged.

        If obs_shape was not set in config, returns obs unchanged (safe fallback).
        """
        if self._n_cells is None:
            return obs  # no shape info — raw passthrough

        n = self._n_cells
        p0    = obs[:, :n]        # (B, n_cells) — 1 where player 0 has piece
        p1    = obs[:, n:2*n]     # (B, n_cells) — 1 where player 1 has piece
        empty = 1.0 - p0 - p1    # (B, n_cells) — 1 where cell is empty

        # Stack → (B, n_cells, 3) → flatten → (B, 3*n_cells)
        pos_encoded = torch.stack([empty, p0, p1], dim=2).reshape(obs.shape[0], -1)
        other = obs[:, 2*n:]      # (B, remaining channels flat)
        return torch.cat([pos_encoded, other], dim=1)

    def forward(self, obs):
        x = self.input_embed(self._encode_obs(obs))
        
        for block in self.res_blocks:
            x = block(x)
        
        x = self.attention(x)
        
        # Separate heads
        policy_features = self.policy_net(x)
        value_features = self.value_net(x)
        
        logits = self.policy_head(policy_features)
        value = self.value_head(value_features).squeeze(-1)
        value = torch.clamp(value, -self.value_clip, self.value_clip)
        
        return logits, value
    
    def get_action_and_value(self, obs, mask, deterministic=False, temperature=1.0):
        """Get action, log probability, entropy, and value.

        Args:
            obs: Observation tensor
            mask: Legal action mask
            deterministic: If True, always pick best action (for evaluation vs minimax)
            temperature: Sampling temperature (0.0 = argmax, 1.0 = policy distribution, >1.0 = more random)
                        Use 0.3-0.5 for playing vs humans to be less predictable but still strong
        """
        logits, value = self.forward(obs)

        # Mask illegal actions
        masked_logits = logits.float()
        masked_logits[mask == 0] = -1e9

        if deterministic:
            action = masked_logits.argmax(dim=-1)
            probs = F.softmax(masked_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
        else:
            # Apply temperature scaling for more/less exploration
            # temperature < 1.0 = sharper (more deterministic)
            # temperature > 1.0 = flatter (more random)
            scaled_logits = masked_logits / max(temperature, 0.01)
            probs = F.softmax(scaled_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value

    def get_action_for_play(self, obs, mask, temperature=0.4):
        """Get action for playing against humans.

        Uses temperature sampling to avoid being too predictable.
        temperature=0.4 is a good balance of strength and unpredictability.
        """
        with torch.no_grad():
            logits, _ = self.forward(obs)
            masked_logits = logits.float()
            masked_logits[mask == 0] = -1e9

            # Temperature sampling
            scaled_logits = masked_logits / max(temperature, 0.01)
            probs = F.softmax(scaled_logits, dim=-1)

            # Sample from distribution instead of argmax
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            return action.item()
