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
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        
        # Simplified attention using sigmoid
        attn = torch.sigmoid((q * k).sum(dim=-1, keepdim=True) * self.scale)
        out = (attn * v).reshape(B, -1)
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
        
        # Input embedding
        self.input_embed = nn.Sequential(
            nn.Linear(obs_size, dim),
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
        
    def forward(self, obs):
        x = self.input_embed(obs)
        
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
    
    def get_action_and_value(self, obs, mask, deterministic=False):
        """Get action, log probability, entropy, and value."""
        logits, value = self.forward(obs)
        
        # Mask illegal actions
        masked_logits = logits.float()
        masked_logits[mask == 0] = -1e9
        
        probs = F.softmax(masked_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        if deterministic:
            action = masked_logits.argmax(dim=-1)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value
