"""
Nine Men's Morris - Random Training Configuration
Simplified config for 10M epochs of random opponent training
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

from dataclasses import dataclass
import torch


@dataclass
class Config:
    """Training configuration for random opponent training."""

    # Training scale - 10 million epochs
    total_episodes: int = 10_000_000
    episodes_per_update: int = 2048
    ppo_epochs: int = 3
    mini_batch_size: int = 512

    # Parallelism - reduced to prevent memory issues
    num_workers: int = 8
    envs_per_worker: int = 16

    # Model architecture
    hidden_dim: int = 128
    num_res_blocks: int = 8
    num_attention_heads: int = 8
    dropout: float = 0.05

    # Learning rate
    lr_policy: float = 3e-4
    lr_value: float = 3e-4

    # PPO hyperparameters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    max_grad_norm: float = 0.5

    # Entropy coefficient (decays over training)
    entropy_coef_start: float = 0.1
    entropy_coef_end: float = 0.01
    entropy_decay_episodes: int = 5_000_000

    # Value function
    value_coef: float = 0.5
    value_loss_clamp: float = 10.0

    # Game settings
    max_game_steps: int = 300

    # Mixed precision
    use_mixed_precision: bool = True
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Clipping
    value_clip: float = 2.0
    advantage_clip: float = 3.0
    ratio_clip: float = 2.0
    log_prob_clip: float = 10.0

    # Return normalization
    normalize_returns: bool = False
    return_norm_clip: float = 5.0

    # Logging intervals
    log_interval: int = 50_000
    save_interval: int = 500_000
    eval_interval: int = 100_000
    eval_games: int = 200

    # Directories
    model_dir: str = "models"
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"

    # Reward configuration for random opponent training
    win_reward_base: float = 1.0
    win_reward_speed_bonus: float = 0.5
    loss_reward: float = -1.0
    draw_penalty: float = -0.8
    mill_reward: float = 0.2
    enemy_mill_penalty: float = -0.2
    block_mill_reward: float = 0.2
    double_mill_reward: float = 0.3
    double_mill_extra_reward: float = 0.3
    setup_capture_reward: float = 0.1
