"""
Nine Men's Morris - Configuration
Simplified config that works with CurriculumManager
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
    """Training configuration - curriculum-aware."""
    
    # Training scale
    total_episodes: int = 10_000_000  # Max episodes (curriculum may finish earlier)
    episodes_per_update: int = 2048
    ppo_epochs: int = 3
    mini_batch_size: int = 512
    
    # Parallelism
    num_workers: int = 16
    envs_per_worker: int = 32
    
    # Model architecture
    hidden_dim: int = 128
    num_res_blocks: int = 8
    num_attention_heads: int = 8
    dropout: float = 0.05
    
    # Base learning rate (curriculum manager will override)
    lr_policy: float = 3e-4
    lr_value: float = 3e-4
    
    # PPO hyperparameters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    max_grad_norm: float = 0.5
    
    # Entropy - gradual decay like random_train for consistent exploration
    entropy_coef_start: float = 0.1
    entropy_coef_end: float = 0.01
    entropy_decay_episodes: int = 5_000_000  # Match random_train (was 2M, too fast)
    
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
    
    # Logging - less frequent, always with minimax eval
    log_interval: int = 25_000
    save_interval: int = 100_000
    eval_interval: int = 50_000
    eval_games: int = 200
    graduation_check_interval: int = 5_000  # Check graduation/promotion every N episodes
    
    # Directories
    model_dir: str = "models"
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    curriculum_dir: str = "curriculum"
