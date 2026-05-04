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

from dataclasses import dataclass, field
from typing import Optional, List
import torch


@dataclass
class Config:
    """Training configuration - curriculum-aware."""
    
    # Training scale
    total_episodes: int = 500_000_000  # Max episodes (curriculum may finish earlier)
    episodes_per_update: int = 8192
    ppo_epochs: int = 3
    mini_batch_size: int = 8192
    
    # Parallelism (optimized for Threadripper 3960X + RTX 3090)
    num_workers: int = 22
    envs_per_worker: int = 48
    
    # Observation shape from pyspiel (set at runtime, e.g. [5, 7, 7] for nine_mens_morris)
    # Channel 0 = player 0 pieces, Channel 1 = player 1 pieces, rest = game state
    obs_shape: Optional[List[int]] = None

    # Model architecture
    hidden_dim: int = 128
    num_res_blocks: int = 8
    num_attention_heads: int = 16
    dropout: float = 0.05
    
    # PPO hyperparameters
    gamma: float = 0.9
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.12
    max_grad_norm: float = 0.5
    
    # Entropy - gradual decay with floor to prevent policy collapse
    entropy_coef_start: float = 0.1
    entropy_coef_end: float = 0.02  # Increased floor from 0.01 to prevent overfitting
    entropy_decay_episodes: int = 5_000_000  # Match random_train (was 2M, too fast)
    
    # Value function
    value_coef: float = 0.5
    value_loss_clamp: float = 10.0
    
    # Game settings
    max_game_steps: int = 300

    # Note: Using pyspiel's nine_mens_morris game
    # Random moves are used to prepare board positions for training
    # The number of random moves is managed by curriculum per phase

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
    
    # RL Plateau Scheduler - Automatic LR management based on hard opponent win rate
    # This scheduler prevents premature LR drop when agent beats easy opponents
    # but hasn't yet learned to beat strong opponents (Minimax-2)
    scheduler_factor: float = 0.5        # LR multiplier when reducing (0.5 = halve)
    scheduler_patience: int = 10         # Evaluation steps before reducing (10 * log_interval)
    scheduler_min_lr: float = 1e-6       # Minimum learning rate
    scheduler_threshold: float = 0.02    # Minimum improvement to reset patience (2%)
    scheduler_target_wr: float = 0.95    # Stop training at this win rate vs Minimax-2
    
    # Directories
    model_dir: str = "models"
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    curriculum_dir: str = "curriculum"

