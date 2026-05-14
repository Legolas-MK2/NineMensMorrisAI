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
    num_workers: int = 16
    envs_per_worker: int = 48

    # Per-worker async-minimax thread pool.
    # Total concurrent minimax searches = num_workers * minimax_threads_per_worker.
    # Keep this product <= (physical_cores - reserved_display_cores) or the
    # X/VNC server starves and the desktop freezes during training.
    # Default 2 -> 16*2 = 32 threads, fits a 48-core box leaving 4 for display.
    minimax_threads_per_worker: int = 2

    # CPU isolation for desktop responsiveness when running in a VNC
    # or VSCodium container. Workers pin to cores [0, ncpu - reserved)
    # and run with `worker_nice` niceness. The display server keeps
    # cores [ncpu - reserved, ncpu) for itself.
    # Set reserved_display_cores=0 to disable on bare-metal setups.
    reserved_display_cores: int = 4
    worker_nice: int = 10  # niceness offset; 0 = same priority as parent

    # Minimax transposition-table size, per (worker, depth) bot.
    # The natural per-search hit rate in this game is ~16-30% (bounded
    # by transposition density, not table size) and plateaus past
    # ~64 MiB. Measured at depth=5 across 200 mid-game states:
    #   64 MiB: 19.8% hits, 11k collisions / 1.7M nodes (0.7%)
    #   256 MiB - 2 GiB: 19.9% hits, identical wall-clock.
    # We default to 128 MiB (safely past the knee, low waste).
    # Worst-case process budget = num_workers * max_active_depths *
    # tt_bytes => 22 * 7 * 128 MiB = ~19.7 GiB. Bigger sizes give
    # essentially zero training speedup, so don't push higher unless
    # you observe high collision rates.
    minimax_tt_bytes_per_bot: int = 128 * 1024 * 1024

    # Cross-worker SharedMoveCache (POSIX shm). One segment, attached by
    # all 22 workers + the trainer. Stores Zobrist key -> last best
    # action chosen at that position. Used for move-ordering hints at
    # the search root: if any worker has searched this position before,
    # the cached move is tried first (alpha-beta benefits when the
    # first move is the best one).
    #
    # 16 bytes/entry. Tradeoff: bigger = more positions remembered, but
    # /dev/shm must accommodate it (typically 50% of physical RAM by
    # default; raise via `mount -o remount,size=128G /dev/shm`).
    #
    # Set move_cache_bytes <= 0 to disable.
    move_cache_name: str = "/nmm_shared_move_cache"
    move_cache_bytes: int = 4 * 1024 * 1024 * 1024  # 4 GiB -> ~256 M entries
    
    # Observation shape from pyspiel (set at runtime, e.g. [5, 7, 7] for nine_mens_morris)
    # Channel 0 = player 0 pieces, Channel 1 = player 1 pieces, rest = game state
    obs_shape: Optional[List[int]] = None

    # Model architecture
    hidden_dim: int = 128
    num_res_blocks: int = 8
    num_attention_heads: int = 16
    dropout: float = 0.05
    
    # PPO hyperparameters
    gamma: float = 0.96
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

    # LR scheduler — warmup + warm-restart cosine, driven by PPO updates.
    # Peak LR after warmup. Cosine each cycle anneals from this down to lr_min.
    lr_peak: float = 3e-4
    lr_min: float = 1e-6
    # Linear warmup from 0 -> lr_peak over this many episodes from training start.
    lr_warmup_episodes: int = 5_000_000
    # Cosine cycle length (in episodes). Once cycle hits lr_min it stays there
    # until a phase-graduation or clone-replacement reset event.
    lr_cycle_t_max_episodes: int = 40_000_000
    # On phase graduation: peak <- peak * lr_phase_reset_factor, fresh cycle.
    lr_phase_reset_factor: float = 0.7
    # On clone replacement: lr <- min(lr_peak, current_lr * lr_clone_bump_factor),
    # fresh cycle from the bumped lr.
    lr_clone_bump_factor: float = 1.3

    # Phase graduation thresholds (Phase 2-10, mixed opponents).
    # All conditions must hold simultaneously to graduate.
    graduation_min_episodes: int = 2_500_000
    graduation_min_wr_top_depth: float = 0.70
    graduation_min_samples_per_depth: int = 20
    graduation_min_clone_generations: int = 5

    # Directories
    model_dir: str = "models"
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    curriculum_dir: str = "curriculum"

