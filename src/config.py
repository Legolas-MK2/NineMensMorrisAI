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
    """Training configuration - curriculum-aware.

    Note: there is deliberately no total-episode cap — training runs until
    the operator stops the (infinite) final curriculum phase.
    """

    # Training scale
    episodes_per_update: int = 8192
    ppo_epochs: int = 3
    mini_batch_size: int = 8192
    
    # Parallelism (optimized for Threadripper 3960X + RTX 3090)
    num_workers: int = 16
    envs_per_worker: int = 48

    # Per-worker async-minimax thread pool.
    # Total concurrent minimax searches = num_workers * minimax_threads_per_worker.
    minimax_threads_per_worker: int = 2

    # Niceness offset applied in each worker process (0 = same priority
    # as parent). The trainer's main process serves batched GPU inference
    # for all workers during collection; the workers' minimax threads
    # oversubscribe the CPU (num_workers * minimax_threads_per_worker),
    # so without this offset they can starve the inference loop and stall
    # the whole pipeline. Keep > 0 unless workers no longer saturate cores.
    worker_nice: int = 10

    # Minimax transposition-table size, per (worker, depth) bot.
    # The natural per-search hit rate in this game is ~16-30% (bounded
    # by transposition density, not table size) and plateaus past
    # ~64 MiB. Measured at depth=5 across 200 mid-game states:
    #   64 MiB: 19.8% hits, 11k collisions / 1.7M nodes (0.7%)
    #   256 MiB - 2 GiB: 19.9% hits, identical wall-clock.
    # We default to 128 MiB (safely past the knee, low waste).
    # Worst-case process budget = num_workers * max_training_depths *
    # tt_bytes => 16 * 5 * 128 MiB = ~10 GiB (D5 is the training cap;
    # bots are created lazily per sampled depth). Bigger sizes give
    # essentially zero training speedup, so don't push higher unless
    # you observe high collision rates.
    minimax_tt_bytes_per_bot: int = 128 * 1024 * 1024

    # Multiplicative jitter applied to the minimax evaluation weights
    # once per log cycle (per worker) so the bot doesn't always pick the
    # same forced line from the same position. 0.0 = deterministic
    # weights (matches the historical hardcoded values). Each weight is
    # resampled as base * Uniform(1 - j, 1 + j) and the transposition
    # table is cleared on each reroll.
    minimax_weight_jitter: float = 0.15

    # Board-symmetry data augmentation. The 24-point board graph has an order-16
    # automorphism group (D4 of the 7x7 grid x inner/outer-ring swap); applying a
    # random sigma to (obs, mask, action) during rollout yields 16x effective
    # position diversity at zero cost. See src/symmetry.py.
    # aug_granularity: 'game' redraws sigma once per game (per env); 'step'
    # redraws per decision -- usually overkill, since one game gives 16 nearly
    # iid positions under the larger-batch shuffle.
    use_symmetry_aug: bool = True
    aug_granularity: str = 'game'

    # Model architecture -- relational token-based ActorCritic.
    # The 24 board points are tokens; a 25th global token carries phase /
    # piece-count features. Attention biases between board tokens add learned
    # per-head scalars on adjacency and mill-cohabitation; both are sigma-
    # invariant, so the bias is correct in every augmented frame.
    d_model: int = 128
    n_layers: int = 5
    n_heads: int = 8
    d_k: int = 32
    ff_mult: int = 4
    node_feat_dim: int = 3
    global_feat_dim: int = 11
    num_positions: int = 24
    # 'pointer' = per-node + from->to inner-product head (final architecture).
    # 'flat'    = pooled -> Linear(-> 600) head (Milestone A; for debugging).
    policy_head: str = 'pointer'
    # 'global' = use the global token only; 'global+mean' = concat global + mean
    # over board tokens before the value MLP.
    value_pool: str = 'global+mean'
    dropout: float = 0.0

    # PPO hyperparameters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.12
    max_grad_norm: float = 0.5
    
    # Entropy - gradual decay with floor to prevent policy collapse
    entropy_coef_start: float = 0.1
    entropy_coef_end: float = 0.02  # Increased floor from 0.01 to prevent overfitting
    entropy_decay_episodes: int = 5_000_000  # was 2M — decayed too fast
    
    # Value function
    value_coef: float = 0.5
    value_loss_clamp: float = 10.0
    
    # Game settings
    max_game_steps: int = 300

    # Note: Using fastnmm's nine_mens_morris C++ engine.
    # Board positions are seeded via the engine's `starting_stones` option;
    # the curriculum controls the per-phase per-player stone distribution.

    # Mixed precision
    use_mixed_precision: bool = True
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Clipping
    value_clip: float = 2.0
    advantage_clip: float = 3.0
    ratio_clip: float = 2.0
    log_prob_clip: float = 10.0

    # Logging cadence. Progressive minimax eval is NOT rerun every log tick —
    # it only runs at training startup and on phase graduation (see
    # PPOTrainer._on_phase_change); each log tick reuses the cached result.
    log_interval: int = 25_000
    save_interval: int = 100_000

    # Progressive minimax eval runs once at training start (seeds the
    # cache for the first log lines) and again on every phase graduation.
    # Log-tick frequency throttling was removed -- phase graduations are
    # the natural milestone for refreshing the depth ladder.

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
    lr_phase_reset_factor: float = 0.8
    # On clone replacement: lr <- min(lr_peak, current_lr * lr_clone_bump_factor),
    # fresh cycle from the bumped lr.
    lr_clone_bump_factor: float = 1.3

    # Phase 2-9 plateau graduation: minimum episodes-in-phase before the
    # per-depth WR plateau check may graduate the phase. Passed to
    # CurriculumManager by the trainer (trend criteria live in
    # curriculum.GRADUATION_CONFIG).
    graduation_min_episodes: int = 2_500_000

    # Directories
    model_dir: str = "models"
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    curriculum_dir: str = "curriculum"

    # Dropping dominated opponents from PPO training is governed by the
    # hysteresis thresholds in curriculum.MIXED_CONFIG
    # ('minimax_depth_dominate_threshold' / '..._recover') — one mechanism
    # for sampling, training AND the log's "(no-train)" notes.

    # Self-play PPO pause: when wr_vs_self exceeds this threshold at a log
    # tick, self-play experiences stop feeding PPO for `selfplay_train_pause_episodes`
    # episodes. Games keep playing (results still hit results_vs_self), so the
    # WR keeps refreshing — if it's still above the threshold at the next log
    # tick, the pause extends. Different from random/minimax cutoffs: those use
    # a hard WR comparison every batch; this uses a time window so we don't
    # bounce in-and-out on noise near the threshold.
    selfplay_train_pause_threshold: float = 0.95
    selfplay_train_pause_episodes: int = 500_000  # 10 log cycles @ 25k each

    # Webserver model file size limit (bytes). Files larger than this in
    # the auto-discovered model directories are ignored as likely
    # incomplete or corrupted.
    max_model_file_bytes: int = 200 * 1024 * 1024

