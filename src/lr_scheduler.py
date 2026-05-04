"""
Nine Men's Morris - RL Plateau Learning Rate Scheduler

This module provides a custom learning rate scheduler designed specifically for
reinforcement learning training in Nine Men's Morris.

Key Features:
- Tracks win rate against the HARDEST opponent (Minimax-2 or older self-play)
- Reduces LR only when performance plateaus against hard opponents
- Ignores easy opponents (Random, Minimax-1) for LR decisions
- Supports curriculum-based opponent sampling linked to LR
- Automatic training termination when target win rate is reached
"""

import torch
from typing import Dict, Callable, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import IntEnum


class OpponentType(IntEnum):
    """Opponent difficulty levels."""
    RANDOM = 0
    MINIMAX_1 = 1
    MINIMAX_2 = 2
    MINIMAX_3 = 3
    MINIMAX_4 = 4
    SELF_PLAY = 5
    OLD_SELF_PLAY = 6


@dataclass
class OpponentMix:
    """Opponent sampling distribution."""
    random: float = 0.0
    minimax_1: float = 0.0
    minimax_2: float = 0.0
    minimax_3: float = 0.0
    minimax_4: float = 0.0
    self_play: float = 0.0
    
    def normalize(self):
        """Normalize probabilities to sum to 1.0."""
        total = sum([
            self.random, self.minimax_1, self.minimax_2,
            self.minimax_3, self.minimax_4, self.self_play
        ])
        if total > 0:
            self.random /= total
            self.minimax_1 /= total
            self.minimax_2 /= total
            self.minimax_3 /= total
            self.minimax_4 /= total
            self.self_play /= total
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            'random': self.random,
            'minimax_1': self.minimax_1,
            'minimax_2': self.minimax_2,
            'minimax_3': self.minimax_3,
            'minimax_4': self.minimax_4,
            'self_play': self.self_play,
        }


class RLPlateauScheduler:
    """
    Custom learning rate scheduler for RL training.
    
    This scheduler reduces the learning rate when the agent's performance
    against the HARDEST opponent stops improving. This prevents premature
    LR reduction when the agent achieves 100% win rate against easy opponents
    but hasn't yet learned to beat strong opponents.
    
    Key differences from standard PyTorch schedulers:
    1. Uses "Hard Opponent Win Rate" instead of loss or average win rate
    2. Only reduces LR when improvement falls below threshold
    3. Supports automatic training termination
    4. Integrates with curriculum-based opponent sampling
    
    Usage:
        scheduler = RLPlateauScheduler(
            optimizer,
            factor=0.5,           # LR multiplier when reducing
            patience=10,          # Steps before reducing (1000 steps each = 10k)
            min_lr=1e-6,          # Minimum LR
            threshold=0.02,       # Minimum improvement to reset patience
            target_win_rate=0.95  # Stop training when reached
        )
        
        # In training loop (every evaluation step, e.g., every 1000 steps)
        hard_opponent_wr = get_win_rate_vs_minimax_2()
        scheduler.step(hard_opponent_wr)
        
        if scheduler.should_stop():
            break
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        factor: float = 0.5,
        patience: int = 10,
        min_lr: float = 1e-6,
        threshold: float = 0.02,
        target_win_rate: float = 0.95,
        metric_name: str = "wr_vs_mm_d2",
        allow_early_stop: bool = False,
    ):
        """
        Initialize the RL Plateau scheduler.
        
        Args:
            optimizer: PyTorch optimizer (AdamW recommended)
            factor: Multiplicative factor for LR reduction (default 0.5)
            patience: Number of evaluation steps without improvement before reducing LR
            min_lr: Minimum learning rate
            threshold: Minimum improvement to count as progress (default 2%)
            target_win_rate: Win rate threshold to stop training (default 95%)
            metric_name: Name of metric being tracked (for logging)
            allow_early_stop: If True, scheduler can stop training on target WR or min LR.
        """
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.threshold = threshold
        self.target_win_rate = target_win_rate
        self.metric_name = metric_name
        self.allow_early_stop = allow_early_stop
        
        # State tracking
        self.wait = 0
        self.best_score = -float('inf')
        self.best_episode = 0
        self.reductions = 0
        
        # History for analysis
        self.history: List[Dict] = []
        
        # Get initial LR
        self.initial_lrs = [
            group['lr'] for group in optimizer.param_groups
        ]
        self.current_lrs = list(self.initial_lrs)
        
        # Stop flag
        self._should_stop = False
        self.stop_reason = ""
    
    def step(self, metric: float, episode: int = 0) -> bool:
        """
        Step the scheduler with a new metric value.

        The scheduler does NOT write to the optimizer directly: the trainer
        owns LR (cosine base * plateau multiplier). A reduction here simply
        increments `reductions`, which changes what `get_plateau_multiplier`
        returns. The trainer's `_update_learning_rate` picks this up on its
        next call and applies it on top of the cosine base.

        Args:
            metric: Win rate against the hardest opponent (e.g., Minimax-2)
            episode: Current episode count (for logging)

        Returns:
            True if a plateau reduction was registered this step.
        """
        lr_reduced = False

        # Record history
        self.history.append({
            'episode': episode,
            'metric': metric,
            'best_score': self.best_score,
            'wait': self.wait,
            'current_lr': self.current_lrs[0],
            'reductions': self.reductions,
        })

        # Ignore persisted stop flags when scheduler auto-stop is disabled.
        if not self.allow_early_stop:
            self._should_stop = False
            self.stop_reason = ""

        # Optional early-stop behavior. Disabled by default so curriculum phases
        # drive termination, not LR scheduler state.
        if self.allow_early_stop:
            if metric >= self.target_win_rate:
                self._should_stop = True
                self.stop_reason = f"Target win rate reached: {metric:.1%} >= {self.target_win_rate:.1%}"
                print(f"  [Scheduler] {self.stop_reason}")
                return False

            if self.current_lrs[0] <= self.min_lr:
                self._should_stop = True
                self.stop_reason = f"LR reached minimum: {self.current_lrs[0]:.1e} <= {self.min_lr:.1e}"
                print(f"  [Scheduler] {self.stop_reason}")
                return False

        # Check for improvement
        if metric > self.best_score + self.threshold:
            self.best_score = metric
            self.best_episode = episode
            self.wait = 0
        else:
            self.wait += 1

            if self.wait >= self.patience:
                # Register a plateau reduction. Actual LR is applied by the
                # trainer via get_plateau_multiplier().
                self.reductions += 1
                lr_reduced = True
                mult = self.get_plateau_multiplier()
                print(
                    f"  [Scheduler] Plateau reduction #{self.reductions} "
                    f"(multiplier now {mult:.3f}x, best {self.metric_name}={self.best_score:.1%} "
                    f"at episode {self.best_episode:,})"
                )
                self.wait = 0

        return lr_reduced

    def get_plateau_multiplier(self) -> float:
        """Multiplicative factor applied by plateau reductions.

        The trainer composes this with the curriculum's cosine base LR:
            final_lr = max(cosine_lr * plateau_multiplier, min_lr)
        """
        return self.factor ** self.reductions

    def notify_current_lr(self, lr: float):
        """Let the trainer report the effective LR back for logging parity."""
        self.current_lrs = [lr] * max(1, len(self.current_lrs))

    def reset(self):
        """Reset scheduler state for a new phase.

        Called on every phase transition so accumulated reductions from
        previous phases (where WR ceilings were different) don't carry
        forward and pin the new-phase LR at min_lr from the start.
        """
        self.best_score = -float('inf')
        self.best_episode = 0
        self.wait = 0
        self.reductions = 0
        self._should_stop = False
        self.stop_reason = ""
        print(f"  [Scheduler] Reset for new phase (reductions → 0, best_score → -∞)")

    def _update_learning_rate(self, new_lr: float):
        """Deprecated: kept for backward-compatible state loading paths."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
    
    def should_stop(self) -> bool:
        """Check if training should stop."""
        return self._should_stop
    
    def get_stop_reason(self) -> str:
        """Get the reason for stopping."""
        return self.stop_reason
    
    def get_best_score(self) -> float:
        """Get the best metric score achieved."""
        return self.best_score
    
    def get_current_lr(self) -> float:
        """Get the current learning rate."""
        return self.current_lrs[0]
    
    def get_state_dict(self) -> Dict:
        """Get scheduler state for checkpointing."""
        return {
            'best_score': self.best_score,
            'best_episode': self.best_episode,
            'wait': self.wait,
            'current_lrs': self.current_lrs,
            'reductions': self.reductions,
            'should_stop': self._should_stop,
            'stop_reason': self.stop_reason,
            'history': self.history[-100:],  # Keep last 100 entries
        }
    
    def load_state_dict(self, state: Dict):
        """Load scheduler state from checkpoint."""
        self.best_score = state['best_score']
        self.best_episode = state['best_episode']
        self.wait = state['wait']
        self.current_lrs = state['current_lrs']
        self.reductions = state['reductions']
        if self.allow_early_stop:
            self._should_stop = state['should_stop']
            self.stop_reason = state['stop_reason']
        else:
            self._should_stop = False
            self.stop_reason = ""
        self.history = state.get('history', [])


class CurriculumLRScheduler:
    """
    Learning rate scheduler linked with curriculum-based opponent sampling.
    
    This scheduler automatically adjusts both the learning rate and opponent
    mixture based on training progress. It implements the following curriculum:
    
    - Early Training: High LR, easy opponents (learn basic rules)
    - Mid Training: Medium LR, mixed opponents (learn strategy)
    - Late Training: Low LR, hard opponents (refine perfection)
    
    Usage:
        scheduler = CurriculumLRScheduler(
            optimizer,
            curriculum_stages={
                'early': {'end_step': 10000, 'lr': 3e-4, 'mix': OpponentMix(random=0.5, minimax_1=0.5)},
                'mid': {'end_step': 50000, 'lr': 1e-4, 'mix': OpponentMix(minimax_2=0.5, self_play=0.5)},
                'late': {'end_step': float('inf'), 'lr': 3e-5, 'mix': OpponentMix(self_play=1.0)},
            }
        )
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        curriculum_stages: Optional[Dict] = None,
        base_scheduler: Optional[RLPlateauScheduler] = None
    ):
        """
        Initialize curriculum LR scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            curriculum_stages: Dict defining curriculum stages
            base_scheduler: Optional RLPlateauScheduler for fine-tuning within stages
        """
        self.optimizer = optimizer
        self.base_scheduler = base_scheduler
        
        # Default curriculum stages
        self.curriculum_stages = curriculum_stages or self._default_curriculum()
        self.current_stage = 0
        self.current_mix = OpponentMix()
        
        # Initialize with first stage
        self._apply_stage(0)
    
    def _default_curriculum(self) -> Dict:
        """Default curriculum configuration."""
        return {
            0: {
                'name': 'early',
                'end_step': 10000,
                'lr': 3e-4,
                'mix': OpponentMix(random=0.5, minimax_1=0.5),
            },
            1: {
                'name': 'mid',
                'end_step': 50000,
                'lr': 1e-4,
                'mix': OpponentMix(minimax_2=0.5, self_play=0.5),
            },
            2: {
                'name': 'late',
                'end_step': float('inf'),
                'lr': 3e-5,
                'mix': OpponentMix(self_play=1.0),
            },
        }
    
    def _apply_stage(self, stage: int):
        """Apply a curriculum stage."""
        if stage not in self.curriculum_stages:
            return
        
        stage_config = self.curriculum_stages[stage]
        new_lr = stage_config['lr']
        new_mix = stage_config['mix']
        
        # Update learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        
        # Update opponent mix
        self.current_mix = new_mix
        self.current_mix.normalize()
        
        self.current_stage = stage
        
        print(
            f"  [CurriculumLR] Stage: {stage_config['name']}, "
            f"LR: {new_lr:.1e}, Mix: {self.current_mix.to_dict()}"
        )
    
    def step(self, step: int, metric: Optional[float] = None) -> OpponentMix:
        """
        Step the scheduler.
        
        Args:
            step: Current training step
            metric: Optional metric for base scheduler
        
        Returns:
            Current opponent mix
        """
        # Check if we should advance to next stage
        if self.current_stage + 1 in self.curriculum_stages:
            next_stage = self.curriculum_stages[self.current_stage + 1]
            if step >= next_stage['end_step']:
                self._apply_stage(self.current_stage + 1)
        
        # Use base scheduler if provided
        if self.base_scheduler and metric is not None:
            self.base_scheduler.step(metric)
        
        return self.current_mix
    
    def get_opponent_mix(self) -> OpponentMix:
        """Get current opponent mix."""
        return self.current_mix
    
    def get_current_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']
    
    def should_stop(self) -> bool:
        """Check if training should stop."""
        if self.base_scheduler:
            return self.base_scheduler.should_stop()
        return False


def get_curriculum_mix_and_lr(step: int) -> Tuple[Dict[str, float], float]:
    """
    Get opponent mix and learning rate based on training step.
    
    This is a convenience function for simple curriculum-based training.
    
    Args:
        step: Current training step
    
    Returns:
        Tuple of (opponent_mix_dict, learning_rate)
    """
    if step < 10000:
        # Early: Learn basic rules against easy opponents
        return {'random': 0.5, 'minimax1': 0.5}, 3e-4
    elif step < 50000:
        # Mid: Learn strategy against mixed opponents
        return {'minimax2': 0.5, 'self': 0.5}, 1e-4
    else:
        # Late: Refine against self-play
        return {'self': 1.0}, 3e-5