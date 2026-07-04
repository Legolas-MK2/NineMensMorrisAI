"""
Nine Men's Morris - Warmup + Warm-Restart Cosine LR Scheduler

Driven by PPO update steps (not episodes). Behavior:

- Warmup: linear 0 -> lr_peak over `warmup_updates`.
- Cosine: each cycle anneals from `cycle_starting_lr` down to `lr_min` over
  `cycle_t_max` updates. Once it hits the floor it stays there until a reset.
- Phase graduation event: phase_peak *= phase_reset_factor; new cycle starts at
  phase_peak. Successive graduations compound (factor^n, e.g. 0.8 * 0.8 * ...).
- Clone replacement event: raise cycle_starting_lr toward phase_peak (capped),
  without resetting cycle progress. Cap is phase_peak — not lr_peak — so clones
  cannot override phase decay. Cycle_step is preserved so frequent clone churn
  cannot pin LR at the ceiling.

Episode-based config knobs are converted to updates via episodes_per_update.
"""

import math
from typing import Dict

import torch


class WarmRestartLRScheduler:
    """Warmup + warm-restart cosine, with phase- and clone-driven resets."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        lr_peak: float,
        lr_min: float,
        warmup_updates: int,
        cycle_t_max_updates: int,
        phase_reset_factor: float,
        clone_bump_factor: float,
    ):
        self.optimizer = optimizer
        self.lr_peak = float(lr_peak)              # original config peak; cap for clone bumps
        self.lr_min = float(lr_min)
        self.warmup_updates = max(1, int(warmup_updates))
        self.cycle_t_max = max(1, int(cycle_t_max_updates))
        self.phase_reset_factor = float(phase_reset_factor)
        self.clone_bump_factor = float(clone_bump_factor)

        # Dynamic state
        self.phase_peak = self.lr_peak             # shrinks at each phase graduation
        self.cycle_starting_lr = self.lr_peak      # peak of current cosine cycle
        self.global_update = 0                     # total step() calls
        self.cycle_step = 0                        # updates within current cycle (post-warmup)
        self.in_warmup = True
        self.last_reset_event = "none"             # "none" | "phase" | "clone"

        self._apply(self._compute_lr())

    # ----- LR computation -----

    def _compute_lr(self) -> float:
        if self.in_warmup:
            progress = self.global_update / self.warmup_updates
            progress = min(1.0, max(0.0, progress))
            return self.lr_peak * progress

        progress = self.cycle_step / self.cycle_t_max
        progress = min(1.0, max(0.0, progress))
        return self.lr_min + 0.5 * (self.cycle_starting_lr - self.lr_min) * (
            1.0 + math.cos(math.pi * progress)
        )

    def _apply(self, lr: float):
        for group in self.optimizer.param_groups:
            group['lr'] = lr

    # ----- Public API -----

    def step(self) -> float:
        """Advance one PPO update. Returns the new LR."""
        self.global_update += 1
        if self.in_warmup:
            if self.global_update >= self.warmup_updates:
                self.in_warmup = False
                self.cycle_step = 0
                self.cycle_starting_lr = self.phase_peak
        else:
            self.cycle_step += 1
        lr = self._compute_lr()
        self._apply(lr)
        return lr

    def notify_phase_graduated(self):
        """Shrink peak by phase_reset_factor and start a fresh cosine cycle."""
        self.phase_peak = max(self.lr_min, self.phase_peak * self.phase_reset_factor)
        self.cycle_starting_lr = self.phase_peak
        self.cycle_step = 0
        self.in_warmup = False
        self.last_reset_event = "phase"
        self._apply(self._compute_lr())

    def notify_clone_replaced(self):
        """Raise cycle ceiling toward phase_peak; cycle progress is preserved.

        Clone events fire far more often than the cosine cycle can anneal
        (every few updates in some phases vs cycle_t_max in the thousands), so
        resetting cycle_step here would pin LR at the ceiling forever. Instead
        we only raise cycle_starting_lr when the bump target exceeds it, and
        cap at phase_peak so clones cannot override phase decay.
        """
        if self.in_warmup:
            return
        current_lr = self.get_lr()
        target = min(self.phase_peak, current_lr * self.clone_bump_factor)
        if target > self.cycle_starting_lr:
            self.cycle_starting_lr = target
        self.last_reset_event = "clone"
        self._apply(self._compute_lr())

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']

    # Aliases preserved for callers expecting the old CosineLRScheduler API.
    def get_current_lr(self) -> float:
        return self.get_lr()

    # ----- Checkpointing -----

    def state_dict(self) -> Dict:
        return {
            'lr_peak': self.lr_peak,
            'lr_min': self.lr_min,
            'warmup_updates': self.warmup_updates,
            'cycle_t_max': self.cycle_t_max,
            'phase_reset_factor': self.phase_reset_factor,
            'clone_bump_factor': self.clone_bump_factor,
            'phase_peak': self.phase_peak,
            'cycle_starting_lr': self.cycle_starting_lr,
            'global_update': self.global_update,
            'cycle_step': self.cycle_step,
            'in_warmup': self.in_warmup,
            'last_reset_event': self.last_reset_event,
        }

    def get_state_dict(self) -> Dict:
        return self.state_dict()

    def load_state_dict(self, state: Dict):
        self.lr_peak = float(state.get('lr_peak', self.lr_peak))
        self.lr_min = float(state.get('lr_min', self.lr_min))
        self.warmup_updates = int(state.get('warmup_updates', self.warmup_updates))
        self.cycle_t_max = int(state.get('cycle_t_max', self.cycle_t_max))
        self.phase_reset_factor = float(state.get('phase_reset_factor', self.phase_reset_factor))
        self.clone_bump_factor = float(state.get('clone_bump_factor', self.clone_bump_factor))
        self.phase_peak = float(state.get('phase_peak', self.lr_peak))
        self.cycle_starting_lr = float(state.get('cycle_starting_lr', self.phase_peak))
        self.global_update = int(state.get('global_update', 0))
        self.cycle_step = int(state.get('cycle_step', 0))
        self.in_warmup = bool(state.get('in_warmup', True))
        self.last_reset_event = str(state.get('last_reset_event', 'none'))
        self._apply(self._compute_lr())
