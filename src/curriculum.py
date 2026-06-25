"""
Nine Men's Morris - Curriculum Manager
Handles phased training with automatic progression.

Boards are seeded via fastnmm's `starting_stones` engine option — no random
moves are played to prepare positions. The per-player stone count is drawn
from a per-phase distribution at every env reset, so adjacent phases overlap
smoothly.

Phase structure:
- Phase 1: Per-player stones drawn uniformly from {3..9}, opponent mix is
          self-play + random only (75% self / 25% random). Used as warmup;
          graduation is gated on WR vs random.
- Phase 2-8: Per-player stone count concentrated around 3..9 with a small
             {-2, -1, 0, +1, +2} spread, mixed opponents. The base mix is
             equal-share across self + unlocked minimax depths + random, with
             self-play weighted x3 (see `compute_opponent_distribution`).
             Currently-dampened slots are pinned at `dampen_cap` (1%).
- Phase 9: Fixed 9 stones per player (full game), same mixed opponents.
- Phase 10: Per-player stones drawn uniformly from {3..9}, mixed opponents.
            Shaping continues its global linear decay (does NOT force to 0
            on phase entry). Duration is anchored to the shaping schedule:
            phase 10 runs for PHASE_10_POST_SHAPING_EPISODES after shaping
            reaches 0. If shaping ended before phase 10 began, phase 10
            lasts exactly that floor; otherwise it lasts long enough that
            the final PHASE_10_POST_SHAPING_EPISODES are shaping-free.
- Phase 11: Infinite phase combining the structure of Phase 9 and Phase 10.
            Runs forever in alternating sub-phases of
            `PHASE_11_FULL_GAME_EPISODES` full-game episodes (9 stones per
            player, like Phase 9) followed by `PHASE_11_MIX_EPISODES`
            uniform-{3..9} episodes (like Phase 10). The cycle repeats
            until the operator stops training; the phase never graduates.

Training only opposes minimax depths D1..D5 (see `minimax_max_depth`); WR
vs D6/D7 is tracked from the periodic progressive minimax eval, never from
training games.
"""

import os
import json
import math
import time
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Tuple, Any, Set
from collections import deque
from enum import IntEnum
import numpy as np

from logging_setup import get_logger

logger = get_logger(__name__)


class Phase(IntEnum):
    """Training phases."""
    PHASE_1 = 1   # Random 3-9 stones per player, self + random only (warmup)
    PHASE_2 = 2   # ~3 stones per player, mixed opponents
    PHASE_3 = 3   # ~4 stones per player, mixed opponents
    PHASE_4 = 4   # ~5 stones per player, mixed opponents
    PHASE_5 = 5   # ~6 stones per player, mixed opponents
    PHASE_6 = 6   # ~7 stones per player, mixed opponents
    PHASE_7 = 7   # ~8 stones per player, mixed opponents
    PHASE_8 = 8   # ~9 stones per player, mixed opponents
    PHASE_9 = 9   # 9 stones per player (full game), mixed opponents
    PHASE_10 = 10 # Random 3-9 stones per player, shaping keeps decaying
    PHASE_11 = 11 # Infinite: alternates 2.5M full-game (9 stones) + 2.5M uniform {3..9}
    COMPLETED = 12


@dataclass
class PhaseConfig:
    """Configuration for a single training phase."""
    phase: Phase
    description: str

    # Opponent type. All phases currently use 'mixed'; Phase 1 just omits
    # minimax from the unlocked set so the mix degenerates to self + random.
    opponent_type: str = 'mixed'

    # Reward multipliers
    win_reward_base: float = 2.0
    win_reward_speed_bonus: float = 1.0
    loss_reward: float = -2.0
    draw_penalty: float = -1.5

    # Base shaping intensity per phase (further reduced by schedule)
    shaping_multiplier: float = 0.0

    # Base shaping rewards
    mill_reward: float = 0.3
    enemy_mill_penalty: float = -0.3
    block_mill_reward: float = 0.2
    double_mill_reward: float = 0.5
    double_mill_extra_reward: float = 0.8
    setup_capture_reward: float = 0.2

    # Graduation criteria
    win_rate_threshold: float = 0.90
    min_games_for_graduation: int = 1000

    # Duration limits
    min_episodes: int = 0  # Minimum episodes before graduation allowed
    max_episodes: int = 0  # 0 means no limit


# Mixed opponent configuration for Phase 2+
#
# The base opponent distribution is no longer hardcoded. It is computed by
# `compute_opponent_distribution()` from the current set of available enemy
# slots: each unlocked minimax depth, self-play, and random. Self-play gets
# a 3× weight; every other slot gets weight 1. Dampened slots are pinned at
# `dampen_cap` (1%) and the remaining mass redistributes by weight. See
# `compute_opponent_distribution` for the full algorithm.
MIXED_CONFIG = {
    # Self-play: clone update at 80% win rate over the rolling 500-game
    # self-play window (same window that's logged as wr_vs_self). Checked at
    # log tick (every log_interval episodes).
    'selfplay_winrate_threshold': 0.92,

    # Minimax depth range — gradual unlock from D1 up to D5 based on win rate.
    # D6 and D7 are intentionally NOT used as training opponents (too slow /
    # too strong to be useful gradient signal). WR vs D6/D7 is still tracked
    # and logged via the evaluation pass.
    'minimax_min_depth': 1,
    'minimax_max_depth': 5,  # D1-D5 used for training; D6/D7 are eval/log only

    # Minimax depth unlock: unlock next depth when WR vs current >= 50% over 100 games
    'minimax_depth_unlock_threshold': 0.50,
    'minimax_depth_unlock_min_games': 100,

    # Stagnation detection: graduate early if model stops improving
    'stagnation_min_episodes': 1_000_000,    # Don't trigger before 1M episodes in phase
    'stagnation_clone_window': 3_000_000,    # Evaluate only after 3M eps since phase start / last clone
    'stagnation_snapshot_interval': 100_000, # Take minimax WR snapshot every 100k episodes
    'stagnation_snapshot_window': 5,         # Compare last 5 snapshots (= 500k episodes)
    'stagnation_threshold': 0.03,            # Must improve combined d1+d2 WR by 3%

    # Per-opponent sampling dampening with hysteresis. When WR vs an opponent
    # crosses `dominate_threshold`, that opponent's sampling probability
    # collapses to `dampen_cap` (1%) and the freed mass is redistributed
    # across the remaining slots by `compute_opponent_distribution`. Recovery
    # happens only when WR falls below `dominate_recover` — the gap prevents
    # flapping at the boundary. Applies to minimax depths AND random.
    'minimax_depth_dominate_threshold': 0.90,
    'minimax_depth_dominate_recover': 0.85,
    'minimax_depth_dominate_min_games': 100,
    'dampen_cap': 0.01,
    # Self-play uses a TIMED dampening instead of hysteresis: once
    # wr_vs_self exceeds `selfplay_train_pause_threshold` (config.py) at a
    # log tick, sampling drops to `dampen_cap` for
    # `selfplay_train_pause_episodes` episodes, then recovers unconditionally.
    # Self-play weighting in the base formula (slot gets 3× the share of any
    # other non-dampened slot).
    'selfplay_weight': 3.0,
}


def compute_opponent_distribution(
    unlocked_depths: List[int],
    dampened: Set[str],
    *,
    cap_value: float = 0.01,
    selfplay_weight: float = 3.0,
    include_selfplay: bool = True,
    include_random: bool = True,
) -> Dict[str, float]:
    """Distribute sampling probability across enemy slots.

    Each slot starts with weight 1 except self-play which gets `selfplay_weight`
    (default 3). Slot keys are 'self', 'random', and 'minimax_d{d}' for each
    unlocked depth d. Slots whose key is in `dampened` are pinned at
    `cap_value`; the remaining `1 - n_dampened * cap_value` mass distributes
    across non-dampened slots proportionally to their weights.

    Worked examples (cap=0.01, selfplay_weight=3):
      unlocked=[1..7], dampened={}        → self=0.2727, mm_d* each=0.0909, random=0.0909
      unlocked=[1..7], dampened={'random'} → self=0.2970, mm_d* each=0.0990, random=0.01
      unlocked=[],     dampened={}        → self=0.75,  random=0.25
    """
    slots: List[Tuple[str, float]] = []
    if include_selfplay:
        slots.append(('self', selfplay_weight))
    for d in unlocked_depths:
        slots.append((f'minimax_d{int(d)}', 1.0))
    if include_random:
        slots.append(('random', 1.0))

    if not slots:
        return {}

    pinned = [k for k, _ in slots if k in dampened]
    free = [(k, w) for k, w in slots if k not in dampened]

    fixed_total = cap_value * len(pinned)
    remaining = max(0.0, 1.0 - fixed_total)
    weight_sum = sum(w for _, w in free)

    dist: Dict[str, float] = {}
    if weight_sum <= 0.0:
        # Every slot is dampened — degenerate, give them all cap_value.
        for k, _ in slots:
            dist[k] = cap_value
        return dist

    for k, _ in slots:
        if k in dampened:
            dist[k] = cap_value
        else:
            w = next(w for kk, w in free if kk == k)
            dist[k] = remaining * w / weight_sum
    return dist


# Graduation criteria (Phase 2-10, mixed opponents).
#
# Each WR sampling tick fills a per-depth window (20 samples = 500k-episode
# lookback). A phase graduates only when BOTH of the following hold for every
# currently-unlocked minimax depth:
#   1. episodes_in_phase >= min_episodes
#   2. slope angle of WR vs depth d < trend_max_angle_degrees, for every d
#
# Combined-WR is no longer used to drive graduation (it inflated easy depths
# and hid weakness on top depths). It may still be logged for dashboards.
GRADUATION_CONFIG = {
    'trend_window_samples': 20,        # samples per depth window (per-depth + legacy combined)
    'trend_max_angle_degrees': 2.0,    # condition 2: allow small upward drift on saturated depths
    'min_episodes': 1_500_000,         # condition 1
}


# Shaping decay schedule, applied across the whole run (not per-phase). Phase 10
# deliberately keeps consuming this schedule — it does NOT force shaping to 0
# the moment phase 10 begins.
SHAPING_DECAY_EPISODES = 20_000_000

# Phase 10 must end with at least this many shaping-free episodes. If shaping
# already ended before phase 10 started, phase 10 lasts exactly this long;
# otherwise phase 10 length = (episodes_with_shaping_in_phase_10) + this.
PHASE_10_POST_SHAPING_EPISODES = 5_000_000

# Phase 10 draw-penalty decay: starts at the per-phase configured draw_penalty
# (-1.5) at episodes_in_phase=0 and linearly decays to PHASE_10_DRAW_PENALTY_END
# over PHASE_10_DRAW_PENALTY_DECAY_EPISODES, then stays at the end value. Lower
# magnitude makes draws less catastrophic so the model can accept a draw against
# strong minimax instead of forcing a losing aggression.
PHASE_10_DRAW_PENALTY_END = -0.2
PHASE_10_DRAW_PENALTY_DECAY_EPISODES = 4_000_000

# Phase 11 sub-phase cycle. Phase 11 alternates indefinitely between two
# sub-phases:
#   sub-phase 'full' -> fixed 9 stones per player (like Phase 9), for
#     PHASE_11_FULL_GAME_EPISODES episodes.
#   sub-phase 'mix'  -> per-player stones uniform over {3..9} (like
#     Phase 10), for PHASE_11_MIX_EPISODES episodes.
# Cycle length = PHASE_11_FULL_GAME_EPISODES + PHASE_11_MIX_EPISODES. The
# phase never graduates; the operator stops training to end it.
PHASE_11_FULL_GAME_EPISODES = 2_500_000
PHASE_11_MIX_EPISODES = 2_500_000
PHASE_11_CYCLE_EPISODES = PHASE_11_FULL_GAME_EPISODES + PHASE_11_MIX_EPISODES

# Phase 11 inherits Phase 10's end-state draw penalty as a flat value (the
# decaying schedule has already played out by the time training reaches
# Phase 11; a flat low-magnitude penalty keeps draws acceptable against
# strong minimax without re-triggering the decay each cycle).
PHASE_11_DRAW_PENALTY = PHASE_10_DRAW_PENALTY_END


# Define all phases
PHASE_CONFIGS = {
    Phase.PHASE_1: PhaseConfig(
        phase=Phase.PHASE_1,
        description="warmup; per-player stones uniform over {3..9}; 75% self / 25% random",
        opponent_type='mixed',
        win_reward_base=2.0,
        win_reward_speed_bonus=1.0,
        loss_reward=-2.0,
        draw_penalty=-1.5,
        shaping_multiplier=1.0,
        win_rate_threshold=0.95,  # Must dominate random before moving on
        min_games_for_graduation=2000,
        min_episodes=100_000,
    ),

    Phase.PHASE_2: PhaseConfig(
        phase=Phase.PHASE_2,
        description="per-player stones {3: 78%, 4: 18%, 5: 5%}; vs mixed",
        opponent_type='mixed',
        win_reward_base=2.0,
        win_reward_speed_bonus=1.0,
        loss_reward=-2.0,
        draw_penalty=-1.5,
        shaping_multiplier=0.7,
        win_rate_threshold=0.80,
        min_games_for_graduation=1000,
    ),

    Phase.PHASE_3: PhaseConfig(
        phase=Phase.PHASE_3,
        description="per-player stones {3: 22%, 4: 55%, 5: 18%, 6: 5%}; vs mixed",
        opponent_type='mixed',
        win_reward_base=2.0,
        win_reward_speed_bonus=1.0,
        loss_reward=-2.0,
        draw_penalty=-1.5,
        shaping_multiplier=0.5,
        win_rate_threshold=0.75,
        min_games_for_graduation=1000,
    ),

    Phase.PHASE_4: PhaseConfig(
        phase=Phase.PHASE_4,
        description="per-player stones {3: 5%, 4: 18%, 5: 55%, 6: 18%, 7: 5%}; vs mixed",
        opponent_type='mixed',
        win_reward_base=2.0,
        win_reward_speed_bonus=1.0,
        loss_reward=-2.0,
        draw_penalty=-1.5,
        shaping_multiplier=0.3,
        win_rate_threshold=0.70,
        min_games_for_graduation=1000,
    ),

    Phase.PHASE_5: PhaseConfig(
        phase=Phase.PHASE_5,
        description="per-player stones {4: 5%, 5: 18%, 6: 55%, 7: 18%, 8: 5%}; vs mixed",
        opponent_type='mixed',
        win_reward_base=2.0,
        win_reward_speed_bonus=1.0,
        loss_reward=-2.0,
        draw_penalty=-1.5,
        shaping_multiplier=0.2,
        win_rate_threshold=0.65,
        min_games_for_graduation=1000,
    ),

    Phase.PHASE_6: PhaseConfig(
        phase=Phase.PHASE_6,
        description="per-player stones {5: 5%, 6: 18%, 7: 55%, 8: 18%, 9: 5%}; vs mixed",
        opponent_type='mixed',
        win_reward_base=2.0,
        win_reward_speed_bonus=1.0,
        loss_reward=-2.0,
        draw_penalty=-1.5,
        shaping_multiplier=0.1,
        win_rate_threshold=0.60,
        min_games_for_graduation=1000,
    ),

    Phase.PHASE_7: PhaseConfig(
        phase=Phase.PHASE_7,
        description="per-player stones {6: 5%, 7: 18%, 8: 55%, 9: 22%}; vs mixed",
        opponent_type='mixed',
        win_reward_base=2.0,
        win_reward_speed_bonus=1.0,
        loss_reward=-2.0,
        draw_penalty=-1.5,
        shaping_multiplier=0.05,
        win_rate_threshold=0.55,
        min_games_for_graduation=1000,
    ),

    Phase.PHASE_8: PhaseConfig(
        phase=Phase.PHASE_8,
        description="per-player stones {7: 5%, 8: 18%, 9: 78%}; vs mixed",
        opponent_type='mixed',
        win_reward_base=2.0,
        win_reward_speed_bonus=1.0,
        loss_reward=-2.0,
        draw_penalty=-1.5,
        shaping_multiplier=0.0,
        win_rate_threshold=0.50,
        min_games_for_graduation=1000,
    ),

    Phase.PHASE_9: PhaseConfig(
        phase=Phase.PHASE_9,
        description="9 stones/player (full game); vs mixed",
        opponent_type='mixed',
        win_reward_base=2.0,
        win_reward_speed_bonus=1.0,
        loss_reward=-2.0,
        draw_penalty=-1.5,
        shaping_multiplier=0.0,
        win_rate_threshold=0.50,
        min_games_for_graduation=1000,
    ),

    Phase.PHASE_10: PhaseConfig(
        phase=Phase.PHASE_10,
        description="per-player stones uniform over {3..9}; vs mixed (D1-D5)",
        opponent_type='mixed',
        win_reward_base=2.0,
        win_reward_speed_bonus=1.0,
        loss_reward=-2.0,
        draw_penalty=-1.5,
        shaping_multiplier=0.0,  # Per-phase field is unused; live multiplier comes from get_shaping_multiplier
        win_rate_threshold=0.50,
        min_games_for_graduation=1000,
        max_episodes=0,  # Duration enforced by should_graduate using PHASE_10_POST_SHAPING_EPISODES
    ),

    Phase.PHASE_11: PhaseConfig(
        phase=Phase.PHASE_11,
        description=(
            f"infinite: alternates {PHASE_11_FULL_GAME_EPISODES / 1_000_000:g}M full-game "
            f"(9 stones) + {PHASE_11_MIX_EPISODES / 1_000_000:g}M uniform {{3..9}}; vs mixed (D1-D5)"
        ),
        opponent_type='mixed',
        win_reward_base=2.0,
        win_reward_speed_bonus=1.0,
        loss_reward=-2.0,
        # Draw penalty is overridden via get_phase11_draw_penalty() (flat
        # PHASE_11_DRAW_PENALTY); the field value here is only consulted as a
        # safety fallback.
        draw_penalty=PHASE_11_DRAW_PENALTY,
        shaping_multiplier=0.0,  # Live multiplier comes from get_shaping_multiplier (already 0 by phase 11)
        win_rate_threshold=0.50,
        min_games_for_graduation=1000,
        max_episodes=0,  # Phase 11 never graduates; runs until the operator stops training.
    ),
}


@dataclass
class MixedTrainingState:
    """Per-phase mixed-opponent training state.

    Used by every phase. Phase 1 leaves the minimax depth slots untouched (no
    minimax is sampled there); Phase 2+ unlock and populate them as WR climbs.
    """
    total_episodes: int = 0

    clone_generation: int = 0
    last_clone_episode: int = 0  # total_episodes when clone was last updated

    # Self-play PPO pause: while total_episodes < this, self-play batches are
    # dropped from PPO training (games still play, results still tracked).
    # Set at log tick when wr_vs_self exceeds the configured threshold.
    selfplay_train_cooldown_until: int = 0

    # Minimax win rate snapshots for stagnation detection: list of (total_episodes, win_rate)
    minimax_winrate_snapshots: List = field(default_factory=list)

    # Minimax tracking
    minimax_wins_by_depth: Dict[int, int] = field(default_factory=lambda: {d: 0 for d in range(1, 8)})

    # Per-opponent game counts
    games_vs_random: int = 0
    games_vs_minimax: int = 0
    games_vs_self: int = 0

    # Active minimax depth ceiling (starts at D1, unlocks D2-D5 progressively
    # via win rate; D6/D7 are never sampled as training opponents).
    active_minimax_max_depth: int = 1

    # Win tracking for last 500 games per opponent type
    results_vs_random: deque = field(default_factory=lambda: deque(maxlen=500))
    results_vs_minimax_d1: deque = field(default_factory=lambda: deque(maxlen=500))
    results_vs_minimax_d2: deque = field(default_factory=lambda: deque(maxlen=500))
    results_vs_minimax_d3: deque = field(default_factory=lambda: deque(maxlen=500))
    results_vs_minimax_d4: deque = field(default_factory=lambda: deque(maxlen=500))
    results_vs_minimax_d5: deque = field(default_factory=lambda: deque(maxlen=500))
    results_vs_minimax_d6: deque = field(default_factory=lambda: deque(maxlen=500))
    results_vs_minimax_d7: deque = field(default_factory=lambda: deque(maxlen=500))
    results_vs_self: deque = field(default_factory=lambda: deque(maxlen=500))

    # Combined-WR history (legacy — logged only, not used for graduation).
    # Samples combined minimax winrate every 25k episodes, keeps 20 samples (500k episode window)
    minimax_wr_history: deque = field(default_factory=lambda: deque(maxlen=20))

    # Per-depth WR sliding windows for graduation. One deque per minimax depth
    # (D1..D7), each maxlen = trend_window_samples (20) so the lookback is
    # 500k episodes. Locked depths simply never receive samples — graduation
    # logic only inspects unlocked depths.
    minimax_wr_history_by_depth: Dict[int, deque] = field(
        default_factory=lambda: {d: deque(maxlen=20) for d in range(1, 8)}
    )

    # Per-depth "dominated" flags driving the sampling-weight dampener.
    # Sticky: set when WR(d) >= dominate_threshold over min_games games,
    # cleared when WR(d) drops below dominate_recover. Persisted with
    # the rest of mixed_state so resumes pick up where training left off.
    minimax_depth_dominated: Dict[int, bool] = field(
        default_factory=lambda: {d: False for d in range(1, 8)}
    )

    # Random "dominated" flag — same hysteresis rules as minimax depths.
    # When True, random's sampling probability is pinned at `dampen_cap`
    # and games vs random are dropped from PPO training.
    random_dominated: bool = False

    def get_selfplay_win_rate(self) -> float:
        """Get win rate from the rolling self-play window (last 500 games)."""
        return self.get_win_rate_vs_opponent('self')

    def should_update_clone(self) -> bool:
        """Check if clone should be updated.

        Designed to be polled at the log tick (every log_interval episodes).
        Uses the same rolling `results_vs_self` window (last 500 self-play
        games) that gets reported as `wr_vs_self` in the CSV/log — no extra
        bookkeeping. Requires the window to be full so the WR signal is
        stable.
        """
        if len(self.results_vs_self) < self.results_vs_self.maxlen:
            return False
        return self.get_win_rate_vs_opponent('self') >= MIXED_CONFIG['selfplay_winrate_threshold']

    def on_clone_updated(self):
        """Called when clone is updated."""
        # Reset the self-play window so the next 500 games reflect the new
        # (harder) clone, not the previous one.
        self.results_vs_self.clear()
        # Reset stagnation history so the next stagnation window starts fresh.
        self.minimax_winrate_snapshots.clear()
        self.clone_generation += 1
        self.last_clone_episode = self.total_episodes

    def record_minimax_result(self, depth: int, won: bool):
        """Record a minimax game result."""
        if won:
            self.minimax_wins_by_depth[depth] = self.minimax_wins_by_depth.get(depth, 0) + 1

    def get_combined_minimax_win_rate(self) -> float:
        """Get combined win rate vs minimax d1 and d2 (last 500 games each)."""
        combined = list(self.results_vs_minimax_d1) + list(self.results_vs_minimax_d2)
        if len(combined) < 20:
            return 0.0
        wins = sum(1 for r in combined if r == 'win')
        return wins / len(combined)

    def get_combined_minimax_win_rate_up_to(self, max_depth: int) -> float:
        """Get combined win rate vs minimax D1 through max_depth (last 500 games each)."""
        _depth_results = [
            self.results_vs_minimax_d1,
            self.results_vs_minimax_d2,
            self.results_vs_minimax_d3,
            self.results_vs_minimax_d4,
            self.results_vs_minimax_d5,
            self.results_vs_minimax_d6,
            self.results_vs_minimax_d7,
        ]
        combined = []
        for d in range(1, min(max_depth, 7) + 1):
            combined += list(_depth_results[d - 1])
        if len(combined) < 20:
            return 0.0
        wins = sum(1 for r in combined if r == 'win')
        return wins / len(combined)

    def get_combined_minimax_win_rate_phase10(self) -> float:
        """Get combined win rate vs minimax d1-d4 for phase 10 (last 500 games each)."""
        return self.get_combined_minimax_win_rate_up_to(4)

    def _depth_results_deque(self, depth: int):
        """Return the results deque for a given minimax depth (1-7)."""
        return [
            None,
            self.results_vs_minimax_d1,
            self.results_vs_minimax_d2,
            self.results_vs_minimax_d3,
            self.results_vs_minimax_d4,
            self.results_vs_minimax_d5,
            self.results_vs_minimax_d6,
            self.results_vs_minimax_d7,
        ][depth] if 1 <= depth <= 7 else None

    def get_win_rate_vs_opponent(self, opponent_type: str, depth: int = 0) -> float:
        """Get win rate vs specific opponent type from last 500 games."""
        if opponent_type == 'random':
            results = self.results_vs_random
        elif opponent_type == 'minimax':
            results = self._depth_results_deque(depth)
            if results is None:
                return 0.0
        elif opponent_type == 'self':
            results = self.results_vs_self
        else:
            return 0.0

        if len(results) < 10:
            return 0.0
        wins = sum(1 for r in results if r == 'win')
        return wins / len(results)

    def record_game_result(self, opponent_type: str, result_str: str, depth: int = 0):
        """Record game result for win rate tracking."""
        if opponent_type == 'random':
            self.results_vs_random.append(result_str)
        elif opponent_type == 'minimax':
            deq = self._depth_results_deque(depth)
            if deq is not None:
                deq.append(result_str)
        elif opponent_type == 'self':
            self.results_vs_self.append(result_str)

    @staticmethod
    def _slope_per_episode(history) -> float:
        """Least-squares slope of WR-vs-sample-index, converted to WR-per-episode.
        Returns +inf if too few samples.
        """
        if len(history) < 10:
            return float('inf')
        n = len(history)
        x = np.arange(n)
        y = np.array(history)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x ** 2)
        denom = n * sum_x2 - sum_x ** 2
        if abs(denom) < 1e-10:
            return 0.0
        slope_per_sample = (n * sum_xy - sum_x * sum_y) / denom
        episodes_per_sample = 25000
        return slope_per_sample / episodes_per_sample

    def calculate_winrate_trend(self, active_max_depth: int) -> float:
        """Slope (WR per episode) of combined-minimax history (legacy / dashboards)."""
        return self._slope_per_episode(list(self.minimax_wr_history))

    def calculate_winrate_trend_for_depth(self, depth: int) -> float:
        """Slope (WR per episode) of WR-vs-depth-d history. +inf if too few samples."""
        history = self.minimax_wr_history_by_depth.get(depth)
        if history is None:
            return float('inf')
        return self._slope_per_episode(list(history))

    def get_slope_angle_for_depth(self, depth: int) -> float:
        """Slope of WR vs depth d, expressed as degrees over a 1M-episode horizon.
        Returns +inf if too few samples.
        """
        slope_per_episode = self.calculate_winrate_trend_for_depth(depth)
        if slope_per_episode == float('inf'):
            return float('inf')
        slope_per_million = slope_per_episode * 1_000_000
        return math.degrees(math.atan(slope_per_million))

    def get_window_size_for_depth(self, depth: int) -> int:
        """How many WR samples are currently in the depth-d window."""
        history = self.minimax_wr_history_by_depth.get(depth)
        return 0 if history is None else len(history)

    def update_dampened_state(self, opp_wr: Dict[str, float]) -> None:
        """Refresh hysteresis flags for minimax depths AND random.

        Reads WR values directly from `opp_wr` — the same dict the trainer
        logs each tick (`wr_vs_mm_d{d}`, `wr_vs_random`). The dampener
        decision now consumes the value that was just logged, so
        "logs WR >= 90%" and "fires the >= 90% protocol" stay in lockstep.

        Called by the trainer once per log_interval (alongside the other
        slow-signal checks), not on every game. A slot becomes dampened
        once WR >= dominate_threshold over at least dominate_min_games
        games; it stays dampened until WR drops below dominate_recover.
        Slots with too few games to judge keep their previous flag.

        Self-play uses a TIMED scheme (`selfplay_train_cooldown_until`) and
        is updated by the trainer, not here.
        """
        threshold = MIXED_CONFIG['minimax_depth_dominate_threshold']
        recover = MIXED_CONFIG['minimax_depth_dominate_recover']
        min_games = MIXED_CONFIG['minimax_depth_dominate_min_games']

        for d in range(1, 8):
            deq = self._depth_results_deque(d)
            n_games = len(deq) if deq is not None else 0
            if n_games < min_games:
                continue
            wr = opp_wr.get(f'wr_vs_mm_d{d}', 0.0)
            if not self.minimax_depth_dominated[d]:
                if wr >= threshold:
                    self.minimax_depth_dominated[d] = True
                    logger.info(
                        "Depth dominated: D%d sampling collapsed "
                        "(WR %.0f%% >= %.0f%%)",
                        d, wr * 100, threshold * 100,
                    )
            else:
                if wr < recover:
                    self.minimax_depth_dominated[d] = False
                    logger.info(
                        "Depth recovered: D%d sampling restored "
                        "(WR %.0f%% < %.0f%%)",
                        d, wr * 100, recover * 100,
                    )

        if len(self.results_vs_random) >= min_games:
            wr_random = opp_wr.get('wr_vs_random', 0.0)
            if not self.random_dominated:
                if wr_random >= threshold:
                    self.random_dominated = True
                    logger.info(
                        "Random dominated: sampling collapsed "
                        "(WR %.0f%% >= %.0f%%)",
                        wr_random * 100, threshold * 100,
                    )
            else:
                if wr_random < recover:
                    self.random_dominated = False
                    logger.info(
                        "Random recovered: sampling restored "
                        "(WR %.0f%% < %.0f%%)",
                        wr_random * 100, recover * 100,
                    )

    def is_selfplay_dampened(self) -> bool:
        """Self-play is dampened while inside the timed cooldown window."""
        return self.total_episodes < self.selfplay_train_cooldown_until

    def get_dampened_set(self) -> Set[str]:
        """Return the set of slot keys currently pinned at `dampen_cap`.

        Reads the cached hysteresis flags — does NOT re-evaluate them.
        The trainer refreshes flags once per `log_interval` via
        `update_dampened_state`; calling on every game would flap the flags
        at the boundary and spam the log. Keys match the slot-name convention
        used by `compute_opponent_distribution`: 'self', 'random', 'minimax_d{d}'.
        """
        dampened: Set[str] = set()
        if self.random_dominated:
            dampened.add('random')
        if self.is_selfplay_dampened():
            dampened.add('self')
        for d in range(1, 8):
            if self.minimax_depth_dominated.get(d, False):
                dampened.add(f'minimax_d{d}')
        return dampened


@dataclass
class PhaseStats:
    """Statistics for current phase."""
    phase: Phase
    episodes_in_phase: int = 0
    total_games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    best_win_rate: float = 0.0
    phase_start_time: float = field(default_factory=time.time)

    # Rolling stats
    recent_results: deque = field(default_factory=lambda: deque(maxlen=2000))

    def add_result(self, result: str):
        """Add game result: 'win', 'loss', or 'draw'."""
        self.total_games += 1
        self.recent_results.append(result)

        if result == 'win':
            self.wins += 1
        elif result == 'loss':
            self.losses += 1
        else:
            self.draws += 1

        if len(self.recent_results) >= 100:
            wr = self.get_win_rate()
            if wr > self.best_win_rate:
                self.best_win_rate = wr

    def get_win_rate(self) -> float:
        """Get win rate from recent games. Draws count as half a win."""
        if not self.recent_results:
            return 0.0
        wins = sum(1 for r in self.recent_results if r == 'win')
        draws = sum(1 for r in self.recent_results if r == 'draw')
        return (wins + 0.5 * draws) / len(self.recent_results)

    def get_draw_rate(self) -> float:
        """Get draw rate from recent games."""
        if not self.recent_results:
            return 0.0
        draws = sum(1 for r in self.recent_results if r == 'draw')
        return draws / len(self.recent_results)

    def get_loss_rate(self) -> float:
        """Get loss rate from recent games."""
        if not self.recent_results:
            return 0.0
        losses = sum(1 for r in self.recent_results if r == 'loss')
        return losses / len(self.recent_results)


class CurriculumManager:
    """Manages phased curriculum training."""

    def __init__(self, start_phase: Phase = Phase.PHASE_1, save_dir: str = "curriculum"):
        self.current_phase = start_phase
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.stats = PhaseStats(phase=start_phase)
        self.phase_history: List[Dict] = []
        self.total_episodes = 0

        # Mixed training state (used by all phases — Phase 1 omits minimax
        # from the unlocked set but still tracks per-opponent windows).
        self.mixed_state = MixedTrainingState()

        # Callbacks
        self.on_phase_change_callbacks = []
        self.on_clone_update_callbacks = []
        self.on_game_settings_change_callbacks = []

        # Snapshot of per-depth state captured at the moment of graduation.
        # Populated by graduate() before mixed_state is reset; consumed by the
        # trainer's phase-change callback for logging.
        self.last_graduation_snapshot: Optional[Dict[str, Any]] = None

    def get_config(self) -> PhaseConfig:
        """Get current phase configuration."""
        if self.current_phase == Phase.COMPLETED:
            return PHASE_CONFIGS[Phase.PHASE_9]
        return PHASE_CONFIGS[self.current_phase]

    def get_stone_distribution_for_phase(self) -> List[Tuple[int, float]]:
        """Per-player starting-stone sampling distribution.

        Each player draws independently from this distribution at every env
        reset, producing asymmetric games and smooth phase transitions.

        Shape for phases 2-8 (centered on the phase's base count):
          offset {-2, -1, 0, +1, +2} → weight {0.05, 0.175, 0.55, 0.175, 0.05}
        Mass falling outside [3, 9] is collapsed onto the boundary. Phases 1
        and 10 keep their "random 3-9" feel via a uniform distribution. Phase
        9 stays fixed at 9 stones (full game). Phase 11 toggles between the
        Phase 9 and Phase 10 distributions according to its sub-phase cycle
        (see `get_phase11_subphase`).
        """
        if self.current_phase == Phase.COMPLETED:
            return [(9, 1.0)]

        phase_num = int(self.current_phase)

        if phase_num == 11:
            if self.get_phase11_subphase() == 'full':
                return [(9, 1.0)]
            return [(s, 1.0 / 7.0) for s in range(3, 10)]

        if phase_num in (1, 10):
            return [(s, 1.0 / 7.0) for s in range(3, 10)]

        if phase_num == 9:
            return [(9, 1.0)]

        base = phase_num + 1  # Phase 2 → 3, Phase 8 → 9
        offset_weights = [(-2, 0.05), (-1, 0.175), (0, 0.55), (+1, 0.175), (+2, 0.05)]
        merged: Dict[int, float] = {}
        for offset, w in offset_weights:
            count = max(3, min(9, base + offset))
            merged[count] = merged.get(count, 0.0) + w
        return sorted(merged.items())

    def get_starting_stones_for_phase(self) -> int:
        """
        Get the per-player starting-stone count for the current phase.

        Boards are initialized via fastnmm's `starting_stones` engine option
        instead of preparing positions with random moves.

        - Phase 1, 10: -1 sentinel — each player gets random stones in [3, 9]
          per game (resampled at every env reset)
        - Phase 2: 3 stones
        - Phase 3: 4 stones
        - Phase 4: 5 stones
        - Phase 5: 6 stones
        - Phase 6: 7 stones
        - Phase 7: 8 stones
        - Phase 8: 9 stones
        - Phase 9: 9 stones (full game)
        - Phase 11: 9 stones during the 'full' sub-phase, -1 during 'mix'
          (see `get_phase11_subphase`).

        Returns:
            Stone count in [1, 9], or -1 for randomize-per-game phases.
        """
        if self.current_phase == Phase.COMPLETED:
            return 9

        phase_num = int(self.current_phase)

        if phase_num == 11:
            return 9 if self.get_phase11_subphase() == 'full' else -1

        if phase_num in (1, 10):
            return -1

        if phase_num == 9:
            return 9

        stones = phase_num + 1  # Phase 2 → 3, Phase 8 → 9
        return max(1, min(9, stones))

    def get_ai_disadvantage(self) -> bool:
        """Return True when the AI should start each game with fewer stones.

        Active only during Phase 11's 'mix' sub-phase. The 'full' sub-phase
        uses a single fixed count (9 stones / player) so disadvantage is
        meaningless there. Outside Phase 11 this always returns False.
        """
        return (
            self.current_phase == Phase.PHASE_11
            and self.get_phase11_subphase() == 'mix'
        )

    def get_phase11_subphase(self) -> str:
        """Return Phase 11's current sub-phase: 'full' or 'mix'.

        Phase 11 alternates indefinitely:
          [0, PHASE_11_FULL_GAME_EPISODES) -> 'full' (9 stones / player)
          [PHASE_11_FULL_GAME_EPISODES, PHASE_11_CYCLE_EPISODES) -> 'mix'
        repeating every PHASE_11_CYCLE_EPISODES of episodes-in-phase.

        Outside phase 11 the helper returns 'full' as a harmless default;
        callers gate on the active phase first.
        """
        if self.current_phase != Phase.PHASE_11:
            return 'full'
        position = self.stats.episodes_in_phase % PHASE_11_CYCLE_EPISODES
        return 'full' if position < PHASE_11_FULL_GAME_EPISODES else 'mix'

    def get_phase11_subphase_progress(self) -> Tuple[int, int]:
        """Return (episodes_into_current_subphase, subphase_length) for Phase 11.

        Used by logging to display progress through the active sub-phase.
        Outside phase 11 returns (0, 0).
        """
        if self.current_phase != Phase.PHASE_11:
            return (0, 0)
        position = self.stats.episodes_in_phase % PHASE_11_CYCLE_EPISODES
        if position < PHASE_11_FULL_GAME_EPISODES:
            return (position, PHASE_11_FULL_GAME_EPISODES)
        return (position - PHASE_11_FULL_GAME_EPISODES, PHASE_11_MIX_EPISODES)

    def get_shaping_multiplier(self) -> float:
        """
        Monotonic decay across total training episodes (no per-phase reset).

        Combined with PBRS, the optimal policy is preserved for any mult ≥ 0,
        so this schedule only controls the *scale* of the dense signal, not
        its correctness. Decaying smoothly across phases — including into
        phase 10 — avoids the value-function shock that phase-boundary resets
        used to cause.

        - Start (episode 0): 1.0
        - Linear decay over SHAPING_DECAY_EPISODES
        - COMPLETED: floor at 0.0
        - Phase 10: keeps consuming the same schedule (no short-circuit). The
          shaping-free tail at the end of phase 10 is enforced by
          should_graduate(), not by zeroing the multiplier here.
        """
        if self.current_phase == Phase.COMPLETED:
            return 0.0

        progress = min(1.0, self.total_episodes / SHAPING_DECAY_EPISODES)
        return max(0.0, 1.0 - progress)

    def get_phase10_draw_penalty(self) -> float:
        """Late-phase draw-penalty schedule (phases 10 and 11).

        - Outside phases 10/11: returns the per-phase configured value
          unchanged.
        - Inside phase 10: decays linearly from `config.draw_penalty` (-1.5)
          to `PHASE_10_DRAW_PENALTY_END` over the first
          `PHASE_10_DRAW_PENALTY_DECAY_EPISODES` episodes of the phase, then
          holds at the end value.
        - Inside phase 11: held flat at `PHASE_11_DRAW_PENALTY` (the same
          end value Phase 10 decayed to); the cycling sub-phases must not
          re-trigger a decay.
        """
        config = self.get_config()
        if self.current_phase == Phase.PHASE_11:
            return PHASE_11_DRAW_PENALTY
        if self.current_phase != Phase.PHASE_10:
            return config.draw_penalty
        progress = min(1.0, self.stats.episodes_in_phase / PHASE_10_DRAW_PENALTY_DECAY_EPISODES)
        return config.draw_penalty + (PHASE_10_DRAW_PENALTY_END - config.draw_penalty) * progress

    def get_reward_config(self) -> Dict[str, float]:
        """Get current reward configuration.

        Per-feature weights are scaled by the shaping multiplier. step_penalty
        is scaled too so the final phase is genuinely sparse. Gamma is included
        so the PBRS shaping in RewardCalculator can compute γ·Φ(s') − Φ(s).

        Phase 10 also overrides `draw_penalty` with a linear decay schedule —
        see `get_phase10_draw_penalty`. Workers receive the updated value via
        the periodic reward-config rebroadcast in the trainer's log tick.
        """
        config = self.get_config()
        mult = self.get_shaping_multiplier()

        return {
            'win_reward_base': config.win_reward_base,
            'win_reward_speed_bonus': config.win_reward_speed_bonus,
            'loss_reward': config.loss_reward,
            'draw_penalty': self.get_phase10_draw_penalty(),
            'mill_reward': config.mill_reward * mult,
            'enemy_mill_penalty': config.enemy_mill_penalty * mult,
            'block_mill_reward': config.block_mill_reward * mult,
            'double_mill_reward': config.double_mill_reward * mult,
            'setup_capture_reward': config.setup_capture_reward * mult,
            'step_penalty': -0.003 * mult,
            'piece_advantage_reward': 0.02 * mult,
            'mobility_reward': 0.05 * mult,
            'max_shaping_abs': 0.20,
            'gamma': 0.99,
        }

    def add_game_result(self, result: float, opponent_type: str = 'random', minimax_depth: int = 0):
        """
        Add a game result.
        result > 0.5 = win, result < -1.5 = loss, else = draw

        Note: draw_penalty is typically -1.5, loss_reward is -2.0
        So draws fall in range [-1.5, 0.5]
        """
        self.total_episodes += 1
        self.stats.episodes_in_phase += 1

        # Convert result to string
        # Win: typically +2.0 to +3.0 (base + speed bonus)
        # Draw: typically -1.5 (draw_penalty)
        # Loss: typically -2.0 (loss_reward)
        if result > 0.5:
            result_str = 'win'
        elif result < -1.5:
            result_str = 'loss'
        else:
            result_str = 'draw'

        # Track in general stats
        self.stats.add_result(result_str)

        config = self.get_config()
        if config.opponent_type != 'mixed':
            # Defensive guard for future non-mixed phases; today every phase
            # in PHASE_CONFIGS is 'mixed', so this branch is unreachable.
            return

        # Track for mixed training
        self.mixed_state.total_episodes += 1

        # Track per-opponent results for win rate stats
        self.mixed_state.record_game_result(opponent_type, result_str, minimax_depth)

        if opponent_type == 'random':
            self.mixed_state.games_vs_random += 1
        elif opponent_type == 'self':
            self.mixed_state.games_vs_self += 1
        elif opponent_type == 'minimax':
            self.mixed_state.games_vs_minimax += 1
            self._handle_minimax_result(result_str, minimax_depth)

        # Take periodic minimax win rate snapshot for stagnation detection
        interval = MIXED_CONFIG['stagnation_snapshot_interval']
        if self.mixed_state.total_episodes % interval == 0:
            wr = self.mixed_state.get_combined_minimax_win_rate()
            self.mixed_state.minimax_winrate_snapshots.append(
                (self.mixed_state.total_episodes, wr)
            )
            max_keep = MIXED_CONFIG['stagnation_snapshot_window'] + 1
            if len(self.mixed_state.minimax_winrate_snapshots) > max_keep:
                self.mixed_state.minimax_winrate_snapshots = (
                    self.mixed_state.minimax_winrate_snapshots[-max_keep:]
                )

    def _handle_minimax_result(self, result_str: str, depth: int):
        """Handle minimax game result (simplified - no progressive rounds)."""
        won = (result_str == 'win')
        self.mixed_state.record_minimax_result(depth, won)

    def should_update_clone(self) -> bool:
        """Check if clone should be updated."""
        config = self.get_config()
        if config.opponent_type != 'mixed':
            return False
        return self.mixed_state.should_update_clone()

    def do_clone_update(self):
        """Called when clone is updated."""
        self.mixed_state.on_clone_updated()
        gen = self.mixed_state.clone_generation
        logger.info("Clone updated to generation %d", gen)

        for callback in self.on_clone_update_callbacks:
            callback()

        self.save_state()

    def is_stagnating(self) -> bool:
        """
        Check if the model has stopped improving enough to warrant moving to next phase.
        Triggers when BOTH conditions hold after a waiting window:
          1. Self-play win rate is still below the clone-update threshold
          2. Combined minimax d1+d2 win rate has not improved by 3%+ over last 500k episodes

        The waiting window is measured since the latest clone update (or phase start),
        so every clone reset restarts the 3M stagnation clock from zero.
        """
        config = self.get_config()
        if config.opponent_type != 'mixed':
            return False

        ms = self.mixed_state
        if ms.total_episodes < MIXED_CONFIG['stagnation_min_episodes']:
            return False

        episodes_since_clone = ms.total_episodes - ms.last_clone_episode
        if episodes_since_clone < MIXED_CONFIG['stagnation_clone_window']:
            return False

        # Condition 1: still not reliably beating current clone.
        selfplay_stuck = ms.get_selfplay_win_rate() < MIXED_CONFIG['selfplay_winrate_threshold']

        # Condition 2: minimax win rate not improving
        window = MIXED_CONFIG['stagnation_snapshot_window']
        snapshots = ms.minimax_winrate_snapshots
        if len(snapshots) < window:
            minimax_flat = False
        else:
            oldest_wr = snapshots[-window][1]
            newest_wr = snapshots[-1][1]
            minimax_flat = (newest_wr - oldest_wr) < MIXED_CONFIG['stagnation_threshold']

        return selfplay_stuck and minimax_flat

    def sample_minimax_winrate(self):
        """
        Sample minimax win rates for trend-based graduation.
        Call this at log_interval (every 25,000 episodes).

        - Combined WR (D1..D7) is appended to `minimax_wr_history` for
          dashboards/legacy logging — it does NOT drive graduation.
        - Per unlocked depth d in [1, active_max], current WR vs that depth is
          appended to `minimax_wr_history_by_depth[d]`. Graduation reads from
          these per-depth windows so weakness on the top depth can't be hidden
          by easy depths.
        """
        config = self.get_config()
        if config.opponent_type != 'mixed':
            return

        ms = self.mixed_state

        # Legacy combined sample (logging only)
        wr_combined = ms.get_combined_minimax_win_rate_up_to(7)
        ms.minimax_wr_history.append(wr_combined)

        # Per-depth samples (graduation signal)
        active_max = ms.active_minimax_max_depth
        for d in range(1, active_max + 1):
            wr_d = ms.get_win_rate_vs_opponent('minimax', d)
            ms.minimax_wr_history_by_depth[d].append(wr_d)

    def _has_plateaued(self) -> bool:
        """
        Check whether every unlocked minimax depth has plateaued.

        Returns True iff, for every unlocked depth d, the slope angle of
        WR-vs-depth-d (over a 1M-episode horizon) is below
        `trend_max_angle_degrees` (i.e., flat or declining).

        If any depth is still climbing, the phase is still making progress.
        """
        config = self.get_config()
        if config.opponent_type != 'mixed':
            return False

        ms = self.mixed_state
        active_max = ms.active_minimax_max_depth
        max_angle = GRADUATION_CONFIG['trend_max_angle_degrees']

        for d in range(1, active_max + 1):
            angle = ms.get_slope_angle_for_depth(d)
            if angle == float('inf') or angle > max_angle:
                return False
        return True

    def should_graduate(self) -> bool:
        """Check if ready to move to next phase."""
        if self.current_phase == Phase.COMPLETED:
            return False

        config = self.get_config()
        stats = self.stats

        # Check minimum games
        if len(stats.recent_results) < config.min_games_for_graduation:
            return False

        # Check minimum episodes (applies to all phases)
        if config.min_episodes > 0 and stats.episodes_in_phase < config.min_episodes:
            return False

        # Phase 1: graduate when WR vs random >= win_rate_threshold.
        # We measure against random specifically because Phase 1 mixes in
        # self-play (70/30), and self-play caps overall WR at ~50% by
        # symmetry, so aggregate WR can never reach the threshold.
        if self.current_phase == Phase.PHASE_1:
            return self.mixed_state.get_win_rate_vs_opponent('random') >= config.win_rate_threshold

        # Phase 10: anchored to the shaping schedule. Phase 10 ends exactly
        # PHASE_10_POST_SHAPING_EPISODES after shaping reaches 0.
        #   - If shaping already ended when phase 10 began, duration = floor.
        #   - If shaping continues N episodes into phase 10, duration = N + floor.
        if self.current_phase == Phase.PHASE_10:
            episodes_in_phase = stats.episodes_in_phase
            total_at_phase_start = self.total_episodes - episodes_in_phase
            shaping_overlap = max(0, SHAPING_DECAY_EPISODES - total_at_phase_start)
            required = shaping_overlap + PHASE_10_POST_SHAPING_EPISODES
            return episodes_in_phase >= required

        # Phase 11: infinite. Cycles full-game / mix sub-phases forever; the
        # operator stops training to end it. Never graduates on its own.
        if self.current_phase == Phase.PHASE_11:
            return False

        # Phase 2-9 graduation: per-depth saturation.
        # Both conditions must hold simultaneously.

        # Condition 1: minimum time-in-phase floor.
        if stats.episodes_in_phase < GRADUATION_CONFIG['min_episodes']:
            return False

        # Condition 2: every unlocked depth has plateaued and has enough
        # samples in its window.
        return self._has_plateaued()

    def graduate(self) -> bool:
        """Move to next phase. Returns True if graduated."""
        if self.current_phase == Phase.COMPLETED:
            return False

        # Determine graduation reason for logging
        config = self.get_config()
        if config.opponent_type == 'mixed':
            graduation_reason = 'per-depth saturation + WR-vs-top-depth threshold'
        else:
            graduation_reason = f"win rate >= {config.win_rate_threshold:.0%}"

        # Per-depth snapshot at the moment of graduation. Captured BEFORE the
        # mixed_state reset so trainer callbacks (and the phase_history entry)
        # can dump it for diagnostics.
        ms = self.mixed_state
        active_max = ms.active_minimax_max_depth if config.opponent_type == 'mixed' else 0
        per_depth_wr_snapshot = {
            d: ms.get_win_rate_vs_opponent('minimax', d)
            for d in range(1, max(active_max, 0) + 1)
        }
        per_depth_slope_angle = {
            d: ms.get_slope_angle_for_depth(d)
            for d in range(1, max(active_max, 0) + 1)
        }
        per_depth_samples = {
            d: ms.get_window_size_for_depth(d)
            for d in range(1, max(active_max, 0) + 1)
        }
        wr_top_depth = (
            ms.get_win_rate_vs_opponent('minimax', active_max) if active_max >= 1 else 0.0
        )
        self.last_graduation_snapshot = {
            'phase_id': int(self.current_phase),
            'episodes_in_phase': self.stats.episodes_in_phase,
            'wr_vs_top_depth_at_graduation': wr_top_depth,
            'per_depth_wr': per_depth_wr_snapshot,
            'per_depth_slope_angle': per_depth_slope_angle,
            'per_depth_samples': per_depth_samples,
            'top_depth': active_max,
            'clone_generations_in_phase': ms.clone_generation,
            'graduation_reason': graduation_reason,
        }

        # Save phase history
        self.phase_history.append({
            'phase': int(self.current_phase),
            'episodes': self.stats.episodes_in_phase,
            'total_games': self.stats.total_games,
            'wins': self.stats.wins,
            'losses': self.stats.losses,
            'draws': self.stats.draws,
            'best_win_rate': self.stats.best_win_rate,
            'clone_generations': ms.clone_generation,
            'graduation_reason': graduation_reason,
            'duration_seconds': time.time() - self.stats.phase_start_time,
            'wr_vs_top_depth_at_graduation': wr_top_depth,
            'per_depth_wr_at_graduation': per_depth_wr_snapshot,
        })

        old_phase = self.current_phase

        # Move to next phase. Phase 11 is the terminal training phase (infinite
        # cycling sub-phases) — should_graduate() returns False there, so this
        # branch is only reached for phases 1..10.
        if self.current_phase == Phase.PHASE_11:
            self.current_phase = Phase.COMPLETED
        else:
            self.current_phase = Phase(int(self.current_phase) + 1)

        # Reset stats for new phase
        self.stats = PhaseStats(phase=self.current_phase)

        # Reset mixed state for new phase, but carry over the minimax unlock
        # progression so a graduated phase doesn't have to re-unlock D1→D5.
        prev_active_max_depth = self.mixed_state.active_minimax_max_depth
        self.mixed_state = MixedTrainingState()
        self.mixed_state.active_minimax_max_depth = prev_active_max_depth

        # Notify callbacks
        for callback in self.on_phase_change_callbacks:
            callback(old_phase, self.current_phase)

        new_config = PHASE_CONFIGS.get(self.current_phase)
        print(f"\n{'='*60}")
        print(f"  GRADUATED from Phase {int(old_phase)} to Phase {int(self.current_phase)}! (reason: {graduation_reason})")
        if new_config:
            print(f"  {new_config.description}")
        print(f"{'='*60}\n")

        self.save_state()
        return True

    def check_and_graduate(self) -> bool:
        """Check graduation criteria and graduate if met."""
        if self.current_phase == Phase.COMPLETED:
            return False

        if self.should_graduate():
            return self.graduate()
        return False

    def get_status_string(self) -> str:
        """Get a status string for logging."""
        if self.current_phase == Phase.COMPLETED:
            return "Training Complete"

        config = self.get_config()
        stats = self.stats
        wr = stats.get_win_rate()
        shaping_mult = self.get_shaping_multiplier()
        stones = self.get_starting_stones_for_phase()
        stones_str = f"{stones}st" if stones >= 0 else "rand-st"

        if self.current_phase == Phase.PHASE_11:
            sub = self.get_phase11_subphase()
            sub_pos, sub_len = self.get_phase11_subphase_progress()
            phase_label = (
                f"Phase 11(inf, {sub} "
                f"{sub_pos // 1000}k/{sub_len // 1000}k)"
            )
        else:
            phase_label = f"Phase {int(self.current_phase)}/11"

        parts = [
            phase_label,
            stones_str,
            f"WR:{wr:.0%}",
            f"Shape:{shaping_mult:.2f}",
        ]

        if config.opponent_type == 'mixed':
            ms = self.mixed_state
            parts.append(f"clone gen:{ms.clone_generation}")
            # Add trend info
            if len(ms.minimax_wr_history) >= 10:
                slope = ms.calculate_winrate_trend(ms.active_minimax_max_depth)
                slope_per_million = slope * 1_000_000
                angle = math.degrees(math.atan(slope_per_million)) if abs(slope_per_million) < 1e6 else float('inf')
                parts.append(f"trend:{angle:.1f}°")

        return " | ".join(parts)

    def get_active_minimax_max_depth(self) -> int:
        """Get currently unlocked maximum minimax depth (starts at 1, cap = D5)."""
        return self.mixed_state.active_minimax_max_depth

    def check_and_unlock_minimax_depth(self) -> bool:
        """
        Progressively unlock harder minimax depths (D1 → D5):
        D(n+1) unlocks when WR vs D(n) >= 50% over at least 100 games.
        D5 is the training cap; D6/D7 are never sampled as training
        opponents (see `minimax_max_depth`).
        Returns True if a new depth was unlocked.
        """
        config = self.get_config()
        if config.opponent_type != 'mixed':
            return False

        ms = self.mixed_state
        threshold = MIXED_CONFIG['minimax_depth_unlock_threshold']
        min_games = MIXED_CONFIG['minimax_depth_unlock_min_games']

        current_max = ms.active_minimax_max_depth
        if current_max >= MIXED_CONFIG['minimax_max_depth']:
            return False  # Already at training-side maximum (D5)

        results = ms._depth_results_deque(current_max)
        if results is None or len(results) < min_games:
            return False

        wr = ms.get_win_rate_vs_opponent('minimax', current_max)
        if wr >= threshold:
            ms.active_minimax_max_depth = current_max + 1
            print(f"\n  [Depth Unlock] D{current_max + 1} minimax unlocked! "
                  f"(WR vs D{current_max}: {wr:.0%})")
            return True

        return False

    def get_opponent_distribution(self) -> Dict[str, float]:
        """Full per-opponent sampling distribution for the current phase.

        Slot keys: 'self', 'random', 'minimax_d1' .. 'minimax_d7'. The
        distribution sums to 1.0. Phase 1 (warmup) omits minimax entirely
        and yields {'self': 0.75, 'random': 0.25}. All other phases use the
        equal-share-with-selfplay×3 base formula across self + unlocked
        minimax depths + random, with currently-dampened slots pinned at
        `dampen_cap` and the remaining mass redistributed.
        """
        if self.current_phase == Phase.PHASE_1:
            unlocked: List[int] = []
        else:
            # Clamp to the training-side cap so a checkpoint that previously
            # unlocked D6/D7 keeps them out of the sampling pool.
            train_cap = MIXED_CONFIG['minimax_max_depth']
            active_max = min(self.mixed_state.active_minimax_max_depth, train_cap)
            unlocked = list(range(1, active_max + 1))
        dampened = self.mixed_state.get_dampened_set()
        return compute_opponent_distribution(
            unlocked,
            dampened,
            cap_value=MIXED_CONFIG['dampen_cap'],
            selfplay_weight=MIXED_CONFIG['selfplay_weight'],
        )

    def get_opponent_win_rates(self) -> Dict[str, float]:
        """Get win rates vs each opponent type (last 500 games)."""
        ms = self.mixed_state
        return {
            'wr_vs_mm_d1': ms.get_win_rate_vs_opponent('minimax', 1),
            'wr_vs_mm_d2': ms.get_win_rate_vs_opponent('minimax', 2),
            'wr_vs_mm_d3': ms.get_win_rate_vs_opponent('minimax', 3),
            'wr_vs_mm_d4': ms.get_win_rate_vs_opponent('minimax', 4),
            'wr_vs_mm_d5': ms.get_win_rate_vs_opponent('minimax', 5),
            'wr_vs_mm_d6': ms.get_win_rate_vs_opponent('minimax', 6),
            'wr_vs_mm_d7': ms.get_win_rate_vs_opponent('minimax', 7),
            'wr_vs_random': ms.get_win_rate_vs_opponent('random'),
            'wr_vs_self': ms.get_win_rate_vs_opponent('self'),
            'active_mm_max_depth': ms.active_minimax_max_depth,
        }

    def to_state_dict(self) -> Dict[str, Any]:
        """Build a serializable curriculum state snapshot."""
        return {
            'current_phase': int(self.current_phase),
            'total_episodes': self.total_episodes,
            'stats': {
                'phase': int(self.stats.phase),
                'episodes_in_phase': self.stats.episodes_in_phase,
                'total_games': self.stats.total_games,
                'wins': self.stats.wins,
                'losses': self.stats.losses,
                'draws': self.stats.draws,
                'best_win_rate': self.stats.best_win_rate,
                'recent_results': list(self.stats.recent_results),
            },
            'phase_history': self.phase_history,
            'mixed_state': {
                'total_episodes': self.mixed_state.total_episodes,
                'clone_generation': self.mixed_state.clone_generation,
                'last_clone_episode': self.mixed_state.last_clone_episode,
                'selfplay_train_cooldown_until': self.mixed_state.selfplay_train_cooldown_until,
                'games_vs_random': self.mixed_state.games_vs_random,
                'games_vs_minimax': self.mixed_state.games_vs_minimax,
                'games_vs_self': self.mixed_state.games_vs_self,
                'minimax_wins_by_depth': dict(self.mixed_state.minimax_wins_by_depth),
                'minimax_winrate_snapshots': self.mixed_state.minimax_winrate_snapshots,
                'active_minimax_max_depth': self.mixed_state.active_minimax_max_depth,
                'results_vs_random': list(self.mixed_state.results_vs_random),
                'results_vs_minimax_d1': list(self.mixed_state.results_vs_minimax_d1),
                'results_vs_minimax_d2': list(self.mixed_state.results_vs_minimax_d2),
                'results_vs_minimax_d3': list(self.mixed_state.results_vs_minimax_d3),
                'results_vs_minimax_d4': list(self.mixed_state.results_vs_minimax_d4),
                'results_vs_minimax_d5': list(self.mixed_state.results_vs_minimax_d5),
                'results_vs_minimax_d6': list(self.mixed_state.results_vs_minimax_d6),
                'results_vs_minimax_d7': list(self.mixed_state.results_vs_minimax_d7),
                'results_vs_self': list(self.mixed_state.results_vs_self),
                'minimax_wr_history': list(self.mixed_state.minimax_wr_history),
                'minimax_wr_history_by_depth': {
                    d: list(dq) for d, dq in self.mixed_state.minimax_wr_history_by_depth.items()
                },
                'minimax_depth_dominated': dict(self.mixed_state.minimax_depth_dominated),
                'random_dominated': self.mixed_state.random_dominated,
            },
        }

    def save_state(self, path: Optional[str] = None):
        """Save curriculum state to file."""
        if path is None:
            path = os.path.join(self.save_dir, "curriculum_state.json")

        state = self.to_state_dict()

        with open(path, 'w') as f:
            json.dump(state, f, indent=2)

    def load_state_dict(self, state: Dict[str, Any]) -> bool:
        """Load curriculum state from a dict."""
        try:
            self.current_phase = Phase(state['current_phase'])
            self.total_episodes = state['total_episodes']
            self.phase_history = state.get('phase_history', [])

            stats_data = state['stats']
            self.stats = PhaseStats(
                phase=Phase(stats_data['phase']),
                episodes_in_phase=stats_data['episodes_in_phase'],
                total_games=stats_data['total_games'],
                wins=stats_data['wins'],
                losses=stats_data['losses'],
                draws=stats_data['draws'],
                best_win_rate=stats_data['best_win_rate'],
                recent_results=deque(stats_data.get('recent_results', []), maxlen=2000),
            )

            if 'mixed_state' in state:
                ms = state['mixed_state']
                self.mixed_state = MixedTrainingState(
                    total_episodes=ms.get('total_episodes', 0),
                    clone_generation=ms.get('clone_generation', 0),
                    last_clone_episode=ms.get('last_clone_episode', 0),
                    selfplay_train_cooldown_until=ms.get('selfplay_train_cooldown_until', 0),
                    games_vs_random=ms.get('games_vs_random', 0),
                    games_vs_minimax=ms.get('games_vs_minimax', 0),
                    games_vs_self=ms.get('games_vs_self', 0),
                    active_minimax_max_depth=ms.get('active_minimax_max_depth', 1),
                    results_vs_random=deque(ms.get('results_vs_random', []), maxlen=500),
                    results_vs_minimax_d1=deque(ms.get('results_vs_minimax_d1', []), maxlen=500),
                    results_vs_minimax_d2=deque(ms.get('results_vs_minimax_d2', []), maxlen=500),
                    results_vs_minimax_d3=deque(ms.get('results_vs_minimax_d3', []), maxlen=500),
                    results_vs_minimax_d4=deque(ms.get('results_vs_minimax_d4', []), maxlen=500),
                    results_vs_minimax_d5=deque(ms.get('results_vs_minimax_d5', []), maxlen=500),
                    results_vs_minimax_d6=deque(ms.get('results_vs_minimax_d6', []), maxlen=500),
                    results_vs_minimax_d7=deque(ms.get('results_vs_minimax_d7', []), maxlen=500),
                    results_vs_self=deque(ms.get('results_vs_self', []), maxlen=500),
                )
                if 'minimax_wins_by_depth' in ms:
                    self.mixed_state.minimax_wins_by_depth = {
                        int(k): v for k, v in ms['minimax_wins_by_depth'].items()
                    }
                self.mixed_state.minimax_winrate_snapshots = ms.get('minimax_winrate_snapshots', [])
                self.mixed_state.minimax_wr_history = deque(
                    ms.get('minimax_wr_history', []),
                    maxlen=GRADUATION_CONFIG['trend_window_samples']
                )
                # Per-depth WR windows (graduation signal). Missing keys default
                # to empty deques — the windows refill organically as training
                # continues, so old checkpoints simply re-warm.
                window_max = GRADUATION_CONFIG['trend_window_samples']
                hist_by_depth_raw = ms.get('minimax_wr_history_by_depth', {})
                self.mixed_state.minimax_wr_history_by_depth = {
                    d: deque(hist_by_depth_raw.get(str(d), hist_by_depth_raw.get(d, [])),
                             maxlen=window_max)
                    for d in range(1, 8)
                }
                dominated_raw = ms.get('minimax_depth_dominated', {})
                if dominated_raw:
                    self.mixed_state.minimax_depth_dominated = {
                        int(k): bool(v) for k, v in dominated_raw.items()
                    }
                self.mixed_state.random_dominated = bool(ms.get('random_dominated', False))

            print(f"  Loaded curriculum: Phase {int(self.current_phase)}, {self.total_episodes:,} episodes")
            return True
        except Exception as e:
            print(f"  Failed to load curriculum state: {e}")
            return False

    def load_state(self, path: Optional[str] = None) -> bool:
        """Load curriculum state from file."""
        if path is None:
            path = os.path.join(self.save_dir, "curriculum_state.json")

        if not os.path.exists(path):
            return False

        try:
            with open(path, 'r') as f:
                state = json.load(f)
            return self.load_state_dict(state)
        except Exception as e:
            print(f"  Failed to load curriculum state: {e}")
            return False

    def print_summary(self):
        """Print training summary."""
        print("\n" + "="*60)
        print("CURRICULUM TRAINING SUMMARY")
        print("="*60)

        for phase_data in self.phase_history:
            phase = Phase(phase_data['phase'])
            config = PHASE_CONFIGS.get(phase)
            duration = phase_data['duration_seconds']
            hours = duration / 3600

            print(f"\nPhase {int(phase)}: {config.description if config else 'Unknown'}")
            print(f"  Episodes: {phase_data['episodes']:,}")
            print(f"  Games: {phase_data['total_games']} (W:{phase_data['wins']} L:{phase_data['losses']} D:{phase_data['draws']})")
            print(f"  Best WR: {phase_data['best_win_rate']:.1%}")
            if phase_data.get('clone_generations', 0) > 0:
                print(f"  Clone Generations: {phase_data['clone_generations']}")
            if 'graduation_reason' in phase_data:
                print(f"  Graduated by: {phase_data['graduation_reason']}")
            print(f"  Duration: {hours:.1f}h")

        if self.current_phase != Phase.COMPLETED:
            print(f"\nCurrent: Phase {int(self.current_phase)} - {self.get_config().description}")
            print(f"  Progress: {self.get_status_string()}")
        else:
            print("\n  TRAINING COMPLETED!")

        print("="*60 + "\n")