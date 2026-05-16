"""
Nine Men's Morris - Curriculum Manager
Handles phased training with automatic progression

Phase Structure:
- Phase 1: Random stones (3-9), jumping phase, vs random only (warmup)
          Min 200k episodes, shaping_multiplier=1.0
- Phase 2-9: 3-9 stones, mixed opponents (30% minimax D1-D2, 65% self-play, 5% random)
             Shaping multiplier: 1.0 -> 0.0 over first 3/4 of phase, then 0.0 for last 1/4
             Resets to 1.0 at start of each new phase
- Phase 10: Full game, no shaping (multiplier=0.0), 1M episodes, minimax D1-D4 (35% minimax, 55% self-play, 10% random)
"""

import os
import json
import math
import time
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Tuple, Any
from collections import deque
from enum import IntEnum
import numpy as np

from logging_setup import get_logger

logger = get_logger(__name__)


class Phase(IntEnum):
    """Training phases."""
    PHASE_1 = 1   # Random 3-9 stones, jumping, vs random (warmup)
    PHASE_2 = 2   # 3 stones, jumping, mixed opponents
    PHASE_3 = 3   # 4 stones, moving, mixed opponents
    PHASE_4 = 4   # 5 stones, moving, mixed opponents
    PHASE_5 = 5   # 6 stones, moving, mixed opponents
    PHASE_6 = 6   # 7 stones, moving, mixed opponents
    PHASE_7 = 7   # 8 stones, moving, mixed opponents
    PHASE_8 = 8   # 9 stones, moving, mixed opponents
    PHASE_9 = 9   # 9 stones, full game (placing), mixed opponents
    PHASE_10 = 10 # Final: full game, no shaping, D1-D6 minimax
    COMPLETED = 11


@dataclass
class PhaseConfig:
    """Configuration for a single training phase."""
    phase: Phase
    description: str

    # Opponent settings for Phase 1 (random only)
    opponent_type: str = 'random'  # 'random' for Phase 1, 'mixed' for Phase 2+

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
MIXED_CONFIG = {
    # Opponent distribution - increased minimax to reduce self-play overfitting
    'opponent_mix': {
        'minimax': 0.37,   # 30% minimax (was 15%)
        'self': 0.60,      # 60% self-play (was 80%)
        'random': 0.03,    # 10% random (was 5%)
    },

    # Self-play: clone update at 80% win rate (was 90%, less aggressive)
    'selfplay_winrate_threshold': 0.8,
    'selfplay_winrate_games': 25000,  # Increased from 500 for stability
    # Minimum episodes between clone updates. Rapid clone churn was collapsing
    # the policy, so require at least this many episodes since the previous
    # clone before making a new one.
    'selfplay_clone_cooldown_episodes': 200_000,

    # Minimax depth range — gradual unlock from D1 up to D7 based on win rate
    'minimax_min_depth': 1,
    'minimax_max_depth': 7,  # D1-D7, unlocked progressively

    # Minimax depth unlock: unlock next depth when WR vs current >= 50% over 100 games
    'minimax_depth_unlock_threshold': 0.50,
    'minimax_depth_unlock_min_games': 100,

    # Stagnation detection: graduate early if model stops improving
    'stagnation_min_episodes': 1_000_000,    # Don't trigger before 1M episodes in phase
    'stagnation_clone_window': 3_000_000,    # Evaluate only after 3M eps since phase start / last clone
    'stagnation_snapshot_interval': 100_000, # Take minimax WR snapshot every 100k episodes
    'stagnation_snapshot_window': 5,         # Compare last 5 snapshots (= 500k episodes)
    'stagnation_threshold': 0.03,            # Must improve combined d1+d2 WR by 3%

    # Per-depth sampling dampening with hysteresis. When WR vs depth d crosses
    # `dominate_threshold`, the depth is "dominated" and its sampling weight
    # collapses to `dominate_weight` (~1% of normal). The freed probability
    # mass shifts to self-play. The depth recovers (weight 1.0) only after WR
    # falls below `dominate_recover` — the gap prevents flapping at the edge.
    'minimax_depth_dominate_threshold': 0.90,
    'minimax_depth_dominate_recover': 0.85,
    'minimax_depth_dominate_min_games': 100,
    'minimax_depth_dominate_weight': 0.01,
}

# Special config for Phase 1 (warmup: self-play + random, no minimax yet)
PHASE_1_CONFIG = {
    'minimax': 0.0,
    'self':    0.70,
    'random':  0.30,
}

# Special config for Phase 10 (final phase with harder minimax)
PHASE_10_CONFIG = {
    'minimax': 0.44,   # 35% minimax (harder opponents for final phase)
    'self': 0.55,      # 55% self-play
    'random': 0.01,    # 01% random
    'selfplay_winrate_threshold': 0.8,
    'selfplay_winrate_games': 25000,
    'minimax_min_depth': 1,
    'minimax_max_depth': 7,  # D1-D7 for final phase
}


# Graduation criteria (Phase 2-10, mixed opponents).
#
# Each WR sampling tick fills a per-depth window (20 samples = 500k-episode
# lookback). A phase graduates only when ALL of the following hold for every
# currently-unlocked minimax depth:
#   1. clone_generations_in_phase >= min_clone_generations
#   2. episodes_in_phase >= min_episodes
#   3. WR vs the top unlocked depth >= min_wr_top_depth
#   4. slope angle of WR vs depth d < trend_max_angle_degrees, for every d
#   5. samples_in_window for depth d >= min_samples_per_depth, for every d
#
# Combined-WR is no longer used to drive graduation (it inflated easy depths
# and hid weakness on top depths). It may still be logged for dashboards.
GRADUATION_CONFIG = {
    'trend_window_samples': 20,        # samples per depth window (per-depth + legacy combined)
    'trend_max_angle_degrees': 0.0,    # condition 4: no measurable improvement
    'min_episodes': 2_500_000,         # condition 2
    'min_wr_top_depth': 0.70,          # condition 3
    'min_samples_per_depth': 20,       # condition 5
    'min_clone_generations': 5,        # condition 1
}


# Define all phases
PHASE_CONFIGS = {
    Phase.PHASE_1: PhaseConfig(
        phase=Phase.PHASE_1,
        description="Warmup: 0-150 random pre-moves, 70% self / 30% random",
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
        description="150 random pre-moves, vs mixed",
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
        description="~129 random pre-moves, vs mixed",
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
        description="~107 random pre-moves, vs mixed",
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
        description="~86 random pre-moves, vs mixed",
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
        description="~64 random pre-moves, vs mixed",
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
        description="~43 random pre-moves, vs mixed",
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
        description="~21 random pre-moves, vs mixed",
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
        description="Full game from start (0 pre-moves), vs mixed",
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
        description="Full game, 0-150 random pre-moves, vs harder minimax",
        opponent_type='mixed',
        win_reward_base=2.0,
        win_reward_speed_bonus=1.0,
        loss_reward=-2.0,
        draw_penalty=-1.5,
        shaping_multiplier=0.0,  # No shaping rewards
        win_rate_threshold=0.50,
        min_games_for_graduation=1000,
        max_episodes=0,  # No fixed limit - uses trend-based graduation
    ),
}


@dataclass
class MixedTrainingState:
    """State for mixed opponent training (Phase 2+)."""
    total_episodes: int = 0

    # Self-play tracking (maxlen matches selfplay_winrate_games = 1000)
    selfplay_results: deque = field(default_factory=lambda: deque(maxlen=1000))
    clone_generation: int = 0
    last_clone_episode: int = 0  # total_episodes when clone was last updated

    # Minimax win rate snapshots for stagnation detection: list of (total_episodes, win_rate)
    minimax_winrate_snapshots: List = field(default_factory=list)

    # Minimax tracking
    minimax_wins_by_depth: Dict[int, int] = field(default_factory=lambda: {d: 0 for d in range(1, 8)})

    # Per-opponent game counts
    games_vs_random: int = 0
    games_vs_minimax: int = 0
    games_vs_self: int = 0

    # Active minimax depth ceiling (starts at D1, unlocks D2-D7 progressively via win rate)
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

    def get_selfplay_win_rate(self) -> float:
        """Get win rate from last N self-play games."""
        if len(self.selfplay_results) < 50:
            return 0.5  # Not enough data
        wins = sum(1 for r in self.selfplay_results if r == 'win')
        return wins / len(self.selfplay_results)

    def should_update_clone(self) -> bool:
        """Check if clone should be updated (85% WR over 1000 games).

        Enforces a cooldown: a new clone cannot be created if the previous
        clone was created within the last `selfplay_clone_cooldown_episodes`
        episodes. The very first clone in a phase (clone_generation == 0)
        is always allowed to be created as soon as the WR threshold is met.
        """
        if len(self.selfplay_results) < MIXED_CONFIG['selfplay_winrate_games']:
            return False
        if self.clone_generation > 0:
            cooldown = MIXED_CONFIG['selfplay_clone_cooldown_episodes']
            if (self.total_episodes - self.last_clone_episode) < cooldown:
                return False
        return self.get_selfplay_win_rate() >= MIXED_CONFIG['selfplay_winrate_threshold']

    def on_clone_updated(self):
        """Called when clone is updated."""
        self.selfplay_results.clear()
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

    def update_minimax_dominated_state(self) -> Dict[int, bool]:
        """Refresh per-depth `minimax_depth_dominated` flags using hysteresis.

        A depth d becomes dominated once WR(d) >= dominate_threshold over at
        least dominate_min_games games. It stays dominated until WR(d) drops
        below dominate_recover; the gap prevents flapping at the boundary.
        Depths with too few games to judge keep their previous flag.
        """
        threshold = MIXED_CONFIG['minimax_depth_dominate_threshold']
        recover = MIXED_CONFIG['minimax_depth_dominate_recover']
        min_games = MIXED_CONFIG['minimax_depth_dominate_min_games']
        for d in range(1, 8):
            deq = self._depth_results_deque(d)
            n_games = len(deq) if deq is not None else 0
            if n_games < min_games:
                continue
            wr = self.get_win_rate_vs_opponent('minimax', d)
            if not self.minimax_depth_dominated[d]:
                if wr >= threshold:
                    self.minimax_depth_dominated[d] = True
                    logger.info(
                        "Depth dominated: D%d sampling collapsed "
                        "(WR %.0f%% >= %.0f%%); freed mass -> self-play",
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
        return dict(self.minimax_depth_dominated)


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

        # Mixed training state (for Phase 2+)
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

        Returns:
            Stone count in [1, 9], or -1 for randomize-per-game phases.
        """
        if self.current_phase == Phase.COMPLETED:
            return 9

        phase_num = int(self.current_phase)

        if phase_num in (1, 10):
            return -1

        if phase_num == 9:
            return 9

        stones = phase_num + 1  # Phase 2 → 3, Phase 8 → 9
        return max(1, min(9, stones))

    def get_shaping_multiplier(self) -> float:
        """
        Monotonic decay across total training episodes (no per-phase reset).

        Combined with PBRS, the optimal policy is preserved for any mult ≥ 0,
        so this schedule only controls the *scale* of the dense signal, not
        its correctness. Decaying smoothly across phases avoids the
        value-function shock that phase-boundary resets used to cause.

        - Start (episode 0): 1.0
        - Linear decay over SHAPING_DECAY_EPISODES
        - Final phase (PHASE_10) and COMPLETED: floor at 0.0
        """
        if self.current_phase == Phase.COMPLETED:
            return 0.0
        if self.current_phase == Phase.PHASE_10:
            return 0.0

        SHAPING_DECAY_EPISODES = 12_000_000
        progress = min(1.0, self.total_episodes / SHAPING_DECAY_EPISODES)
        return max(0.0, 1.0 - progress)

    def get_reward_config(self) -> Dict[str, float]:
        """Get current reward configuration.

        Per-feature weights are scaled by the shaping multiplier. step_penalty
        is scaled too so the final phase is genuinely sparse. Gamma is included
        so the PBRS shaping in RewardCalculator can compute γ·Φ(s') − Φ(s).
        """
        config = self.get_config()
        mult = self.get_shaping_multiplier()

        return {
            'win_reward_base': config.win_reward_base,
            'win_reward_speed_bonus': config.win_reward_speed_bonus,
            'loss_reward': config.loss_reward,
            'draw_penalty': config.draw_penalty,
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
            return  # Phase 1: no mixed tracking needed

        # Track for mixed training
        self.mixed_state.total_episodes += 1

        # Track per-opponent results for win rate stats
        self.mixed_state.record_game_result(opponent_type, result_str, minimax_depth)

        if opponent_type == 'random':
            self.mixed_state.games_vs_random += 1
        elif opponent_type == 'self':
            self.mixed_state.games_vs_self += 1
            self.mixed_state.selfplay_results.append(result_str)
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

        Returns True iff, for every unlocked depth d:
          - the per-depth window has at least `min_samples_per_depth` samples
          - the slope angle of WR-vs-depth-d (over a 1M-episode horizon) is
            below `trend_max_angle_degrees` (i.e., flat or declining).

        If any depth is still climbing, the phase is still making progress.
        """
        config = self.get_config()
        if config.opponent_type != 'mixed':
            return False

        ms = self.mixed_state
        active_max = ms.active_minimax_max_depth
        min_samples = GRADUATION_CONFIG['min_samples_per_depth']
        max_angle = GRADUATION_CONFIG['trend_max_angle_degrees']

        for d in range(1, active_max + 1):
            if ms.get_window_size_for_depth(d) < min_samples:
                return False
            angle = ms.get_slope_angle_for_depth(d)
            if angle == float('inf') or angle >= max_angle:
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

        # Phase 2-10 graduation: per-depth saturation + competence at top depth.
        # All five conditions must hold simultaneously.
        ms = self.mixed_state

        # Condition 1: enough clone generations have happened in this phase
        # (so the agent has actually faced progressively harder snapshots,
        # not just its frozen initial clone).
        if ms.clone_generation < GRADUATION_CONFIG['min_clone_generations']:
            return False

        # Condition 2: minimum time-in-phase floor.
        if stats.episodes_in_phase < GRADUATION_CONFIG['min_episodes']:
            return False

        # Condition 3: competence at the hardest currently-unlocked opponent.
        top_depth = ms.active_minimax_max_depth
        wr_top = ms.get_win_rate_vs_opponent('minimax', top_depth)
        if wr_top < GRADUATION_CONFIG['min_wr_top_depth']:
            return False

        # Conditions 4 & 5: every unlocked depth has plateaued and has enough
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

        # Move to next phase
        if self.current_phase == Phase.PHASE_10:
            self.current_phase = Phase.COMPLETED
        else:
            self.current_phase = Phase(int(self.current_phase) + 1)

        # Reset stats for new phase
        self.stats = PhaseStats(phase=self.current_phase)

        # Reset mixed state for new phase
        self.mixed_state = MixedTrainingState()

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

        parts = [
            f"Phase {int(self.current_phase)}/10",
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
        """Get currently unlocked maximum minimax depth (starts at 2, max 4)."""
        return self.mixed_state.active_minimax_max_depth

    def check_and_unlock_minimax_depth(self) -> bool:
        """
        Progressively unlock harder minimax depths (D1 → D7):
        D(n+1) unlocks when WR vs D(n) >= 50% over at least 100 games.
        Returns True if a new depth was unlocked.
        """
        config = self.get_config()
        if config.opponent_type != 'mixed':
            return False

        ms = self.mixed_state
        threshold = MIXED_CONFIG['minimax_depth_unlock_threshold']
        min_games = MIXED_CONFIG['minimax_depth_unlock_min_games']

        current_max = ms.active_minimax_max_depth
        if current_max >= 7:
            return False  # Already at maximum

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

    def get_minimax_depth_weights(self) -> Dict[int, float]:
        """Per-depth sampling weights for minimax opponents (Phase 2+).

        Updates the dominated-state hysteresis, then maps each depth to:
          - `minimax_depth_dominate_weight` (default 0.01) when dominated
          - 1.0 otherwise

        Workers receive this dict and redistribute the freed probability mass
        from dominated depths to self-play; the random share is unchanged.
        Outside mixed phases this returns all 1.0s (no-op).
        """
        config = self.get_config()
        if config.opponent_type != 'mixed':
            return {d: 1.0 for d in range(1, 8)}
        self.mixed_state.update_minimax_dominated_state()
        dominated_weight = MIXED_CONFIG['minimax_depth_dominate_weight']
        return {
            d: (dominated_weight if self.mixed_state.minimax_depth_dominated[d] else 1.0)
            for d in range(1, 8)
        }

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
                'games_vs_random': self.mixed_state.games_vs_random,
                'games_vs_minimax': self.mixed_state.games_vs_minimax,
                'games_vs_self': self.mixed_state.games_vs_self,
                'minimax_wins_by_depth': dict(self.mixed_state.minimax_wins_by_depth),
                'minimax_winrate_snapshots': self.mixed_state.minimax_winrate_snapshots,
                'active_minimax_max_depth': self.mixed_state.active_minimax_max_depth,
                'selfplay_results': list(self.mixed_state.selfplay_results),
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
                    games_vs_random=ms.get('games_vs_random', 0),
                    games_vs_minimax=ms.get('games_vs_minimax', 0),
                    games_vs_self=ms.get('games_vs_self', 0),
                    active_minimax_max_depth=ms.get('active_minimax_max_depth', 1),
                    selfplay_results=deque(ms.get('selfplay_results', []), maxlen=MIXED_CONFIG['selfplay_winrate_games']),
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