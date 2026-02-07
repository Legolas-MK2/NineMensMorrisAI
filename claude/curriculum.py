"""
Nine Men's Morris - Curriculum Manager
Handles phased training with automatic progression

Phase Structure:
- Phase 1: Random stones (3-9), jumping phase, vs random only (warmup)
          Min 200k episodes, shaping_multiplier=1.0
- Phase 2-9: 3-9 stones, mixed opponents (30% minimax D1-D2, 65% self-play, 5% random)
             Shaping multiplier: 1.0 -> 0.0 over first 3/4 of phase, then 0.0 for last 1/4
             Resets to 1.0 at start of each new phase
- Phase 10: Full game, no shaping (multiplier=0.0), 1M episodes, minimax D1-D6 (30% minimax, 65% self-play, 5% random)
"""

import os
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Tuple, Any
from collections import deque
from enum import IntEnum
import numpy as np


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

    # Game settings
    stones_per_player: int = 3
    start_phase: str = 'jumping'  # 'placing', 'moving', or 'jumping'

    # Opponent settings for Phase 1 (random only)
    opponent_type: str = 'random'  # 'random' for Phase 1, 'mixed' for Phase 2+

    # Learning rate
    lr_start: float = 3e-4
    lr_end: float = 7e-5

    # Reward multipliers
    win_reward_base: float = 1.0
    win_reward_speed_bonus: float = 1.0
    loss_reward: float = -1.0
    draw_penalty: float = -0.5

    # Shaping reward multiplier (0.0 = no shaping, 1.0 = full shaping)
    shaping_multiplier: float = 1.0

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
    lr_decay_episodes: int = 500_000


# Mixed opponent configuration for Phase 2+
MIXED_CONFIG = {
    # Opponent distribution
    'opponent_mix': {
        'minimax': 0.30,   # 30% minimax
        'self': 0.65,      # 65% self-play (vs clone)
        'random': 0.05,    # 5% random
    },

    # Self-play: clone update when 80% win rate over last 500 games
    'selfplay_winrate_threshold': 0.80,
    'selfplay_winrate_games': 500,

    # Minimax depth range (random selection, no progressive)
    'minimax_min_depth': 1,
    'minimax_max_depth': 2,  # D1-D2 for phases 2-9

    # Graduation: 2M episodes per phase
    'graduation_episodes': 2_000_000,
}

# Special config for Phase 10 (final phase with harder minimax)
PHASE_10_CONFIG = {
    'minimax': 0.30,   # 30% minimax
    'self': 0.65,      # 65% self-play (vs clone)
    'random': 0.05,    # 5% random
    'selfplay_winrate_threshold': 0.80,
    'selfplay_winrate_games': 500,
    'minimax_min_depth': 1,
    'minimax_max_depth': 6,  # D1-D6 for final phase
    'graduation_episodes': 1_000_000,
}


# Define all phases
PHASE_CONFIGS = {
    Phase.PHASE_1: PhaseConfig(
        phase=Phase.PHASE_1,
        description="Warmup: random 3-9 stones jumping vs random",
        stones_per_player=3,  # Base value, actual is random 3-9
        start_phase='jumping',
        opponent_type='random',
        lr_start=3e-4,
        lr_end=1e-4,
        win_reward_base=1.0,
        win_reward_speed_bonus=0.5,
        loss_reward=-1.0,
        draw_penalty=-0.8,
        shaping_multiplier=1.0,
        win_rate_threshold=0.85,
        min_games_for_graduation=2000,
        min_episodes=200_000,  # At least 200k episodes before graduation
        lr_decay_episodes=500_000,
    ),

    Phase.PHASE_2: PhaseConfig(
        phase=Phase.PHASE_2,
        description="3 stones jumping, mixed opponents",
        stones_per_player=3,
        start_phase='jumping',
        opponent_type='mixed',
        lr_start=1e-4,
        lr_end=5e-5,
        win_reward_base=1.0,
        win_reward_speed_bonus=0.5,
        loss_reward=-1.0,
        draw_penalty=-0.8,
        shaping_multiplier=0.7,
        win_rate_threshold=0.80,
        min_games_for_graduation=1000,
        lr_decay_episodes=500_000,
    ),

    Phase.PHASE_3: PhaseConfig(
        phase=Phase.PHASE_3,
        description="4 stones moving, mixed opponents",
        stones_per_player=4,
        start_phase='moving',
        opponent_type='mixed',
        lr_start=5e-5,
        lr_end=3e-5,
        win_reward_base=1.0,
        win_reward_speed_bonus=0.5,
        loss_reward=-1.0,
        draw_penalty=-0.8,
        shaping_multiplier=0.5,
        win_rate_threshold=0.75,
        min_games_for_graduation=1000,
        lr_decay_episodes=500_000,
    ),

    Phase.PHASE_4: PhaseConfig(
        phase=Phase.PHASE_4,
        description="5 stones moving, mixed opponents",
        stones_per_player=5,
        start_phase='moving',
        opponent_type='mixed',
        lr_start=3e-5,
        lr_end=2e-5,
        win_reward_base=1.0,
        win_reward_speed_bonus=0.5,
        loss_reward=-1.0,
        draw_penalty=-0.8,
        shaping_multiplier=0.3,
        win_rate_threshold=0.70,
        min_games_for_graduation=1000,
        lr_decay_episodes=500_000,
    ),

    Phase.PHASE_5: PhaseConfig(
        phase=Phase.PHASE_5,
        description="6 stones moving, mixed opponents",
        stones_per_player=6,
        start_phase='moving',
        opponent_type='mixed',
        lr_start=2e-5,
        lr_end=1e-5,
        win_reward_base=1.0,
        win_reward_speed_bonus=0.5,
        loss_reward=-1.0,
        draw_penalty=-0.8,
        shaping_multiplier=0.2,
        win_rate_threshold=0.65,
        min_games_for_graduation=1000,
        lr_decay_episodes=500_000,
    ),

    Phase.PHASE_6: PhaseConfig(
        phase=Phase.PHASE_6,
        description="7 stones moving, mixed opponents",
        stones_per_player=7,
        start_phase='moving',
        opponent_type='mixed',
        lr_start=1e-5,
        lr_end=7e-6,
        win_reward_base=1.0,
        win_reward_speed_bonus=0.5,
        loss_reward=-1.0,
        draw_penalty=-0.8,
        shaping_multiplier=0.1,
        win_rate_threshold=0.60,
        min_games_for_graduation=1000,
        lr_decay_episodes=500_000,
    ),

    Phase.PHASE_7: PhaseConfig(
        phase=Phase.PHASE_7,
        description="8 stones moving, mixed opponents",
        stones_per_player=8,
        start_phase='moving',
        opponent_type='mixed',
        lr_start=7e-6,
        lr_end=5e-6,
        win_reward_base=1.0,
        win_reward_speed_bonus=0.5,
        loss_reward=-1.0,
        draw_penalty=-0.8,
        shaping_multiplier=0.05,
        win_rate_threshold=0.55,
        min_games_for_graduation=1000,
        lr_decay_episodes=500_000,
    ),

    Phase.PHASE_8: PhaseConfig(
        phase=Phase.PHASE_8,
        description="9 stones moving, mixed opponents",
        stones_per_player=9,
        start_phase='moving',
        opponent_type='mixed',
        lr_start=5e-6,
        lr_end=3e-6,
        win_reward_base=1.0,
        win_reward_speed_bonus=0.5,
        loss_reward=-1.0,
        draw_penalty=-0.8,
        shaping_multiplier=0.0,
        win_rate_threshold=0.50,
        min_games_for_graduation=1000,
        lr_decay_episodes=500_000,
    ),

    Phase.PHASE_9: PhaseConfig(
        phase=Phase.PHASE_9,
        description="Full game (9 stones from placing), mixed opponents",
        stones_per_player=9,
        start_phase='placing',
        opponent_type='mixed',
        lr_start=3e-6,
        lr_end=1e-6,
        win_reward_base=1.0,
        win_reward_speed_bonus=0.5,
        loss_reward=-1.0,
        draw_penalty=-0.8,
        shaping_multiplier=0.0,
        win_rate_threshold=0.50,
        min_games_for_graduation=1000,
        lr_decay_episodes=5_000_000,
    ),

    Phase.PHASE_10: PhaseConfig(
        phase=Phase.PHASE_10,
        description="Final: full game, no shaping, D1-D6 minimax",
        stones_per_player=9,
        start_phase='placing',
        opponent_type='mixed',
        lr_start=1e-6,
        lr_end=5e-7,
        win_reward_base=1.0,
        win_reward_speed_bonus=0.5,
        loss_reward=-1.0,
        draw_penalty=-0.8,
        shaping_multiplier=0.0,  # No shaping rewards
        win_rate_threshold=0.50,
        min_games_for_graduation=1000,
        max_episodes=1_000_000,  # Fixed 1M episodes
        lr_decay_episodes=1_000_000,
    ),
}


@dataclass
class MixedTrainingState:
    """State for mixed opponent training (Phase 2+)."""
    total_episodes: int = 0

    # Self-play tracking (maxlen matches selfplay_winrate_games)
    selfplay_results: deque = field(default_factory=lambda: deque(maxlen=500))
    clone_generation: int = 0

    # Minimax tracking (simplified - no progressive rounds)
    minimax_wins_by_depth: Dict[int, int] = field(default_factory=lambda: {d: 0 for d in range(1, 7)})

    # Per-opponent game counts
    games_vs_random: int = 0
    games_vs_minimax: int = 0
    games_vs_self: int = 0

    # Win tracking for last 500 games per opponent type
    results_vs_random: deque = field(default_factory=lambda: deque(maxlen=500))
    results_vs_minimax_d1: deque = field(default_factory=lambda: deque(maxlen=500))
    results_vs_minimax_d2: deque = field(default_factory=lambda: deque(maxlen=500))
    results_vs_self: deque = field(default_factory=lambda: deque(maxlen=500))

    def get_selfplay_win_rate(self) -> float:
        """Get win rate from last N self-play games."""
        if len(self.selfplay_results) < 50:
            return 0.5  # Not enough data
        wins = sum(1 for r in self.selfplay_results if r == 'win')
        return wins / len(self.selfplay_results)

    def should_update_clone(self) -> bool:
        """Check if clone should be updated (80% WR over 500 games)."""
        if len(self.selfplay_results) < MIXED_CONFIG['selfplay_winrate_games']:
            return False
        return self.get_selfplay_win_rate() >= MIXED_CONFIG['selfplay_winrate_threshold']

    def on_clone_updated(self):
        """Called when clone is updated."""
        self.selfplay_results.clear()
        self.clone_generation += 1

    def record_minimax_result(self, depth: int, won: bool):
        """Record a minimax game result."""
        if won:
            self.minimax_wins_by_depth[depth] = self.minimax_wins_by_depth.get(depth, 0) + 1

    def get_win_rate_vs_opponent(self, opponent_type: str, depth: int = 0) -> float:
        """Get win rate vs specific opponent type from last 500 games."""
        if opponent_type == 'random':
            results = self.results_vs_random
        elif opponent_type == 'minimax' and depth == 1:
            results = self.results_vs_minimax_d1
        elif opponent_type == 'minimax' and depth == 2:
            results = self.results_vs_minimax_d2
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
        elif opponent_type == 'minimax' and depth == 1:
            self.results_vs_minimax_d1.append(result_str)
        elif opponent_type == 'minimax' and depth == 2:
            self.results_vs_minimax_d2.append(result_str)
        elif opponent_type == 'self':
            self.results_vs_self.append(result_str)


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

    def get_config(self) -> PhaseConfig:
        """Get current phase configuration."""
        if self.current_phase == Phase.COMPLETED:
            return PHASE_CONFIGS[Phase.PHASE_9]
        return PHASE_CONFIGS[self.current_phase]

    def get_game_settings(self) -> Tuple[int, str]:
        """Get stones_per_player and start_phase for current phase."""
        config = self.get_config()
        return config.stones_per_player, config.start_phase

    def get_random_moves_for_phase(self) -> int:
        """
        Get the number of random moves to prepare the board for current phase.

        - Phase 1-2: 150 random moves
        - Phase 3-8: Linear decrease from 150 to 0
          Phase 3: ~129, Phase 4: ~107, Phase 5: ~86, Phase 6: ~64, Phase 7: ~43, Phase 8: ~21
        - Phase 9: 0 random moves
        - Phase 10: Random between 0 and 150 (returned as -1 to signal random)

        Returns:
            Number of random moves, or -1 for Phase 10 (meaning pick random 0-150)
        """
        if self.current_phase == Phase.COMPLETED:
            return 0

        phase_num = int(self.current_phase)

        # Phase 1-2: 150 random moves
        if phase_num <= 2:
            return 150

        # Phase 9: 0 random moves
        if phase_num == 9:
            return 0

        # Phase 10: Random between 0 and 150 (signal with -1)
        if phase_num == 10:
            return -1

        # Phase 3-8: Linear decrease from 150 to 0
        # Phase 3 starts at ~129, Phase 8 ends at ~21
        # Linear interpolation: phase 3 -> 150*(6/7), phase 8 -> 150*(1/7)
        # Formula: 150 * (9 - phase) / 7
        moves = int(150 * (9 - phase_num) / 7)
        return max(0, moves)

    def get_learning_rate(self) -> float:
        """Get current learning rate."""
        config = self.get_config()
        progress = min(1.0, self.stats.episodes_in_phase / config.lr_decay_episodes)
        return config.lr_start + progress * (config.lr_end - config.lr_start)

    def get_shaping_multiplier(self) -> float:
        """
        Get current shaping multiplier based on phase and progress.

        - Phase 1: Static 1.0 (full shaping)
        - Phase 2-9: Starts at 1.0, linearly decreases to 0 at 3/4 of phase,
                     then stays at 0 for the last 1/4. Resets each phase.
        - Phase 10: Always 0.0 (no shaping)
        """
        if self.current_phase == Phase.COMPLETED:
            return 0.0

        if self.current_phase == Phase.PHASE_1:
            return 1.0

        if self.current_phase == Phase.PHASE_10:
            return 0.0

        # Phase 2-9: Dynamic shaping based on progress
        graduation_episodes = MIXED_CONFIG['graduation_episodes']
        progress = self.mixed_state.total_episodes / graduation_episodes

        # First 3/4: linear decay from 1.0 to 0.0
        # Last 1/4: stay at 0.0
        if progress >= 0.75:
            return 0.0
        else:
            # Linear interpolation: 1.0 at progress=0, 0.0 at progress=0.75
            return 1.0 - (progress / 0.75)

    def get_reward_config(self) -> Dict[str, float]:
        """Get current reward configuration."""
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
            'double_mill_extra_reward': config.double_mill_extra_reward * mult,
            'setup_capture_reward': config.setup_capture_reward * mult,
        }

    def select_opponent_for_game(self) -> Tuple[str, int]:
        """
        Select opponent for next game.
        Returns (opponent_type, minimax_depth).

        For Phase 1: always ('random', 0)
        For Phase 2+: mixed selection with random minimax depth
        """
        config = self.get_config()

        if config.opponent_type == 'random':
            return ('random', 0)

        # Get config based on phase (Phase 10 uses harder minimax)
        if self.current_phase == Phase.PHASE_10:
            mix_config = PHASE_10_CONFIG
        else:
            mix_config = MIXED_CONFIG

        # Mixed opponent selection
        mix = mix_config['opponent_mix']
        roll = np.random.random()

        if roll < mix['minimax']:
            # Random depth selection (no progressive rounds)
            min_depth = mix_config['minimax_min_depth']
            max_depth = mix_config['minimax_max_depth']
            depth = np.random.randint(min_depth, max_depth + 1)
            return ('minimax', depth)
        elif roll < mix['minimax'] + mix['self']:
            return ('self', 0)
        else:
            return ('random', 0)

    def add_game_result(self, result: float, opponent_type: str = 'random', minimax_depth: int = 0):
        """
        Add a game result.
        result > 0.5 = win, result < -0.9 = loss, else = draw

        Note: draw_penalty is typically -0.8, loss_reward is -1.0
        So draws fall in range [-0.9, 0.5]
        """
        self.total_episodes += 1
        self.stats.episodes_in_phase += 1

        # Convert result to string
        # Win: typically +1.0 to +2.0 (base + speed bonus)
        # Draw: typically -0.8 (draw_penalty)
        # Loss: typically -1.0 (loss_reward)
        if result > 0.5:
            result_str = 'win'
        elif result < -0.9:
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
        print(f"  Clone updated to generation {gen}")

        for callback in self.on_clone_update_callbacks:
            callback()

        self.save_state()

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

        # Phase 1: Need win rate threshold vs random AND min_episodes
        if config.opponent_type == 'random':
            return stats.get_win_rate() >= config.win_rate_threshold

        # Phase 10: Fixed 1M episodes
        if self.current_phase == Phase.PHASE_10:
            return self.mixed_state.total_episodes >= PHASE_10_CONFIG['graduation_episodes']

        # Phase 2-9: Episode limit based graduation
        if self.mixed_state.total_episodes >= MIXED_CONFIG['graduation_episodes']:
            return True

        return False

    def graduate(self) -> bool:
        """Move to next phase. Returns True if graduated."""
        if self.current_phase == Phase.COMPLETED:
            return False

        # Save phase history
        self.phase_history.append({
            'phase': int(self.current_phase),
            'episodes': self.stats.episodes_in_phase,
            'total_games': self.stats.total_games,
            'wins': self.stats.wins,
            'losses': self.stats.losses,
            'draws': self.stats.draws,
            'best_win_rate': self.stats.best_win_rate,
            'clone_generations': self.mixed_state.clone_generation,
            'duration_seconds': time.time() - self.stats.phase_start_time,
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
        print(f"  GRADUATED from Phase {int(old_phase)} to Phase {int(self.current_phase)}!")
        if new_config:
            print(f"  {new_config.description}")
            print(f"  Stones: {new_config.stones_per_player}, Start: {new_config.start_phase}")
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

        parts = [
            f"Phase {int(self.current_phase)}/10",
            f"{config.stones_per_player}s/{config.start_phase[:4]}",
            f"WR:{wr:.0%}",
            f"Shape:{shaping_mult:.2f}",
        ]

        if config.opponent_type == 'mixed':
            ms = self.mixed_state
            parts.append(f"Clone:{ms.clone_generation}")

        return " | ".join(parts)

    def get_opponent_win_rates(self) -> Dict[str, float]:
        """Get win rates vs each opponent type (last 500 games)."""
        ms = self.mixed_state
        return {
            'wr_vs_mm_d1': ms.get_win_rate_vs_opponent('minimax', 1),
            'wr_vs_mm_d2': ms.get_win_rate_vs_opponent('minimax', 2),
            'wr_vs_random': ms.get_win_rate_vs_opponent('random'),
            'wr_vs_self': ms.get_win_rate_vs_opponent('self'),
        }

    def save_state(self, path: Optional[str] = None):
        """Save curriculum state to file."""
        if path is None:
            path = os.path.join(self.save_dir, "curriculum_state.json")

        state = {
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
            },
            'phase_history': self.phase_history,
            'mixed_state': {
                'total_episodes': self.mixed_state.total_episodes,
                'clone_generation': self.mixed_state.clone_generation,
                'games_vs_random': self.mixed_state.games_vs_random,
                'games_vs_minimax': self.mixed_state.games_vs_minimax,
                'games_vs_self': self.mixed_state.games_vs_self,
                'minimax_wins_by_depth': dict(self.mixed_state.minimax_wins_by_depth),
            },
        }

        with open(path, 'w') as f:
            json.dump(state, f, indent=2)

    def load_state(self, path: Optional[str] = None) -> bool:
        """Load curriculum state from file."""
        if path is None:
            path = os.path.join(self.save_dir, "curriculum_state.json")

        if not os.path.exists(path):
            return False

        try:
            with open(path, 'r') as f:
                state = json.load(f)

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
            )

            if 'mixed_state' in state:
                ms = state['mixed_state']
                self.mixed_state = MixedTrainingState(
                    total_episodes=ms.get('total_episodes', 0),
                    clone_generation=ms.get('clone_generation', 0),
                    games_vs_random=ms.get('games_vs_random', 0),
                    games_vs_minimax=ms.get('games_vs_minimax', 0),
                    games_vs_self=ms.get('games_vs_self', 0),
                )
                if 'minimax_wins_by_depth' in ms:
                    self.mixed_state.minimax_wins_by_depth = {
                        int(k): v for k, v in ms['minimax_wins_by_depth'].items()
                    }

            config = self.get_config()
            print(f"  Loaded curriculum: Phase {int(self.current_phase)}, {self.total_episodes:,} episodes")
            print(f"  Game: {config.stones_per_player} stones, {config.start_phase}")
            return True
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
            print(f"  Duration: {hours:.1f}h")

        if self.current_phase != Phase.COMPLETED:
            print(f"\nCurrent: Phase {int(self.current_phase)} - {self.get_config().description}")
            print(f"  Progress: {self.get_status_string()}")
        else:
            print("\n  TRAINING COMPLETED!")

        print("="*60 + "\n")


def calculate_win_reward(steps: int, max_steps: int = 200) -> float:
    """Calculate win reward based on game length."""
    if steps < 50:
        return 2.0
    elif steps < 100:
        return 1.5 + 0.5 * (100 - steps) / 50
    elif steps < 150:
        return 1.0 + 0.5 * (150 - steps) / 50
    else:
        return 1.0


def calculate_loss_penalty(steps: int, base_penalty: float = -1.5) -> float:
    """Calculate loss penalty based on game length."""
    if steps < 30:
        return base_penalty * 1.33
    elif steps < 60:
        return base_penalty
    else:
        return base_penalty * 0.67
