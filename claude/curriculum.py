"""
Nine Men's Morris - Curriculum Manager
Handles phased training with automatic progression
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
    PHASE_1_RANDOM = 1       # Learn basics vs random
    PHASE_2_MINIMAX_EASY = 2 # Beat minimax D1-D2
    PHASE_3_REDUCED_REWARDS = 3  # Less reward shaping vs D2-D3
    PHASE_4_SPARSE_REWARDS = 4   # Win/loss only vs D3-D4
    PHASE_5_SELF_PLAY = 5    # Refinement via self-play
    COMPLETED = 6


@dataclass
class PhaseConfig:
    """Configuration for a single training phase."""
    phase: Phase
    description: str
    
    # Opponent settings
    opponent_type: str  # 'random', 'minimax', 'self'
    minimax_start_depth: int = 1
    minimax_max_depth: int = 1
    
    # Learning rate
    lr_start: float = 3e-4
    lr_end: float = 7e-5
    
    # Reward multipliers (applied to base rewards)
    win_reward_base: float = 1.0
    win_reward_speed_bonus: float = 1.0  # Extra for fast wins
    loss_reward: float = -1.0
    draw_penalty: float = -0.5
    
    # Shaping reward multiplier (0.0 = no shaping, 1.0 = full shaping)
    shaping_multiplier: float = 1.0
    
    # Base shaping rewards (multiplied by shaping_multiplier)
    mill_reward: float = 0.3
    enemy_mill_penalty: float = -0.3
    block_mill_reward: float = 0.2
    double_mill_reward: float = 0.5
    double_mill_extra_reward: float = 0.8
    setup_capture_reward: float = 0.2
    
    # Graduation criteria
    win_rate_threshold: float = 0.95
    min_games_for_graduation: int = 500
    minimax_depth_to_beat: int = 0  # 0 means not applicable
    no_loss_streak_required: int = 0  # 0 means not required
    
    # Duration limits
    max_episodes: int = 0  # 0 means no limit
    lr_decay_episodes: int = 500_000  # Episodes over which to decay LR (not tied to win rate)


# Define all phases
PHASE_CONFIGS = {
    Phase.PHASE_1_RANDOM: PhaseConfig(
        phase=Phase.PHASE_1_RANDOM,
        description="Learning basics against random opponent",
        opponent_type='random',
        lr_start=3e-4,
        lr_end=1e-4,
        win_reward_base=1.0,
        win_reward_speed_bonus=0.5,
        loss_reward=-1.0,
        draw_penalty=-0.8,
        shaping_multiplier=1.0,
        mill_reward=0.2,
        enemy_mill_penalty=-0.2,
        block_mill_reward=0.2,
        double_mill_reward=0.3,
        double_mill_extra_reward=0.3,
        setup_capture_reward=0.1,
        win_rate_threshold=0.92,
        min_games_for_graduation=1000,
        lr_decay_episodes=1_000_000,  # Decay LR over 1M episodes (keeps it high early)
    ),
    
    Phase.PHASE_2_MINIMAX_EASY: PhaseConfig(
        phase=Phase.PHASE_2_MINIMAX_EASY,
        description="Learning strategy against weak minimax (D1-D2)",
        opponent_type='minimax',
        minimax_start_depth=1,
        minimax_max_depth=2,
        lr_start=1e-4,  # Start where Phase 1 ended
        lr_end=5e-5,
        win_reward_base=1.0,
        win_reward_speed_bonus=0.5,
        loss_reward=-1.0,
        draw_penalty=-0.8,
        shaping_multiplier=0.7,
        mill_reward=0.2,
        enemy_mill_penalty=-0.2,
        block_mill_reward=0.2,
        double_mill_reward=0.3,
        double_mill_extra_reward=0.3,
        setup_capture_reward=0.1,
        win_rate_threshold=0.80,
        min_games_for_graduation=500,
        minimax_depth_to_beat=2,
        lr_decay_episodes=500_000,  # Decay over 500k episodes
    ),
    
    Phase.PHASE_3_REDUCED_REWARDS: PhaseConfig(
        phase=Phase.PHASE_3_REDUCED_REWARDS,
        description="Reducing reward dependency against minimax (D2-D3)",
        opponent_type='minimax',
        minimax_start_depth=1,
        minimax_max_depth=3,
        lr_start=5e-5,  # Start where Phase 2 ended
        lr_end=1e-5,
        win_reward_base=1.0,
        win_reward_speed_bonus=0.5,
        loss_reward=-1.0,
        draw_penalty=-0.8,
        shaping_multiplier=0.3,
        mill_reward=0.2,
        enemy_mill_penalty=-0.2,
        block_mill_reward=0.2,
        double_mill_reward=0.3,
        double_mill_extra_reward=0.3,
        setup_capture_reward=0.1,
        win_rate_threshold=0.75,
        min_games_for_graduation=500,
        minimax_depth_to_beat=3,
        lr_decay_episodes=500_000,  # Decay over 500k episodes
    ),
    
    Phase.PHASE_4_SPARSE_REWARDS: PhaseConfig(
        phase=Phase.PHASE_4_SPARSE_REWARDS,
        description="Sparse rewards only against minimax (D3-D4)",
        opponent_type='minimax',
        minimax_start_depth=1,
        minimax_max_depth=4,
        lr_start=1e-5,  # Start where Phase 3 ended
        lr_end=5e-6,
        win_reward_base=1.0,
        win_reward_speed_bonus=0.5,
        loss_reward=-1.0,
        draw_penalty=-0.8,
        shaping_multiplier=0.1,
        mill_reward=0.2,
        enemy_mill_penalty=0.2,   # Positive penalty as per table (0.2)
        block_mill_reward=0.2,
        double_mill_reward=0.3,
        double_mill_extra_reward=0.3,
        setup_capture_reward=0.1,
        win_rate_threshold=0.70,
        min_games_for_graduation=500,
        minimax_depth_to_beat=4,
        lr_decay_episodes=500_000,  # Decay over 500k episodes
    ),
    
    Phase.PHASE_5_SELF_PLAY: PhaseConfig(
        phase=Phase.PHASE_5_SELF_PLAY,
        description="Self-play evolution (100 generations)",
        opponent_type='self',
        lr_start=5e-6,  # Start where Phase 4 ended
        lr_end=1e-6,
        win_reward_base=1.0,
        win_reward_speed_bonus=0.5,
        loss_reward=-1.0,
        draw_penalty=-0.8,
        shaping_multiplier=0.0,
        mill_reward=0.2,
        enemy_mill_penalty=0.2,   # Positive penalty as per table (0.2)
        block_mill_reward=0.2,
        double_mill_reward=0.3,
        double_mill_extra_reward=0.3,
        setup_capture_reward=0.1,
        win_rate_threshold=0.95,  # 90% win rate to clone
        min_games_for_graduation=500,  # Min games before checking win rate
        max_episodes=0,  # No max - controlled by generations
        lr_decay_episodes=5_000_000,  # Decay over 5M episodes (long self-play phase)
    ),
}


# Phase 5 Mixed Opponent Training Configuration
PHASE5_CONFIG = {
    # Mixed opponent distribution (must sum to 1.0)
    'opponent_mix': {
        'self': 0.50,      # 50% self-play (vs clone)
        'minimax': 0.35,   # 35% minimax (cycling D1-D6)
        'random': 0.15,    # 15% random (to prevent overfitting)
    },

    # Minimax cycling: train D1 until mastery, then D2, etc.
    'minimax_mastery_threshold': 0.90,   # 90% WR to move to next depth
    'minimax_mastery_games': 200,        # Games needed to evaluate mastery
    'minimax_max_depth': 6,              # Maximum depth to train against
    'minimax_cycle_on_mastery': True,    # After D6, cycle back to D1

    # Graduation criteria
    'graduation_minimax_depth': 5,       # Must master D5 to graduate
    'graduation_total_episodes': 5_000_000,  # Or complete 5M episodes

    # Periodic full evaluation
    'full_eval_interval': 250_000,       # Full minimax eval every 250k epochs
    'full_eval_depths': [1, 2, 3, 4, 5, 6],

    # Clone update (for self-play portion)
    'clone_update_interval': 100_000,    # Update clone every 100k epochs
}


@dataclass
class MixedTrainingState:
    """State for Phase 5 mixed opponent training."""
    total_episodes_phase5: int = 0
    episodes_since_clone_update: int = 0
    episodes_since_full_eval: int = 0

    # Current minimax depth being trained
    current_minimax_depth: int = 1
    highest_mastered_depth: int = 0

    # Per-depth tracking for minimax mastery
    minimax_results: Dict = field(default_factory=lambda: {
        d: deque(maxlen=PHASE5_CONFIG['minimax_mastery_games'])
        for d in range(1, PHASE5_CONFIG['minimax_max_depth'] + 1)
    })

    # Self-play tracking
    games_vs_clone: int = 0
    recent_vs_clone: deque = field(default_factory=lambda: deque(maxlen=500))

    # Random opponent tracking
    games_vs_random: int = 0
    recent_vs_random: deque = field(default_factory=lambda: deque(maxlen=200))

    def add_result(self, opponent_type: str, result: str, depth: int = 0):
        """Add a game result for the appropriate opponent type."""
        if opponent_type == 'self':
            self.games_vs_clone += 1
            self.recent_vs_clone.append(result)
        elif opponent_type == 'random':
            self.games_vs_random += 1
            self.recent_vs_random.append(result)
        elif opponent_type == 'minimax' and 1 <= depth <= PHASE5_CONFIG['minimax_max_depth']:
            self.minimax_results[depth].append(result)

    def get_minimax_win_rate(self, depth: int) -> Tuple[float, int]:
        """Get win rate against specific minimax depth. Returns (win_rate, num_games)."""
        if depth not in self.minimax_results:
            return 0.0, 0
        results = self.minimax_results[depth]
        if len(results) == 0:
            return 0.0, 0
        wins = sum(1 for r in results if r == 'win')
        draws = sum(1 for r in results if r == 'draw')
        # Draws count as half a win
        return (wins + 0.5 * draws) / len(results), len(results)

    def get_clone_win_rate(self) -> float:
        """Get win rate against clone."""
        if len(self.recent_vs_clone) < 50:
            return 0.5  # Not enough data
        wins = sum(1 for r in self.recent_vs_clone if r == 'win')
        return wins / len(self.recent_vs_clone)

    def check_minimax_mastery(self) -> bool:
        """Check if current depth is mastered and should advance."""
        depth = self.current_minimax_depth
        win_rate, num_games = self.get_minimax_win_rate(depth)

        if num_games < PHASE5_CONFIG['minimax_mastery_games']:
            return False

        if win_rate >= PHASE5_CONFIG['minimax_mastery_threshold']:
            # Mastered this depth!
            if depth > self.highest_mastered_depth:
                self.highest_mastered_depth = depth
            return True
        return False

    def advance_minimax_depth(self):
        """Advance to next minimax depth (with cycling)."""
        old_depth = self.current_minimax_depth
        max_depth = PHASE5_CONFIG['minimax_max_depth']

        if self.current_minimax_depth >= max_depth:
            if PHASE5_CONFIG['minimax_cycle_on_mastery']:
                self.current_minimax_depth = 1  # Cycle back
            # else stay at max depth
        else:
            self.current_minimax_depth += 1

        # Clear results for the new depth to get fresh evaluation
        self.minimax_results[self.current_minimax_depth].clear()

        print(f"📈 Advanced from D{old_depth} to D{self.current_minimax_depth} "
              f"(highest mastered: D{self.highest_mastered_depth})")

    def should_update_clone(self) -> bool:
        """Check if clone should be updated."""
        return self.episodes_since_clone_update >= PHASE5_CONFIG['clone_update_interval']

    def should_do_full_eval(self) -> bool:
        """Check if full minimax evaluation should run."""
        return self.episodes_since_full_eval >= PHASE5_CONFIG['full_eval_interval']

    def is_graduated(self) -> bool:
        """Check if Phase 5 is complete."""
        # Graduate if mastered graduation depth OR hit episode limit
        if self.highest_mastered_depth >= PHASE5_CONFIG['graduation_minimax_depth']:
            return True
        if self.total_episodes_phase5 >= PHASE5_CONFIG['graduation_total_episodes']:
            return True
        return False


@dataclass
class PhaseStats:
    """Statistics for current phase."""
    phase: Phase
    episodes_in_phase: int = 0
    total_games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    current_minimax_depth: int = 1
    best_win_rate: float = 0.0
    games_since_last_loss: int = 0
    phase_start_time: float = field(default_factory=time.time)

    # Rolling stats - must be >= max graduation requirement
    recent_results: deque = field(default_factory=lambda: deque(maxlen=2000))
    
    def add_result(self, result: str):
        """Add game result: 'win', 'loss', or 'draw'."""
        self.total_games += 1
        self.recent_results.append(result)
        
        if result == 'win':
            self.wins += 1
            self.games_since_last_loss += 1
        elif result == 'loss':
            self.losses += 1
            self.games_since_last_loss = 0
        else:  # draw
            self.draws += 1
            self.games_since_last_loss += 1
        
        # Update best win rate
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

    def __init__(self, start_phase: Phase = Phase.PHASE_1_RANDOM, save_dir: str = "curriculum"):
        self.current_phase = start_phase
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.stats = PhaseStats(phase=start_phase)
        self.phase_history: List[Dict] = []
        self.total_episodes = 0

        # Phase 5 mixed training state
        self.mixed_state = MixedTrainingState()

        # Callbacks
        self.on_phase_change_callbacks = []
        self.on_clone_update_callbacks = []  # Called when clone should be updated
        self.on_depth_change_callbacks = []  # Called when minimax depth changes
    
    def get_config(self) -> PhaseConfig:
        """Get current phase configuration."""
        if self.current_phase == Phase.COMPLETED:
            # Return Phase 5 config as a fallback when training is complete
            return PHASE_CONFIGS[Phase.PHASE_5_SELF_PLAY]
        return PHASE_CONFIGS[self.current_phase]
    
    def get_learning_rate(self) -> float:
        """Get current learning rate based on episode count (not win rate)."""
        config = self.get_config()

        # Linear decay based on episodes in phase (like random_train)
        # This keeps LR high during early learning instead of dropping it
        # when win rate is still low
        progress = min(1.0, self.stats.episodes_in_phase / config.lr_decay_episodes)

        lr = config.lr_start + progress * (config.lr_end - config.lr_start)
        return lr
    
    def get_minimax_depth(self) -> int:
        """Get current minimax depth for the phase."""
        config = self.get_config()
        if config.opponent_type != 'minimax':
            return 1
        return self.stats.current_minimax_depth
    
    def get_reward_config(self) -> Dict[str, float]:
        """Get current reward configuration."""
        config = self.get_config()
        mult = config.shaping_multiplier
        
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
    
    def add_game_result(self, result: float, opponent_type: str = 'random', minimax_depth: int = 0):
        """
        Add a game result.
        result > 0.5 = win, result < -0.5 = loss, else = draw
        opponent_type: 'random', 'minimax', or 'self'
        minimax_depth: depth if opponent_type is 'minimax'
        """
        self.total_episodes += 1
        self.stats.episodes_in_phase += 1

        # Convert result to string
        if result > 0.5:
            result_str = 'win'
        elif result < -0.5:
            result_str = 'loss'
        else:
            result_str = 'draw'

        # Phase 5: Track mixed training
        if self.current_phase == Phase.PHASE_5_SELF_PLAY:
            self.mixed_state.total_episodes_phase5 += 1
            self.mixed_state.episodes_since_clone_update += 1
            self.mixed_state.episodes_since_full_eval += 1
            self.mixed_state.add_result(opponent_type, result_str, minimax_depth)
            return

        # For other phases, only track non-self-play results
        if opponent_type != 'self':
            self.stats.add_result(result_str)

    # ========== Phase 5 Mixed Training Methods ==========

    def get_phase5_opponent(self) -> Tuple[str, int]:
        """
        Get opponent type for Phase 5 mixed training.
        Returns (opponent_type, minimax_depth)
        """
        if self.current_phase != Phase.PHASE_5_SELF_PLAY:
            return ('random', 0)

        mix = PHASE5_CONFIG['opponent_mix']
        roll = np.random.random()

        if roll < mix['self']:
            return ('self', 0)
        elif roll < mix['self'] + mix['minimax']:
            return ('minimax', self.mixed_state.current_minimax_depth)
        else:
            return ('random', 0)

    def should_update_clone(self) -> bool:
        """Check if clone model should be updated."""
        if self.current_phase != Phase.PHASE_5_SELF_PLAY:
            return False
        return self.mixed_state.should_update_clone()

    def do_clone_update(self):
        """Called when clone is updated - reset counter."""
        if self.current_phase != Phase.PHASE_5_SELF_PLAY:
            return

        self.mixed_state.episodes_since_clone_update = 0
        self.mixed_state.recent_vs_clone.clear()

        print(f"🔄 Clone model updated (every {PHASE5_CONFIG['clone_update_interval']:,} epochs)")

        # Notify callbacks (trainer should update the clone)
        for callback in self.on_clone_update_callbacks:
            callback()

        self.save_state()

    def should_do_full_eval(self) -> bool:
        """Check if it's time for full minimax evaluation."""
        if self.current_phase != Phase.PHASE_5_SELF_PLAY:
            return False
        return self.mixed_state.should_do_full_eval()

    def reset_full_eval_counter(self):
        """Reset the full eval counter after evaluation."""
        self.mixed_state.episodes_since_full_eval = 0

    def check_minimax_mastery(self) -> bool:
        """Check if current minimax depth is mastered and advance if so."""
        if self.current_phase != Phase.PHASE_5_SELF_PLAY:
            return False

        if self.mixed_state.check_minimax_mastery():
            old_depth = self.mixed_state.current_minimax_depth
            self.mixed_state.advance_minimax_depth()
            new_depth = self.mixed_state.current_minimax_depth

            # Notify callbacks
            for callback in self.on_depth_change_callbacks:
                callback(old_depth, new_depth)

            self.save_state()
            return True
        return False

    def get_current_minimax_depth(self) -> int:
        """Get current minimax depth for Phase 5."""
        return self.mixed_state.current_minimax_depth

    def get_highest_mastered_depth(self) -> int:
        """Get highest mastered minimax depth."""
        return self.mixed_state.highest_mastered_depth

    def is_phase5_completed(self) -> bool:
        """Check if Phase 5 is complete."""
        if self.current_phase != Phase.PHASE_5_SELF_PLAY:
            return False
        return self.mixed_state.is_graduated()
    
    def should_graduate(self) -> bool:
        """Check if ready to move to next phase."""
        if self.current_phase == Phase.COMPLETED:
            return False

        # Phase 5: Graduate when 100 generations complete
        if self.current_phase == Phase.PHASE_5_SELF_PLAY:
            return self.is_phase5_completed()

        config = self.get_config()
        stats = self.stats

        # Check minimum games
        if len(stats.recent_results) < config.min_games_for_graduation:
            return False

        # Check max episodes (for self-play phase)
        if config.max_episodes > 0 and stats.episodes_in_phase >= config.max_episodes:
            return True

        # Check win rate threshold
        win_rate = stats.get_win_rate()

        # For minimax phases at higher depths, adjust threshold based on depth
        # Higher depths have more draws, making high WR thresholds unrealistic
        effective_threshold = config.win_rate_threshold
        if config.opponent_type == 'minimax' and stats.current_minimax_depth >= 3:
            # At D3+, 60% WR (with draws counting as 0.5) shows solid mastery
            # This means: mostly wins + some draws, few losses
            effective_threshold = min(config.win_rate_threshold, 0.60)

        if win_rate < effective_threshold:
            return False

        # For minimax phases: must be at target depth with good win rate
        if config.minimax_depth_to_beat > 0:
            if stats.current_minimax_depth < config.minimax_depth_to_beat:
                return False  # Haven't reached target depth yet

        # Check no-loss streak requirement
        if config.no_loss_streak_required > 0:
            if stats.games_since_last_loss < config.no_loss_streak_required:
                return False

        return True
    
    def graduate(self) -> bool:
        """Move to next phase. Returns True if graduated, False if already completed."""
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
            'final_minimax_depth': self.stats.current_minimax_depth,
            'duration_seconds': time.time() - self.stats.phase_start_time,
        })
        
        old_phase = self.current_phase
        
        # Move to next phase
        if self.current_phase == Phase.PHASE_5_SELF_PLAY:
            self.current_phase = Phase.COMPLETED
        else:
            self.current_phase = Phase(int(self.current_phase) + 1)
        
        # Reset stats for new phase
        new_config = PHASE_CONFIGS.get(self.current_phase)
        self.stats = PhaseStats(
            phase=self.current_phase,
            current_minimax_depth=new_config.minimax_start_depth if new_config else 1
        )
        
        # Notify callbacks
        for callback in self.on_phase_change_callbacks:
            callback(old_phase, self.current_phase)
        
        print(f"\n{'='*60}")
        print(f"🎓 GRADUATED from Phase {int(old_phase)} to Phase {int(self.current_phase)}!")
        if new_config:
            print(f"   {new_config.description}")
        print(f"{'='*60}\n")
        
        self.save_state()
        return True
    
    def check_and_graduate(self) -> bool:
        """Check graduation criteria and graduate if met."""
        if self.current_phase == Phase.COMPLETED:
            return False
        
        # First, check if we should promote minimax depth (for minimax phases)
        config = self.get_config()
        if config.opponent_type == 'minimax':
            self._check_depth_promotion()
        
        # Now check graduation
        if self.should_graduate():
            return self.graduate()
        return False
    
    def _check_depth_promotion(self):
        """Check if we should increase minimax depth."""
        config = self.get_config()
        stats = self.stats

        # Need enough games to evaluate
        if len(stats.recent_results) < 100:
            return

        win_rate = stats.get_win_rate()
        current_depth = stats.current_minimax_depth

        # Already at max depth for this phase
        if current_depth >= config.minimax_max_depth:
            return

        # Promote threshold: 55% WR is enough (draws count as 0.5)
        # At higher depths, games are more likely to draw, so we can't require 75%
        # A 55% WR means winning more than losing, which shows mastery
        promotion_threshold = 0.55

        if win_rate >= promotion_threshold:
            new_depth = current_depth + 1
            stats.current_minimax_depth = new_depth
            stats.recent_results.clear()  # Reset for new depth
            stats.wins = 0
            stats.losses = 0
            stats.draws = 0
            print(f"📈 Promoted to minimax depth {new_depth}! (was {win_rate:.0%} vs D{current_depth})")
    
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
                'current_minimax_depth': self.stats.current_minimax_depth,
                'best_win_rate': self.stats.best_win_rate,
                'games_since_last_loss': self.stats.games_since_last_loss,
            },
            'phase_history': self.phase_history,
            # Phase 5 mixed training state
            'mixed_state': {
                'total_episodes_phase5': self.mixed_state.total_episodes_phase5,
                'episodes_since_clone_update': self.mixed_state.episodes_since_clone_update,
                'episodes_since_full_eval': self.mixed_state.episodes_since_full_eval,
                'current_minimax_depth': self.mixed_state.current_minimax_depth,
                'highest_mastered_depth': self.mixed_state.highest_mastered_depth,
                'games_vs_clone': self.mixed_state.games_vs_clone,
                'games_vs_random': self.mixed_state.games_vs_random,
            },
        }

        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, path: Optional[str] = None) -> bool:
        """Load curriculum state from file. Returns True if loaded successfully."""
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
                current_minimax_depth=stats_data['current_minimax_depth'],
                best_win_rate=stats_data['best_win_rate'],
                games_since_last_loss=stats_data['games_since_last_loss'],
            )

            # Load mixed training state if present
            if 'mixed_state' in state:
                ms = state['mixed_state']
                self.mixed_state = MixedTrainingState(
                    total_episodes_phase5=ms.get('total_episodes_phase5', 0),
                    episodes_since_clone_update=ms.get('episodes_since_clone_update', 0),
                    episodes_since_full_eval=ms.get('episodes_since_full_eval', 0),
                    current_minimax_depth=ms.get('current_minimax_depth', 1),
                    highest_mastered_depth=ms.get('highest_mastered_depth', 0),
                    games_vs_clone=ms.get('games_vs_clone', 0),
                    games_vs_random=ms.get('games_vs_random', 0),
                )

            print(f"✓ Loaded curriculum state: Phase {int(self.current_phase)}, {self.total_episodes} total episodes")
            if self.current_phase == Phase.PHASE_5_SELF_PLAY:
                print(f"  Mixed Training: D{self.mixed_state.current_minimax_depth}, "
                      f"Best: D{self.mixed_state.highest_mastered_depth}, "
                      f"{self.mixed_state.total_episodes_phase5:,} episodes")
            return True
        except Exception as e:
            print(f"✗ Failed to load curriculum state: {e}")
            return False
    
    def get_status_string(self) -> str:
        """Get a status string for logging."""
        # Handle completed phase
        if self.current_phase == Phase.COMPLETED:
            return "Training Complete"

        config = self.get_config()
        stats = self.stats

        # Phase 5: Special status for mixed training
        if self.current_phase == Phase.PHASE_5_SELF_PLAY:
            ms = self.mixed_state
            current_wr, current_games = ms.get_minimax_win_rate(ms.current_minimax_depth)

            parts = [
                f"Phase 5 Mixed",
                f"D{ms.current_minimax_depth}→D{PHASE5_CONFIG['graduation_minimax_depth']}",
                f"Best:D{ms.highest_mastered_depth}",
            ]

            # Current depth progress
            if current_games > 0:
                parts.append(f"D{ms.current_minimax_depth}:{current_wr:.0%}({current_games}g)")

            # Episodes progress
            total = ms.total_episodes_phase5
            target = PHASE5_CONFIG['graduation_total_episodes']
            parts.append(f"Ep:{total/1e6:.1f}M/{target/1e6:.0f}M")

            return " | ".join(parts)

        wr = stats.get_win_rate()
        lr = stats.get_loss_rate()

        parts = [
            f"Phase {int(self.current_phase)}/{int(Phase.PHASE_5_SELF_PLAY)}",
            f"WR:{wr:.0%} LR:{lr:.0%}",
        ]

        if config.opponent_type == 'minimax':
            depth_str = f"D{stats.current_minimax_depth}"
            if config.minimax_depth_to_beat > 0:
                depth_str += f"→D{config.minimax_depth_to_beat}"
            parts.append(depth_str)

        # Show what's needed to graduate
        needs = []
        if wr < config.win_rate_threshold:
            needs.append(f"WR≥{config.win_rate_threshold:.0%}")
        if config.minimax_depth_to_beat > 0 and stats.current_minimax_depth < config.minimax_depth_to_beat:
            needs.append(f"reach D{config.minimax_depth_to_beat}")
        if len(stats.recent_results) < config.min_games_for_graduation:
            needs.append(f"N≥{config.min_games_for_graduation}")

        if needs:
            parts.append(f"Need: {', '.join(needs)}")
        else:
            parts.append("Ready to graduate!")

        return " | ".join(parts)
    
    def print_summary(self):
        """Print training summary."""
        print("\n" + "="*60)
        print("CURRICULUM TRAINING SUMMARY")
        print("="*60)
        
        for phase_data in self.phase_history:
            phase = Phase(phase_data['phase'])
            duration = phase_data['duration_seconds']
            hours = duration / 3600
            
            print(f"\nPhase {int(phase)}: {PHASE_CONFIGS[phase].description}")
            print(f"  Episodes: {phase_data['episodes']:,}")
            print(f"  Games: {phase_data['total_games']} (W:{phase_data['wins']} L:{phase_data['losses']} D:{phase_data['draws']})")
            print(f"  Best WR: {phase_data['best_win_rate']:.1%}")
            if phase_data['final_minimax_depth'] > 1:
                print(f"  Final Minimax Depth: {phase_data['final_minimax_depth']}")
            print(f"  Duration: {hours:.1f}h")
        
        if self.current_phase != Phase.COMPLETED:
            print(f"\nCurrent: Phase {int(self.current_phase)} - {self.get_config().description}")
            print(f"  Progress: {self.get_status_string()}")
        else:
            print("\n✅ TRAINING COMPLETED!")
        
        print("="*60 + "\n")


def calculate_win_reward(steps: int, max_steps: int = 200) -> float:
    """
    Calculate win reward based on game length.
    Fast wins (< 50 moves) get bonus, slow wins get less.
    Returns reward between 1.0 and 2.0
    """
    if steps < 50:
        # Fast win: full bonus
        return 2.0
    elif steps < 100:
        # Medium win: partial bonus
        return 1.5 + 0.5 * (100 - steps) / 50
    elif steps < 150:
        # Slow win: small bonus
        return 1.0 + 0.5 * (150 - steps) / 50
    else:
        # Very slow win: base reward
        return 1.0


def calculate_loss_penalty(steps: int, base_penalty: float = -1.5) -> float:
    """
    Calculate loss penalty based on game length.
    Fast losses are worse (didn't even try).
    Returns penalty between -2.0 and -1.0
    """
    if steps < 30:
        # Very fast loss: harsh penalty
        return base_penalty * 1.33  # e.g., -2.0
    elif steps < 60:
        # Fast loss: normal penalty
        return base_penalty
    else:
        # Slow loss (fought hard): reduced penalty
        return base_penalty * 0.67  # e.g., -1.0
