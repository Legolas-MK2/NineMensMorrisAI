"""
Nine Men's Morris - Utilities
Game helpers, reward calculation, and experience data structures
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict
import random
import numpy as np


def get_legal_mask(state, num_actions: int) -> np.ndarray:
    """Create a binary mask of legal actions."""
    mask = np.zeros(num_actions, dtype=np.float32)
    legal = state.legal_actions()
    if legal:
        mask[legal] = 1.0
    return mask


def count_pieces_from_state(state, player: int) -> Tuple[int, int]:
    """
    Count pieces for specified player and opponent directly from game state.
    Returns (my_pieces, opponent_pieces)
    """
    state_str = str(state)
    # In Nine Men's Morris: 'o'/'O' = player 0, 'x'/'X' = player 1
    p0_pieces = state_str.count('o') + state_str.count('O')
    p1_pieces = state_str.count('x') + state_str.count('X')

    if player == 0:
        return p0_pieces, p1_pieces
    else:
        return p1_pieces, p0_pieces


class RewardCalculator:
    """
    Calculates rewards based on curriculum phase settings.
    """
    
    def __init__(self, reward_config: Dict[str, float]):
        self.config = reward_config
    
    def update_config(self, reward_config: Dict[str, float]):
        """Update reward configuration (e.g., when phase changes)."""
        self.config = reward_config
    
    def calculate_terminal_reward(
        self,
        returns: List[float],
        player: int,
        steps: int,
        max_steps: int = 200
    ) -> float:
        """Calculate reward for terminal game state."""
        my_return = returns[player]
        opp_return = returns[1 - player]
        
        if my_return > opp_return:
            # Win - bonus for fast wins
            base = self.config['win_reward_base']
            bonus = self.config['win_reward_speed_bonus']
            
            # Simple speed bonus: 1.0 down to 0.0 based on steps (0-300)
            # Table says bonus is 0.5, curriculum says 0.5 or 1.0.
            # We use the config value.
            speed_bonus = bonus * max(0, 1.0 - (steps / max_steps))
            return base + speed_bonus
                
        elif my_return < opp_return:
            # Loss - penalty (fixed in table)
            return self.config['loss_reward']
        else:
            # Draw
            return self.config['draw_penalty']
    
    def calculate_shaping_reward(
        self,
        prev_state_info: Dict,
        new_state_info: Dict,
        player: int
    ) -> float:
        """
        Calculate reward shaping for piece captures and game progress.

        state_info should contain:
        - my_pieces: int
        - opp_pieces: int
        """
        reward = 0.0

        # Small per-step cost to discourage stalling and draws
        reward += self.config.get('step_penalty', -0.003)

        # Piece changes
        prev_opp = prev_state_info.get('opp_pieces', 0)
        new_opp = new_state_info.get('opp_pieces', 0)

        # Did we capture? (opponent lost a piece = we formed a mill)
        pieces_captured = prev_opp - new_opp
        if pieces_captured > 0:
            # Base mill reward
            reward += self.config['mill_reward'] * pieces_captured

            # Bonus for multiple captures (rare but powerful)
            if pieces_captured >= 2:
                reward += self.config['double_mill_reward']

        # Continuous piece advantage signal — pushes model toward capturing
        my_pieces = new_state_info.get('my_pieces', 0)
        opp_pieces = new_state_info.get('opp_pieces', 0)
        piece_diff = my_pieces - opp_pieces
        reward += piece_diff * self.config.get('piece_advantage_reward', 0.02)

        return reward
    
    def calculate_timeout_penalty(self) -> float:
        """Penalty for game timing out (too long)."""
        return self.config['draw_penalty'] * 0.8  # Slightly better than draw


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Generalized Advantage Estimation for a single episode.
    This must be done per-episode, not across concatenated batches.

    Returns (advantages, returns)
    """
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    last_gae = 0.0

    for t in reversed(range(n)):
        if t == n - 1:
            # Terminal state - no bootstrap value
            next_value = 0.0
        else:
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value - values[t]
        last_gae = delta + gamma * gae_lambda * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


@dataclass
class ExperienceBatch:
    """Batch of experiences from a single episode for one player."""
    obs: np.ndarray
    actions: np.ndarray
    logprobs: np.ndarray
    values: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    masks: np.ndarray
    advantages: np.ndarray  # Pre-computed per-episode GAE advantages
    returns: np.ndarray     # Pre-computed per-episode returns
    game_result: float      # Final reward (win/loss/draw)
    game_steps: int = 0     # Number of steps in the game
    opponent_type: str = 'unknown'  # Track opponent type ('random', 'minimax', 'self')
    minimax_depth: int = 0  # Depth if opponent is minimax


def prepare_game_state(state, random_moves: int):
    """Play random vs random moves to prepare a mid-game board position (not recorded)."""
    moves_made = 0
    while moves_made < random_moves and not state.is_terminal():
        # Stop early if either player is down to 3 stones (about to enter jumping phase)
        try:
            obs = state.observation_tensor(0)
            p0_pieces = sum(1 for i in range(24) if obs[i] == 1)
            p1_pieces = sum(1 for i in range(24) if obs[i + 24] == 1)
            if p0_pieces <= 3 or p1_pieces <= 3:
                break
        except:
            pass

        legal_actions = state.legal_actions()
        if not legal_actions:
            break

        action = random.choice(legal_actions)
        state.apply_action(action)
        moves_made += 1
