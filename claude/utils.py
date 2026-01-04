"""
Nine Men's Morris - Utilities
Game helpers, reward calculation, and experience data structures
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
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


def detect_mill(state, player: int) -> bool:
    """
    Check if a mill was just formed by looking at state.
    This is a simplified check - in practice you'd track state changes.
    """
    # Mill detection would require comparing before/after states
    # For now, we detect mills through piece count changes
    return False


def count_mills(state_str: str, player: int) -> int:
    """
    Count the number of mills for a player.
    This is a heuristic based on board patterns.
    """
    # Nine Men's Morris mill positions (simplified)
    # A mill is 3 pieces in a row on the board
    # This would need proper board parsing - placeholder for now
    return 0


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
        Calculate reward shaping for piece captures.
        Simplified to focus on what matters: capturing opponent pieces.
        
        state_info should contain:
        - my_pieces: int
        - opp_pieces: int
        """
        reward = 0.0
        
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
        
        # Note: Enemy captures are now handled in worker.py when opponent moves
        # This gives cleaner credit assignment
        
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


class BoardAnalyzer:
    """
    Analyzes Nine Men's Morris board state for reward shaping.
    """
    
    # Mill positions on the board (groups of 3 that form mills)
    # These are the indices in the standard board representation
    MILL_POSITIONS = [
        # Outer square
        [0, 1, 2], [2, 3, 4], [4, 5, 6], [6, 7, 0],
        # Middle square  
        [8, 9, 10], [10, 11, 12], [12, 13, 14], [14, 15, 8],
        # Inner square
        [16, 17, 18], [18, 19, 20], [20, 21, 22], [22, 23, 16],
        # Connections
        [1, 9, 17], [3, 11, 19], [5, 13, 21], [7, 15, 23],
    ]
    
    @staticmethod
    def parse_board(state_str: str) -> Dict[int, Optional[int]]:
        """
        Parse board state string into position -> player mapping.
        Returns dict with position index -> player (0, 1, or None for empty)
        """
        # This is a simplified parser - actual implementation depends on
        # how pyspiel formats the Nine Men's Morris state string
        board = {}
        
        # Count pieces as a simple heuristic
        for i, char in enumerate(state_str):
            if char in 'oO':
                board[i] = 0
            elif char in 'xX':
                board[i] = 1
        
        return board
    
    @classmethod
    def count_potential_mills(cls, board: Dict[int, Optional[int]], player: int) -> int:
        """Count positions where player is one move away from a mill."""
        potential = 0
        
        for mill in cls.MILL_POSITIONS:
            player_count = sum(1 for pos in mill if board.get(pos) == player)
            empty_count = sum(1 for pos in mill if board.get(pos) is None)
            
            # Two pieces + one empty = potential mill
            if player_count == 2 and empty_count == 1:
                potential += 1
        
        return potential
    
    @classmethod
    def can_block_mill(cls, board: Dict[int, Optional[int]], player: int) -> bool:
        """Check if player can block opponent's potential mill."""
        opponent = 1 - player
        
        for mill in cls.MILL_POSITIONS:
            opp_count = sum(1 for pos in mill if board.get(pos) == opponent)
            empty_count = sum(1 for pos in mill if board.get(pos) is None)
            
            if opp_count == 2 and empty_count == 1:
                return True
        
        return False
