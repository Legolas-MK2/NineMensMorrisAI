"""
Nine Men's Morris - Utilities
Game helpers, reward calculation, and experience data structures
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import random
import numpy as np


def get_legal_mask(state, num_actions: int) -> np.ndarray:
    """Create a binary mask of legal actions."""
    mask = np.zeros(num_actions, dtype=np.float32)
    legal = state.legal_actions()
    if legal:
        mask[legal] = 1.0
    return mask


# Board topology and parsing aligned with minimax.py
MILLS: Tuple[Tuple[int, int, int], ...] = (
    (0, 1, 2),
    (0, 9, 21),
    (2, 14, 23),
    (21, 22, 23),
    (3, 4, 5),
    (3, 10, 18),
    (5, 13, 20),
    (18, 19, 20),
    (6, 7, 8),
    (6, 11, 15),
    (8, 12, 17),
    (15, 16, 17),
    (1, 4, 7),
    (9, 10, 11),
    (12, 13, 14),
    (16, 19, 22),
)

ADJACENCY: Tuple[Tuple[int, ...], ...] = (
    (1, 9),
    (0, 2, 4),
    (1, 14),
    (4, 10),
    (1, 3, 5, 7),
    (4, 13),
    (7, 11),
    (4, 6, 8),
    (7, 12),
    (0, 10, 21),
    (3, 9, 11, 18),
    (6, 10, 15),
    (8, 13, 17),
    (5, 12, 14, 20),
    (2, 13, 23),
    (11, 16),
    (15, 17, 19),
    (12, 16),
    (10, 19),
    (16, 18, 20, 22),
    (13, 19),
    (9, 22),
    (19, 21, 23),
    (14, 22),
)

POSITION_TO_MILLS: Tuple[Tuple[Tuple[int, int, int], ...], ...] = tuple(
    tuple(mill for mill in MILLS if pos in mill) for pos in range(24)
)

BOARD_POS_TO_GRID: Tuple[Tuple[int, int], ...] = (
    (0, 0),
    (0, 3),
    (0, 6),
    (1, 1),
    (1, 3),
    (1, 5),
    (2, 2),
    (2, 3),
    (2, 4),
    (3, 0),
    (3, 1),
    (3, 2),
    (3, 4),
    (3, 5),
    (3, 6),
    (4, 2),
    (4, 3),
    (4, 4),
    (5, 1),
    (5, 3),
    (5, 5),
    (6, 0),
    (6, 3),
    (6, 6),
)


def parse_board_from_state(state) -> Tuple[Optional[int], ...]:
    """Parse absolute board (0=player0, 1=player1, None=empty) from state."""
    try:
        obs = state.observation_tensor(0)
        obs_array = np.asarray(obs).reshape(5, 7, 7)
        board = []
        for pos in range(24):
            r, c = BOARD_POS_TO_GRID[pos]
            if obs_array[0, r, c] > 0.5:
                board.append(0)
            elif obs_array[1, r, c] > 0.5:
                board.append(1)
            else:
                board.append(None)
        return tuple(board)
    except Exception:
        return tuple(None for _ in range(24))


def _count_pieces(board: Tuple[Optional[int], ...], player: int) -> int:
    return sum(1 for p in board if p == player)


def _count_mills(board: Tuple[Optional[int], ...], player: int) -> int:
    return sum(1 for mill in MILLS if all(board[p] == player for p in mill))


def _count_potential_mills(board: Tuple[Optional[int], ...], player: int) -> int:
    count = 0
    for mill in MILLS:
        player_count = sum(1 for p in mill if board[p] == player)
        empty_count = sum(1 for p in mill if board[p] is None)
        if player_count == 2 and empty_count == 1:
            count += 1
    return count


def _count_blocked_mills(board: Tuple[Optional[int], ...], player: int) -> int:
    opponent = 1 - player
    count = 0
    for mill in MILLS:
        opp_count = sum(1 for p in mill if board[p] == opponent)
        our_count = sum(1 for p in mill if board[p] == player)
        if opp_count == 2 and our_count == 1:
            count += 1
    return count


def _count_unblocked_threats(board: Tuple[Optional[int], ...], player: int) -> int:
    opponent = 1 - player
    count = 0
    for mill in MILLS:
        opp_count = sum(1 for p in mill if board[p] == opponent)
        empty_count = sum(1 for p in mill if board[p] is None)
        if opp_count == 2 and empty_count == 1:
            count += 1
    return count


def _count_double_mills(board: Tuple[Optional[int], ...], player: int) -> int:
    count = 0
    for x in range(24):
        if board[x] != player:
            continue

        in_complete_mill = any(
            all(board[p] == player for p in mill) for mill in POSITION_TO_MILLS[x]
        )
        if not in_complete_mill:
            continue

        found = False
        for n in ADJACENCY[x]:
            if board[n] is not None:
                continue

            for mill_b in POSITION_TO_MILLS[n]:
                if x in mill_b:
                    continue
                if all(board[p] == player for p in mill_b if p != n):
                    found = True
                    break
            if found:
                break

        if found:
            count += 1

    return count


def _get_mobility(board: Tuple[Optional[int], ...], player: int) -> int:
    piece_count = _count_pieces(board, player)
    moves = 0

    for pos in range(24):
        if board[pos] != player:
            continue

        if piece_count <= 3:
            for target in range(24):
                if board[target] is None:
                    moves += 1
                    if moves >= 12:
                        return moves
        else:
            for adj in ADJACENCY[pos]:
                if board[adj] is None:
                    moves += 1

    return moves


def extract_state_features(state, player: int) -> Dict[str, float]:
    """Extract low-cost board structure features for shaping, aligned with minimax."""
    board = parse_board_from_state(state)
    opponent = 1 - player

    return {
        "my_pieces": float(_count_pieces(board, player)),
        "opp_pieces": float(_count_pieces(board, opponent)),
        "my_mills": float(_count_mills(board, player)),
        "opp_mills": float(_count_mills(board, opponent)),
        "my_potential_mills": float(_count_potential_mills(board, player)),
        "opp_potential_mills": float(_count_potential_mills(board, opponent)),
        "my_blocked_mills": float(_count_blocked_mills(board, player)),
        "opp_blocked_mills": float(_count_blocked_mills(board, opponent)),
        "my_unblocked_threats": float(_count_unblocked_threats(board, player)),
        "opp_unblocked_threats": float(_count_unblocked_threats(board, opponent)),
        "my_double_mills": float(_count_double_mills(board, player)),
        "opp_double_mills": float(_count_double_mills(board, opponent)),
        "my_mobility": float(_get_mobility(board, player)),
        "opp_mobility": float(_get_mobility(board, opponent)),
    }


def count_pieces_from_state(state, player: int) -> Tuple[int, int]:
    """
    Count pieces for specified player/opponent from parsed absolute board state.
    Returns (my_pieces, opponent_pieces)
    """
    board = parse_board_from_state(state)
    my_pieces = _count_pieces(board, player)
    opp_pieces = _count_pieces(board, 1 - player)
    return my_pieces, opp_pieces


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
        prev_state_info: Dict[str, Any],
        new_state_info: Dict[str, Any],
        player: int
    ) -> float:
        """
        Low-intensity shaping from board-structure deltas (same primitives as minimax).
        """
        reward = float(self.config.get('step_penalty', -0.003))

        def delta(key: str) -> float:
            return float(new_state_info.get(key, 0.0)) - float(prev_state_info.get(key, 0.0))

        # Positive structure gains
        reward += self.config.get('mill_reward', 0.0) * delta('my_mills')
        reward += self.config.get('block_mill_reward', 0.0) * delta('my_blocked_mills')
        reward += 0.50 * self.config.get('setup_capture_reward', 0.0) * delta('my_potential_mills')
        reward += 0.25 * self.config.get('double_mill_reward', 0.0) * delta('my_double_mills')

        # Threat control
        reward -= 0.50 * self.config.get('block_mill_reward', 0.0) * delta('opp_unblocked_threats')

        # Gentle global pressure signals
        old_piece_diff = float(prev_state_info.get('my_pieces', 0.0)) - float(prev_state_info.get('opp_pieces', 0.0))
        new_piece_diff = float(new_state_info.get('my_pieces', 0.0)) - float(new_state_info.get('opp_pieces', 0.0))
        reward += self.config.get('piece_advantage_reward', 0.0) * (new_piece_diff - old_piece_diff)

        mobility_delta = delta('my_mobility') - 0.5 * delta('opp_mobility')
        reward += self.config.get('mobility_reward', 0.0) * mobility_delta

        shaping_cap = float(self.config.get('max_shaping_abs', 0.20))
        return float(np.clip(reward, -shaping_cap, shaping_cap))
    
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
            p0_pieces, p1_pieces = count_pieces_from_state(state, 0)
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
