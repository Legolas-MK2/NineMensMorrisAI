"""
Nine Men's Morris - Utilities
Game helpers, reward calculation, and experience data structures
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict
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
    opponent_type: str = 'random'  # Track opponent type
    minimax_depth: int = 0  # Depth if opponent is minimax
