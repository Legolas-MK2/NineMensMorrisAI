"""
Nine Men's Morris - Utilities
Game helpers, reward calculation, and experience data structures
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# Optional C++ fast path for AI/PPO reward feature extraction.
# If the installed fastnmm is older than the EvaluateForAI bindings
# (separate from minimax `evaluate`), fall back to the pure-Python loops
# defined further down in this file.
try:
    from fastnmm import evaluate_for_ai_breakdown as _cpp_ai_breakdown
    _HAS_CPP_AI_EVAL = True
except Exception:
    _cpp_ai_breakdown = None
    _HAS_CPP_AI_EVAL = False


def get_legal_mask(state, num_actions: int) -> np.ndarray:
    """Create a binary mask of legal actions.

    Returned as np.bool_ (1 byte/element) — 4× smaller than the old float32
    encoding. All callsites consume it as a boolean index (`mask == 0`) or
    via fancy indexing in symmetry permutations; both work on bool dtype.
    """
    mask = np.zeros(num_actions, dtype=np.bool_)
    legal = state.legal_actions()
    if legal:
        mask[legal] = True
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


def _build_adj_matrix() -> np.ndarray:
    m = np.zeros((24, 24), dtype=bool)
    for u, nbrs in enumerate(ADJACENCY):
        for v in nbrs:
            m[u, v] = True
    return m


def _build_mill_matrix() -> np.ndarray:
    m = np.zeros((24, 24), dtype=bool)
    for mill in MILLS:
        for a in mill:
            for b in mill:
                if a != b:
                    m[a, b] = True
    return m


# Structural adjacency / mill-cohabitation matrices for the relational
# trunk's attention bias. Indexed by canonical board position; invariant
# under every board automorphism (each sigma is a graph automorphism that
# preserves both ADJACENCY and MILLS), so the bias matrices are correct
# in every augmented frame.
ADJ_MATRIX: np.ndarray = _build_adj_matrix()
MILL_MATRIX: np.ndarray = _build_mill_matrix()

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


def relativize_obs(state, player: int) -> np.ndarray:
    """Flat player-relative observation: [my_pieces, opp_pieces, ...] per cell.

    fastnmm's `observation_tensor_numpy` returns an absolute board (channel 0
    = p0 pieces, channel 1 = p1 pieces) regardless of which player is asking.
    Swap channels 0/1 when `player == 1` so the network always sees the same
    [self, opponent, ...] layout.

    Legacy flat-obs helper: no production path uses it anymore (training,
    eval and play all use `build_token_obs`). Kept because the symmetry
    test suite validates `permute_obs` against it.
    """
    obs = state.observation_tensor_numpy(player)  # shape (5, 7, 7)
    if player == 1:
        obs = obs.copy()
        obs[[0, 1]] = obs[[1, 0]]
    return obs.reshape(-1)


# Token-obs dimensions produced by `build_token_obs`. Must stay in sync with
# the corresponding Config fields (global_feat_dim / node_feat_dim /
# num_positions), which the model reads independently. Used by the tests.
GLOBAL_FEAT_DIM: int = 11
NODE_FEAT_DIM: int = 3
NUM_POSITIONS: int = 24


def build_token_obs(state, player: int) -> Tuple[np.ndarray, np.ndarray]:
    """Player-relative token observation for the relational trunk.

    Returns:
        node_feats:   float32[24, 3]  -- per board point [empty, mine, opponent]
        global_feats: float32[11]     -- phase one-hot (3) +
                                         my/opp unplaced+onboard scaled to /9 (4) +
                                         my/opp flying flags (2) + turn/max_turns (1) +
                                         1 reserved slot.

    "Mine" / "opponent" are relative to `player`; this is the token-obs analog of
    `relativize_obs`'s channel swap.
    """
    board = parse_board_from_state(state)
    opponent = 1 - player

    node_feats = np.zeros((24, 3), dtype=np.float32)
    my_onboard = 0
    opp_onboard = 0
    for p in range(24):
        v = board[p]
        if v is None:
            node_feats[p, 0] = 1.0
        elif v == player:
            node_feats[p, 1] = 1.0
            my_onboard += 1
        else:
            node_feats[p, 2] = 1.0
            opp_onboard += 1

    phase_obj = state.current_phase()
    phase_name = getattr(phase_obj, "name", str(phase_obj)).upper()
    placing = 1.0 if "PLAC" in phase_name else 0.0
    moving  = 1.0 if "MOV"  in phase_name else 0.0
    capturing = 1.0 if "CAP" in phase_name else 0.0

    my_unplaced  = float(state.men_to_deploy(player)) / 9.0
    opp_unplaced = float(state.men_to_deploy(opponent)) / 9.0
    my_onboard_n  = float(my_onboard) / 9.0
    opp_onboard_n = float(opp_onboard) / 9.0

    placed_done = state.men_to_deploy(player) == 0 and state.men_to_deploy(opponent) == 0
    my_flying  = 1.0 if (placed_done and my_onboard  == 3) else 0.0
    opp_flying = 1.0 if (placed_done and opp_onboard == 3) else 0.0

    max_turns = max(1, int(state.max_turns()))
    turn_norm = float(state.turn()) / float(max_turns)

    global_feats = np.array([
        placing, moving, capturing,
        my_unplaced, opp_unplaced, my_onboard_n, opp_onboard_n,
        my_flying, opp_flying,
        turn_norm,
        0.0,  # reserved
    ], dtype=np.float32)

    return node_feats, global_feats


def parse_board_from_state(state) -> Tuple[Optional[int], ...]:
    """Parse absolute board (0=player0, 1=player1, None=empty) from state.

    Uses fastnmm's `observation_tensor_numpy` for a direct (5,7,7) float32
    array, avoiding the list-allocation that the OpenSpiel-compatible
    `observation_tensor` performs.
    """
    try:
        obs_array = state.observation_tensor_numpy(0)  # shape (5, 7, 7), float32
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
    """
    Count mills that `player` can actually close on their very next move.

    Placement phase (total pieces on board < 18) or flying phase (player has
    exactly 3 pieces on board): any mill with 2 player pieces + 1 empty counts,
    because the player can drop/fly a stone into the empty spot.

    Movement phase (player has > 3 pieces, past placement): a mill with
    2 player pieces + 1 empty only counts if there is a player stone
    *outside* this mill that is adjacent to the empty spot, so it can
    be slid in to complete the mill.
    """
    piece_count = _count_pieces(board, player)
    total_on_board = sum(1 for p in board if p is not None)

    # Before all 18 stones have been placed, we treat it as placement phase
    # (captures during placement are rare enough that this heuristic is fine).
    is_placement = total_on_board < 18
    # Flying: exactly 3 stones and placement is over.
    is_flying = (not is_placement) and piece_count == 3
    any_empty_counts = is_placement or is_flying

    count = 0
    for mill in MILLS:
        players_in_mill = [p for p in mill if board[p] == player]
        if len(players_in_mill) != 2:
            continue
        empties = [p for p in mill if board[p] is None]
        if len(empties) != 1:
            continue

        empty_pos = empties[0]

        if any_empty_counts:
            count += 1
            continue

        # Movement phase: need a player piece outside this mill that is
        # adjacent to empty_pos and can slide into it.
        for adj in ADJACENCY[empty_pos]:
            if board[adj] == player and adj not in players_in_mill:
                count += 1
                break
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
    """Count mills the opponent can close on their next move (see
    `_count_potential_mills` for the definition)."""
    return _count_potential_mills(board, 1 - player)


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
    """Extract low-cost board structure features for shaping, aligned with minimax.

    Fast path: `fastnmm.evaluate_for_ai_breakdown(state, player)` (pure C++,
    GIL-released) replaces the parse-board + 7 Python loops + dict build.
    Falls back to the Python implementation if the C++ binding is missing
    (e.g., a worker started before fastnmm was rebuilt).
    """
    if _HAS_CPP_AI_EVAL:
        try:
            b = _cpp_ai_breakdown(state, int(player))
            return {
                "my_pieces":             float(b.my_pieces),
                "opp_pieces":            float(b.opp_pieces),
                "my_mills":              float(b.my_mills),
                "opp_mills":             float(b.opp_mills),
                "my_potential_mills":    float(b.my_potential_mills),
                "opp_potential_mills":   float(b.opp_potential_mills),
                "my_blocked_mills":      float(b.my_blocked_mills),
                "opp_blocked_mills":     float(b.opp_blocked_mills),
                "my_unblocked_threats":  float(b.my_unblocked_threats),
                "opp_unblocked_threats": float(b.opp_unblocked_threats),
                "my_double_mills":       float(b.my_double_mills),
                "opp_double_mills":      float(b.opp_double_mills),
                "my_mobility":           float(b.my_mobility),
                "opp_mobility":          float(b.opp_mobility),
            }
        except Exception:
            pass  # graceful fallback to pure Python below

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

    Shaping is PBRS-style:
        r_shape = γ·Φ(s') − Φ(s) + step_penalty

    Caveat: this is NOT strictly potential-based in the formal sense. Φ is
    evaluated only around the AI's own action (the opponent's intervening
    move changes the state between decisions, so the telescoping sum does
    not cancel exactly), and the worker additionally applies a direct
    `enemy_mill_penalty` on piece loss. Treat it as a heuristic dense
    signal whose scale decays to 0 over training (see the curriculum's
    shaping multiplier) — the final policy is trained on sparse terminal
    rewards only.
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
            base = self.config['win_reward_base']
            bonus = self.config['win_reward_speed_bonus']
            speed_bonus = bonus * max(0, 1.0 - (steps / max_steps))
            return base + speed_bonus

        elif my_return < opp_return:
            return self.config['loss_reward']
        else:
            return self.config['draw_penalty']

    def calculate_potential(self, state_info: Dict[str, Any]) -> float:
        """
        Φ(s): scalar potential for the acting player, derived from minimax-aligned
        board features. Weights come from the reward config (already scaled by the
        curriculum's shaping multiplier).
        """
        cfg = self.config
        phi = 0.0
        phi += cfg.get('mill_reward', 0.0) * state_info.get('my_mills', 0.0)
        phi -= cfg.get('mill_reward', 0.0) * state_info.get('opp_mills', 0.0)
        phi += cfg.get('setup_capture_reward', 0.0) * state_info.get('my_potential_mills', 0.0)
        phi -= cfg.get('block_mill_reward', 0.0) * state_info.get('opp_unblocked_threats', 0.0)
        phi += cfg.get('double_mill_reward', 0.0) * state_info.get('my_double_mills', 0.0)
        phi += cfg.get('piece_advantage_reward', 0.0) * (
            state_info.get('my_pieces', 0.0) - state_info.get('opp_pieces', 0.0)
        )
        phi += cfg.get('mobility_reward', 0.0) * (
            state_info.get('my_mobility', 0.0) - state_info.get('opp_mobility', 0.0)
        )
        return float(phi)

    def calculate_shaping_reward(
        self,
        prev_state_info: Dict[str, Any],
        new_state_info: Dict[str, Any],
        player: int
    ) -> float:
        """PBRS: r = γ·Φ(s') − Φ(s) + step_penalty, clipped to ±max_shaping_abs."""
        gamma = float(self.config.get('gamma', 0.99))
        prev_phi = self.calculate_potential(prev_state_info)
        new_phi = self.calculate_potential(new_state_info)

        reward = gamma * new_phi - prev_phi
        reward += float(self.config.get('step_penalty', 0.0))

        cap = float(self.config.get('max_shaping_abs', 0.20))
        return float(np.clip(reward, -cap, cap))

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
    """Batch of experiences from a single episode for one player.

    Observations are stored as token tensors (per-position + global) — the
    relational ActorCritic ingests these directly. All tensors are in the
    same (possibly augmented) frame as `actions` / `masks`.
    """
    node_feats: np.ndarray   # float32[T, 24, 3]
    global_feats: np.ndarray # float32[T, G]
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
    outcome: str = 'draw'   # Explicit result for this batch's player: 'win'/'loss'/'draw'


