"""
Nine Men's Morris - Minimax Bot
Alpha-beta pruning implementation for training opponent and evaluation
Supports multithreaded search at root level for improved performance.
"""

from typing import Tuple, Dict, List, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
import random
from utils import get_legal_mask


# ============================================================================
# NINE MEN'S MORRIS BOARD TOPOLOGY
# ============================================================================
#
# Board positions (0-23):
#
#    0-----------1-----------2
#    |           |           |
#    |   8-------9------10   |
#    |   |       |       |   |
#    |   |  16--17--18   |   |
#    |   |   |       |   |   |
#    7--15--23      19--11---3
#    |   |   |       |   |   |
#    |   |  22--21--20   |   |
#    |   |       |       |   |
#    |  14------13------12   |
#    |           |           |
#    6-----------5-----------4
#
# ============================================================================

# All 16 possible mills (3 pieces in a row)
MILLS = [
    # Outer square (horizontal and vertical)
    (0, 1, 2), (2, 3, 4), (4, 5, 6), (6, 7, 0),
    # Middle square
    (8, 9, 10), (10, 11, 12), (12, 13, 14), (14, 15, 8),
    # Inner square
    (16, 17, 18), (18, 19, 20), (20, 21, 22), (22, 23, 16),
    # Radial connections (connecting the squares)
    (1, 9, 17), (3, 11, 19), (5, 13, 21), (7, 15, 23),
]

# Adjacent positions for each board position (for movement)
ADJACENCY = {
    0: [1, 7],
    1: [0, 2, 9],
    2: [1, 3],
    3: [2, 4, 11],
    4: [3, 5],
    5: [4, 6, 13],
    6: [5, 7],
    7: [6, 0, 15],
    8: [9, 15],
    9: [8, 10, 1, 17],
    10: [9, 11],
    11: [10, 12, 3, 19],
    12: [11, 13],
    13: [12, 14, 5, 21],
    14: [13, 15],
    15: [14, 8, 7, 23],
    16: [17, 23],
    17: [16, 18, 9],
    18: [17, 19],
    19: [18, 20, 11],
    20: [19, 21],
    21: [20, 22, 13],
    22: [21, 23],
    23: [22, 16, 15],
}

# Mills that each position belongs to (for quick lookup)
POSITION_TO_MILLS: Dict[int, List[Tuple[int, int, int]]] = {i: [] for i in range(24)}
for mill in MILLS:
    for pos in mill:
        POSITION_TO_MILLS[pos].append(mill)

# Strategic value of positions (more connections = better control)
# Corner positions have 2 connections, middle positions have 3-4
POSITION_VALUES = {
    # Outer corners (2 connections each)
    0: 2, 2: 2, 4: 2, 6: 2,
    # Outer midpoints (3 connections - can form radial mills)
    1: 3, 3: 3, 5: 3, 7: 3,
    # Middle corners (2 connections)
    8: 2, 10: 2, 12: 2, 14: 2,
    # Middle midpoints (4 connections - most valuable)
    9: 4, 11: 4, 13: 4, 15: 4,
    # Inner corners (2 connections)
    16: 2, 18: 2, 20: 2, 22: 2,
    # Inner midpoints (3 connections)
    17: 3, 19: 3, 21: 3, 23: 3,
}


def parse_board_from_state(state) -> Dict[int, Optional[int]]:
    """
    Parse the game state into a board dictionary.
    Returns {position: player} where player is 0, 1, or None for empty.
    """
    state_str = str(state)
    board = {i: None for i in range(24)}

    # Extract just the board portion (positions with pieces)
    pos_idx = 0
    for char in state_str:
        if char == '.':
            pos_idx += 1
        elif char in 'oO':
            if pos_idx < 24:
                board[pos_idx] = 0
            pos_idx += 1
        elif char in 'xX':
            if pos_idx < 24:
                board[pos_idx] = 1
            pos_idx += 1

    return board


def parse_board_from_observation(state, player: int) -> Dict[int, Optional[int]]:
    """
    Parse board from observation tensor (more reliable than string parsing).
    """
    try:
        obs = state.observation_tensor(player)
        board = {i: None for i in range(24)}

        # Observation tensor layout for Nine Men's Morris:
        # First 24 values: current player's pieces
        # Next 24 values: opponent's pieces
        for i in range(24):
            if obs[i] == 1:
                board[i] = player
            elif obs[i + 24] == 1:
                board[i] = 1 - player

        return board
    except:
        # Fallback to string parsing
        return parse_board_from_state(state)


class MinimaxBot:
    """Minimax bot with alpha-beta pruning for Nine Men's Morris.

    Supports multithreaded search at the root level for improved performance
    on multi-core systems.

    Args:
        max_depth: Maximum search depth
        num_threads: Number of threads for parallel root-level search
        random_move_prob: Probability of making a random move instead of minimax.
                         Use 0.0 for evaluation, ~0.3 for training to prevent overfitting.
    """

    # ========================================================================
    # EVALUATION WEIGHTS (tuned for strong play)
    # ========================================================================
    WEIGHTS = {
        # Terminal states
        'win': 100000,
        'loss': -100000,
        'draw': 0,

        # Piece count (fundamental)
        'piece_value': 1000,

        # Mill-related
        'mill': 500,                    # Complete mill (3 in a row)
        'potential_mill': 150,          # 2 pieces + 1 empty (one move from mill)
        'double_mill': 800,             # Two mills sharing a piece (can alternate)
        'double_mill_with_extra': 1200, # Double mill + another mill ready

        # Blocking
        'blocked_opponent_mill': 200,   # Blocked opponent's potential mill
        'blocked_piece': 50,            # Opponent piece that cannot move

        # Mobility
        'mobility': 10,                 # Each legal move available
        'opponent_immobility': 30,      # Each move opponent lacks

        # Positional (control of key intersections)
        'position_value': 5,            # Based on POSITION_VALUES map

        # Phase-specific adjustments
        'placing_phase_mill_bonus': 100,  # Extra value for mills in placing phase
        'endgame_piece_bonus': 200,       # Pieces worth more when few remain
    }

    def __init__(self, max_depth: int = 3, num_threads: int = 4, random_move_prob: float = 0.0):
        self.max_depth = max_depth
        self.num_threads = num_threads
        self.random_move_prob = random_move_prob
        self.nodes_evaluated = 0
        self._lock = threading.Lock()

    def _get_player_positions(self, board: Dict[int, Optional[int]], player: int) -> Set[int]:
        """Get all positions occupied by a player."""
        return {pos for pos, p in board.items() if p == player}

    def _count_mills(self, board: Dict[int, Optional[int]], player: int) -> int:
        """Count complete mills for a player."""
        count = 0
        for mill in MILLS:
            if all(board.get(pos) == player for pos in mill):
                count += 1
        return count

    def _count_potential_mills(self, board: Dict[int, Optional[int]], player: int) -> int:
        """
        Count potential mills (2 pieces + 1 empty).
        These are one move away from completion.
        """
        count = 0
        for mill in MILLS:
            player_count = sum(1 for pos in mill if board.get(pos) == player)
            empty_count = sum(1 for pos in mill if board.get(pos) is None)
            if player_count == 2 and empty_count == 1:
                count += 1
        return count

    def _count_double_mills(self, board: Dict[int, Optional[int]], player: int) -> int:
        """
        Count double mills - positions where a piece belongs to two mills
        and can alternate between them by moving back and forth.

        A double mill is extremely powerful because each move creates a mill,
        allowing continuous captures.
        """
        double_mill_count = 0
        player_positions = self._get_player_positions(board, player)

        for pos in player_positions:
            mills_with_pos = POSITION_TO_MILLS[pos]
            complete_mills = 0

            for mill in mills_with_pos:
                # Check if this mill is complete
                if all(board.get(p) == player for p in mill):
                    complete_mills += 1

            # If piece is part of 2+ complete mills, it's a double mill piece
            if complete_mills >= 2:
                double_mill_count += 1

        return double_mill_count

    def _count_double_mill_potential(self, board: Dict[int, Optional[int]], player: int) -> int:
        """
        Count positions that could become double mills.
        A piece that is in one complete mill and one potential mill.
        """
        count = 0
        player_positions = self._get_player_positions(board, player)

        for pos in player_positions:
            mills_with_pos = POSITION_TO_MILLS[pos]
            complete = 0
            potential = 0

            for mill in mills_with_pos:
                player_count = sum(1 for p in mill if board.get(p) == player)
                empty_count = sum(1 for p in mill if board.get(p) is None)

                if player_count == 3:
                    complete += 1
                elif player_count == 2 and empty_count == 1:
                    potential += 1

            # One complete + one potential = about to have double mill
            if complete >= 1 and potential >= 1:
                count += 1

        return count

    def _count_blocked_pieces(self, board: Dict[int, Optional[int]], player: int) -> int:
        """
        Count pieces that cannot move (all adjacent positions occupied).
        Only relevant in moving phase (not placing or flying).
        """
        blocked = 0
        player_positions = self._get_player_positions(board, player)

        for pos in player_positions:
            adjacent = ADJACENCY[pos]
            if all(board.get(adj) is not None for adj in adjacent):
                blocked += 1

        return blocked

    def _get_mobility(self, board: Dict[int, Optional[int]], player: int, piece_count: int) -> int:
        """
        Calculate mobility (number of possible moves).
        """
        player_positions = self._get_player_positions(board, player)

        if piece_count <= 3:
            # Flying phase: can move to any empty position
            empty_positions = sum(1 for p in board.values() if p is None)
            return len(player_positions) * empty_positions

        # Normal moving phase: count adjacent empty positions
        moves = 0
        for pos in player_positions:
            for adj in ADJACENCY[pos]:
                if board.get(adj) is None:
                    moves += 1
        return moves

    def _positional_score(self, board: Dict[int, Optional[int]], player: int) -> int:
        """Calculate positional advantage based on board control."""
        score = 0
        for pos, owner in board.items():
            if owner == player:
                score += POSITION_VALUES[pos]
            elif owner == (1 - player):
                score -= POSITION_VALUES[pos]
        return score

    def _count_blocked_opponent_mills(self, board: Dict[int, Optional[int]], player: int) -> int:
        """
        Count opponent's potential mills that we are blocking
        (opponent has 2 pieces but we have 1 blocking piece).
        """
        opponent = 1 - player
        blocked = 0

        for mill in MILLS:
            opp_count = sum(1 for pos in mill if board.get(pos) == opponent)
            my_count = sum(1 for pos in mill if board.get(pos) == player)

            # Opponent has 2 pieces, we have 1 blocking
            if opp_count == 2 and my_count == 1:
                blocked += 1

        return blocked

    def _detect_game_phase(self, state, my_pieces: int, opp_pieces: int) -> str:
        """
        Detect the current game phase.
        - 'placing': Still placing pieces (< 18 total pieces placed)
        - 'moving': Normal movement phase
        - 'flying': Endgame when a player has exactly 3 pieces
        """
        total_pieces = my_pieces + opp_pieces

        # Check if still in placing phase
        # In placing phase, pieces_in_hand > 0 for either player
        state_str = str(state)

        # Heuristic: if total pieces on board < 18 and no captures yet,
        # likely still placing. This is approximate.
        if total_pieces < 6:
            return 'placing'
        elif my_pieces == 3 or opp_pieces == 3:
            return 'flying'
        else:
            return 'moving'

    def evaluate(self, state, player: int) -> float:
        """
        Comprehensive heuristic evaluation of board position.
        Considers: pieces, mills, double mills, potential mills,
        blocked pieces, mobility, and positional control.

        Positive = good for player, negative = bad.
        """
        # Terminal state check
        if state.is_terminal():
            returns = state.returns()
            if returns[player] > returns[1 - player]:
                return self.WEIGHTS['win']
            elif returns[player] < returns[1 - player]:
                return self.WEIGHTS['loss']
            else:
                return self.WEIGHTS['draw']

        # Parse board state
        board = parse_board_from_observation(state, player)
        opponent = 1 - player

        # Count pieces
        my_positions = self._get_player_positions(board, player)
        opp_positions = self._get_player_positions(board, opponent)
        my_pieces = len(my_positions)
        opp_pieces = len(opp_positions)

        # Detect game phase
        phase = self._detect_game_phase(state, my_pieces, opp_pieces)

        score = 0.0

        # ===== PIECE COUNT (fundamental) =====
        piece_diff = my_pieces - opp_pieces
        piece_weight = self.WEIGHTS['piece_value']

        # Pieces worth more in endgame
        if phase == 'flying' or my_pieces <= 4 or opp_pieces <= 4:
            piece_weight += self.WEIGHTS['endgame_piece_bonus']

        score += piece_diff * piece_weight

        # ===== MILLS =====
        my_mills = self._count_mills(board, player)
        opp_mills = self._count_mills(board, opponent)
        mill_bonus = self.WEIGHTS['mill']

        if phase == 'placing':
            mill_bonus += self.WEIGHTS['placing_phase_mill_bonus']

        score += (my_mills - opp_mills) * mill_bonus

        # ===== POTENTIAL MILLS (one move from completion) =====
        my_potential = self._count_potential_mills(board, player)
        opp_potential = self._count_potential_mills(board, opponent)
        score += (my_potential - opp_potential) * self.WEIGHTS['potential_mill']

        # ===== DOUBLE MILLS (extremely powerful) =====
        my_double_mills = self._count_double_mills(board, player)
        opp_double_mills = self._count_double_mills(board, opponent)
        score += (my_double_mills - opp_double_mills) * self.WEIGHTS['double_mill']

        # ===== DOUBLE MILL POTENTIAL =====
        my_double_potential = self._count_double_mill_potential(board, player)
        opp_double_potential = self._count_double_mill_potential(board, opponent)
        score += (my_double_potential - opp_double_potential) * self.WEIGHTS['double_mill_with_extra']

        # ===== BLOCKED OPPONENT MILLS =====
        my_blocks = self._count_blocked_opponent_mills(board, player)
        opp_blocks = self._count_blocked_opponent_mills(board, opponent)
        score += (my_blocks - opp_blocks) * self.WEIGHTS['blocked_opponent_mill']

        # ===== BLOCKED PIECES (only in moving phase) =====
        if phase == 'moving':
            my_blocked = self._count_blocked_pieces(board, player)
            opp_blocked = self._count_blocked_pieces(board, opponent)
            score -= my_blocked * self.WEIGHTS['blocked_piece']
            score += opp_blocked * self.WEIGHTS['blocked_piece']

        # ===== MOBILITY =====
        my_mobility = self._get_mobility(board, player, my_pieces)
        opp_mobility = self._get_mobility(board, opponent, opp_pieces)
        score += my_mobility * self.WEIGHTS['mobility']
        score += (10 - opp_mobility) * self.WEIGHTS['opponent_immobility'] if opp_mobility < 10 else 0

        # ===== POSITIONAL CONTROL =====
        score += self._positional_score(board, player) * self.WEIGHTS['position_value']

        return score
    
    def _increment_nodes(self):
        """Thread-safe increment of nodes evaluated counter."""
        with self._lock:
            self.nodes_evaluated += 1

    def minimax(self, state, depth: int, alpha: float, beta: float,
                maximizing: bool, player: int) -> Tuple[float, int]:
        """Minimax with alpha-beta pruning. Returns (score, best_action)."""
        self._increment_nodes()
        
        if depth == 0 or state.is_terminal():
            return self.evaluate(state, player), -1
        
        legal_actions = state.legal_actions()
        if not legal_actions:
            return self.evaluate(state, player), -1
        
        best_action = legal_actions[0]
        
        if maximizing:
            max_eval = float('-inf')
            for action in legal_actions:
                child = state.clone()
                child.apply_action(action)
                eval_score, _ = self.minimax(child, depth - 1, alpha, beta, False, player)
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_action = action
                
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            
            return max_eval, best_action
        else:
            min_eval = float('inf')
            for action in legal_actions:
                child = state.clone()
                child.apply_action(action)
                eval_score, _ = self.minimax(child, depth - 1, alpha, beta, True, player)
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_action = action
                
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            
            return min_eval, best_action

    def _evaluate_action(self, state, action: int, depth: int, player: int) -> Tuple[int, float]:
        """Evaluate a single action. Used for parallel root-level search.

        Returns (action, score) tuple.
        """
        child = state.clone()
        child.apply_action(action)
        # After our move, opponent minimizes
        score, _ = self.minimax(
            child, depth - 1,
            float('-inf'), float('inf'),
            False, player
        )
        return action, score

    def minimax_parallel(self, state, depth: int, player: int) -> Tuple[float, int]:
        """Parallel minimax search at root level.

        Spawns threads to evaluate each root action in parallel.
        Returns (best_score, best_action).
        """
        legal_actions = state.legal_actions()
        if not legal_actions:
            return self.evaluate(state, player), -1

        if len(legal_actions) == 1:
            # Only one move, no need for parallel search
            return self._evaluate_action(state, legal_actions[0], depth, player)

        best_score = float('-inf')
        best_action = legal_actions[0]

        # Use thread pool to evaluate actions in parallel
        num_workers = min(self.num_threads, len(legal_actions))
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(self._evaluate_action, state, action, depth, player): action
                for action in legal_actions
            }

            for future in as_completed(futures):
                action, score = future.result()
                if score > best_score:
                    best_score = score
                    best_action = action

        return best_score, best_action

    def get_action(self, state, use_threading: bool = True) -> int:
        """Get best action for current player.

        Args:
            state: Current game state
            use_threading: If True, use multithreaded search at root level
        """

        legal_actions = state.legal_actions()

        # Optional random move for training (prevents overfitting)
        if self.random_move_prob > 0 and random.random() < self.random_move_prob:
            return random.choice(legal_actions)

        self.nodes_evaluated = 0
        player = state.current_player()

        if use_threading and self.num_threads > 1 and len(legal_actions) > 1:
            # Use parallel search at root level
            _, action = self.minimax_parallel(state, self.max_depth, player)
        else:
            # Use sequential search
            _, action = self.minimax(
                state, self.max_depth,
                float('-inf'), float('inf'),
                True, player
            )
        return action


def evaluate_vs_minimax(
    model,
    device,
    num_actions: int,
    max_depth: int = 6,
    games_per_depth: int = 10,
    max_steps: int = 200,
    use_mixed_precision: bool = True,
    unlimited: bool = True
) -> Tuple[int, Dict]:
    """
    Progressive evaluation against minimax bots.
    Returns (max_depth_beaten, detailed_results).

    If unlimited=True, will keep testing higher depths beyond max_depth
    as long as the AI keeps winning (>50% win rate).
    """
    import pyspiel
    game = pyspiel.load_game("nine_mens_morris")
    results = {}
    max_depth_beaten = 0

    model.eval()

    depth = 1
    while True:
        # Stop if we've hit max_depth and unlimited is False
        if not unlimited and depth > max_depth:
            break

        bot = MinimaxBot(max_depth=depth)
        wins, draws, losses = 0, 0, 0

        with torch.no_grad():
            for game_idx in range(games_per_depth):
                state = game.new_initial_state()
                ai_player = game_idx % 2
                steps = 0

                while not state.is_terminal() and steps < max_steps:
                    current = state.current_player()

                    if current == ai_player:
                        obs = torch.tensor(
                            state.observation_tensor(current),
                            dtype=torch.float32, device=device
                        ).unsqueeze(0)
                        mask = torch.tensor(
                            get_legal_mask(state, num_actions),
                            dtype=torch.float32, device=device
                        ).unsqueeze(0)

                        with autocast('cuda', enabled=use_mixed_precision and device.type == 'cuda'):
                            logits, _ = model(obs)

                        masked = logits.squeeze(0).float()
                        masked[mask.squeeze(0) == 0] = -1e9
                        action = masked.argmax().item()
                    else:
                        action = bot.get_action(state)

                    state.apply_action(action)
                    steps += 1

                if state.is_terminal():
                    returns = state.returns()
                    if returns[ai_player] > returns[1 - ai_player]:
                        wins += 1
                    elif returns[ai_player] < returns[1 - ai_player]:
                        losses += 1
                    else:
                        draws += 1
                else:
                    draws += 1  # Timeout = draw

        win_rate = (wins + 0.5 * draws) / games_per_depth
        results[depth] = {
            'wins': wins,
            'draws': draws,
            'losses': losses,
            'win_rate': win_rate
        }

        if win_rate > 0.5:
            max_depth_beaten = depth
            depth += 1  # Try next depth
        else:
            break  # Stop if we're losing

    return max_depth_beaten, results


def format_minimax_results(results: Dict) -> str:
    """Format minimax evaluation results as a string."""
    parts = []
    for depth in sorted(results.keys()):
        r = results[depth]
        parts.append(f"D{depth}:{r['wins']}W/{r['draws']}D/{r['losses']}L")
    return " | ".join(parts)
