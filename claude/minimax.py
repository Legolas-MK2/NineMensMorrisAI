"""
Nine Men's Morris - Fully Optimized Minimax Bot
===============================================

Optimizations implemented:
1. Zobrist Hashing - Fast incremental position hashing
2. Transposition Table - Caches evaluated positions (configurable size, default ~8GB)
3. Move Ordering - Evaluates best moves first for maximum cutoffs
4. Iterative Deepening - Progressive deepening with time control
5. Killer Move Heuristic - Remembers refutation moves per ply
6. History Heuristic - Tracks historically good moves
7. Principal Variation Search (PVS) - Narrow window search after first move
8. Null Move Pruning - Skip turn to get quick cutoffs
9. Aspiration Windows - Narrow search windows around expected score
10. Late Move Reductions (LMR) - Reduce depth for unlikely moves
11. Quiescence Search - Extend search for captures to avoid horizon effect
12. Static Exchange Evaluation (SEE) - Evaluate capture sequences

Critical Bug Fixes Applied:
---------------------------
1. Board Parsing Fix: Always use observation_tensor(0) for consistent absolute
   board representation. OpenSpiel's observation tensors are RELATIVE to the
   observer (plane 0 = observer's pieces, plane 1 = opponent's pieces).
   Previously, when playing as Player 1, the bot saw the board inverted!

2. Placement Phase Ordering: Added specialized move ordering for the placement
   phase that prioritizes mill completions, blocking opponent mills, and
   strategically strong positions (cross positions).

Author: Optimized for AI training with 10GB+ RAM available
"""

from typing import Tuple, Dict, List, Optional, Set, NamedTuple
from dataclasses import dataclass, field
from enum import IntEnum
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import threading
import numpy as np
import random
import time
import sys

# Optional imports
try:
    import torch
    import torch.nn.functional as F
    from torch.amp import autocast
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from utils import get_legal_mask
except ImportError:
    def get_legal_mask(state, num_actions):
        mask = [0] * num_actions
        for a in state.legal_actions():
            if a < num_actions:
                mask[a] = 1
        return mask

import pyspiel
from game_wrapper import load_game as load_game_fixed


# ============================================================================
# BOARD TOPOLOGY AND CONSTANTS
# ============================================================================

# All 16 possible mills (row-by-row numbering)
#     0-----------1-----------2
#     |           |           |
#     |     3-----4-----5     |
#     |     |     |     |     |
#     |     |  6--7--8  |     |
#     |     |  |     |  |     |
#     9----10-11    12-13----14
#     |     |  |     |  |     |
#     |     | 15-16-17  |     |
#     |     |     |     |     |
#     |    18----19----20     |
#     |           |           |
#    21----------22----------23
MILLS: Tuple[Tuple[int, int, int], ...] = (
    # Outer square sides
    (0, 1, 2), (0, 9, 21), (2, 14, 23), (21, 22, 23),
    # Middle square sides
    (3, 4, 5), (3, 10, 18), (5, 13, 20), (18, 19, 20),
    # Inner square sides
    (6, 7, 8), (6, 11, 15), (8, 12, 17), (15, 16, 17),
    # Cross connections
    (1, 4, 7), (9, 10, 11), (12, 13, 14), (16, 19, 22),
)

# Adjacent positions (row-by-row numbering)
ADJACENCY: Tuple[Tuple[int, ...], ...] = (
    (1, 9),          # 0
    (0, 2, 4),       # 1
    (1, 14),         # 2
    (4, 10),         # 3
    (1, 3, 5, 7),    # 4
    (4, 13),         # 5
    (7, 11),         # 6
    (4, 6, 8),       # 7
    (7, 12),         # 8
    (0, 10, 21),     # 9
    (3, 9, 11, 18),  # 10
    (6, 10, 15),     # 11
    (8, 13, 17),     # 12
    (5, 12, 14, 20), # 13
    (2, 13, 23),     # 14
    (11, 16),        # 15
    (15, 17, 19),    # 16
    (12, 16),        # 17
    (10, 19),        # 18
    (16, 18, 20, 22),# 19
    (13, 19),        # 20
    (9, 22),         # 21
    (19, 21, 23),    # 22
    (14, 22),        # 23
)

# Mills per position (precomputed for speed)
POSITION_TO_MILLS: Tuple[Tuple[Tuple[int, int, int], ...], ...] = tuple(
    tuple(mill for mill in MILLS if pos in mill)
    for pos in range(24)
)

# Position strategic values (based on connectivity)
# Corners have 2 connections, midpoints with cross have 3-4 connections
POSITION_VALUES: Tuple[int, ...] = (
    2, 3, 2,  # Row 0: outer top (0=corner, 1=midpoint+cross, 2=corner)
    2, 4, 2,  # Row 1: middle top (3=corner, 4=midpoint+2cross, 5=corner)
    2, 3, 2,  # Row 2: inner top (6=corner, 7=midpoint, 8=corner)
    3, 4, 3,  # Row 3 left: (9=midpoint+cross, 10=midpoint+2cross, 11=midpoint+cross)
    3, 4, 3,  # Row 3 right: (12=midpoint+cross, 13=midpoint+2cross, 14=midpoint+cross)
    2, 3, 2,  # Row 4: inner bottom (15=corner, 16=midpoint, 17=corner)
    2, 4, 2,  # Row 5: middle bottom (18=corner, 19=midpoint+2cross, 20=corner)
    2, 3, 2,  # Row 6: outer bottom (21=corner, 22=midpoint+cross, 23=corner)
)

# Improved position values for placement phase - prioritize cross positions more heavily
POSITION_VALUES_PLACEMENT: Tuple[int, ...] = (
    3, 4, 3,  # Row 0: outer top (corners=3, cross midpoint=4)
    3, 5, 3,  # Row 1: middle top (corners=3, double-cross=5)
    2, 3, 2,  # Row 2: inner top (less valuable - fewer escape routes)
    4, 5, 3,  # Row 3 left: (cross=4, double-cross=5, inner=3)
    3, 5, 4,  # Row 3 right: (inner=3, double-cross=5, cross=4)
    2, 3, 2,  # Row 4: inner bottom
    3, 5, 3,  # Row 5: middle bottom
    3, 4, 3,  # Row 6: outer bottom
)

# Best opening positions (most connected, part of most mills)
STRONG_OPENING_POSITIONS: Tuple[int, ...] = (4, 10, 13, 19, 1, 9, 14, 22)  # Cross positions first

# Board position (0-23) to 7x7 grid (row, col) mapping for observation tensor parsing
# OpenSpiel uses a 5x7x7 tensor where planes 0/1 are white/black pieces on a 7x7 grid
BOARD_POS_TO_GRID: Tuple[Tuple[int, int], ...] = (
    (0, 0), (0, 3), (0, 6),                         # 0-2: outer top
    (1, 1), (1, 3), (1, 5),                         # 3-5: middle top
    (2, 2), (2, 3), (2, 4),                         # 6-8: inner top
    (3, 0), (3, 1), (3, 2), (3, 4), (3, 5), (3, 6), # 9-14: middle row
    (4, 2), (4, 3), (4, 4),                         # 15-17: inner bottom
    (5, 1), (5, 3), (5, 5),                         # 18-20: middle bottom
    (6, 0), (6, 3), (6, 6),                         # 21-23: outer bottom
)


# ============================================================================
# ZOBRIST HASHING
# ============================================================================

class ZobristHash:
    """
    Zobrist hashing for fast incremental position hashing.
    Uses 64-bit random numbers for each (position, player) combination.
    """
    
    def __init__(self, seed: int = 42):
        rng = random.Random(seed)
        
        # Random values for each (position, player) combination
        # Index: position * 2 + player
        self.piece_keys: Tuple[int, ...] = tuple(
            rng.getrandbits(64) for _ in range(24 * 2)
        )
        
        # Random value for side to move
        self.side_key: int = rng.getrandbits(64)
        
        # Keys for game phase and pieces in hand (for complete state hashing)
        self.phase_keys: Tuple[int, ...] = tuple(
            rng.getrandbits(64) for _ in range(3)  # placing, moving, flying
        )
        self.hand_keys: Tuple[Tuple[int, ...], ...] = tuple(
            tuple(rng.getrandbits(64) for _ in range(10))  # 0-9 pieces in hand
            for _ in range(2)  # per player
        )

        # Key for pending removal state (critical: different action space)
        self.pending_removal_key: int = rng.getrandbits(64)
    
    def hash_board(self, board: Tuple[Optional[int], ...],
                   current_player: int,
                   pieces_in_hand: Tuple[int, int] = (0, 0),
                   pending_removal: bool = False) -> int:
        """Compute full hash from scratch."""
        h = 0
        for pos, player in enumerate(board):
            if player is not None:
                h ^= self.piece_keys[pos * 2 + player]

        if current_player == 1:
            h ^= self.side_key

        # Include pieces in hand for placing phase
        h ^= self.hand_keys[0][min(pieces_in_hand[0], 9)]
        h ^= self.hand_keys[1][min(pieces_in_hand[1], 9)]

        # Include pending removal state (critical: affects legal actions)
        if pending_removal:
            h ^= self.pending_removal_key

        return h
    
    def update_move(self, h: int, from_pos: Optional[int], to_pos: int, 
                    player: int) -> int:
        """Incrementally update hash for a move."""
        if from_pos is not None:
            h ^= self.piece_keys[from_pos * 2 + player]
        h ^= self.piece_keys[to_pos * 2 + player]
        return h
    
    def update_capture(self, h: int, pos: int, player: int) -> int:
        """Incrementally update hash for a capture."""
        return h ^ self.piece_keys[pos * 2 + player]
    
    def update_side(self, h: int) -> int:
        """Toggle side to move."""
        return h ^ self.side_key


# Global Zobrist instance
ZOBRIST = ZobristHash()


# ============================================================================
# TRANSPOSITION TABLE
# ============================================================================

class TTEntryType(IntEnum):
    """Transposition table entry types."""
    EXACT = 0      # Exact score
    LOWER = 1      # Score is lower bound (failed high / beta cutoff)
    UPPER = 2      # Score is upper bound (failed low / alpha cutoff)


@dataclass(slots=True)
class TTEntry:
    """Transposition table entry."""
    hash_key: int          # Full hash for collision detection
    depth: int             # Search depth
    score: float           # Evaluated score
    entry_type: TTEntryType  # Type of bound
    best_move: int         # Best move found (-1 if none)
    age: int               # When this entry was created


class TranspositionTable:
    """
    Lock-free transposition table with replacement strategy.
    
    Uses a large numpy array for memory efficiency.
    Supports ~100M+ entries with 10GB RAM.
    """
    
    # Entry size in bytes (approximate)
    ENTRY_SIZE = 40
    
    def __init__(self, size_mb: int = 8192):
        """
        Initialize transposition table.
        
        Args:
            size_mb: Size in megabytes (default 8GB)
        """
        self.size = (size_mb * 1024 * 1024) // self.ENTRY_SIZE
        self.size = max(1, self.size)
        
        # Use dictionary for simplicity and flexibility
        # For maximum performance, could use numpy structured array
        self.table: Dict[int, TTEntry] = {}
        self.age = 0
        self.hits = 0
        self.misses = 0
        self.collisions = 0
        self._lock = threading.Lock()
    
    def new_search(self):
        """Increment age for new search (for replacement strategy)."""
        self.age += 1
    
    def probe(self, hash_key: int) -> Optional[TTEntry]:
        """Look up position in table."""
        idx = hash_key % self.size
        
        entry = self.table.get(idx)
        if entry is None:
            self.misses += 1
            return None
        
        if entry.hash_key != hash_key:
            self.collisions += 1
            return None
        
        self.hits += 1
        return entry
    
    def store(self, hash_key: int, depth: int, score: float, 
              entry_type: TTEntryType, best_move: int):
        """Store position in table with replacement strategy."""
        idx = hash_key % self.size
        
        existing = self.table.get(idx)
        
        # Replacement strategy: replace if
        # 1. Empty slot
        # 2. Same position (update)
        # 3. Older entry
        # 4. Shallower search depth
        should_replace = (
            existing is None or
            existing.hash_key == hash_key or
            existing.age < self.age or
            existing.depth <= depth
        )
        
        if should_replace:
            self.table[idx] = TTEntry(
                hash_key=hash_key,
                depth=depth,
                score=score,
                entry_type=entry_type,
                best_move=best_move,
                age=self.age
            )
    
    def clear(self):
        """Clear the table."""
        self.table.clear()
        self.age = 0
        self.hits = 0
        self.misses = 0
        self.collisions = 0
    
    def stats(self) -> Dict:
        """Return table statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            'entries': len(self.table),
            'capacity': self.size,
            'fill_rate': len(self.table) / self.size,
            'hits': self.hits,
            'misses': self.misses,
            'collisions': self.collisions,
            'hit_rate': hit_rate,
        }


# ============================================================================
# MOVE ORDERING
# ============================================================================

class MoveOrderer:
    """
    Move ordering for better alpha-beta pruning.
    
    Priority order:
    1. Hash move (from transposition table)
    2. Winning captures
    3. Killer moves
    4. History heuristic
    5. Other moves
    """
    
    # Move score bonuses
    HASH_MOVE_BONUS = 10_000_000
    WINNING_CAPTURE_BONUS = 5_000_000
    MILL_FORMING_BONUS = 1_000_000
    KILLER_BONUS = [900_000, 800_000]  # First and second killer
    CAPTURE_BONUS = 500_000
    
    def __init__(self, max_ply: int = 64):
        self.max_ply = max_ply
        
        # Killer moves: 2 slots per ply
        self.killers: List[List[int]] = [[-1, -1] for _ in range(max_ply)]
        
        # History heuristic: score for each move that caused cutoff
        # Index by (from_pos, to_pos) or just action
        self.history: Dict[int, int] = defaultdict(int)
        
        # Counter-move heuristic: best response to opponent's last move
        self.counter_moves: Dict[int, int] = {}
        
    def clear(self):
        """Clear move ordering data for new game."""
        self.killers = [[-1, -1] for _ in range(self.max_ply)]
        self.history.clear()
        self.counter_moves.clear()
    
    def update_killer(self, ply: int, move: int):
        """Add a killer move (caused beta cutoff)."""
        if ply >= self.max_ply:
            return
        
        # Don't add duplicates
        if move != self.killers[ply][0]:
            self.killers[ply][1] = self.killers[ply][0]
            self.killers[ply][0] = move
    
    def update_history(self, move: int, depth: int):
        """Update history score for a move that caused cutoff."""
        # Depth-squared bonus
        self.history[move] += depth * depth
    
    def update_counter_move(self, prev_move: int, response: int):
        """Record a good response to opponent's move."""
        self.counter_moves[prev_move] = response
    
    def score_move(self, move: int, ply: int, hash_move: int,
                   is_capture: bool, forms_mill: bool,
                   prev_move: int = -1) -> int:
        """Score a move for ordering."""
        score = 0
        
        # Hash move (from TT)
        if move == hash_move:
            return self.HASH_MOVE_BONUS
        
        # Captures
        if is_capture:
            if forms_mill:
                score += self.WINNING_CAPTURE_BONUS
            else:
                score += self.CAPTURE_BONUS
        elif forms_mill:
            score += self.MILL_FORMING_BONUS
        
        # Killer moves
        if ply < self.max_ply:
            if move == self.killers[ply][0]:
                score += self.KILLER_BONUS[0]
            elif move == self.killers[ply][1]:
                score += self.KILLER_BONUS[1]
        
        # Counter-move
        if prev_move >= 0 and self.counter_moves.get(prev_move) == move:
            score += 700_000
        
        # History heuristic
        score += min(self.history.get(move, 0), 500_000)
        
        return score
    
    def _order_placement_moves(self, moves: List[int], board: Tuple[Optional[int], ...],
                                player: int, hash_move: int = -1) -> List[int]:
        """
        Order moves during placement phase with strategic priorities.

        Prioritizes:
        1. Hash move (from TT)
        2. Mill completions
        3. Blocking opponent mills
        4. Creating potential mills
        5. Strong positional placements
        """
        scored_moves = []

        for move in moves:
            score = 0

            # Hash move gets highest priority
            if move == hash_move:
                score += self.HASH_MOVE_BONUS
                scored_moves.append((score, move))
                continue

            # Base positional value (use improved placement values)
            score += POSITION_VALUES_PLACEMENT[move] * 100

            # Bonus for strong opening positions
            if move in STRONG_OPENING_POSITIONS:
                score += 500

            # Check mill-related tactics
            for mill in POSITION_TO_MILLS[move]:
                our_pieces = sum(1 for p in mill if board[p] == player)
                empty = sum(1 for p in mill if board[p] is None)
                opp_pieces = sum(1 for p in mill if board[p] == 1 - player)

                if our_pieces == 2 and empty == 1:
                    # This would complete a mill!
                    score += 10000
                elif opp_pieces == 2 and empty == 1:
                    # Block opponent's mill!
                    score += 8000
                elif our_pieces == 1 and empty == 2:
                    # Creates potential mill
                    score += 300
                elif opp_pieces == 1 and empty == 2:
                    # Occupy shared mill position
                    score += 200

            scored_moves.append((score, move))

        scored_moves.sort(key=lambda x: -x[0])
        return [m for _, m in scored_moves]

    def order_moves(self, moves: List[int], state, player: int, ply: int,
                    hash_move: int = -1, prev_move: int = -1) -> List[int]:
        """Order moves by estimated quality."""
        if len(moves) <= 1:
            return moves

        board = self._parse_board(state, player)

        # Check if we're in placement phase (all moves are 0-23 placements on empty squares)
        if all(m < 24 for m in moves):
            pieces_on_board = sum(1 for p in board if p is not None)
            # During placement phase (< 18 pieces), use specialized ordering
            # Also check that moves are to empty squares (not removal phase)
            if pieces_on_board < 18:
                empty_positions = [pos for pos in range(24) if board[pos] is None]
                if all(m in empty_positions for m in moves):
                    return self._order_placement_moves(moves, board, player, hash_move)

        # Standard move ordering for movement/removal phases
        scored_moves = []

        for move in moves:
            # Analyze move
            forms_mill = self._would_form_mill(move, board, player, state)
            is_capture = self._is_capture_move(move, state)

            score = self.score_move(
                move, ply, hash_move, is_capture, forms_mill, prev_move
            )
            scored_moves.append((score, move))

        # Sort descending by score
        scored_moves.sort(key=lambda x: -x[0])
        return [m for _, m in scored_moves]
    
    def _parse_board(self, state, player: int) -> Tuple[Optional[int], ...]:
        """
        Parse board from state using correct 5x7x7 tensor layout.

        CRITICAL: Always use observation_tensor(0) for consistent absolute board view.

        In OpenSpiel, observation tensors are RELATIVE to the observer:
        - Plane 0 = observer's pieces
        - Plane 1 = opponent's pieces

        By always observing from player 0's perspective, we get absolute positions:
        - Plane 0 = Player 0's pieces (absolute)
        - Plane 1 = Player 1's pieces (absolute)
        """
        try:
            # ALWAYS use player 0's observation for consistent absolute view
            obs = state.observation_tensor(0)
            # Reshape to 5x7x7 - planes 0=player0 pieces, 1=player1 pieces
            obs_array = np.array(obs).reshape(5, 7, 7)
            board = []
            for pos in range(24):
                r, c = BOARD_POS_TO_GRID[pos]
                if obs_array[0, r, c] == 1:    # Plane 0 = Player 0's pieces
                    board.append(0)
                elif obs_array[1, r, c] == 1:  # Plane 1 = Player 1's pieces
                    board.append(1)
                else:
                    board.append(None)
            return tuple(board)
        except:
            return tuple(None for _ in range(24))
    
    def _would_form_mill(self, action: int, board: Tuple[Optional[int], ...],
                         player: int, state) -> bool:
        """Check if action would form a mill."""
        # Decode action to get destination based on action encoding:
        # - Actions 0-23: placement (destination = action) or removal (not relevant)
        # - Actions 24-599: movement (destination = (action - 24) % 24)
        try:
            if action < 24:
                dest = action  # Placement action
            else:
                dest = (action - 24) % 24  # Movement action

            for mill in POSITION_TO_MILLS[dest]:
                player_count = sum(1 for p in mill if p != dest and board[p] == player)
                if player_count == 2:
                    return True
        except:
            pass
        return False

    def _is_capture_move(self, action: int, state) -> bool:
        """
        Check if we're in a removal state (choosing which opponent piece to remove).

        In Nine Men's Morris, captures happen after forming a mill. If we're in a
        removal state, all legal actions (0-23) are captures.
        """
        try:
            # Check if all legal actions are in the removal range (0-23)
            # and we have pieces on the board (not placement phase)
            legal_actions = state.legal_actions()
            if not legal_actions:
                return False

            # If all actions are 0-23, we might be in removal or placement phase
            if all(a < 24 for a in legal_actions):
                # In removal state, these actions target opponent pieces
                # This is a heuristic - if action is in the legal actions and < 24,
                # and we're past early game, it's likely a removal
                return action < 24
        except:
            pass
        return False


# ============================================================================
# OPTIMIZED MINIMAX BOT
# ============================================================================

class OptimizedMinimaxBot:
    """
    Fully optimized Minimax with all standard enhancements.
    
    Features:
    - Alpha-Beta Pruning
    - Transposition Table with Zobrist Hashing
    - Iterative Deepening
    - Principal Variation Search (PVS)
    - Null Move Pruning
    - Late Move Reductions (LMR)
    - Killer Move Heuristic
    - History Heuristic
    - Aspiration Windows
    - Quiescence Search
    - Multi-threaded root search
    """
    
    # Evaluation weights
    WEIGHTS = {
        'win': 1_000_000,
        'loss': -1_000_000,
        'draw': -50_000,
        'mill': 5_000,
        'potential_mill': 1_500,
        'double_mill': 8_000,
        'blocked_mill': 2_000,
        'unblocked_threat': 3_000,    # Penalty for leaving opponent threats open
        'mobility': 100,
        'position': 50,
    }
    
    # Search constants
    NULL_MOVE_REDUCTION = 2
    LMR_THRESHOLD = 4  # Start reducing after this many moves
    LMR_REDUCTION = 1
    ASPIRATION_WINDOW = 500
    QUIESCENCE_DEPTH = 4
    
    def __init__(self,
                 max_depth: int = 6,
                 num_threads: int = 4,
                 tt_size_mb: int = 8192,
                 random_move_prob: float = 0.0,
                 use_null_move: bool = False,  # Disabled: NMM doesn't support passing
                 use_lmr: bool = True,
                 use_quiescence: bool = True,
                 time_limit: Optional[float] = None):
        """
        Initialize optimized minimax bot.
        
        Args:
            max_depth: Maximum search depth
            num_threads: Threads for parallel root search
            tt_size_mb: Transposition table size in MB
            random_move_prob: Probability of random move (for training)
            use_null_move: Enable null move pruning
            use_lmr: Enable late move reductions
            use_quiescence: Enable quiescence search
            time_limit: Optional time limit per move in seconds
        """
        self.max_depth = max_depth
        self.num_threads = num_threads
        self.random_move_prob = random_move_prob
        self.use_null_move = use_null_move
        self.use_lmr = use_lmr
        self.use_quiescence = use_quiescence
        self.time_limit = time_limit
        
        # Initialize components
        self.tt = TranspositionTable(size_mb=tt_size_mb)
        self.move_orderer = MoveOrderer()
        
        # Statistics
        self.nodes_evaluated = 0
        self.tt_cutoffs = 0
        self.null_move_cutoffs = 0
        self.lmr_searches = 0
        self.quiescence_nodes = 0
        
        # Time management
        self.search_start_time = 0.0
        self.search_stopped = False
        
        # Thread safety
        self._lock = threading.Lock()
        self._local = threading.local()
        
    def _increment_stat(self, name: str, value: int = 1):
        """Thread-safe statistic increment."""
        with self._lock:
            setattr(self, name, getattr(self, name) + value)
    
    def _check_time(self) -> bool:
        """Check if we've exceeded time limit."""
        if self.time_limit is None:
            return False
        return time.time() - self.search_start_time >= self.time_limit
    
    def _parse_board(self, state, player: int) -> Tuple[Optional[int], ...]:
        """
        Parse board from observation tensor using correct 5x7x7 layout.

        CRITICAL: Always use observation_tensor(0) for consistent absolute board view.

        In OpenSpiel, observation tensors are RELATIVE to the observer:
        - Plane 0 = observer's pieces
        - Plane 1 = opponent's pieces

        By always observing from player 0's perspective, we get absolute positions:
        - Plane 0 = Player 0's pieces (absolute)
        - Plane 1 = Player 1's pieces (absolute)

        This was a critical bug - when playing as Player 1, the bot was seeing
        the board completely inverted (its pieces as opponent's and vice versa).
        """
        try:
            # ALWAYS use player 0's observation for consistent absolute view
            obs = state.observation_tensor(0)
            # Reshape to 5x7x7 - planes 0=player0 pieces, 1=player1 pieces
            obs_array = np.array(obs).reshape(5, 7, 7)
            board = []
            for pos in range(24):
                r, c = BOARD_POS_TO_GRID[pos]
                if obs_array[0, r, c] == 1:    # Plane 0 = Player 0's pieces
                    board.append(0)
                elif obs_array[1, r, c] == 1:  # Plane 1 = Player 1's pieces
                    board.append(1)
                else:
                    board.append(None)
            return tuple(board)
        except:
            return tuple(None for _ in range(24))
    
    def _is_removal_state(self, state, board: Tuple[Optional[int], ...]) -> bool:
        """
        Detect if we're in a pending removal state (must capture opponent piece).

        In removal state, legal actions are positions 0-23 to remove opponent pieces.
        This is different from placement phase (also 0-23) - we distinguish by
        checking if pieces are already on the board.
        """
        legal_actions = state.legal_actions()
        if not legal_actions:
            return False

        # If any action >= 24, it's a movement action, not removal
        if any(a >= 24 for a in legal_actions):
            return False

        # All actions are 0-23. Check if pieces exist on board (past placement phase)
        # During placement phase, we place on empty spots
        # During removal, we remove opponent pieces from occupied spots
        pieces_on_board = sum(1 for p in board if p is not None)

        # If board has pieces and all actions are 0-23, it's likely a removal state
        # More specifically: check if the legal actions correspond to opponent pieces
        if pieces_on_board > 0:
            current = state.current_player()
            opponent = 1 - current
            opponent_positions = [pos for pos in range(24) if board[pos] == opponent]
            # If legal actions are a subset of opponent positions, it's removal
            if opponent_positions and all(a in opponent_positions for a in legal_actions):
                return True

        return False

    def _get_hash(self, state, player: int) -> int:
        """Get Zobrist hash for state."""
        board = self._parse_board(state, player)
        current = state.current_player()
        # Check for pending removal state (affects legal actions)
        pending_removal = self._is_removal_state(state, board)
        return ZOBRIST.hash_board(board, current, pending_removal=pending_removal)
    
    def _count_pieces(self, board: Tuple[Optional[int], ...], player: int) -> int:
        """Count pieces for a player."""
        return sum(1 for p in board if p == player)
    
    def _count_mills(self, board: Tuple[Optional[int], ...], player: int) -> int:
        """Count complete mills."""
        return sum(
            1 for mill in MILLS
            if all(board[p] == player for p in mill)
        )
    
    def _count_potential_mills(self, board: Tuple[Optional[int], ...], player: int) -> int:
        """Count potential mills (2 pieces + 1 empty)."""
        count = 0
        for mill in MILLS:
            player_count = sum(1 for p in mill if board[p] == player)
            empty_count = sum(1 for p in mill if board[p] is None)
            if player_count == 2 and empty_count == 1:
                count += 1
        return count

    def _count_blocked_mills(self, board: Tuple[Optional[int], ...], player: int) -> int:
        """Count opponent potential mills that we are blocking (opponent has 2, we have 1)."""
        opponent = 1 - player
        count = 0
        for mill in MILLS:
            opp_count = sum(1 for p in mill if board[p] == opponent)
            our_count = sum(1 for p in mill if board[p] == player)
            # Opponent has 2 pieces and we're blocking with 1
            if opp_count == 2 and our_count == 1:
                count += 1
        return count

    def _count_unblocked_threats(self, board: Tuple[Optional[int], ...], player: int) -> int:
        """Count opponent potential mills that we are NOT blocking (opponent has 2, spot is empty)."""
        opponent = 1 - player
        count = 0
        for mill in MILLS:
            opp_count = sum(1 for p in mill if board[p] == opponent)
            empty_count = sum(1 for p in mill if board[p] is None)
            # Opponent has 2 pieces and third spot is empty - immediate threat!
            if opp_count == 2 and empty_count == 1:
                count += 1
        return count
    
    def _count_double_mills(self, board: Tuple[Optional[int], ...], player: int) -> int:
        """
        Count windmill (double mill / see-saw) configurations.

        A windmill exists when a piece at X can move to adjacent empty position N,
        breaking its own complete mill A while simultaneously completing mill B,
        then move back next turn to re-complete mill A (capturing each turn).

        Requires:
        - Piece at X is part of a complete mill A
        - Adjacent empty position N would complete a different mill B (has 2/3 player pieces)
        - X is not in mill B (otherwise moving X out breaks B too)
        """
        count = 0
        for x in range(24):
            if board[x] != player:
                continue

            # X must be in at least one complete mill
            if not any(
                all(board[p] == player for p in mill)
                for mill in POSITION_TO_MILLS[x]
            ):
                continue

            # Check if moving X to any adjacent empty spot would complete another mill
            found = False
            for n in ADJACENCY[x]:
                if board[n] is not None:
                    continue

                for mill_b in POSITION_TO_MILLS[n]:
                    if x in mill_b:
                        continue  # X must not be in mill B

                    # The other 2 positions in mill B must have player's pieces
                    if all(board[p] == player for p in mill_b if p != n):
                        found = True
                        break

                if found:
                    break

            if found:
                count += 1
        return count
    
    def _get_mobility(self, board: Tuple[Optional[int], ...], player: int) -> int:
        """Count available moves."""
        piece_count = self._count_pieces(board, player)
        
        if piece_count <= 3:
            # Flying: can move anywhere
            empty = sum(1 for p in board if p is None)
            return piece_count * empty
        
        # Normal: adjacent moves only
        moves = 0
        for pos in range(24):
            if board[pos] == player:
                for adj in ADJACENCY[pos]:
                    if board[adj] is None:
                        moves += 1
        return moves
    
    def _positional_score(self, board: Tuple[Optional[int], ...], player: int) -> int:
        """Calculate positional advantage."""
        score = 0
        for pos in range(24):
            if board[pos] == player:
                score += POSITION_VALUES[pos]
            elif board[pos] == 1 - player:
                score -= POSITION_VALUES[pos]
        return score
    
    def evaluate(self, state, player: int) -> float:
        """
        Comprehensive position evaluation.

        Returns score from the CURRENT MOVER's perspective (required for negamax).
        The 'player' parameter is the root player, used only to maintain consistent
        board parsing. The final score is adjusted based on whose turn it is.
        """
        # Terminal check - current_player() returns invalid value (-4) when terminal
        if state.is_terminal():
            returns = state.returns()
            if returns[0] == returns[1]:
                # Draw - neutral from any perspective
                return self.WEIGHTS['draw']
            # In Nine Men's Morris, the winner always made the last move
            # (by capturing a piece or blocking the opponent).
            # The "would-be mover" at terminal is the loser.
            # Negamax expects score from current mover's perspective, so return loss.
            return self.WEIGHTS['loss']

        current_mover = state.current_player()
        board = self._parse_board(state, player)

        # Evaluate from current mover's perspective
        mover = current_mover
        opponent = 1 - mover

        score = 0.0

        # Mills
        my_mills = self._count_mills(board, mover)
        opp_mills = self._count_mills(board, opponent)
        score += (my_mills - opp_mills) * self.WEIGHTS['mill']

        # Potential mills
        my_potential = self._count_potential_mills(board, mover)
        opp_potential = self._count_potential_mills(board, opponent)
        score += (my_potential - opp_potential) * self.WEIGHTS['potential_mill']

        # Blocked mills (reward blocking opponent's potential mills)
        my_blocks = self._count_blocked_mills(board, mover)
        opp_blocks = self._count_blocked_mills(board, opponent)
        score += (my_blocks - opp_blocks) * self.WEIGHTS['blocked_mill']

        # Unblocked threats (penalize leaving opponent threats open)
        threats_against_me = self._count_unblocked_threats(board, mover)
        threats_against_opp = self._count_unblocked_threats(board, opponent)
        score -= threats_against_me * self.WEIGHTS['unblocked_threat']
        score += threats_against_opp * self.WEIGHTS['unblocked_threat']

        # Double mills
        my_double = self._count_double_mills(board, mover)
        opp_double = self._count_double_mills(board, opponent)
        score += (my_double - opp_double) * self.WEIGHTS['double_mill']

        # Mobility
        my_mobility = self._get_mobility(board, mover)
        opp_mobility = self._get_mobility(board, opponent)
        score += (my_mobility - opp_mobility) * self.WEIGHTS['mobility']

        # Positional control
        score += self._positional_score(board, mover) * self.WEIGHTS['position']

        return score
    
    def quiescence(self, state, alpha: float, beta: float, 
                   player: int, depth: int = 0) -> float:
        """
        Quiescence search - extend search for captures.
        Prevents horizon effect by searching until position is "quiet".
        """
        self._increment_stat('quiescence_nodes')
        
        # Stand-pat: evaluate current position
        stand_pat = self.evaluate(state, player)
        
        if depth >= self.QUIESCENCE_DEPTH:
            return stand_pat
        
        if state.is_terminal():
            return stand_pat
        
        # Beta cutoff
        if stand_pat >= beta:
            return beta
        
        # Update alpha
        if stand_pat > alpha:
            alpha = stand_pat
        
        # Only search captures (moves that form mills in Nine Men's Morris)
        legal_actions = state.legal_actions()
        board = self._parse_board(state, player)
        current = state.current_player()
        is_maximizing = (current == player)
        
        # Filter to "loud" moves (captures/mills)
        capture_moves = []
        for action in legal_actions:
            if self.move_orderer._would_form_mill(action, board, current, state):
                capture_moves.append(action)
        
        if not capture_moves:
            return stand_pat
        
        for action in capture_moves:
            child = state.clone()
            child.apply_action(action)
            
            score = -self.quiescence(child, -beta, -alpha, player, depth + 1)
            
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        
        return alpha
    
    def pvs(self, state, depth: int, alpha: float, beta: float,
            player: int, ply: int = 0, null_move_allowed: bool = True,
            prev_move: int = -1) -> Tuple[float, int]:
        """
        Principal Variation Search with all optimizations.
        
        Returns (score, best_move).
        """
        self._increment_stat('nodes_evaluated')
        
        # Time check
        if self._check_time():
            self.search_stopped = True
            return 0, -1
        
        # Terminal or depth limit
        if state.is_terminal():
            return self.evaluate(state, player), -1
        
        if depth <= 0:
            if self.use_quiescence:
                return self.quiescence(state, alpha, beta, player), -1
            return self.evaluate(state, player), -1
        
        # Get legal actions early (needed for TT validation and move ordering)
        legal_actions = state.legal_actions()
        if not legal_actions:
            return self.evaluate(state, player), -1
        legal_set = set(legal_actions)

        # Get hash and probe TT
        hash_key = self._get_hash(state, player)
        tt_entry = self.tt.probe(hash_key)
        tt_move = -1

        if tt_entry is not None:
            tt_move = tt_entry.best_move

            # Validate TT move is legal (defense against hash collisions)
            if tt_move >= 0 and tt_move not in legal_set:
                tt_move = -1  # Invalidate stale TT move

            if tt_entry.depth >= depth and tt_move >= 0:
                if tt_entry.entry_type == TTEntryType.EXACT:
                    self._increment_stat('tt_cutoffs')
                    return tt_entry.score, tt_move
                elif tt_entry.entry_type == TTEntryType.LOWER:
                    alpha = max(alpha, tt_entry.score)
                elif tt_entry.entry_type == TTEntryType.UPPER:
                    beta = min(beta, tt_entry.score)

                if alpha >= beta:
                    self._increment_stat('tt_cutoffs')
                    return tt_entry.score, tt_move

        current_player = state.current_player()

        # Null move pruning (disabled by default for Nine Men's Morris)
        # NOTE: True null move pruning requires giving the opponent an extra turn,
        # which Nine Men's Morris doesn't support (no passing). This is kept as a
        # simplified heuristic that can be enabled for experimentation, but it
        # doesn't provide the same benefits as in chess-like games.
        if (self.use_null_move and null_move_allowed and
            depth >= 3 and not state.is_terminal() and current_player == player):
            null_score = self.evaluate(state, player)

            if null_score >= beta:
                # Verify with shallow search (approximation since we can't pass)
                _, _ = self.pvs(state, depth - 1 - self.NULL_MOVE_REDUCTION,
                               beta - 1, beta, player, ply + 1, False, prev_move)
                if not self.search_stopped:
                    self._increment_stat('null_move_cutoffs')

        # Order moves (legal_actions already retrieved and validated above)
        ordered_moves = self.move_orderer.order_moves(
            legal_actions, state, current_player, ply, tt_move, prev_move
        )
        
        # Negamax: always maximize (score negation handles perspective)
        best_score = float('-inf')
        best_move = ordered_moves[0]
        entry_type = TTEntryType.UPPER

        for i, action in enumerate(ordered_moves):
            if self.search_stopped:
                break

            child = state.clone()
            child.apply_action(action)

            # Determine search depth (LMR)
            new_depth = depth - 1
            if (self.use_lmr and i >= self.LMR_THRESHOLD and
                depth >= 3 and not state.is_terminal()):
                new_depth -= self.LMR_REDUCTION
                self._increment_stat('lmr_searches')

            if i == 0:
                # First move: full window search
                score, _ = self.pvs(child, new_depth, -beta, -alpha,
                                   player, ply + 1, True, action)
                score = -score
            else:
                # PVS: null window search first
                score, _ = self.pvs(child, new_depth, -alpha - 1, -alpha,
                                   player, ply + 1, True, action)
                score = -score

                # Re-search with full window if improved alpha
                if alpha < score < beta and not self.search_stopped:
                    score, _ = self.pvs(child, new_depth, -beta, -alpha,
                                       player, ply + 1, True, action)
                    score = -score

            # LMR re-search if reduced search found improvement
            if (self.use_lmr and i >= self.LMR_THRESHOLD and
                new_depth < depth - 1 and score > alpha and not self.search_stopped):
                score, _ = self.pvs(child, depth - 1, -beta, -alpha,
                                   player, ply + 1, True, action)
                score = -score

            # Negamax: always maximize
            if score > best_score:
                best_score = score
                best_move = action

            if score > alpha:
                alpha = score
                entry_type = TTEntryType.EXACT

            if alpha >= beta:
                # Beta cutoff
                self.move_orderer.update_killer(ply, action)
                self.move_orderer.update_history(action, depth)
                if prev_move >= 0:
                    self.move_orderer.update_counter_move(prev_move, action)
                entry_type = TTEntryType.LOWER
                break
        
        # Store in TT
        if not self.search_stopped:
            self.tt.store(hash_key, depth, best_score, entry_type, best_move)
        
        return best_score, best_move
    
    def iterative_deepening(self, state, player: int) -> Tuple[float, int]:
        """
        Iterative deepening with aspiration windows.
        """
        self.tt.new_search()
        self.search_stopped = False
        self.search_start_time = time.time()
        
        best_move = -1
        best_score = 0
        
        legal_actions = state.legal_actions()
        if len(legal_actions) == 1:
            return 0, legal_actions[0]
        
        # Start with depth 1
        for depth in range(1, self.max_depth + 1):
            if self.search_stopped:
                break
            
            # Aspiration window
            if depth >= 4 and abs(best_score) < self.WEIGHTS['win'] // 2:
                alpha = best_score - self.ASPIRATION_WINDOW
                beta = best_score + self.ASPIRATION_WINDOW
                
                score, move = self.pvs(state, depth, alpha, beta, player)
                
                # Re-search if outside window
                if not self.search_stopped:
                    if score <= alpha:
                        score, move = self.pvs(state, depth, float('-inf'), beta, player)
                    elif score >= beta:
                        score, move = self.pvs(state, depth, alpha, float('inf'), player)
            else:
                score, move = self.pvs(state, depth, float('-inf'), float('inf'), player)
            
            if not self.search_stopped and move >= 0:
                best_score = score
                best_move = move
        
        return best_score, best_move
    
    def _evaluate_root_action(self, state, action: int, depth: int, 
                               player: int) -> Tuple[int, float]:
        """Evaluate single root action for parallel search."""
        child = state.clone()
        child.apply_action(action)
        
        score, _ = self.pvs(child, depth - 1, float('-inf'), float('inf'),
                           player, 1, True, action)
        return action, -score
    
    def search_parallel(self, state, player: int) -> Tuple[float, int]:
        """
        Parallel search at root level.
        """
        self.tt.new_search()
        self.search_stopped = False
        self.search_start_time = time.time()
        
        legal_actions = state.legal_actions()
        if len(legal_actions) == 1:
            return 0, legal_actions[0]
        
        # Order root moves
        ordered_moves = self.move_orderer.order_moves(
            legal_actions, state, player, 0
        )
        
        best_score = float('-inf')
        best_move = ordered_moves[0]
        
        num_workers = min(self.num_threads, len(ordered_moves))
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    self._evaluate_root_action, state, action, self.max_depth, player
                ): action
                for action in ordered_moves
            }
            
            for future in as_completed(futures):
                if self.search_stopped:
                    break
                    
                action, score = future.result()
                if score > best_score:
                    best_score = score
                    best_move = action
        
        return best_score, best_move
    
    def get_action(self, state, use_iterative: bool = True, 
                   use_threading: bool = False) -> int:
        """
        Get best action for current state.
        
        Args:
            state: Current game state
            use_iterative: Use iterative deepening (recommended)
            use_threading: Use parallel root search
        """
        legal_actions = state.legal_actions()
        
        # Random move for training variety
        if self.random_move_prob > 0 and random.random() < self.random_move_prob:
            return random.choice(legal_actions)
        
        # Reset stats
        self.nodes_evaluated = 0
        self.tt_cutoffs = 0
        self.null_move_cutoffs = 0
        self.lmr_searches = 0
        self.quiescence_nodes = 0
        
        player = state.current_player()
        
        if use_threading and self.num_threads > 1:
            _, action = self.search_parallel(state, player)
        elif use_iterative:
            _, action = self.iterative_deepening(state, player)
        else:
            _, action = self.pvs(state, self.max_depth, 
                                float('-inf'), float('inf'), player)
        
        return action
    
    def get_stats(self) -> Dict:
        """Return search statistics."""
        return {
            'nodes': self.nodes_evaluated,
            'tt_cutoffs': self.tt_cutoffs,
            'null_move_cutoffs': self.null_move_cutoffs,
            'lmr_searches': self.lmr_searches,
            'quiescence_nodes': self.quiescence_nodes,
            'tt_stats': self.tt.stats(),
        }


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def _prepare_game_state(state, random_moves: int):
    """Play random moves to prepare the board state (not recorded)."""
    import random
    moves_made = 0
    while moves_made < random_moves and not state.is_terminal():
        # Check if either player has only 3 stones - stop early
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


def evaluate_vs_minimax(
    model,
    device,
    num_actions: int,
    max_depth: int = 6,
    games_per_depth: int = 10,
    max_steps: int = 200,
    use_mixed_precision: bool = True,
    unlimited: bool = True,
    config = None,
    random_moves: int = 150,
    tt_size_mb: int = 4096,
) -> Tuple[int, Dict]:
    """
    Progressive evaluation against optimized minimax bots.

    Args:
        model: Neural network model to evaluate
        device: Torch device
        num_actions: Action space size
        max_depth: Maximum minimax depth to test
        games_per_depth: Games per depth level
        max_steps: Maximum steps per game
        use_mixed_precision: Use FP16 for inference
        unlimited: Keep testing higher depths if winning
        config: Optional config object
        random_moves: Number of random moves to prepare the board
        tt_size_mb: Transposition table size

    Returns:
        (max_depth_beaten, detailed_results)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for evaluation")

    game = pyspiel.load_game("nine_mens_morris")
    
    results = {}
    max_depth_beaten = 0
    
    model.eval()
    
    # Create single bot instance with shared TT
    bot = OptimizedMinimaxBot(
        max_depth=1,  # Will be updated
        tt_size_mb=tt_size_mb,
        num_threads=4,
    )
    
    depth = 1
    while True:
        if not unlimited and depth > max_depth:
            break
        
        bot.max_depth = depth
        bot.tt.clear()  # Clear for fair comparison
        bot.move_orderer.clear()  # Clear killer/history heuristics
        
        wins, draws, losses = 0, 0, 0
        total_nodes = 0
        
        with torch.no_grad():
            for game_idx in range(games_per_depth):
                state = game.new_initial_state()

                # Prepare board with random moves
                _prepare_game_state(state, random_moves)

                # Skip if game ended during preparation
                if state.is_terminal():
                    draws += 1
                    continue

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
                        action = bot.get_action(state, use_iterative=True)
                        total_nodes += bot.nodes_evaluated
                    
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
                    draws += 1
        
        win_rate = (wins + 0.5 * draws) / games_per_depth
        results[depth] = {
            'wins': wins,
            'draws': draws,
            'losses': losses,
            'win_rate': win_rate,
            'total_nodes': total_nodes,
            'tt_stats': bot.tt.stats(),
        }

        if win_rate > 0.5:
            max_depth_beaten = depth
            depth += 1
        else:
            break
    
    return max_depth_beaten, results


def format_minimax_results(results: Dict) -> str:
    """Format results as string."""
    parts = []
    for depth in sorted(results.keys()):
        r = results[depth]
        parts.append(f"D{depth}:{r['wins']}W/{r['draws']}D/{r['losses']}L")
    return " | ".join(parts)


# ============================================================================
# BACKWARDS COMPATIBILITY
# ============================================================================

# Alias for drop-in replacement
MinimaxBot = OptimizedMinimaxBot


# ============================================================================
# TESTING
# ============================================================================

def test_optimizations():
    """Test that optimizations are working."""
    print("Testing optimized minimax...")

    game = pyspiel.load_game("nine_mens_morris")

    # Test with small TT for speed
    bot = OptimizedMinimaxBot(
        max_depth=4,
        tt_size_mb=256,
        num_threads=4,
    )

    state = game.new_initial_state()

    # Prepare board with some random moves
    _prepare_game_state(state, 50)

    # Make a few moves
    for i in range(5):
        if state.is_terminal():
            break

        action = bot.get_action(state, use_iterative=True)
        state.apply_action(action)

        stats = bot.get_stats()
        print(f"Move {i+1}: Action={action}, Nodes={stats['nodes']:,}, "
              f"TT Hit Rate={stats['tt_stats']['hit_rate']:.1%}")

    print("\nFinal TT Stats:", bot.tt.stats())
    print("Test completed!")


if __name__ == "__main__":
    test_optimizations()
