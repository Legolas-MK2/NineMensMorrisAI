"""
Nine Men's Morris - Model Testing Web UI
Interactive interface for testing trained models against various opponents.
"""

import os
import sys
import json
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import torch
from flask import Flask, render_template_string, jsonify, request

# Add claude directory to path
sys.path.insert(0, str(Path(__file__).parent / "claude"))

from model import ActorCritic
from config import Config
from utils import get_legal_mask
from minimax import MinimaxBot
import pyspiel
from game_wrapper import load_game as load_game_fixed

# Import random_train model with alias to avoid conflicts
sys.path.insert(0, str(Path(__file__).parent / "random_train"))
try:
    from random_train.model import ActorCritic as RandomTrainActorCritic
    from random_train.config import Config as RandomTrainConfig
    RANDOM_TRAIN_AVAILABLE = True
except ImportError:
    RANDOM_TRAIN_AVAILABLE = False
    print("Warning: Random train model not available")

app = Flask(__name__)

# Game constants - using pyspiel with position 0 bug fix
GAME = load_game_fixed("nine_mens_morris")
NUM_ACTIONS = GAME.num_distinct_actions()  # 600
OBS_SIZE = GAME.observation_tensor_size()  # 104
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Shared minimax bot with transposition table for better performance
# This allows the TT to persist across moves
_minimax_bot = None

# Board position coordinates for SVG rendering (row, col format)
# Row-by-row numbering: top-to-bottom, left-to-right within each row
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
POINT_TO_COORD = {
    0: (0, 0), 1: (0, 3), 2: (0, 6),
    3: (1, 1), 4: (1, 3), 5: (1, 5),
    6: (2, 2), 7: (2, 3), 8: (2, 4),
    9: (3, 0), 10: (3, 1), 11: (3, 2),
    12: (3, 4), 13: (3, 5), 14: (3, 6),
    15: (4, 2), 16: (4, 3), 17: (4, 4),
    18: (5, 1), 19: (5, 3), 20: (5, 5),
    21: (6, 0), 22: (6, 3), 23: (6, 6),
}

COORD_TO_POINT = {v: k for k, v in POINT_TO_COORD.items()}

# Adjacent positions for each point (row-by-row numbering)
# Cross connections: 1<->4<->7, 9<->10<->11, 12<->13<->14, 16<->19<->22
ADJACENCY = {
    # Top rows
    0: [1, 9], 1: [0, 2, 4], 2: [1, 14],
    3: [4, 10], 4: [1, 3, 5, 7], 5: [4, 13],
    6: [7, 11], 7: [4, 6, 8, 16], 8: [7, 12],
    # Middle row
    9: [0, 10, 21], 10: [3, 9, 11, 18], 11: [6, 10, 15],
    12: [8, 13, 17], 13: [5, 12, 14, 20], 14: [2, 13, 23],
    # Bottom rows
    15: [11, 16], 16: [7, 15, 17, 19], 17: [12, 16],
    18: [10, 19], 19: [16, 18, 20, 22], 20: [13, 19],
    21: [9, 22], 22: [19, 21, 23], 23: [14, 22],
}


@dataclass
class GameState:
    """Holds the current game state and configuration."""
    state: Any = None
    player_types: Dict[int, str] = None  # 0: player0 type, 1: player1 type
    player_models: Dict[int, Any] = None  # Loaded AI models
    player_minimax_depth: Dict[int, int] = None  # Minimax depth per player
    selected_position: Optional[int] = None  # For human moves
    last_move_probs: Optional[Dict[int, float]] = None  # AI move probabilities
    game_phase: str = "placement"  # placement, movement, capture
    waiting_for_capture: bool = False

    def __post_init__(self):
        if self.player_types is None:
            self.player_types = {0: "human", 1: "random"}
        if self.player_models is None:
            self.player_models = {0: None, 1: None}
        if self.player_minimax_depth is None:
            self.player_minimax_depth = {0: 3, 1: 3}


# Global game state
game_state = GameState()


def get_available_models() -> List[Dict[str, str]]:
    """Scan for available trained models."""
    models = []

    # Check claude/models directory
    claude_models_dir = Path(__file__).parent / "claude" / "models"
    if claude_models_dir.exists():
        for pt_file in sorted(claude_models_dir.glob("*.pt"), reverse=True):
            models.append({
                "name": f"claude/{pt_file.name}",
                "path": str(pt_file),
                "type": "claude"
            })

    # Check claude/checkpoints directory
    claude_checkpoints_dir = Path(__file__).parent / "claude" / "checkpoints"
    if claude_checkpoints_dir.exists():
        for pt_file in sorted(claude_checkpoints_dir.glob("*.pt"), reverse=True):
            models.append({
                "name": f"claude/checkpoints/{pt_file.name}",
                "path": str(pt_file),
                "type": "claude"
            })

    # Check random_train/models directory
    random_train_models_dir = Path(__file__).parent / "random_train" / "models"
    if random_train_models_dir.exists():
        for pt_file in sorted(random_train_models_dir.glob("*.pt"), reverse=True):
            models.append({
                "name": f"random_train/{pt_file.name}",
                "path": str(pt_file),
                "type": "random_train"
            })

    # Check random_train/checkpoints directory
    random_train_checkpoints_dir = Path(__file__).parent / "random_train" / "checkpoints"
    if random_train_checkpoints_dir.exists():
        for pt_file in sorted(random_train_checkpoints_dir.glob("*.pt"), reverse=True):
            models.append({
                "name": f"random_train/checkpoints/{pt_file.name}",
                "path": str(pt_file),
                "type": "random_train"
            })

    # Check models directory
    models_dir = Path(__file__).parent / "models"
    if models_dir.exists():
        for pt_file in sorted(models_dir.glob("*.pt"), reverse=True):
            # Skip very large files (likely incomplete or corrupted)
            if pt_file.stat().st_size < 100_000_000:  # 100MB limit
                models.append({
                    "name": f"models/{pt_file.name}",
                    "path": str(pt_file),
                    "type": "gemini"
                })

    return models


def load_model(model_info: Dict[str, str]) -> Tuple[Any, str]:
    """Load a model from disk."""
    model_type = model_info["type"]

    if model_type == "random_train":
        if not RANDOM_TRAIN_AVAILABLE:
            raise ImportError("Random train model not available")
        config = RandomTrainConfig()
        model = RandomTrainActorCritic(OBS_SIZE, NUM_ACTIONS, config).to(DEVICE)
    else:
        # Default to claude model for 'claude' and other types
        config = Config()
        model = ActorCritic(OBS_SIZE, NUM_ACTIONS, config).to(DEVICE)

    checkpoint = torch.load(model_info["path"], map_location=DEVICE, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model, model_info["type"]


def decode_action(action: int, is_capture_phase: bool = False) -> Dict[str, Any]:
    """Decode action ID into human-readable format.

    Note: In nine_mens_morris, captures use the same 0-23 range as placements.
    The is_capture_phase flag determines interpretation.
    """
    if action < 24:
        if is_capture_phase:
            return {"type": "capture", "position": action}
        else:
            return {"type": "place", "position": action}
    else:
        action_offset = action - 24
        from_pos = action_offset // 24
        to_pos = action_offset % 24
        return {"type": "move", "from": from_pos, "to": to_pos}


def encode_action(action_info: Dict[str, Any]) -> int:
    """Encode action info into action ID."""
    if action_info["type"] == "place":
        return action_info["position"]
    elif action_info["type"] == "move":
        return 24 + action_info["from"] * 24 + action_info["to"]
    else:  # capture - same encoding as place
        return action_info["position"]


def get_board_state() -> Dict[str, Any]:
    """Get current board state for rendering."""
    if game_state.state is None:
        return {"positions": {}, "current_player": 0, "phase": "not_started"}

    state = game_state.state
    state_str = str(state)

    # Get legal actions to understand board state
    legal = state.legal_actions()

    # Count pieces - W for player 0, B for player 1
    p0_pieces = state_str.count('W')
    p1_pieces = state_str.count('B')

    # Determine phase by checking the state string for capture indicator
    # and by analyzing legal actions
    phase = "placement"
    if "Capture time" in state_str or "capture" in state_str.lower():
        phase = "capture"
    elif legal and all(a >= 24 for a in legal):
        phase = "movement"

    if state.is_terminal():
        phase = "terminal"

    return {
        "state_str": state_str,
        "current_player": int(state.current_player()) if not state.is_terminal() else -1,
        "phase": phase,
        "is_terminal": state.is_terminal(),
        "legal_actions": [int(a) for a in legal],
        "p0_pieces": p0_pieces,
        "p1_pieces": p1_pieces,
        "returns": [float(r) for r in state.returns()] if state.is_terminal() else None
    }


def parse_board_positions(state) -> Dict[int, int]:
    """Parse board to get piece positions: {position: player}."""
    positions = {}

    # Use observation tensor from player 0's perspective
    # Observation shape is (channels, rows, cols) - e.g. (5, 7, 7)
    # Channel 0: Player 0's pieces (white)
    # Channel 1: Player 1's pieces (black)
    obs = np.array(state.observation_tensor(0)).reshape(GAME.observation_tensor_shape())

    # Iterate through board positions using POINT_TO_COORD mapping
    for pos, (row, col) in POINT_TO_COORD.items():
        if obs[0, row, col] == 1:
            # Player 0's piece (white)
            positions[pos] = 0
        elif obs[1, row, col] == 1:
            # Player 1's piece (black)
            positions[pos] = 1

    return positions


def get_ai_move_with_probs(model, state, player: int, temperature: float = 0.4) -> Tuple[int, Dict[int, float]]:
    """Get AI move and probability distribution.

    Args:
        model: The neural network model
        state: Current game state
        player: Player ID
        temperature: Sampling temperature for action selection
            - 0.0 = always pick best action (deterministic, good vs minimax)
            - 0.3-0.5 = balanced play (less predictable but still strong, good vs humans)
            - 1.0 = sample from full policy distribution

        The overfitting problem occurs when training only against deterministic
        minimax and then using argmax during play. The AI learns ONE specific line
        that beats minimax, but fails when humans play unpredictably.

        Using temperature=0.4 introduces controlled randomness, making the AI
        less exploitable by humans who notice its patterns.
    """
    obs = torch.tensor(
        state.observation_tensor(player),
        dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)

    mask = torch.tensor(
        get_legal_mask(state, NUM_ACTIONS),
        dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits, value = model(obs)
        masked_logits = logits.float()
        masked_logits[mask == 0] = -1e9

        # Apply temperature scaling for controlled randomness
        # Lower temperature = sharper distribution (more deterministic)
        # Higher temperature = flatter distribution (more random)
        if temperature < 0.01:
            # Deterministic mode (argmax)
            action = int(masked_logits.argmax(dim=-1).item())
            probs = torch.softmax(masked_logits, dim=-1)
        else:
            scaled_logits = masked_logits / temperature
            probs = torch.softmax(scaled_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = int(dist.sample().item())

        # Get probability distribution for display (use original probs)
        display_probs = torch.softmax(masked_logits, dim=-1)
        legal_actions = state.legal_actions()
        move_probs = {}
        for act in legal_actions:
            move_probs[int(act)] = float(display_probs[0, act].item())

    return action, move_probs


def get_random_move(state) -> int:
    """Get a random legal move."""
    legal = state.legal_actions()
    return random.choice(legal) if legal else -1


def get_minimax_move(state, depth: int) -> int:
    """Get minimax move with specified depth using optimized minimax.

    Uses a shared bot instance to benefit from transposition table
    persistence across moves in the same game.
    """
    global _minimax_bot

    if _minimax_bot is None or _minimax_bot.max_depth != depth:
        # Create new bot with the requested depth
        # Using moderate TT size for webui (512MB)
        _minimax_bot = MinimaxBot(
            max_depth=depth
        )
    else:
        # Update depth if needed
        _minimax_bot.max_depth = depth

    # Use iterative deepening for best move quality
    return _minimax_bot.get_action(state, use_iterative=True)


# HTML Template with embedded CSS and JavaScript
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Nine Men's Morris - Model Tester</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #e0e0e0;
            padding: 20px;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: 300px 1fr 350px;
            gap: 20px;
        }

        .panel {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        h1 {
            text-align: center;
            margin-bottom: 20px;
            color: #64b5f6;
            grid-column: 1 / -1;
        }

        h2 {
            color: #81c784;
            margin-bottom: 15px;
            font-size: 1.2em;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            padding-bottom: 10px;
        }

        h3 {
            color: #ffb74d;
            margin: 15px 0 10px 0;
            font-size: 1em;
        }

        .player-config {
            margin-bottom: 20px;
        }

        .player-config.player-0 {
            border-left: 4px solid #fff;
            padding-left: 15px;
        }

        .player-config.player-1 {
            border-left: 4px solid #333;
            padding-left: 15px;
        }

        label {
            display: block;
            margin-bottom: 5px;
            color: #b0bec5;
            font-size: 0.9em;
        }

        select, input[type="number"] {
            width: 100%;
            padding: 10px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 8px;
            background: rgba(0, 0, 0, 0.3);
            color: #e0e0e0;
            font-size: 0.9em;
            margin-bottom: 10px;
        }

        select:focus, input:focus {
            outline: none;
            border-color: #64b5f6;
        }

        button {
            width: 100%;
            padding: 12px;
            border: none;
            border-radius: 8px;
            font-size: 1em;
            cursor: pointer;
            transition: all 0.3s;
            margin-bottom: 10px;
        }

        .btn-primary {
            background: linear-gradient(135deg, #4caf50, #45a049);
            color: white;
        }

        .btn-primary:hover {
            background: linear-gradient(135deg, #45a049, #3d8b40);
            transform: translateY(-2px);
        }

        .btn-secondary {
            background: linear-gradient(135deg, #2196f3, #1976d2);
            color: white;
        }

        .btn-secondary:hover {
            background: linear-gradient(135deg, #1976d2, #1565c0);
        }

        .btn-warning {
            background: linear-gradient(135deg, #ff9800, #f57c00);
            color: white;
        }

        /* Board Styles */
        .board-container {
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 500px;
        }

        #board-svg {
            max-width: 100%;
            height: auto;
        }

        .board-point {
            cursor: pointer;
            transition: all 0.2s;
        }

        .board-point:hover {
            filter: brightness(1.3);
        }

        .board-point.legal {
            animation: pulse 1s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.6; }
        }

        .piece {
            pointer-events: none;
        }

        /* Status Panel */
        .status {
            padding: 15px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 10px;
            margin-bottom: 15px;
        }

        .status-item {
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
        }

        .status-label {
            color: #90a4ae;
        }

        .status-value {
            font-weight: bold;
        }

        .current-turn {
            text-align: center;
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 15px;
            font-weight: bold;
        }

        .turn-0 {
            background: rgba(255, 255, 255, 0.2);
            color: #fff;
        }

        .turn-1 {
            background: rgba(50, 50, 50, 0.8);
            color: #aaa;
        }

        /* Probability Display */
        .prob-list {
            max-height: 400px;
            overflow-y: auto;
        }

        .prob-item {
            display: flex;
            align-items: center;
            margin-bottom: 8px;
            padding: 8px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 6px;
        }

        .prob-bar-container {
            flex: 1;
            height: 20px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
            overflow: hidden;
            margin: 0 10px;
        }

        .prob-bar {
            height: 100%;
            background: linear-gradient(90deg, #4caf50, #8bc34a);
            transition: width 0.3s;
        }

        .prob-action {
            width: 120px;
            font-size: 0.85em;
            color: #b0bec5;
        }

        .prob-value {
            width: 60px;
            text-align: right;
            font-weight: bold;
            color: #81c784;
        }

        /* Game Log */
        .game-log {
            max-height: 200px;
            overflow-y: auto;
            font-family: monospace;
            font-size: 0.85em;
            background: rgba(0, 0, 0, 0.3);
            padding: 10px;
            border-radius: 8px;
        }

        .log-entry {
            padding: 3px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }

        .log-player-0 {
            color: #e0e0e0;
        }

        .log-player-1 {
            color: #888;
        }

        /* Terminal State */
        .game-over {
            text-align: center;
            padding: 20px;
            background: rgba(76, 175, 80, 0.2);
            border-radius: 10px;
            margin-bottom: 15px;
        }

        .game-over.winner-0 {
            background: rgba(255, 255, 255, 0.2);
        }

        .game-over.winner-1 {
            background: rgba(100, 100, 100, 0.3);
        }

        .winner-text {
            font-size: 1.5em;
            font-weight: bold;
            margin-bottom: 10px;
        }

        /* Instructions */
        .instructions {
            font-size: 0.85em;
            color: #90a4ae;
            margin-top: 15px;
            padding: 10px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 8px;
        }

        .hidden {
            display: none !important;
        }

        /* Minimax depth */
        .minimax-options {
            margin-top: 10px;
            padding: 10px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 8px;
        }

        /* Model options */
        .model-options {
            margin-top: 10px;
        }

        /* Loading indicator */
        .loading {
            text-align: center;
            padding: 20px;
        }

        .spinner {
            border: 3px solid rgba(255, 255, 255, 0.1);
            border-top: 3px solid #64b5f6;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Nine Men's Morris - Model Tester</h1>

        <!-- Left Panel: Configuration -->
        <div class="panel config-panel">
            <h2>Game Configuration</h2>

            <div class="player-config player-0">
                <h3>Player 1 (White)</h3>
                <label for="player0-type">Player Type</label>
                <select id="player0-type" onchange="updatePlayerOptions(0)">
                    <option value="human">Human</option>
                    <option value="ai">AI Model</option>
                    <option value="minimax">Minimax</option>
                    <option value="random">Random</option>
                </select>

                <div id="player0-model-options" class="model-options hidden">
                    <label for="player0-model">Select Model</label>
                    <select id="player0-model"></select>
                </div>

                <div id="player0-minimax-options" class="minimax-options hidden">
                    <label for="player0-depth">Minimax Depth</label>
                    <input type="number" id="player0-depth" min="1" max="8" value="3">
                </div>
            </div>

            <div class="player-config player-1">
                <h3>Player 2 (Black)</h3>
                <label for="player1-type">Player Type</label>
                <select id="player1-type" onchange="updatePlayerOptions(1)">
                    <option value="human">Human</option>
                    <option value="ai" selected>AI Model</option>
                    <option value="minimax">Minimax</option>
                    <option value="random">Random</option>
                </select>

                <div id="player1-model-options" class="model-options">
                    <label for="player1-model">Select Model</label>
                    <select id="player1-model"></select>
                </div>

                <div id="player1-minimax-options" class="minimax-options hidden">
                    <label for="player1-depth">Minimax Depth</label>
                    <input type="number" id="player1-depth" min="1" max="8" value="3">
                </div>
            </div>

            <button class="btn-primary" onclick="newGame()">New Game</button>
            <button class="btn-secondary" onclick="makeAIMove()">AI Move / Next Step</button>

            <div class="instructions">
                <strong>How to play:</strong><br>
                • Click on empty positions to place/move pieces<br>
                • When moving, click your piece first, then destination<br>
                • After forming a mill (3 in a row), click opponent's piece to capture<br>
                • AI probabilities show on the right panel
            </div>
        </div>

        <!-- Center: Game Board -->
        <div class="panel board-panel">
            <div class="board-container">
                <svg id="board-svg" viewBox="-30 -30 560 560" width="500" height="500">
                    <!-- Board background -->
                    <rect x="-30" y="-30" width="560" height="560" fill="#2d3748"/>

                    <!-- Board lines (row-by-row numbering) -->
                    <g stroke="#4a5568" stroke-width="4" fill="none">
                        <!-- Outer square (positions: 0,1,2,9,14,21,22,23) -->
                        <rect x="0" y="0" width="500" height="500"/>
                        <!-- Middle square (positions: 3,4,5,10,13,18,19,20) -->
                        <rect x="83.33" y="83.33" width="333.33" height="333.33"/>
                        <!-- Inner square (positions: 6,7,8,11,12,15,16,17) -->
                        <rect x="166.66" y="166.66" width="166.66" height="166.66"/>
                        <!-- Cross connections: 1-4-7 (top), 22-19-16 (bottom), 9-10-11 (left), 14-13-12 (right) -->
                        <line x1="250" y1="0" x2="250" y2="166.66"/>
                        <line x1="250" y1="333.33" x2="250" y2="500"/>
                        <line x1="0" y1="250" x2="166.66" y2="250"/>
                        <line x1="333.33" y1="250" x2="500" y2="250"/>
                    </g>

                    <!-- Board points (clickable) -->
                    <g id="board-points"></g>

                    <!-- Highlights for legal moves -->
                    <g id="highlights"></g>

                    <!-- Pieces -->
                    <g id="pieces"></g>
                </svg>
            </div>
        </div>

        <!-- Right Panel: Status and Probabilities -->
        <div class="panel info-panel">
            <h2>Game Status</h2>

            <div id="game-over-display" class="game-over hidden">
                <div class="winner-text" id="winner-text"></div>
            </div>

            <div id="current-turn" class="current-turn turn-0">
                Player 1's Turn (White)
            </div>

            <div class="status">
                <div class="status-item">
                    <span class="status-label">Phase:</span>
                    <span class="status-value" id="phase-display">Placement</span>
                </div>
                <div class="status-item">
                    <span class="status-label">White Pieces:</span>
                    <span class="status-value" id="p0-pieces">0</span>
                </div>
                <div class="status-item">
                    <span class="status-label">Black Pieces:</span>
                    <span class="status-value" id="p1-pieces">0</span>
                </div>
            </div>

            <h2>AI Move Probabilities</h2>
            <div id="prob-container" class="prob-list">
                <p style="color: #90a4ae; text-align: center;">
                    Probabilities will appear when AI makes a move
                </p>
            </div>

            <h2>Game Log</h2>
            <div id="game-log" class="game-log"></div>
        </div>
    </div>

    <script>
        // Game state
        let selectedPosition = null;
        let legalActions = [];
        let boardPositions = {};
        let currentPlayer = 0;
        let gamePhase = 'placement';
        let isTerminal = false;

        // Board coordinates (matching Python POINT_TO_COORD) - (row, col) format
        // Row-by-row numbering: top-to-bottom, left-to-right
        const POINT_TO_COORD = {
            0: [0, 0], 1: [0, 3], 2: [0, 6],
            3: [1, 1], 4: [1, 3], 5: [1, 5],
            6: [2, 2], 7: [2, 3], 8: [2, 4],
            9: [3, 0], 10: [3, 1], 11: [3, 2],
            12: [3, 4], 13: [3, 5], 14: [3, 6],
            15: [4, 2], 16: [4, 3], 17: [4, 4],
            18: [5, 1], 19: [5, 3], 20: [5, 5],
            21: [6, 0], 22: [6, 3], 23: [6, 6],
        };

        // Convert grid coords (row, col) to SVG coords
        function gridToSvg(row, col) {
            return [col * 83.33, row * 83.33];
        }

        // Initialize the board
        function initBoard() {
            const pointsGroup = document.getElementById('board-points');
            pointsGroup.innerHTML = '';

            for (let pos = 0; pos < 24; pos++) {
                const [row, col] = POINT_TO_COORD[pos];
                const [sx, sy] = gridToSvg(row, col);

                const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                circle.setAttribute('cx', sx);
                circle.setAttribute('cy', sy);
                circle.setAttribute('r', 15);
                circle.setAttribute('fill', '#2d3748');
                circle.setAttribute('stroke', '#4a5568');
                circle.setAttribute('stroke-width', 2);
                circle.setAttribute('class', 'board-point');
                circle.setAttribute('data-pos', pos);
                circle.onclick = () => handleClick(pos);

                pointsGroup.appendChild(circle);
            }
        }

        // Update board display
        function updateBoard(state) {
            if (!state) return;

            legalActions = state.legal_actions || [];
            currentPlayer = state.current_player;
            gamePhase = state.phase;
            isTerminal = state.is_terminal;

            // Update status
            document.getElementById('phase-display').textContent =
                gamePhase.charAt(0).toUpperCase() + gamePhase.slice(1);
            document.getElementById('p0-pieces').textContent = state.p0_pieces || 0;
            document.getElementById('p1-pieces').textContent = state.p1_pieces || 0;

            // Update turn indicator
            const turnDiv = document.getElementById('current-turn');
            if (isTerminal) {
                turnDiv.textContent = 'Game Over';
                turnDiv.className = 'current-turn';

                // Show winner
                const gameOverDiv = document.getElementById('game-over-display');
                const winnerText = document.getElementById('winner-text');
                gameOverDiv.classList.remove('hidden');

                if (state.returns) {
                    if (state.returns[0] > state.returns[1]) {
                        winnerText.textContent = 'Player 1 (White) Wins!';
                        gameOverDiv.className = 'game-over winner-0';
                    } else if (state.returns[1] > state.returns[0]) {
                        winnerText.textContent = 'Player 2 (Black) Wins!';
                        gameOverDiv.className = 'game-over winner-1';
                    } else {
                        winnerText.textContent = 'Draw!';
                        gameOverDiv.className = 'game-over';
                    }
                }
            } else {
                document.getElementById('game-over-display').classList.add('hidden');
                if (currentPlayer === 0) {
                    turnDiv.textContent = "Player 1's Turn (White)";
                    turnDiv.className = 'current-turn turn-0';
                } else {
                    turnDiv.textContent = "Player 2's Turn (Black)";
                    turnDiv.className = 'current-turn turn-1';
                }
            }

            // Clear and redraw pieces
            updatePieces(state.positions || {});

            // Highlight legal moves
            updateHighlights();
        }

        // Update pieces on board
        function updatePieces(positions) {
            boardPositions = positions;
            const piecesGroup = document.getElementById('pieces');
            piecesGroup.innerHTML = '';

            for (const [pos, player] of Object.entries(positions)) {
                const [row, col] = POINT_TO_COORD[pos];
                const [sx, sy] = gridToSvg(row, col);

                const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                circle.setAttribute('cx', sx);
                circle.setAttribute('cy', sy);
                circle.setAttribute('r', 25);
                circle.setAttribute('class', 'piece');

                if (player === 0) {
                    circle.setAttribute('fill', '#ffffff');
                    circle.setAttribute('stroke', '#cccccc');
                } else {
                    circle.setAttribute('fill', '#333333');
                    circle.setAttribute('stroke', '#555555');
                }
                circle.setAttribute('stroke-width', 3);

                piecesGroup.appendChild(circle);
            }
        }

        // Update highlights for legal moves
        function updateHighlights() {
            const highlightsGroup = document.getElementById('highlights');
            highlightsGroup.innerHTML = '';

            if (isTerminal) return;

            // Get player type for current player
            const playerType = document.getElementById(`player${currentPlayer}-type`).value;
            if (playerType !== 'human') return;  // Only highlight for human players

            // Determine what to highlight based on phase and selection
            let positionsToHighlight = [];

            if (gamePhase === 'placement') {
                // Highlight all positions where we can place
                for (const action of legalActions) {
                    if (action < 24) {
                        positionsToHighlight.push({pos: action, color: '#4caf50'});
                    }
                }
            } else if (gamePhase === 'movement') {
                if (selectedPosition === null) {
                    // Highlight pieces that can move
                    for (const action of legalActions) {
                        if (action >= 24 && action < 600) {
                            const fromPos = Math.floor((action - 24) / 24);
                            positionsToHighlight.push({pos: fromPos, color: '#2196f3'});
                        }
                    }
                } else {
                    // Highlight valid destinations
                    for (const action of legalActions) {
                        if (action >= 24 && action < 600) {
                            const fromPos = Math.floor((action - 24) / 24);
                            const toPos = (action - 24) % 24;
                            if (fromPos === selectedPosition) {
                                positionsToHighlight.push({pos: toPos, color: '#4caf50'});
                            }
                        }
                    }
                    // Also highlight selected piece
                    positionsToHighlight.push({pos: selectedPosition, color: '#ffeb3b'});
                }
            } else if (gamePhase === 'capture') {
                // Highlight capturable pieces
                for (const action of legalActions) {
                    if (action < 24) {
                        positionsToHighlight.push({pos: action, color: '#f44336'});
                    }
                }
            }

            // Remove duplicates
            const seen = new Set();
            positionsToHighlight = positionsToHighlight.filter(item => {
                if (seen.has(item.pos)) return false;
                seen.add(item.pos);
                return true;
            });

            // Draw highlights
            for (const {pos, color} of positionsToHighlight) {
                const [row, col] = POINT_TO_COORD[pos];
                const [sx, sy] = gridToSvg(row, col);

                const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                circle.setAttribute('cx', sx);
                circle.setAttribute('cy', sy);
                circle.setAttribute('r', 30);
                circle.setAttribute('fill', 'none');
                circle.setAttribute('stroke', color);
                circle.setAttribute('stroke-width', 4);
                circle.setAttribute('opacity', 0.7);
                circle.setAttribute('class', 'legal');

                highlightsGroup.appendChild(circle);
            }
        }

        // Handle click on board position
        async function handleClick(pos) {
            if (isTerminal) return;

            const playerType = document.getElementById(`player${currentPlayer}-type`).value;
            if (playerType !== 'human') {
                console.log('Not human turn');
                return;
            }

            let action = null;

            if (gamePhase === 'placement') {
                // Check if this is a valid placement
                if (legalActions.includes(pos)) {
                    action = pos;
                }
            } else if (gamePhase === 'movement') {
                if (selectedPosition === null) {
                    // Select a piece to move
                    const canMoveFrom = legalActions.some(a => {
                        if (a >= 24 && a < 600) {
                            return Math.floor((a - 24) / 24) === pos;
                        }
                        return false;
                    });
                    if (canMoveFrom) {
                        selectedPosition = pos;
                        updateHighlights();
                    }
                } else {
                    // Try to move to this position
                    const moveAction = 24 + selectedPosition * 24 + pos;
                    if (legalActions.includes(moveAction)) {
                        action = moveAction;
                    }
                    selectedPosition = null;
                }
            } else if (gamePhase === 'capture') {
                if (legalActions.includes(pos)) {
                    action = pos;
                }
            }

            if (action !== null) {
                await makeMove(action);
            } else {
                updateHighlights();
            }
        }

        // Make a move
        async function makeMove(action) {
            try {
                const response = await fetch('/api/move', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({action: action})
                });
                const data = await response.json();

                if (data.success) {
                    updateBoard(data.state);
                    addLogEntry(data.move_description, data.player);
                    selectedPosition = null;

                    // Check if next player is AI
                    if (!data.state.is_terminal) {
                        const nextPlayer = data.state.current_player;
                        const nextType = document.getElementById(`player${nextPlayer}-type`).value;
                        if (nextType !== 'human') {
                            // Small delay then make AI move
                            setTimeout(() => makeAIMove(), 500);
                        }
                    }
                } else {
                    console.error('Move failed:', data.error);
                }
            } catch (err) {
                console.error('Error making move:', err);
            }
        }

        // Make AI move
        async function makeAIMove() {
            if (isTerminal) return;

            const playerType = document.getElementById(`player${currentPlayer}-type`).value;
            if (playerType === 'human') {
                console.log('Human turn - not making AI move');
                return;
            }

            try {
                const depth = document.getElementById(`player${currentPlayer}-depth`)?.value || 3;
                const modelSelect = document.getElementById(`player${currentPlayer}-model`);
                const modelPath = modelSelect?.value || '';

                const response = await fetch('/api/ai_move', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        player_type: playerType,
                        model_path: modelPath,
                        minimax_depth: parseInt(depth)
                    })
                });
                const data = await response.json();

                if (data.success) {
                    updateBoard(data.state);
                    addLogEntry(data.move_description, data.player);

                    // Update probabilities display
                    if (data.probabilities) {
                        displayProbabilities(data.probabilities);
                    }

                    // Check if next player is also AI
                    if (!data.state.is_terminal) {
                        const nextPlayer = data.state.current_player;
                        const nextType = document.getElementById(`player${nextPlayer}-type`).value;
                        if (nextType !== 'human') {
                            setTimeout(() => makeAIMove(), 500);
                        }
                    }
                } else {
                    console.error('AI move failed:', data.error);
                }
            } catch (err) {
                console.error('Error making AI move:', err);
            }
        }

        // Display move probabilities
        function displayProbabilities(probs) {
            const container = document.getElementById('prob-container');

            if (!probs || Object.keys(probs).length === 0) {
                container.innerHTML = '<p style="color: #90a4ae; text-align: center;">No probabilities available</p>';
                return;
            }

            // Sort by probability
            const sorted = Object.entries(probs)
                .sort((a, b) => b[1] - a[1])
                .slice(0, 10);  // Top 10

            container.innerHTML = sorted.map(([action, prob]) => {
                const actionInfo = decodeAction(parseInt(action));
                const percentage = (prob * 100).toFixed(1);
                return `
                    <div class="prob-item">
                        <span class="prob-action">${actionInfo}</span>
                        <div class="prob-bar-container">
                            <div class="prob-bar" style="width: ${percentage}%"></div>
                        </div>
                        <span class="prob-value">${percentage}%</span>
                    </div>
                `;
            }).join('');
        }

        // Decode action to human readable
        // Note: captures use the same 0-23 range as placements, depends on game phase
        function decodeAction(action) {
            if (action < 24) {
                // Could be place or capture depending on phase
                if (gamePhase === 'capture') {
                    return `Capture @ ${action}`;
                } else {
                    return `Place @ ${action}`;
                }
            } else {
                const from = Math.floor((action - 24) / 24);
                const to = (action - 24) % 24;
                return `Move ${from} → ${to}`;
            }
        }

        // Add log entry
        function addLogEntry(description, player) {
            const log = document.getElementById('game-log');
            const entry = document.createElement('div');
            entry.className = `log-entry log-player-${player}`;
            entry.textContent = `P${player + 1}: ${description}`;
            log.insertBefore(entry, log.firstChild);
        }

        // Update player options visibility
        function updatePlayerOptions(player) {
            const type = document.getElementById(`player${player}-type`).value;
            const modelOptions = document.getElementById(`player${player}-model-options`);
            const minimaxOptions = document.getElementById(`player${player}-minimax-options`);

            modelOptions.classList.add('hidden');
            minimaxOptions.classList.add('hidden');

            if (type === 'ai') {
                modelOptions.classList.remove('hidden');
            } else if (type === 'minimax') {
                minimaxOptions.classList.remove('hidden');
            }
        }

        // Load available models
        async function loadModels() {
            try {
                const response = await fetch('/api/models');
                const models = await response.json();

                for (let player = 0; player < 2; player++) {
                    const select = document.getElementById(`player${player}-model`);
                    select.innerHTML = models.map(m =>
                        `<option value="${m.path}">${m.name}</option>`
                    ).join('');
                }
            } catch (err) {
                console.error('Error loading models:', err);
            }
        }

        // Start new game
        async function newGame() {
            try {
                // Get player configurations
                const config = {
                    player0_type: document.getElementById('player0-type').value,
                    player0_model: document.getElementById('player0-model').value,
                    player0_depth: parseInt(document.getElementById('player0-depth').value),
                    player1_type: document.getElementById('player1-type').value,
                    player1_model: document.getElementById('player1-model').value,
                    player1_depth: parseInt(document.getElementById('player1-depth').value)
                };

                const response = await fetch('/api/new_game', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(config)
                });
                const data = await response.json();

                if (data.success) {
                    updateBoard(data.state);
                    document.getElementById('game-log').innerHTML = '';
                    document.getElementById('prob-container').innerHTML =
                        '<p style="color: #90a4ae; text-align: center;">Probabilities will appear when AI makes a move</p>';
                    selectedPosition = null;

                    // If first player is AI, make their move
                    if (config.player0_type !== 'human') {
                        setTimeout(() => makeAIMove(), 500);
                    }
                }
            } catch (err) {
                console.error('Error starting new game:', err);
            }
        }

        // Initialize on page load
        document.addEventListener('DOMContentLoaded', async () => {
            initBoard();
            await loadModels();
            updatePlayerOptions(0);
            updatePlayerOptions(1);
            await newGame();
        });
    </script>
</body>
</html>
'''


# API Routes

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/models')
def api_models():
    return jsonify(get_available_models())


@app.route('/api/new_game', methods=['POST'])
def api_new_game():
    global game_state, _minimax_bot

    config = request.json or {}

    # Clear minimax transposition table for new game
    if _minimax_bot is not None:
        _minimax_bot.tt.clear()
        _minimax_bot.move_orderer.clear()

    # Initialize new game
    game_state = GameState()
    game_state.state = GAME.new_initial_state()

    # Configure players
    game_state.player_types[0] = config.get('player0_type', 'human')
    game_state.player_types[1] = config.get('player1_type', 'ai')
    game_state.player_minimax_depth[0] = config.get('player0_depth', 3)
    game_state.player_minimax_depth[1] = config.get('player1_depth', 3)

    # Load AI models if needed
    models = get_available_models()
    for player in [0, 1]:
        if game_state.player_types[player] == 'ai':
            model_path = config.get(f'player{player}_model', '')
            if model_path:
                # Find matching model
                for m in models:
                    if m['path'] == model_path:
                        try:
                            game_state.player_models[player] = load_model(m)
                        except Exception as e:
                            print(f"Error loading model: {e}")
                        break

    # Get board state
    state_info = get_board_state()
    state_info['positions'] = parse_board_positions(game_state.state)

    return jsonify({
        'success': True,
        'state': state_info
    })


@app.route('/api/move', methods=['POST'])
def api_move():
    global game_state

    if game_state.state is None or game_state.state.is_terminal():
        return jsonify({'success': False, 'error': 'Game not active'})

    data = request.json or {}
    action = data.get('action')

    if action is None:
        return jsonify({'success': False, 'error': 'No action provided'})

    # Check if action is legal
    if action not in game_state.state.legal_actions():
        return jsonify({'success': False, 'error': 'Illegal action'})

    # Get current player and determine phase before move
    player = game_state.state.current_player()
    state_before = get_board_state()
    is_capture = state_before['phase'] == 'capture'

    # Apply action
    game_state.state.apply_action(action)

    # Get updated state
    state_info = get_board_state()
    state_info['positions'] = parse_board_positions(game_state.state)

    # Describe move based on phase before action
    action_info = decode_action(action, is_capture_phase=is_capture)
    if action_info['type'] == 'place':
        desc = f"Placed piece at position {action_info['position']}"
    elif action_info['type'] == 'move':
        desc = f"Moved piece from {action_info['from']} to {action_info['to']}"
    else:
        desc = f"Captured piece at position {action_info['position']}"

    return jsonify({
        'success': True,
        'state': state_info,
        'player': int(player),
        'move_description': desc
    })


@app.route('/api/ai_move', methods=['POST'])
def api_ai_move():
    global game_state

    if game_state.state is None or game_state.state.is_terminal():
        return jsonify({'success': False, 'error': 'Game not active'})

    data = request.json or {}
    player_type = data.get('player_type', 'random')

    current_player = game_state.state.current_player()
    action = None
    probabilities = None

    if player_type == 'ai':
        model_path = data.get('model_path', '')

        # Check if we have a loaded model
        model_data = game_state.player_models.get(current_player)

        if model_data is None and model_path:
            # Try to load model
            models = get_available_models()
            for m in models:
                if m['path'] == model_path:
                    try:
                        model_data = load_model(m)
                        game_state.player_models[current_player] = model_data
                    except Exception as e:
                        print(f"Error loading model: {e}")
                    break

        if model_data:
            model, model_type = model_data
            action, probabilities = get_ai_move_with_probs(model, game_state.state, current_player)
        else:
            # Fallback to random
            action = get_random_move(game_state.state)

    elif player_type == 'minimax':
        depth = data.get('minimax_depth', game_state.player_minimax_depth.get(current_player, 3))
        action = get_minimax_move(game_state.state, depth)

    else:  # random
        action = get_random_move(game_state.state)

    if action is None or action not in game_state.state.legal_actions():
        return jsonify({'success': False, 'error': 'Could not determine valid action'})

    # Check phase before action
    state_before = get_board_state()
    is_capture = state_before['phase'] == 'capture'

    # Apply action
    game_state.state.apply_action(action)

    # Get updated state
    state_info = get_board_state()
    state_info['positions'] = parse_board_positions(game_state.state)

    # Describe move based on phase before action
    action_info = decode_action(action, is_capture_phase=is_capture)
    if action_info['type'] == 'place':
        desc = f"Placed piece at position {action_info['position']}"
    elif action_info['type'] == 'move':
        desc = f"Moved piece from {action_info['from']} to {action_info['to']}"
    else:
        desc = f"Captured piece at position {action_info['position']}"

    return jsonify({
        'success': True,
        'state': state_info,
        'player': int(current_player),
        'move_description': desc,
        'probabilities': probabilities
    })


@app.route('/api/state')
def api_state():
    if game_state.state is None:
        return jsonify({'error': 'No game in progress'})

    state_info = get_board_state()
    state_info['positions'] = parse_board_positions(game_state.state)
    return jsonify(state_info)


if __name__ == '__main__':
    print("=" * 60)
    print("Nine Men's Morris - Model Testing Web UI")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Available models: {len(get_available_models())}")
    print()
    print("Open http://192.168.178.23:7860 in your browser")
    print("=" * 60)

    app.run(host='0.0.0.0', port=7860, debug=True)
