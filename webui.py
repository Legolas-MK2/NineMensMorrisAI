"""
Nine Men's Morris - Model Testing Web UI
Interactive interface for testing trained models against various opponents.
"""

import os
# Limit OpenMP threads to prevent "Thread creation failed" errors
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['VECLIB_MAXIMUM_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '4'

import sys
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import torch
from flask import Flask, render_template_string, jsonify, request

# Add model directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import fastnmm

from model import ActorCritic
from config import Config
from utils import get_legal_mask, relativize_obs
from minimax import MinimaxBot
from board_utils import (
    POINT_TO_COORD,
    parse_board_positions as _parse_board_positions,
    decode_action,
    encode_action,
)
from model_loader import discover_models, load_actor_critic


app = Flask(__name__)

# Game constants - fastnmm fixes the OpenSpiel position-0 bug natively.
GAME = fastnmm.load_game("nine_mens_morris")
NUM_ACTIONS = GAME.num_distinct_actions()
OBS_SIZE = GAME.observation_tensor_size()
OBS_SHAPE = list(GAME.observation_tensor_shape())  # e.g. [5, 7, 7] = 245 dims
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Shared minimax bot with transposition table for better performance
# This allows the TT to persist across moves
_minimax_bot = None

COORD_TO_POINT = {v: k for k, v in POINT_TO_COORD.items()}


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


_REPO_ROOT = Path(__file__).parent


def get_available_models() -> List[Dict[str, str]]:
    """Scan canonical directories for available trained models."""
    return discover_models(_REPO_ROOT, max_bytes=Config().max_model_file_bytes)


def load_model(model_info: Dict[str, str]) -> Any:
    """Load a model from disk. Returns (model, 'src')."""
    model = load_actor_critic(
        model_info["path"], OBS_SIZE, NUM_ACTIONS, OBS_SHAPE, DEVICE,
    )
    return model, 'src'


def get_board_state() -> Dict[str, Any]:
    """Get current board state for rendering."""
    if game_state.state is None:
        return {"positions": {}, "current_player": 0, "phase": "not_started"}

    state = game_state.state
    state_str = str(state)

    # Get legal actions to understand board state
    legal = state.legal_actions()

    # Count pieces on the board via the engine (fastnmm renders W/B in str()).
    p0_pieces = state.men_on_board(0)
    p1_pieces = state.men_on_board(1)

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
    return _parse_board_positions(state, OBS_SIZE, OBS_SHAPE)


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
    obs = torch.from_numpy(
        relativize_obs(state, player)
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

    return _minimax_bot.get_action(state)


_TEMPLATE_PATH = Path(__file__).parent / 'templates' / 'webui.html'
HTML_TEMPLATE = _TEMPLATE_PATH.read_text(encoding='utf-8')


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

    # Drop any cached minimax bot so the next move builds a fresh one;
    # fastnmm manages its own search state internally.
    _minimax_bot = None

    # Initialize new game
    game_state = GameState()
    game_state.state = GAME.new_initial_state()

    # Configure players
    game_state.player_types[0] = config.get('player0_type', 'human')
    game_state.player_types[1] = config.get('player1_type', 'ai')
    game_state.player_minimax_depth[0] = config.get('player0_depth', 3)
    game_state.player_minimax_depth[1] = config.get('player1_depth', 3)

    # Prepare board with random moves to produce a mid-game position.
    prepare_moves = int(config.get('prepare_moves', 0) or 0)
    if prepare_moves > 0:
        state = game_state.state
        for _ in range(prepare_moves):
            if state.is_terminal():
                break
            legal = state.legal_actions()
            if not legal:
                break
            state.apply_action(random.choice(legal))

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
            model, _ = model_data
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
    host = os.environ.get('NMM_WEBUI_HOST', '0.0.0.0')
    port = int(os.environ.get('NMM_WEBUI_PORT', '7860'))
    print("=" * 60)
    print("Nine Men's Morris - Model Testing Web UI")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Available models: {len(get_available_models())}")
    print()
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(f"Open http://{local_ip}:{port} in your browser")
    print("=" * 60)

    app.run(host=host, port=port, debug=True, use_reloader=False)
