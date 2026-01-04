import gradio as gr
import pyspiel
import torch
import numpy as np
import os
import glob
import random
import sys

# Add claude folder to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'claude'))

# Import gemini model (simple architecture)
from gemini import ActorCritic as GeminiActorCritic, legal_action_mask, masked_categorical

# Import claude model (complex architecture with config)
try:
    from claude.model import ActorCritic as ClaudeActorCritic
    from claude.config import Config as ClaudeConfig
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False
    print("Warning: Claude model not available")

# Initialize Game
GAME_NAME = "nine_mens_morris"
game = pyspiel.load_game(GAME_NAME)
obs_size = game.observation_tensor_size()
num_actions = game.num_distinct_actions()
device = torch.device("cpu")

# Board Mapping - position ID to (row, col)
POINT_TO_COORD = {
    0: (0,0), 1: (0,3), 2: (0,6),
    3: (1,1), 4: (1,3), 5: (1,5),
    6: (2,2), 7: (2,3), 8: (2,4),
    9: (3,0), 10: (3,1), 11: (3,2),
    12: (3,4), 13: (3,5), 14: (3,6),
    15: (4,2), 16: (4,3), 17: (4,4),
    18: (5,1), 19: (5,3), 20: (5,5),
    21: (6,0), 22: (6,3), 23: (6,6)
}

COORD_TO_POINT = {v: k for k, v in POINT_TO_COORD.items()}

# Action encoding for Nine Men's Morris:
# Phase 1 (placement): action = position (0-23)
# Phase 2 (movement): action = from_pos * 24 + to_pos (24-599)
# Capture: action = capture_pos (0-23) - same as placement but in different state

def decode_action(action, is_capture=False):
    """Decode action into (type, from_pos, to_pos, capture_pos)"""
    if action < 24:
        if is_capture:
            return ('capture', None, None, action)
        else:
            return ('place', None, action, None)
    else:
        action_offset = action - 24
        from_pos = action_offset // 24
        to_pos = action_offset % 24
        return ('move', from_pos, to_pos, None)

def encode_place_action(pos):
    return pos

def encode_move_action(from_pos, to_pos):
    return 24 + from_pos * 24 + to_pos

def encode_capture_action(capture_pos):
    return capture_pos


def generate_board_svg(state, selected_pos=None, legal_targets=None, legal_captures=None):
    """Generate interactive SVG board"""
    tensor = np.array(state.observation_tensor(0)).reshape(game.observation_tensor_shape())

    svg_width = 700
    svg_height = 700
    grid_step = 100
    offset = 50

    # Define Lines
    lines_svg = ""
    squares = [
        [(0,0), (0,6), (6,6), (6,0)],
        [(1,1), (1,5), (5,5), (5,1)],
        [(2,2), (2,4), (4,4), (4,2)]
    ]

    for sq in squares:
        points_str = ""
        for r, c in sq:
            x = c * grid_step + offset
            y = r * grid_step + offset
            points_str += f"{x},{y} "
        r, c = sq[0]
        x = c * grid_step + offset
        y = r * grid_step + offset
        points_str += f"{x},{y}"
        lines_svg += f'<polyline points="{points_str}" fill="none" stroke="#5a3825" stroke-width="4" />\n'

    # Cross connections
    connections = [
        ((0,3), (2,3)), ((6,3), (4,3)),
        ((3,0), (3,2)), ((3,6), (3,4))
    ]
    for start, end in connections:
        x1 = start[1] * grid_step + offset
        y1 = start[0] * grid_step + offset
        x2 = end[1] * grid_step + offset
        y2 = end[0] * grid_step + offset
        lines_svg += f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#5a3825" stroke-width="4" />\n'

    # Draw Points/Pieces with click handlers
    pieces_svg = ""
    legal_targets = legal_targets or set()
    legal_captures = legal_captures or set()

    for pid, (r, c) in POINT_TO_COORD.items():
        cx = c * grid_step + offset
        cy = r * grid_step + offset

        is_white = tensor[0, r, c] == 1
        is_black = tensor[1, r, c] == 1

        # Determine styling based on state
        is_selected = (selected_pos == pid)
        is_target = pid in legal_targets
        is_capture = pid in legal_captures

        if is_white:
            fill = "#f5f5dc" if not is_selected else "#90EE90"
            stroke = "#228B22" if is_selected else ("#ff4444" if is_capture else "#333")
            stroke_width = 4 if is_selected or is_capture else 2
            pieces_svg += f'<circle cx="{cx}" cy="{cy}" r="30" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}" style="cursor:pointer" data-pos="{pid}" class="piece white-piece" />\n'
        elif is_black:
            fill = "#222" if not is_selected else "#4a7c4a"
            stroke = "#228B22" if is_selected else ("#ff4444" if is_capture else "#888")
            stroke_width = 4 if is_selected or is_capture else 2
            pieces_svg += f'<circle cx="{cx}" cy="{cy}" r="30" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}" style="cursor:pointer" data-pos="{pid}" class="piece black-piece" />\n'
        else:
            # Empty spot
            if is_target:
                # Highlight as valid move target
                pieces_svg += f'<circle cx="{cx}" cy="{cy}" r="20" fill="#90EE90" stroke="#228B22" stroke-width="3" style="cursor:pointer" data-pos="{pid}" class="empty-spot target" />\n'
            else:
                pieces_svg += f'<circle cx="{cx}" cy="{cy}" r="10" fill="#5a3825" style="cursor:pointer" data-pos="{pid}" class="empty-spot" />\n'

    svg = f"""
    <svg width="100%" height="100%" viewBox="0 0 {svg_width} {svg_height}" version="1.1" xmlns="http://www.w3.org/2000/svg" id="game-board">
        <rect width="100%" height="100%" fill="#deb887" />
        {lines_svg}
        {pieces_svg}
    </svg>
    """
    return svg


def get_models():
    """Get all available models from both gemini and claude directories"""
    models = ["Human", "Random"]

    # Get script directory for absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Gemini models (models/*.pt in script directory)
    gemini_models_dir = os.path.join(script_dir, "models")
    if os.path.exists(gemini_models_dir):
        files = glob.glob(os.path.join(gemini_models_dir, "*.pt"))
        for f in files:
            models.append(f"gemini:{os.path.basename(f)}")

    # Claude models (claude/models/*.pt and claude/checkpoints/*.pt)
    if CLAUDE_AVAILABLE:
        claude_models_dir = os.path.join(script_dir, "claude", "models")
        if os.path.exists(claude_models_dir):
            files = glob.glob(os.path.join(claude_models_dir, "*.pt"))
            for f in files:
                models.append(f"claude:{os.path.basename(f)}")

        claude_checkpoints_dir = os.path.join(script_dir, "claude", "checkpoints")
        if os.path.exists(claude_checkpoints_dir):
            files = glob.glob(os.path.join(claude_checkpoints_dir, "*.pt"))
            for f in files:
                models.append(f"claude:checkpoints/{os.path.basename(f)}")

    return models


def load_agent(model_name):
    """Load model based on prefix (gemini: or claude:)"""
    if model_name == "Human" or model_name == "Random":
        return None, None

    # Get script directory for absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if model_name.startswith("gemini:"):
        # Load gemini model
        filename = model_name[7:]  # Remove "gemini:" prefix
        path = os.path.join(script_dir, "models", filename)
        model = GeminiActorCritic(obs_size, num_actions, hidden=512, dropout=0.1).to(device)

        try:
            state_dict = torch.load(path, map_location=device)
            model.load_state_dict(state_dict)
            model.eval()
            return model, "gemini"
        except Exception as e:
            print(f"Error loading gemini model {model_name}: {e}")
            return None, None

    elif model_name.startswith("claude:"):
        if not CLAUDE_AVAILABLE:
            print("Claude models not available")
            return None, None

        # Load claude model
        filename = model_name[7:]  # Remove "claude:" prefix
        if filename.startswith("checkpoints/"):
            path = os.path.join(script_dir, "claude", filename)
        else:
            path = os.path.join(script_dir, "claude", "models", filename)

        config = ClaudeConfig()
        model = ClaudeActorCritic(obs_size, num_actions, config).to(device)

        try:
            checkpoint = torch.load(path, map_location=device)
            # Handle both direct state_dict and checkpoint dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            model.eval()
            return model, "claude"
        except Exception as e:
            print(f"Error loading claude model {model_name}: {e}")
            return None, None

    return None, None


def get_ai_action(model, model_type, state, pid):
    """Get action from AI model"""
    obs = torch.tensor(state.observation_tensor(pid), dtype=torch.float32, device=device).unsqueeze(0)
    mask = torch.from_numpy(legal_action_mask(state, num_actions)).to(device).unsqueeze(0)

    with torch.no_grad():
        logits, _ = model(obs)

        if model_type == "claude":
            # Claude model uses different masking
            masked_logits = logits.float()
            masked_logits[mask == 0] = -1e9
            probs = torch.softmax(masked_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
        else:
            # Gemini model
            dist = masked_categorical(logits.squeeze(0), mask.squeeze(0))

        action = int(dist.sample().item())

    return action


def get_legal_action_info(state):
    """Parse legal actions into categories for UI"""
    legal_actions = state.legal_actions()
    
    state_str = str(state)
    is_capture = "Capture time" in state_str or "capture" in state_str.lower()

    place_positions = set()
    move_sources = set()
    move_targets = {}  # from_pos -> set of to_pos
    capture_positions = set()

    for action in legal_actions:
        action_type, from_pos, to_pos, capture_pos = decode_action(action, is_capture=is_capture)

        if action_type == 'place':
            place_positions.add(to_pos)
        elif action_type == 'move':
            move_sources.add(from_pos)
            if from_pos not in move_targets:
                move_targets[from_pos] = set()
            move_targets[from_pos].add(to_pos)
        elif action_type == 'capture':
            capture_positions.add(capture_pos)

    return {
        'place': place_positions,
        'move_sources': move_sources,
        'move_targets': move_targets,
        'capture': capture_positions,
        'raw': set(legal_actions)
    }


def format_board(state, selected_pos=None, legal_info=None):
    """Format board with optional selection highlighting"""
    legal_targets = set()
    legal_captures = set()

    if legal_info:
        if selected_pos is not None and selected_pos in legal_info.get('move_targets', {}):
            legal_targets = legal_info['move_targets'][selected_pos]
        elif legal_info.get('place'):
            legal_targets = legal_info['place']

        if legal_info.get('capture'):
            legal_captures = legal_info['capture']

    return generate_board_svg(state, selected_pos, legal_targets, legal_captures)


def reconstruct_state(history):
    """Reconstruct game state from action history"""
    state = game.new_initial_state()
    for action in history:
        state.apply_action(action)
    return state


def process_click(history, selected_pos, white_agent, black_agent, click_pos):
    """Process a board click"""
    state = reconstruct_state(history)

    if state.is_terminal():
        return history, selected_pos, format_board(state), get_status(state, white_agent, black_agent), ""

    pid = state.current_player()
    agent_name = white_agent if pid == 0 else black_agent

    # Only process clicks for human players
    if agent_name != "Human":
        return history, selected_pos, format_board(state), get_status(state, white_agent, black_agent), "It's not your turn!"

    legal_info = get_legal_action_info(state)
    click_pos = int(click_pos)

    action_taken = None
    new_selected = selected_pos
    message = ""

    # Check what phase we're in based on legal actions
    if legal_info['capture']:
        # Must capture - click on enemy piece
        if click_pos in legal_info['capture']:
            action_taken = encode_capture_action(click_pos)
            new_selected = None
            message = f"Captured piece at position {click_pos}"
        else:
            message = "You must capture an opponent's piece (highlighted in red)"

    elif legal_info['place']:
        # Placement phase - click on empty spot
        if click_pos in legal_info['place']:
            action_taken = encode_place_action(click_pos)
            new_selected = None
            message = f"Placed piece at position {click_pos}"
        else:
            message = "Click on a highlighted empty position to place your piece"

    elif legal_info['move_sources']:
        # Movement phase
        if selected_pos is None:
            # Select a piece to move
            if click_pos in legal_info['move_sources']:
                new_selected = click_pos
                message = f"Selected piece at {click_pos}. Click a highlighted position to move."
            else:
                message = "Click on one of your pieces that can move"
        else:
            # A piece is selected - either move or reselect
            if click_pos in legal_info['move_targets'].get(selected_pos, set()):
                action_taken = encode_move_action(selected_pos, click_pos)
                new_selected = None
                message = f"Moved from {selected_pos} to {click_pos}"
            elif click_pos in legal_info['move_sources']:
                # Reselect different piece
                new_selected = click_pos
                message = f"Selected piece at {click_pos}. Click a highlighted position to move."
            elif click_pos == selected_pos:
                # Deselect
                new_selected = None
                message = "Deselected piece"
            else:
                message = "Invalid move. Click a highlighted position or select a different piece."

    # Apply action if one was made
    if action_taken is not None:
        state.apply_action(action_taken)
        history = history + [action_taken]
        new_selected = None

    # Update legal info after action
    if not state.is_terminal():
        legal_info = get_legal_action_info(state)
    else:
        legal_info = None

    board_html = format_board(state, new_selected, legal_info)
    status = get_status(state, white_agent, black_agent)

    return history, new_selected, board_html, status, message


def get_status(state, white_agent, black_agent):
    """Get current game status"""
    if state.is_terminal():
        returns = state.returns()
        if returns[0] > returns[1]:
            return f"Game Over! White ({white_agent}) wins!"
        elif returns[1] > returns[0]:
            return f"Game Over! Black ({black_agent}) wins!"
        else:
            return "Game Over! Draw!"
    else:
        pid = state.current_player()
        agent = white_agent if pid == 0 else black_agent
        color = "White" if pid == 0 else "Black"

        legal_info = get_legal_action_info(state)
        if legal_info['capture']:
            phase = "Capture"
        elif legal_info['place']:
            phase = "Place"
        else:
            phase = "Move"

        return f"{color}'s turn ({agent}) - {phase} phase"


def ai_step(history, selected_pos, white_agent, black_agent):
    """Execute AI move"""
    state = reconstruct_state(history)

    if state.is_terminal():
        return history, selected_pos, format_board(state), get_status(state, white_agent, black_agent), "Game is over"

    pid = state.current_player()
    agent_name = white_agent if pid == 0 else black_agent

    if agent_name == "Human":
        legal_info = get_legal_action_info(state)
        return history, selected_pos, format_board(state, selected_pos, legal_info), get_status(state, white_agent, black_agent), "It's your turn! Click on the board."

    action_taken = None
    message = ""

    if agent_name == "Random":
        legal_actions = state.legal_actions()
        if legal_actions:
            action_taken = random.choice(legal_actions)
            message = f"Random played action {action_taken}"
    else:
        model, model_type = load_agent(agent_name)
        if model:
            action_taken = get_ai_action(model, model_type, state, pid)
            message = f"{agent_name} played action {action_taken}"
        else:
            return history, selected_pos, format_board(state), f"Failed to load {agent_name}", "Error loading model"

    if action_taken is not None:
        state.apply_action(action_taken)
        history = history + [action_taken]

    # Check if next player is also AI
    if not state.is_terminal():
        legal_info = get_legal_action_info(state)
    else:
        legal_info = None

    return history, None, format_board(state, None, legal_info), get_status(state, white_agent, black_agent), message


def reset_game(white_agent, black_agent):
    """Reset the game"""
    state = game.new_initial_state()
    history = []
    legal_info = get_legal_action_info(state)
    return history, None, format_board(state, None, legal_info), get_status(state, white_agent, black_agent), "Game started! White moves first."


def auto_play_ai(history, selected_pos, white_agent, black_agent):
    """Automatically play AI moves until it's a human's turn or game over"""
    state = reconstruct_state(history)
    moves_made = 0
    max_moves = 200  # Safety limit

    while not state.is_terminal() and moves_made < max_moves:
        pid = state.current_player()
        agent_name = white_agent if pid == 0 else black_agent

        if agent_name == "Human":
            break

        if agent_name == "Random":
            legal_actions = state.legal_actions()
            if legal_actions:
                action = random.choice(legal_actions)
                state.apply_action(action)
                history = history + [action]
                moves_made += 1
        else:
            model, model_type = load_agent(agent_name)
            if model:
                action = get_ai_action(model, model_type, state, pid)
                state.apply_action(action)
                history = history + [action]
                moves_made += 1
            else:
                break

    if not state.is_terminal():
        legal_info = get_legal_action_info(state)
    else:
        legal_info = None

    message = f"AI made {moves_made} moves" if moves_made > 0 else "Your turn"
    return history, None, format_board(state, None, legal_info), get_status(state, white_agent, black_agent), message


# Build the UI
with gr.Blocks(title="Nine Men's Morris AI") as demo:

    gr.Markdown("# Nine Men's Morris")
    gr.Markdown("Select positions using the buttons below the board. Green = valid targets, Red = capturable pieces.")

    # Hidden state
    game_history = gr.State([])
    selected_position = gr.State(None)

    with gr.Row():
        with gr.Column(scale=1):
            white_dd = gr.Dropdown(choices=get_models(), label="White Player", value="Human")
            black_dd = gr.Dropdown(choices=get_models(), label="Black Player", value="Random")

            with gr.Row():
                start_btn = gr.Button("New Game", variant="primary")
                ai_btn = gr.Button("AI Move / Next Step", variant="secondary")

            status_text = gr.Textbox(label="Status", value="Click 'New Game' to start", interactive=False)
            message_text = gr.Textbox(label="Message", value="", interactive=False)

        with gr.Column(scale=2):
            board_display = gr.HTML(
                label="Game Board",
                value=generate_board_svg(game.new_initial_state())
            )

    # Position buttons grid - matches board layout
    gr.Markdown("### Click a position:")

    # Row 0: positions 0, 1, 2
    with gr.Row():
        pos_btns = {}
        pos_btns[0] = gr.Button("0", size="sm", min_width=60)
        gr.Button("", size="sm", min_width=60, interactive=False, visible=True)
        gr.Button("", size="sm", min_width=60, interactive=False, visible=True)
        pos_btns[1] = gr.Button("1", size="sm", min_width=60)
        gr.Button("", size="sm", min_width=60, interactive=False, visible=True)
        gr.Button("", size="sm", min_width=60, interactive=False, visible=True)
        pos_btns[2] = gr.Button("2", size="sm", min_width=60)

    # Row 1: positions 3, 4, 5
    with gr.Row():
        gr.Button("", size="sm", min_width=60, interactive=False, visible=True)
        pos_btns[3] = gr.Button("3", size="sm", min_width=60)
        gr.Button("", size="sm", min_width=60, interactive=False, visible=True)
        pos_btns[4] = gr.Button("4", size="sm", min_width=60)
        gr.Button("", size="sm", min_width=60, interactive=False, visible=True)
        pos_btns[5] = gr.Button("5", size="sm", min_width=60)
        gr.Button("", size="sm", min_width=60, interactive=False, visible=True)

    # Row 2: positions 6, 7, 8
    with gr.Row():
        gr.Button("", size="sm", min_width=60, interactive=False, visible=True)
        gr.Button("", size="sm", min_width=60, interactive=False, visible=True)
        pos_btns[6] = gr.Button("6", size="sm", min_width=60)
        pos_btns[7] = gr.Button("7", size="sm", min_width=60)
        pos_btns[8] = gr.Button("8", size="sm", min_width=60)
        gr.Button("", size="sm", min_width=60, interactive=False, visible=True)
        gr.Button("", size="sm", min_width=60, interactive=False, visible=True)

    # Row 3: positions 9, 10, 11, -, 12, 13, 14
    with gr.Row():
        pos_btns[9] = gr.Button("9", size="sm", min_width=60)
        pos_btns[10] = gr.Button("10", size="sm", min_width=60)
        pos_btns[11] = gr.Button("11", size="sm", min_width=60)
        gr.Button("", size="sm", min_width=60, interactive=False, visible=True)
        pos_btns[12] = gr.Button("12", size="sm", min_width=60)
        pos_btns[13] = gr.Button("13", size="sm", min_width=60)
        pos_btns[14] = gr.Button("14", size="sm", min_width=60)

    # Row 4: positions 15, 16, 17
    with gr.Row():
        gr.Button("", size="sm", min_width=60, interactive=False, visible=True)
        gr.Button("", size="sm", min_width=60, interactive=False, visible=True)
        pos_btns[15] = gr.Button("15", size="sm", min_width=60)
        pos_btns[16] = gr.Button("16", size="sm", min_width=60)
        pos_btns[17] = gr.Button("17", size="sm", min_width=60)
        gr.Button("", size="sm", min_width=60, interactive=False, visible=True)
        gr.Button("", size="sm", min_width=60, interactive=False, visible=True)

    # Row 5: positions 18, 19, 20
    with gr.Row():
        gr.Button("", size="sm", min_width=60, interactive=False, visible=True)
        pos_btns[18] = gr.Button("18", size="sm", min_width=60)
        gr.Button("", size="sm", min_width=60, interactive=False, visible=True)
        pos_btns[19] = gr.Button("19", size="sm", min_width=60)
        gr.Button("", size="sm", min_width=60, interactive=False, visible=True)
        pos_btns[20] = gr.Button("20", size="sm", min_width=60)
        gr.Button("", size="sm", min_width=60, interactive=False, visible=True)

    # Row 6: positions 21, 22, 23
    with gr.Row():
        pos_btns[21] = gr.Button("21", size="sm", min_width=60)
        gr.Button("", size="sm", min_width=60, interactive=False, visible=True)
        gr.Button("", size="sm", min_width=60, interactive=False, visible=True)
        pos_btns[22] = gr.Button("22", size="sm", min_width=60)
        gr.Button("", size="sm", min_width=60, interactive=False, visible=True)
        gr.Button("", size="sm", min_width=60, interactive=False, visible=True)
        pos_btns[23] = gr.Button("23", size="sm", min_width=60)

    # Event handlers
    start_btn.click(
        reset_game,
        inputs=[white_dd, black_dd],
        outputs=[game_history, selected_position, board_display, status_text, message_text]
    )

    ai_btn.click(
        ai_step,
        inputs=[game_history, selected_position, white_dd, black_dd],
        outputs=[game_history, selected_position, board_display, status_text, message_text]
    )

    # Create click handlers for each position button
    def make_click_handler(pos):
        def handler(history, selected, white_agent, black_agent):
            return process_click(history, selected, white_agent, black_agent, str(pos))
        return handler

    for pos_id, btn in pos_btns.items():
        btn.click(
            make_click_handler(pos_id),
            inputs=[game_history, selected_position, white_dd, black_dd],
            outputs=[game_history, selected_position, board_display, status_text, message_text]
        )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
