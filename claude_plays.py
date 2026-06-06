"""
claude_plays.py — Nine Men's Morris: Turn-by-turn API for an external LLM agent

The external caller (Claude or any agent) drives every move via HTTP requests.
The server maintains game state between requests. After each agent move the
opponent responds immediately; the response contains everything needed to choose
the next move.

Usage:
    python claude_plays.py [--port 5001] [--host 0.0.0.0]

Workflow:
    1. GET  /api/models              → list trained models
    2. POST /api/game/start          → create session, get initial board
    3. POST /api/game/move           → submit your action, get opponent move + new state
    4. Repeat step 3 until "terminal" == true
    5. POST /api/game/start          → start a fresh game

All endpoints return JSON. No background threads. No Anthropic SDK required here.
"""

import os
import sys
import re
import json
import random
import argparse
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')

import numpy as np
import torch
from flask import Flask, jsonify, request, render_template_string

SRC_DIR = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC_DIR))

import fastnmm

from model import ActorCritic
from config import Config
from utils import get_legal_mask, relativize_obs
from minimax import MinimaxBot
from board_utils import (
    POINT_TO_COORD, POSITION_NAMES, MILLS,
    parse_board_positions as _parse_board_positions,
    decode_action as _decode_action,
)
from model_loader import discover_models, load_actor_critic

# ──────────────────────────────────────────────────────────────────────────────
# Game constants
# ──────────────────────────────────────────────────────────────────────────────

GAME = fastnmm.load_game("nine_mens_morris")
NUM_ACTIONS = GAME.num_distinct_actions()
OBS_SIZE    = GAME.observation_tensor_size()
OBS_SHAPE   = list(GAME.observation_tensor_shape())   # [5, 7, 7]
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ──────────────────────────────────────────────────────────────────────────────
# Board utilities
# ──────────────────────────────────────────────────────────────────────────────

def parse_board_positions(state) -> Dict[int, int]:
    return _parse_board_positions(state, OBS_SIZE, OBS_SHAPE)


def render_board_ascii(positions: Dict[int, int], agent_player: int) -> str:
    """O = agent's pieces, X = opponent pieces, . = empty."""
    def s(p):
        if p not in positions: return '.'
        return 'O' if positions[p] == agent_player else 'X'
    return "\n".join([
        f" {s(0)}———————————{s(1)}———————————{s(2)}",
        f" |           |           |",
        f" |   {s(3)}———————{s(4)}———————{s(5)}   |",
        f" |   |       |       |   |",
        f" |   |   {s(6)}———{s(7)}———{s(8)}   |   |",
        f" |   |   |       |   |   |",
        f" {s(9)}———{s(10)}———{s(11)}       {s(12)}———{s(13)}———{s(14)}",
        f" |   |   |       |   |   |",
        f" |   |   {s(15)}———{s(16)}———{s(17)}   |   |",
        f" |   |       |       |   |",
        f" |   {s(18)}———————{s(19)}———————{s(20)}   |",
        f" |           |           |",
        f" {s(21)}———————————{s(22)}———————————{s(23)}",
    ])


def detect_phase(state) -> str:
    if state.is_terminal(): return "terminal"
    legal = state.legal_actions()
    s = str(state)
    if "Capture time" in s or "capture" in s.lower(): return "capture"
    if legal and all(a >= 24 for a in legal):          return "movement"
    return "placement"


def decode_action(action: int, is_capture: bool = False) -> Dict[str, Any]:
    return _decode_action(action, is_capture_phase=is_capture)


def describe_action(action: int, is_capture: bool = False) -> str:
    info = decode_action(action, is_capture)
    if info["type"] == "place":
        p = info["position"]
        return f"Place at {p} ({POSITION_NAMES[p]})"
    if info["type"] == "capture":
        p = info["position"]
        return f"Capture {p} ({POSITION_NAMES[p]})"
    f, t = info["from"], info["to"]
    return f"Move {f} ({POSITION_NAMES[f]}) → {t} ({POSITION_NAMES[t]})"


def get_piece_counts(state) -> Tuple[int, int, int, int]:
    """(p0_on_board, p1_on_board, p0_to_place, p1_to_place)"""
    lines = str(state).split('\n')
    board = '\n'.join(lines[:13])
    on0, on1 = board.count('W'), board.count('B')
    to0 = to1 = 0
    for line in lines:
        if "Men to deploy" in line:
            nums = re.findall(r'\d+', line)
            if len(nums) >= 2:
                to0, to1 = int(nums[0]), int(nums[1])
            break
    return on0, on1, to0, to1


def mill_hints(positions: Dict[int, int], agent_player: int,
               legal: List[int]) -> List[str]:
    hints = []
    for m in MILLS:
        mine  = [p for p in m if positions.get(p) == agent_player]
        opp   = [p for p in m if positions.get(p) == (1 - agent_player)]
        empty = [p for p in m if p not in positions]
        if len(mine) == 2 and len(empty) == 1 and empty[0] in legal:
            hints.append(f"Complete mill {m} → place/move to {empty[0]}")
        if len(opp) == 2 and len(empty) == 1 and empty[0] in legal:
            hints.append(f"Block opponent mill {m} → play {empty[0]}")
    return hints


def build_state_response(state, agent_player: int,
                         opponent_last_move: Optional[Dict] = None,
                         move_number: int = 0) -> Dict:
    """
    Build the full JSON response describing the current game state.
    This is returned after every move and on /api/game/state.
    """
    terminal   = state.is_terminal()
    phase      = detect_phase(state)
    positions  = parse_board_positions(state)
    legal      = state.legal_actions() if not terminal else []
    is_capture = phase == "capture"

    on0, on1, to0, to1 = get_piece_counts(state)
    my_on  = on0 if agent_player == 0 else on1
    opp_on = on1 if agent_player == 0 else on0
    my_to  = to0 if agent_player == 0 else to1
    opp_to = to1 if agent_player == 0 else to0

    # Winner
    winner = result = None
    if terminal:
        returns = state.returns()
        if   returns[0] > 0: winner, result = 0, "Player 0 wins"
        elif returns[1] > 0: winner, result = 1, "Player 1 wins"
        else:                winner, result = None, "Draw"

    current_player = int(state.current_player()) if not terminal else -1
    your_turn = (current_player == agent_player)

    # Legal moves as rich objects
    legal_moves = [
        {
            "action":      a,
            "description": describe_action(a, is_capture),
            **decode_action(a, is_capture),
        }
        for a in legal
    ]

    resp = {
        # ── identity ──────────────────────────────────────────────
        "move_number":       move_number,
        "agent_player":      agent_player,          # 0 or 1 — you are always this
        "current_player":    current_player,        # whose turn it is right now
        "your_turn":         your_turn,

        # ── board ─────────────────────────────────────────────────
        "phase":             phase,
        "board_ascii":       render_board_ascii(positions, agent_player),
        "positions":         {str(k): v for k, v in positions.items()},
        # O = agent, X = opponent — mirrors board_ascii legend

        # ── piece counts ──────────────────────────────────────────
        "pieces": {
            "you":      {"on_board": my_on,  "to_place": my_to},
            "opponent": {"on_board": opp_on, "to_place": opp_to},
        },

        # ── legal moves (only meaningful when your_turn == true) ──
        "legal_moves": legal_moves,

        # ── mill opportunities ─────────────────────────────────────
        "mill_hints": mill_hints(positions, agent_player, legal)
                      if your_turn and not terminal else [],

        # ── opponent's last move (set after opponent responds) ─────
        "opponent_last_move": opponent_last_move,   # null on first request

        # ── terminal ──────────────────────────────────────────────
        "terminal": terminal,
        "winner":   winner,         # 0, 1, or null (draw)
        "result":   result,
    }
    return resp


# ──────────────────────────────────────────────────────────────────────────────
# Model & opponent helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_available_models() -> List[Dict[str, str]]:
    return discover_models(Path(__file__).parent, max_bytes=Config().max_model_file_bytes)


_model_cache: Dict[str, Any] = {}

def load_model(path: str):
    return load_actor_critic(
        path, OBS_SIZE, NUM_ACTIONS, OBS_SHAPE, DEVICE, cache=_model_cache,
    )


_minimax_bots: Dict[int, MinimaxBot] = {}

def get_opponent_move(state, opponent_type: str,
                      model_path: Optional[str], depth: int,
                      opponent_player: int) -> int:
    if opponent_type == "random":
        return random.choice(state.legal_actions())
    if opponent_type == "minimax":
        if depth not in _minimax_bots:
            _minimax_bots[depth] = MinimaxBot(max_depth=depth)
        return _minimax_bots[depth].get_action(state, use_iterative=True)
    if opponent_type == "ppo" and model_path:
        model = load_model(model_path)
        obs   = torch.from_numpy(relativize_obs(state, opponent_player)).unsqueeze(0).to(DEVICE)
        mask  = torch.tensor(get_legal_mask(state, NUM_ACTIONS),
                             dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits, _ = model(obs)
            logits = logits.float()
            logits[mask == 0] = -1e9
            return int(logits.argmax(dim=-1).item())
    return random.choice(state.legal_actions())


# ──────────────────────────────────────────────────────────────────────────────
# Session state  (single game session — one game at a time)
# ──────────────────────────────────────────────────────────────────────────────

class Session:
    def __init__(self):
        self.state          = None
        self.agent_player   = 0
        self.opponent_type  = "ppo"
        self.opponent_model = None
        self.minimax_depth  = 3
        self.move_number    = 0
        self.history: List[Dict] = []   # list of move records

    def reset(self, agent_player, opponent_type, opponent_model, minimax_depth):
        self.state          = GAME.new_initial_state()
        self.agent_player   = agent_player
        self.opponent_type  = opponent_type
        self.opponent_model = opponent_model
        self.minimax_depth  = minimax_depth
        self.move_number    = 0
        self.history        = []


SESSION = Session()

# ──────────────────────────────────────────────────────────────────────────────
# Flask app
# ──────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)


@app.route('/api/models')
def api_models():
    """List all available trained .pt model files."""
    return jsonify(get_available_models())


@app.route('/api/game/start', methods=['POST'])
def api_start():
    """
    Create a new game session.

    Body (JSON, all optional):
        agent_player      int   0 or 1  — which player YOU control (default 0)
        opponent_type     str   "ppo" | "minimax" | "random"  (default "ppo")
        opponent_model    str   path to .pt file (auto-selects latest if omitted)
        minimax_depth     int   1-6  (default 3, only relevant for minimax)

    Response: initial state dict.
    If agent_player == 1 the opponent moves first, so the response already
    contains the opponent's opening move.
    """
    data          = request.get_json(force=True) or {}
    agent_player  = int(data.get('agent_player', 0))
    opponent_type = data.get('opponent_type', 'ppo')
    minimax_depth = int(data.get('minimax_depth', 3))

    opponent_model = data.get('opponent_model') or None
    if opponent_type == 'ppo' and not opponent_model:
        models = get_available_models()
        if not models:
            return jsonify({"error": "No trained models found"}), 400
        opponent_model = models[0]['path']

    SESSION.reset(agent_player, opponent_type, opponent_model, minimax_depth)

    opponent_last_move = None

    # If agent plays second, run opponent turns until it's the agent's turn
    opp_player_id = 1 - agent_player
    while (agent_player == 1
           and not SESSION.state.is_terminal()
           and SESSION.state.current_player() != agent_player):
        opp_action = get_opponent_move(
            SESSION.state, opponent_type, opponent_model,
            minimax_depth, opponent_player=opp_player_id,
        )
        phase_before = detect_phase(SESSION.state)
        SESSION.state.apply_action(opp_action)
        SESSION.move_number += 1
        opponent_last_move = {
            "action":      opp_action,
            "description": describe_action(opp_action, phase_before == "capture"),
            **decode_action(opp_action, phase_before == "capture"),
        }
        SESSION.history.append({"player": opp_player_id, "role": "opponent", **opponent_last_move})

    resp = build_state_response(
        SESSION.state, agent_player,
        opponent_last_move=opponent_last_move,
        move_number=SESSION.move_number,
    )
    resp["session"] = {
        "agent_player":  agent_player,
        "opponent_type": opponent_type,
        "opponent_model": Path(opponent_model).name if opponent_model else None,
        "minimax_depth": minimax_depth,
    }
    return jsonify(resp)


@app.route('/api/game/move', methods=['POST'])
def api_move():
    """
    Submit the agent's chosen action and advance the game.

    Body (JSON):
        action   int   — the action id to play (must be in legal_moves)

    What happens:
        1. Validate and apply the agent's action.
        2. If the game isn't over, the opponent immediately responds.
        3. Return the new state (with opponent_last_move filled in).

    You keep calling this endpoint until "terminal": true.
    """
    if SESSION.state is None:
        return jsonify({"error": "No active game. Call /api/game/start first."}), 400
    if SESSION.state.is_terminal():
        return jsonify({"error": "Game is already over. Call /api/game/start."}), 400

    data = request.get_json(force=True) or {}
    if 'action' not in data:
        return jsonify({"error": "'action' field required"}), 400

    action = int(data['action'])
    legal  = SESSION.state.legal_actions()

    if action not in legal:
        return jsonify({
            "error":         f"Illegal action {action}",
            "legal_actions": legal,
        }), 400

    # ── 1. Apply agent's move ─────────────────────────────────────────────────
    phase_before = detect_phase(SESSION.state)
    SESSION.state.apply_action(action)
    SESSION.move_number += 1
    SESSION.history.append({
        "player": SESSION.agent_player,
        "role":   "agent",
        "action": action,
        "description": describe_action(action, phase_before == "capture"),
        **decode_action(action, phase_before == "capture"),
    })

    opponent_last_move = None

    # ── 2. Opponent responds — loop until it's the agent's turn again ─────────
    # After a mill is formed the same player gets an extra "capture" action,
    # so we can't assume a simple single-step alternation.
    opp_player = 1 - SESSION.agent_player
    while (not SESSION.state.is_terminal()
           and SESSION.state.current_player() != SESSION.agent_player):
        opp_action = get_opponent_move(
            SESSION.state,
            SESSION.opponent_type,
            SESSION.opponent_model,
            SESSION.minimax_depth,
            opponent_player=opp_player,
        )
        phase_opp = detect_phase(SESSION.state)
        SESSION.state.apply_action(opp_action)
        SESSION.move_number += 1
        opponent_last_move = {
            "action":      opp_action,
            "description": describe_action(opp_action, phase_opp == "capture"),
            **decode_action(opp_action, phase_opp == "capture"),
        }
        SESSION.history.append({
            "player": opp_player,
            "role":   "opponent",
            **opponent_last_move,
        })

    return jsonify(build_state_response(
        SESSION.state, SESSION.agent_player,
        opponent_last_move=opponent_last_move,
        move_number=SESSION.move_number,
    ))


@app.route('/api/game/state')
def api_state():
    """Get the current game state without making a move."""
    if SESSION.state is None:
        return jsonify({"error": "No active game. Call /api/game/start first."}), 400
    return jsonify(build_state_response(
        SESSION.state, SESSION.agent_player,
        move_number=SESSION.move_number,
    ))


@app.route('/api/game/history')
def api_history():
    """Return the full move history of the current game."""
    return jsonify({
        "move_number": SESSION.move_number,
        "history":     SESSION.history,
    })


# ──────────────────────────────────────────────────────────────────────────────
# Dashboard (quick reference page)
# ──────────────────────────────────────────────────────────────────────────────

DASHBOARD = """<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>Nine Men's Morris — Agent API</title>
<style>
body{font-family:monospace;background:#1a1a2e;color:#e0e0e0;padding:30px;max-width:900px;margin:auto}
h1{color:#64b5f6} h2{color:#81c784;margin-top:30px} h3{color:#ffb74d}
pre{background:#0d1117;padding:14px;border-radius:8px;overflow-x:auto;font-size:13px;line-height:1.5}
code{color:#80cbc4} .dim{color:#666} .url{color:#ce93d8}
table{border-collapse:collapse;width:100%} td,th{border:1px solid #333;padding:6px 12px;text-align:left}
th{background:#16213e;color:#ffb74d}
</style></head>
<body>
<h1>Nine Men&rsquo;s Morris &mdash; Agent API</h1>
<p class="dim">Turn-by-turn REST API. One POST per move. No streaming, no background threads.</p>

<h2>Endpoints</h2>
<table>
<tr><th>Method</th><th>Path</th><th>Description</th></tr>
<tr><td>GET</td> <td class="url">/api/models</td>       <td>List available trained models</td></tr>
<tr><td>POST</td><td class="url">/api/game/start</td>   <td>Start a new game session</td></tr>
<tr><td>POST</td><td class="url">/api/game/move</td>    <td>Submit your action → get opponent response + new state</td></tr>
<tr><td>GET</td> <td class="url">/api/game/state</td>   <td>Current state (no move)</td></tr>
<tr><td>GET</td> <td class="url">/api/game/history</td> <td>Full move log of current game</td></tr>
</table>

<h2>1. Start a game</h2>
<pre>POST /api/game/start
{
  "agent_player":  0,          <span class="dim">// 0 = you move first, 1 = opponent moves first</span>
  "opponent_type": "ppo",      <span class="dim">// "ppo" | "minimax" | "random"</span>
  "opponent_model": null,      <span class="dim">// path to .pt file — null = auto-select latest</span>
  "minimax_depth": 3           <span class="dim">// only for opponent_type "minimax"</span>
}</pre>

<h2>2. Make a move</h2>
<pre>POST /api/game/move
{ "action": 4 }               <span class="dim">// action id from legal_moves list</span></pre>

<h2>Response fields</h2>
<pre>{
  "move_number": 3,
  "agent_player": 0,          <span class="dim">// always your player id</span>
  "current_player": 0,        <span class="dim">// whose turn it is now</span>
  "your_turn": true,

  "phase": "placement",       <span class="dim">// placement | movement | capture | terminal</span>
  "board_ascii": " O—…",      <span class="dim">// O=you, X=opponent, .=empty</span>
  "positions": {"4": 0, …},   <span class="dim">// {board_pos: player_id}</span>

  "pieces": {
    "you":      {"on_board": 2, "to_place": 7},
    "opponent": {"on_board": 1, "to_place": 8}
  },

  "legal_moves": [            <span class="dim">// only meaningful when your_turn==true</span>
    {"action": 0, "type": "place", "position": 0, "description": "Place at 0 (outer top-left)"},
    …
  ],
  "mill_hints": [             <span class="dim">// tactical tips for your turn</span>
    "Complete mill (0,1,2) → place/move to 2"
  ],

  "opponent_last_move": {     <span class="dim">// what the opponent just played (null at start)</span>
    "action": 7,
    "type": "place",
    "position": 7,
    "description": "Place at 7 (inner top-mid)"
  },

  "terminal": false,
  "winner": null,             <span class="dim">// 0, 1, or null (draw) when terminal</span>
  "result": null
}</pre>

<h2>Quick curl example</h2>
<pre><span class="dim"># Start</span>
curl -s -X POST http://localhost:5001/api/game/start \\
     -H "Content-Type: application/json" \\
     -d '{"agent_player":0,"opponent_type":"ppo"}' | python3 -m json.tool

<span class="dim"># Play action 4</span>
curl -s -X POST http://localhost:5001/api/game/move \\
     -H "Content-Type: application/json" \\
     -d '{"action":4}' | python3 -m json.tool</pre>
</body></html>"""


@app.route('/')
def index():
    return render_template_string(DASHBOARD)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Nine Men's Morris Agent API")
    parser.add_argument('--port', type=int, default=5001)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    models = get_available_models()
    print(f"Device : {DEVICE}")
    print(f"Models : {len(models)} found — latest: {models[0]['name'] if models else 'none'}")
    print(f"Server : http://{args.host}:{args.port}")
    print()
    print("Quick start:")
    print(f"  curl -X POST http://localhost:{args.port}/api/game/start \\")
    print(f"       -H 'Content-Type: application/json' \\")
    print(f"       -d '{{\"agent_player\":0,\"opponent_type\":\"ppo\"}}'")
    print()

    app.run(host=args.host, port=args.port, debug=args.debug)
