"""
Nine Men's Morris - Minimax Analysis Web Server
Interactive visualization of the minimax bot's evaluation, reward system, and search algorithm.

Uses the actual code from claude/minimax.py and claude/curriculum.py.
"""

import sys
import os
from pathlib import Path

# Add claude/ to path so we can import the actual minimax code
sys.path.insert(0, str(Path(__file__).parent / "claude"))

from flask import Flask, render_template_string, jsonify, request

# Import directly from the actual minimax module
from minimax import (
    MILLS, ADJACENCY, POSITION_TO_MILLS,
    POSITION_VALUES, POSITION_VALUES_PLACEMENT,
    STRONG_OPENING_POSITIONS, BOARD_POS_TO_GRID,
    OptimizedMinimaxBot, MoveOrderer,
)

# Import curriculum phase configs
from curriculum import PHASE_CONFIGS as CURRICULUM_PHASES, Phase

app = Flask(__name__)

# Instantiate a bot so we can use its evaluation helper methods directly
_bot = OptimizedMinimaxBot(max_depth=1, tt_size_mb=1)

# Build PHASE_CONFIGS list for the frontend from the actual curriculum data
PHASE_CONFIGS = []
for phase_enum in sorted(CURRICULUM_PHASES.keys(), key=lambda p: int(p)):
    pc = CURRICULUM_PHASES[phase_enum]
    PHASE_CONFIGS.append({
        "phase": int(pc.phase),
        "desc": pc.description,
        "stones": pc.stones_per_player,
        "start": pc.start_phase,
        "opponent": pc.opponent_type,
        "shaping_mult": pc.shaping_multiplier,
        "lr_start": pc.lr_start,
        "lr_end": pc.lr_end,
        "win": pc.win_reward_base,
        "win_speed": pc.win_reward_speed_bonus,
        "loss": pc.loss_reward,
        "draw": pc.draw_penalty,
        "mill_rw": pc.mill_reward,
        "enemy_mill": pc.enemy_mill_penalty,
        "block_mill": pc.block_mill_reward,
        "dbl_mill": pc.double_mill_reward,
        "dbl_mill_extra": pc.double_mill_extra_reward,
        "setup_capture": pc.setup_capture_reward,
        "min_ep": pc.min_episodes,
        "grad_wr": pc.win_rate_threshold,
    })

# Read weights directly from the bot class
WEIGHTS = OptimizedMinimaxBot.WEIGHTS

POSITION_NAMES = [
    "a7", "d7", "g7",
    "b6", "d6", "f6",
    "c5", "d5", "e5",
    "a4", "b4", "c4", "e4", "f4", "g4",
    "c3", "d3", "e3",
    "b2", "d2", "f2",
    "a1", "d1", "g1",
]


# ============================================================================
# EVALUATION LOGIC — delegates to the actual bot methods from minimax.py
# ============================================================================

def full_evaluate(board, mover, weights=None):
    """
    Evaluate a board position using the same logic as OptimizedMinimaxBot.evaluate(),
    but with a plain board list instead of a pyspiel state.
    Delegates counting to the actual bot helper methods.
    """
    if weights is None:
        weights = WEIGHTS
    board_tuple = tuple(board)
    opp = 1 - mover

    my_pieces = _bot._count_pieces(board_tuple, mover)
    opp_pieces = _bot._count_pieces(board_tuple, opp)

    # Check for terminal conditions
    # In Nine Men's Morris, a player loses when they have <= 2 pieces or no legal moves
    components = {}

    # Terminal check: opponent has <= 2 pieces (I win)
    if opp_pieces <= 2:
        components['terminal'] = {
            'my': my_pieces, 'opp': opp_pieces, 'result': 'WIN',
            'weight': weights['win'], 'score': weights['win'],
            'desc': f'Opponent has {opp_pieces} pieces (≤2) - WIN'
        }
        return {'components': components, 'total': weights['win']}

    # Terminal check: I have <= 2 pieces (I lose)
    if my_pieces <= 2:
        components['terminal'] = {
            'my': my_pieces, 'opp': opp_pieces, 'result': 'LOSS',
            'weight': weights['loss'], 'score': weights['loss'],
            'desc': f'I have {my_pieces} pieces (≤2) - LOSS'
        }
        return {'components': components, 'total': weights['loss']}

    # Calculate mobility (needed for both terminal check and normal evaluation)
    my_mobility = _bot._get_mobility(board_tuple, mover)
    opp_mobility = _bot._get_mobility(board_tuple, opp)

    # Terminal check: opponent has no legal moves (I win)
    if opp_mobility == 0:
        components['terminal'] = {
            'my': my_pieces, 'opp': opp_pieces, 'result': 'WIN',
            'weight': weights['win'], 'score': weights['win'],
            'desc': f'Opponent has 0 mobility - WIN'
        }
        return {'components': components, 'total': weights['win']}

    # Terminal check: I have no legal moves (I lose)
    if my_mobility == 0:
        components['terminal'] = {
            'my': my_pieces, 'opp': opp_pieces, 'result': 'LOSS',
            'weight': weights['loss'], 'score': weights['loss'],
            'desc': f'I have 0 mobility - LOSS'
        }
        return {'components': components, 'total': weights['loss']}

    # Mills
    my_mills = _bot._count_mills(board_tuple, mover)
    opp_mills = _bot._count_mills(board_tuple, opp)
    components['mills'] = {
        'my': my_mills, 'opp': opp_mills, 'diff': my_mills - opp_mills,
        'weight': weights['mill'], 'score': (my_mills - opp_mills) * weights['mill'],
        'desc': f'({my_mills} - {opp_mills}) x {weights["mill"]:,}'
    }

    # Potential mills
    my_pot = _bot._count_potential_mills(board_tuple, mover)
    opp_pot = _bot._count_potential_mills(board_tuple, opp)
    components['potential_mills'] = {
        'my': my_pot, 'opp': opp_pot, 'diff': my_pot - opp_pot,
        'weight': weights['potential_mill'], 'score': (my_pot - opp_pot) * weights['potential_mill'],
        'desc': f'({my_pot} - {opp_pot}) x {weights["potential_mill"]:,}'
    }

    # Blocked mills
    my_blocks = _bot._count_blocked_mills(board_tuple, mover)
    opp_blocks = _bot._count_blocked_mills(board_tuple, opp)
    components['blocked_mills'] = {
        'my': my_blocks, 'opp': opp_blocks, 'diff': my_blocks - opp_blocks,
        'weight': weights['blocked_mill'], 'score': (my_blocks - opp_blocks) * weights['blocked_mill'],
        'desc': f'({my_blocks} - {opp_blocks}) x {weights["blocked_mill"]:,}'
    }

    # Unblocked threats
    threats_me = _bot._count_unblocked_threats(board_tuple, mover)
    threats_opp = _bot._count_unblocked_threats(board_tuple, opp)
    threat_score = -threats_me * weights['unblocked_threat'] + threats_opp * weights['unblocked_threat']
    components['unblocked_threats'] = {
        'my_threats': threats_me, 'opp_threats': threats_opp,
        'weight': weights['unblocked_threat'], 'score': threat_score,
        'desc': f'(-{threats_me} + {threats_opp}) x {weights["unblocked_threat"]:,}'
    }

    # Double mills
    my_dbl = _bot._count_double_mills(board_tuple, mover)
    opp_dbl = _bot._count_double_mills(board_tuple, opp)
    components['double_mills'] = {
        'my': my_dbl, 'opp': opp_dbl, 'diff': my_dbl - opp_dbl,
        'weight': weights['double_mill'], 'score': (my_dbl - opp_dbl) * weights['double_mill'],
        'desc': f'({my_dbl} - {opp_dbl}) x {weights["double_mill"]:,}'
    }

    # Mobility (already calculated for terminal check)
    components['mobility'] = {
        'my': my_mobility, 'opp': opp_mobility, 'diff': my_mobility - opp_mobility,
        'weight': weights['mobility'], 'score': (my_mobility - opp_mobility) * weights['mobility'],
        'desc': f'({my_mobility} - {opp_mobility}) x {weights["mobility"]:,}'
    }

    # Positional
    pos_sc = _bot._positional_score(board_tuple, mover)
    components['position'] = {
        'raw': pos_sc, 'weight': weights['position'],
        'score': pos_sc * weights['position'],
        'desc': f'{pos_sc} x {weights["position"]:,}'
    }

    total = sum(c['score'] for c in components.values())
    return {'components': components, 'total': total}


def compute_terminal_reward(result, steps, max_steps, win_base, win_speed, loss_rw, draw_pen):
    if result == 'win':
        speed_bonus = win_speed * max(0, 1.0 - (steps / max_steps))
        return win_base + speed_bonus
    elif result == 'loss':
        return loss_rw
    else:
        return draw_pen


def compute_shaping_reward(pieces_captured, phase_config, progress):
    """Compute shaping rewards for a capture event."""
    if phase_config['phase'] == 1:
        mult = 1.0
    elif phase_config['phase'] == 10:
        mult = 0.0
    else:
        if progress >= 0.75:
            mult = 0.0
        else:
            mult = 1.0 - (progress / 0.75)

    mill_rw = phase_config['mill_rw'] * mult
    dbl_rw = phase_config['dbl_mill'] * mult

    reward = mill_rw * pieces_captured
    if pieces_captured >= 2:
        reward += dbl_rw
    return reward, mult


# ============================================================================
# API ROUTES
# ============================================================================

@app.route('/api/evaluate', methods=['POST'])
def api_evaluate():
    data = request.json
    board_raw = data.get('board', [None] * 24)
    board = [b if b is not None else None for b in board_raw]
    mover = data.get('mover', 0)
    custom_weights = data.get('weights', None)
    result = full_evaluate(board, mover, custom_weights)
    return jsonify(result)


@app.route('/api/terminal_reward', methods=['POST'])
def api_terminal_reward():
    data = request.json
    result = data.get('result', 'win')
    steps = data.get('steps', 50)
    max_steps = data.get('max_steps', 200)
    win_base = data.get('win_base', 1.0)
    win_speed = data.get('win_speed', 0.5)
    loss_rw = data.get('loss_rw', -1.0)
    draw_pen = data.get('draw_pen', -0.8)
    reward = compute_terminal_reward(result, steps, max_steps, win_base, win_speed, loss_rw, draw_pen)
    return jsonify({'reward': reward})


@app.route('/api/shaping_reward', methods=['POST'])
def api_shaping_reward():
    data = request.json
    pieces_captured = data.get('pieces_captured', 1)
    phase_idx = data.get('phase_idx', 0)
    progress = data.get('progress', 0.0)
    pc = PHASE_CONFIGS[phase_idx]
    reward, mult = compute_shaping_reward(pieces_captured, pc, progress)
    return jsonify({'reward': reward, 'multiplier': mult})


@app.route('/api/board_info', methods=['POST'])
def api_board_info():
    pos = request.json.get('position', 0)
    return jsonify({
        'position': pos,
        'name': POSITION_NAMES[pos],
        'adjacency': list(ADJACENCY[pos]),
        'mills': [list(m) for m in POSITION_TO_MILLS[pos]],
        'value_movement': POSITION_VALUES[pos],
        'value_placement': POSITION_VALUES_PLACEMENT[pos],
    })


@app.route('/api/constants')
def api_constants():
    return jsonify({
        'mills': [list(m) for m in MILLS],
        'adjacency': [list(a) for a in ADJACENCY],
        'position_values': list(POSITION_VALUES),
        'position_values_placement': list(POSITION_VALUES_PLACEMENT),
        'weights': WEIGHTS,
        'position_names': POSITION_NAMES,
        'phase_configs': PHASE_CONFIGS,
    })


# ============================================================================
# HTML TEMPLATE
# ============================================================================

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Nine Men's Morris - Minimax Analysis</title>
<style>
:root {
  --bg: #1a1a2e;
  --surface: #16213e;
  --surface2: #0f3460;
  --accent: #e94560;
  --accent2: #533483;
  --text: #eee;
  --text-dim: #aaa;
  --green: #4ecca3;
  --red: #e94560;
  --blue: #4fc3f7;
  --yellow: #ffd54f;
  --orange: #ff9800;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; }
.container { max-width: 1400px; margin: 0 auto; padding: 20px; }
h1 { text-align: center; margin-bottom: 8px; font-size: 1.8em; background: linear-gradient(135deg, var(--accent), var(--accent2)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.subtitle { text-align: center; color: var(--text-dim); margin-bottom: 24px; font-size: 0.95em; }

/* Navigation */
.nav { display: flex; gap: 8px; justify-content: center; flex-wrap: wrap; margin-bottom: 24px; }
.nav button { background: var(--surface); border: 1px solid var(--surface2); color: var(--text); padding: 10px 20px; border-radius: 8px; cursor: pointer; font-size: 0.9em; transition: all 0.2s; }
.nav button:hover { border-color: var(--accent); }
.nav button.active { background: var(--accent); border-color: var(--accent); font-weight: bold; }

/* Sections */
.section { display: none; }
.section.active { display: block; }

/* Cards */
.card { background: var(--surface); border-radius: 12px; padding: 20px; margin-bottom: 16px; border: 1px solid var(--surface2); }
.card h2 { font-size: 1.2em; margin-bottom: 12px; color: var(--blue); }
.card h3 { font-size: 1em; margin-bottom: 8px; color: var(--yellow); }

/* Grid layout */
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.grid3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }
@media (max-width: 900px) { .grid2, .grid3 { grid-template-columns: 1fr; } }

/* SVG Board */
.board-wrap { display: flex; justify-content: center; align-items: flex-start; gap: 20px; flex-wrap: wrap; }
svg.board { background: var(--surface); border-radius: 12px; cursor: crosshair; }
svg.board .line { stroke: #555; stroke-width: 2; }
svg.board .cross-line { stroke: #555; stroke-width: 2; stroke-dasharray: 6 3; }
svg.board .point { fill: #444; stroke: #666; stroke-width: 2; cursor: pointer; transition: all 0.15s; }
svg.board .point:hover { fill: #888; r: 16; }
svg.board .point.p0 { fill: #fff; stroke: #ccc; }
svg.board .point.p1 { fill: #222; stroke: var(--accent); }
svg.board .point.highlight { stroke: var(--yellow); stroke-width: 3; }
svg.board .point.adj { stroke: var(--green); stroke-width: 3; }
svg.board .point.mill-hl { stroke: var(--orange); stroke-width: 3; }
svg.board text.label { fill: var(--text-dim); font-size: 11px; pointer-events: none; text-anchor: middle; }
svg.board text.value-label { fill: var(--yellow); font-size: 10px; pointer-events: none; text-anchor: middle; font-weight: bold; }
.mill-line { stroke: var(--orange); stroke-width: 4; opacity: 0.7; stroke-linecap: round; }

/* Eval bar */
.eval-row { display: flex; align-items: center; gap: 10px; margin: 6px 0; font-size: 0.9em; }
.eval-label { width: 160px; text-align: right; color: var(--text-dim); }
.eval-bar-wrap { flex: 1; height: 22px; background: #333; border-radius: 4px; overflow: hidden; position: relative; }
.eval-bar { height: 100%; border-radius: 4px; transition: width 0.3s; min-width: 2px; }
.eval-bar.positive { background: var(--green); }
.eval-bar.negative { background: var(--red); }
.eval-value { width: 100px; font-family: monospace; font-size: 0.85em; }
.eval-desc { width: 200px; color: var(--text-dim); font-size: 0.8em; font-family: monospace; }
.eval-total { font-size: 1.3em; font-weight: bold; text-align: center; padding: 12px; border-top: 1px solid var(--surface2); margin-top: 12px; }
.eval-total.positive { color: var(--green); }
.eval-total.negative { color: var(--red); }

/* Weights table */
table.weights { width: 100%; border-collapse: collapse; }
table.weights th, table.weights td { padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--surface2); }
table.weights th { color: var(--blue); font-size: 0.85em; }
table.weights input { background: var(--bg); border: 1px solid var(--surface2); color: var(--text); padding: 4px 8px; width: 100px; border-radius: 4px; font-family: monospace; }

/* Reward chart */
.reward-chart { width: 100%; height: 200px; position: relative; }
.reward-chart canvas { width: 100% !important; height: 100% !important; }

/* Controls */
.controls { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin-bottom: 12px; }
.controls label { color: var(--text-dim); font-size: 0.9em; }
.controls select, .controls input[type=range], .controls input[type=number] {
  background: var(--bg); border: 1px solid var(--surface2); color: var(--text);
  padding: 6px 10px; border-radius: 6px; font-size: 0.9em;
}
.controls input[type=range] { width: 200px; }
button.btn { background: var(--accent); border: none; color: #fff; padding: 8px 18px; border-radius: 6px; cursor: pointer; font-size: 0.9em; transition: opacity 0.2s; }
button.btn:hover { opacity: 0.8; }
button.btn-secondary { background: var(--surface2); }

/* Phase table */
table.phase-table { width: 100%; border-collapse: collapse; font-size: 0.82em; }
table.phase-table th, table.phase-table td { padding: 6px 8px; border-bottom: 1px solid var(--surface2); text-align: center; }
table.phase-table th { color: var(--blue); position: sticky; top: 0; background: var(--surface); }
table.phase-table tr:hover { background: var(--surface2); }
table.phase-table .hl { background: rgba(233, 69, 96, 0.2); }

/* Info panel */
.info-panel { background: var(--bg); border-radius: 8px; padding: 16px; font-size: 0.9em; line-height: 1.6; }
.info-panel code { background: var(--surface2); padding: 2px 6px; border-radius: 3px; font-size: 0.85em; }
.info-panel .formula { color: var(--yellow); font-family: monospace; margin: 8px 0; padding: 8px; background: var(--surface); border-radius: 4px; }

/* Mill list */
.mill-list { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
.mill-chip { background: var(--surface2); padding: 4px 10px; border-radius: 4px; font-size: 0.8em; cursor: pointer; transition: all 0.2s; font-family: monospace; }
.mill-chip:hover, .mill-chip.active { background: var(--orange); color: #000; }

/* Board piece selector */
.piece-sel { display: flex; gap: 8px; margin-bottom: 12px; align-items: center; }
.piece-btn { width: 32px; height: 32px; border-radius: 50%; cursor: pointer; border: 2px solid var(--surface2); transition: all 0.2s; }
.piece-btn.sel { border-color: var(--yellow); box-shadow: 0 0 8px var(--yellow); }
.piece-btn.white { background: #fff; }
.piece-btn.black { background: #222; border-color: var(--accent); }
.piece-btn.black.sel { border-color: var(--yellow); }
.piece-btn.empty { background: var(--surface); }

/* Shaping graph */
.shaping-graph { position: relative; }
.shaping-graph canvas { border-radius: 8px; }

/* Scrollable */
.scroll-x { overflow-x: auto; }
</style>
</head>
<body>
<div class="container">
<h1>Minimax Analysis</h1>
<p class="subtitle">Interactive visualization of Nine Men's Morris minimax evaluation, rewards & search</p>

<div class="nav">
  <button class="active" onclick="showSection('topology')">Board Topology</button>
  <button onclick="showSection('evaluation')">Evaluation</button>
  <button onclick="showSection('rewards')">Rewards</button>
  <button onclick="showSection('curriculum')">Curriculum</button>
  <button onclick="showSection('reference')">Reference</button>
</div>

<!-- =============== BOARD TOPOLOGY =============== -->
<div id="sec-topology" class="section active">
  <div class="grid2">
    <div class="card">
      <h2>Board Positions & Adjacency</h2>
      <p style="color:var(--text-dim);font-size:0.85em;margin-bottom:12px;">Click a position to see its connections, mills, and strategic value.</p>
      <div class="board-wrap">
        <svg id="topo-board" class="board" width="420" height="420" viewBox="-20 -20 440 440"></svg>
      </div>
    </div>
    <div>
      <div class="card" id="topo-info">
        <h2>Position Info</h2>
        <div class="info-panel" id="topo-detail">Click a position on the board to see details.</div>
      </div>
      <div class="card">
        <h2>All 16 Mills</h2>
        <div class="mill-list" id="mill-list"></div>
      </div>
      <div class="card">
        <h2>Position Values</h2>
        <div class="controls">
          <label>Show:</label>
          <select id="val-mode" onchange="drawTopoBoard()">
            <option value="movement">Movement Phase</option>
            <option value="placement">Placement Phase</option>
          </select>
        </div>
        <div style="color:var(--text-dim);font-size:0.85em;">Values shown as yellow numbers on the board. Higher = more strategically valuable (more connections, part of more mills).</div>
      </div>
    </div>
  </div>
</div>

<!-- =============== EVALUATION =============== -->
<div id="sec-evaluation" class="section">
  <div class="grid2">
    <div class="card">
      <h2>Interactive Board</h2>
      <div class="piece-sel">
        <span style="color:var(--text-dim);font-size:0.9em;">Place:</span>
        <div class="piece-btn white sel" onclick="setPiece(0)" id="sel-white"></div>
        <span style="font-size:0.8em;">White (P0)</span>
        <div class="piece-btn black" onclick="setPiece(1)" id="sel-black"></div>
        <span style="font-size:0.8em;">Black (P1)</span>
        <div class="piece-btn empty" onclick="setPiece(null)" id="sel-empty"></div>
        <span style="font-size:0.8em;">Clear</span>
        <button class="btn btn-secondary" onclick="clearEvalBoard()" style="margin-left:auto;">Reset</button>
      </div>
      <div class="controls">
        <label>Evaluate for:</label>
        <select id="eval-mover" onchange="runEval()">
          <option value="0">White (P0) to move</option>
          <option value="1">Black (P1) to move</option>
        </select>
      </div>
      <div class="board-wrap">
        <svg id="eval-board" class="board" width="420" height="420" viewBox="-20 -20 440 440"></svg>
      </div>
    </div>
    <div>
      <div class="card">
        <h2>Evaluation Breakdown</h2>
        <div id="eval-bars"></div>
        <div class="eval-total" id="eval-total">Total: 0</div>
      </div>
      <div class="card">
        <h2>Weights <span style="font-size:0.7em;color:var(--text-dim);">(edit to experiment)</span></h2>
        <div class="scroll-x">
          <table class="weights" id="weights-table"></table>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- =============== REWARDS =============== -->
<div id="sec-rewards" class="section">
  <div class="grid2">
    <div class="card">
      <h2>Terminal Rewards</h2>
      <div class="info-panel" style="margin-bottom:12px;">
        <p>When a game ends, the agent receives a terminal reward based on outcome and game length.</p>
        <div class="formula">reward = win_base + win_speed_bonus * max(0, 1 - steps/max_steps)</div>
      </div>
      <div class="controls">
        <label>Outcome:</label>
        <select id="rw-outcome" onchange="calcTerminalReward()">
          <option value="win">Win</option>
          <option value="loss">Loss</option>
          <option value="draw">Draw</option>
        </select>
        <label>Steps:</label>
        <input type="range" id="rw-steps" min="1" max="300" value="50" oninput="calcTerminalReward()">
        <span id="rw-steps-val">50</span>
      </div>
      <div class="controls">
        <label>Win Base:</label>
        <input type="number" id="rw-win-base" value="1.0" step="0.1" style="width:70px" onchange="calcTerminalReward()">
        <label>Speed Bonus:</label>
        <input type="number" id="rw-win-speed" value="0.5" step="0.1" style="width:70px" onchange="calcTerminalReward()">
        <label>Loss:</label>
        <input type="number" id="rw-loss" value="-1.0" step="0.1" style="width:70px" onchange="calcTerminalReward()">
        <label>Draw:</label>
        <input type="number" id="rw-draw" value="-0.8" step="0.1" style="width:70px" onchange="calcTerminalReward()">
      </div>
      <div class="eval-total" id="rw-result" style="font-size:1.5em;">0</div>
      <canvas id="rw-chart" height="180" style="margin-top:12px;"></canvas>
    </div>
    <div class="card">
      <h2>Shaping Rewards</h2>
      <div class="info-panel" style="margin-bottom:12px;">
        <p>During training, additional rewards shape behavior. The shaping multiplier decays over each phase.</p>
        <div class="formula">shaping_reward = mill_reward * multiplier * pieces_captured</div>
        <p>Phase 1: mult=1.0 (constant). Phase 2-9: 1.0 &rarr; 0.0 over first 75% of phase. Phase 10: mult=0.0.</p>
      </div>
      <div class="controls">
        <label>Phase:</label>
        <select id="sh-phase" onchange="calcShaping()">
        </select>
        <label>Progress:</label>
        <input type="range" id="sh-progress" min="0" max="100" value="0" oninput="calcShaping()">
        <span id="sh-progress-val">0%</span>
      </div>
      <div class="controls">
        <label>Pieces captured:</label>
        <select id="sh-captures" onchange="calcShaping()">
          <option value="1">1 (single mill)</option>
          <option value="2">2 (double mill)</option>
        </select>
      </div>
      <div id="sh-result" class="eval-total" style="font-size:1.2em;">Shaping reward: 0</div>
      <canvas id="sh-chart" height="180" style="margin-top:12px;"></canvas>
    </div>
  </div>
</div>

<!-- =============== CURRICULUM =============== -->
<div id="sec-curriculum" class="section">
  <div class="card">
    <h2>Training Curriculum - All 10 Phases</h2>
    <p style="color:var(--text-dim);font-size:0.85em;margin-bottom:12px;">Click a row to highlight it. Phases progress from simple (jumping with few stones) to full game.</p>
    <div class="scroll-x">
      <table class="phase-table" id="phase-table"></table>
    </div>
  </div>
  <div class="grid2">
    <div class="card">
      <h2>Shaping Multiplier Decay</h2>
      <p style="color:var(--text-dim);font-size:0.85em;margin-bottom:12px;">Shows how the shaping reward multiplier decays within a phase. Resets to 1.0 at each new phase (except Phase 1 and 10).</p>
      <canvas id="curriculum-chart" height="220"></canvas>
    </div>
    <div class="card">
      <h2>Learning Rate Schedule</h2>
      <p style="color:var(--text-dim);font-size:0.85em;margin-bottom:12px;">Each phase has its own LR range that linearly decays over its lr_decay_episodes.</p>
      <canvas id="lr-chart" height="220"></canvas>
    </div>
  </div>
</div>

<!-- =============== REFERENCE =============== -->
<div id="sec-reference" class="section">
  <div class="grid2">
    <div class="card">
      <h2>Evaluation Weights Reference</h2>
      <div class="info-panel">
        <p><strong>How evaluate() works:</strong></p>
        <p>The evaluation function scores a position from the <strong>current mover's perspective</strong> (negamax convention). Each component compares the mover vs opponent and multiplies by its weight.</p>
        <br>
        <table class="weights" style="font-size:0.85em;">
          <tr><th>Weight</th><th>Value</th><th>Meaning</th></tr>
          <tr><td><code>win</code></td><td>1,000,000</td><td>Terminal win score</td></tr>
          <tr><td><code>loss</code></td><td>-1,000,000</td><td>Terminal loss score (current mover lost)</td></tr>
          <tr><td><code>draw</code></td><td>-50,000</td><td>Draw is slightly negative (prefer winning)</td></tr>
          <tr><td><code>mill</code></td><td>5,000</td><td>Per complete mill (3 in a row)</td></tr>
          <tr><td><code>potential_mill</code></td><td>1,500</td><td>2 pieces + 1 empty in a mill line</td></tr>
          <tr><td><code>double_mill</code></td><td>8,000</td><td>Position that is part of 2+ complete mills</td></tr>
          <tr><td><code>blocked_mill</code></td><td>2,000</td><td>Opponent has 2 in mill, we block with 1</td></tr>
          <tr><td><code>unblocked_threat</code></td><td>3,000</td><td>Penalty: opponent has 2 in mill, spot is empty</td></tr>
          <tr><td><code>mobility</code></td><td>100</td><td>Number of available moves (adj moves, or flying)</td></tr>
          <tr><td><code>position</code></td><td>50</td><td>Strategic position values (connectivity-based)</td></tr>
        </table>
      </div>
    </div>
    <div class="card">
      <h2>Action Encoding</h2>
      <div class="info-panel">
        <p><strong>OpenSpiel Nine Men's Morris actions (0-599):</strong></p>
        <br>
        <p><code>0-23</code>: <strong>Placement</strong> (place piece at position) OR <strong>Removal</strong> (remove opponent piece at position)</p>
        <p>Whether it's placement or removal depends on game state - in removal state after forming a mill, actions 0-23 target opponent pieces.</p>
        <br>
        <p><code>24-599</code>: <strong>Movement</strong> encoded as <code>24 + from_pos * 24 + to_pos</code></p>
        <p>Decode: <code>from_pos = (action - 24) // 24</code>, <code>to_pos = (action - 24) % 24</code></p>
        <br>
        <p><strong>Game phases:</strong></p>
        <ul style="margin-left:20px;margin-top:4px;line-height:1.8;">
          <li><strong>Placing</strong>: Each player places 9 pieces (actions 0-23)</li>
          <li><strong>Moving</strong>: Slide pieces to adjacent positions (actions 24-599)</li>
          <li><strong>Flying</strong>: When down to 3 pieces, can move anywhere (actions 24-599)</li>
          <li><strong>Removal</strong>: After forming a mill, remove opponent piece (actions 0-23)</li>
        </ul>
      </div>
    </div>
  </div>
  <div class="card">
    <h2>Search Algorithm: PVS + Iterative Deepening</h2>
    <div class="info-panel">
      <p><strong>Search flow for each move:</strong></p>
      <ol style="margin-left:20px;margin-top:8px;line-height:2;">
        <li><strong>Iterative Deepening:</strong> Search depth 1, then 2, then 3... up to max_depth. Each iteration uses results from previous (via TT).</li>
        <li><strong>Aspiration Windows (depth >= 4):</strong> Start with narrow window [prev_score - 500, prev_score + 500]. If score falls outside, re-search with full window.</li>
        <li><strong>Transposition Table Probe:</strong> Check if this position was already evaluated. If TT hit with sufficient depth, use stored result directly.</li>
        <li><strong>Move Ordering:</strong> Order moves: TT move first, then mill-forming moves, killer moves, history heuristic, positional value.</li>
        <li><strong>Principal Variation Search:</strong> First move gets full [alpha, beta] window. Subsequent moves get null window [alpha, alpha+1]. Re-search if null window fails high.</li>
        <li><strong>Late Move Reductions:</strong> After 4th move at depth >= 3, reduce search depth by 1. Re-search at full depth if it improves alpha.</li>
        <li><strong>Quiescence Search:</strong> At leaf nodes, extend search for mill-forming moves to avoid horizon effect. Max 4 extra plies.</li>
      </ol>
    </div>
  </div>
</div>

</div>

<script>
// ============================================================================
// STATE
// ============================================================================
let CONSTS = null;
let selectedPiece = 0; // 0=white, 1=black, null=clear
let evalBoard = new Array(24).fill(null);
let topoSelected = -1;
let highlightedMill = -1;

const POS_COORDS = [
  [0,0],[3,0],[6,0],
  [1,1],[3,1],[5,1],
  [2,2],[3,2],[4,2],
  [0,3],[1,3],[2,3],[4,3],[5,3],[6,3],
  [2,4],[3,4],[4,4],
  [1,5],[3,5],[5,5],
  [0,6],[3,6],[6,6],
];

function toSVG(gx, gy) {
  return [gx * 400/6, gy * 400/6];
}

// ============================================================================
// NAVIGATION
// ============================================================================
function showSection(name) {
  document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
  document.getElementById('sec-' + name).classList.add('active');
  document.querySelectorAll('.nav button').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
}

// ============================================================================
// BOARD DRAWING
// ============================================================================
function drawBoard(svgId, board, onClick, options={}) {
  const svg = document.getElementById(svgId);
  svg.innerHTML = '';
  const vals = options.showValues;
  const hlPos = options.highlight || -1;
  const adjHL = options.adjHighlight || [];
  const millHL = options.millHighlight || [];

  // Draw lines
  const lines = [
    [0,1],[1,2],[3,4],[4,5],[6,7],[7,8],
    [9,10],[10,11],[12,13],[13,14],
    [15,16],[16,17],[18,19],[19,20],[21,22],[22,23],
    [0,9],[9,21],[3,10],[10,18],[6,11],[11,15],
    [2,14],[14,23],[5,13],[13,20],[8,12],[12,17],
  ];
  const crossLines = [[1,4],[4,7],[16,19],[19,22],[9,10],[10,11],[12,13],[13,14]];
  // Actually cross lines are: 1-4, 4-7, 9-10, 10-11, 12-13, 13-14, 16-19, 19-22
  // But these overlap with adjacency. Let me just draw all adjacency edges.

  const drawn = new Set();
  for (let i = 0; i < 24; i++) {
    const [x1,y1] = toSVG(...POS_COORDS[i]);
    for (const j of (CONSTS ? CONSTS.adjacency[i] : [])) {
      const key = Math.min(i,j) + '-' + Math.max(i,j);
      if (drawn.has(key)) continue;
      drawn.add(key);
      const [x2,y2] = toSVG(...POS_COORDS[j]);
      const line = document.createElementNS('http://www.w3.org/2000/svg','line');
      line.setAttribute('x1',x1); line.setAttribute('y1',y1);
      line.setAttribute('x2',x2); line.setAttribute('y2',y2);
      line.setAttribute('class','line');
      svg.appendChild(line);
    }
  }

  // Draw mill highlights
  if (millHL.length > 0) {
    for (const mill of millHL) {
      for (let k = 0; k < mill.length; k++) {
        const a = mill[k], b = mill[(k+1)%mill.length];
        if (k < mill.length - 1) {
          const [x1,y1] = toSVG(...POS_COORDS[a]);
          const [x2,y2] = toSVG(...POS_COORDS[b]);
          const line = document.createElementNS('http://www.w3.org/2000/svg','line');
          line.setAttribute('x1',x1); line.setAttribute('y1',y1);
          line.setAttribute('x2',x2); line.setAttribute('y2',y2);
          line.setAttribute('class','mill-line');
          svg.appendChild(line);
        }
      }
    }
  }

  // Draw points
  for (let i = 0; i < 24; i++) {
    const [x,y] = toSVG(...POS_COORDS[i]);
    const circ = document.createElementNS('http://www.w3.org/2000/svg','circle');
    circ.setAttribute('cx',x); circ.setAttribute('cy',y); circ.setAttribute('r',14);
    let cls = 'point';
    if (board && board[i] === 0) cls += ' p0';
    else if (board && board[i] === 1) cls += ' p1';
    if (i === hlPos) cls += ' highlight';
    if (adjHL.includes(i)) cls += ' adj';
    if (millHL.flat && millHL.flat().includes(i) && i !== hlPos) cls += ' mill-hl';
    circ.setAttribute('class', cls);
    circ.addEventListener('click', () => onClick(i));
    svg.appendChild(circ);

    // Index label
    const txt = document.createElementNS('http://www.w3.org/2000/svg','text');
    txt.setAttribute('x', x); txt.setAttribute('y', y - 20);
    txt.setAttribute('class','label');
    txt.textContent = i;
    svg.appendChild(txt);

    // Value label
    if (vals) {
      const vt = document.createElementNS('http://www.w3.org/2000/svg','text');
      vt.setAttribute('x', x); vt.setAttribute('y', y + 5);
      vt.setAttribute('class','value-label');
      vt.textContent = vals[i];
      svg.appendChild(vt);
    }
  }
}

// ============================================================================
// TOPOLOGY SECTION
// ============================================================================
function drawTopoBoard() {
  if (!CONSTS) return;
  const mode = document.getElementById('val-mode').value;
  const vals = mode === 'placement' ? CONSTS.position_values_placement : CONSTS.position_values;
  const mills = topoSelected >= 0 ? CONSTS.mills.filter(m => m.includes(topoSelected)) : [];
  if (highlightedMill >= 0) mills.push(CONSTS.mills[highlightedMill]);
  const adj = topoSelected >= 0 ? CONSTS.adjacency[topoSelected] : [];

  drawBoard('topo-board', null, onTopoClick, {
    showValues: vals,
    highlight: topoSelected,
    adjHighlight: adj,
    millHighlight: mills,
  });
}

function onTopoClick(pos) {
  topoSelected = (topoSelected === pos) ? -1 : pos;
  highlightedMill = -1;
  drawTopoBoard();
  updateTopoInfo(pos);
}

function updateTopoInfo(pos) {
  if (!CONSTS || topoSelected < 0) {
    document.getElementById('topo-detail').innerHTML = 'Click a position on the board to see details.';
    return;
  }
  const adj = CONSTS.adjacency[pos];
  const mills = CONSTS.mills.filter(m => m.includes(pos));
  const vMov = CONSTS.position_values[pos];
  const vPla = CONSTS.position_values_placement[pos];
  const name = CONSTS.position_names[pos];

  let html = `<h3 style="color:var(--yellow);">Position ${pos} (${name})</h3>`;
  html += `<p><strong>Adjacent:</strong> ${adj.map(a => `<code>${a}</code>`).join(', ')}</p>`;
  html += `<p><strong>Connectivity:</strong> ${adj.length} neighbors</p>`;
  html += `<p><strong>Mills containing this position:</strong></p><ul>`;
  mills.forEach(m => {
    html += `<li><code>[${m.join(', ')}]</code> (${m.map(p => CONSTS.position_names[p]).join(' - ')})</li>`;
  });
  html += `</ul>`;
  html += `<p><strong>Strategic Value (movement):</strong> ${vMov}</p>`;
  html += `<p><strong>Strategic Value (placement):</strong> ${vPla}</p>`;
  html += `<p style="color:var(--text-dim);font-size:0.85em;margin-top:8px;">Green = adjacent, Orange = mill connections</p>`;
  document.getElementById('topo-detail').innerHTML = html;
}

function buildMillList() {
  if (!CONSTS) return;
  const el = document.getElementById('mill-list');
  el.innerHTML = '';
  CONSTS.mills.forEach((mill, idx) => {
    const chip = document.createElement('span');
    chip.className = 'mill-chip';
    chip.textContent = `[${mill.join(',')}]`;
    chip.addEventListener('click', () => {
      highlightedMill = (highlightedMill === idx) ? -1 : idx;
      document.querySelectorAll('.mill-chip').forEach(c => c.classList.remove('active'));
      if (highlightedMill >= 0) chip.classList.add('active');
      drawTopoBoard();
    });
    el.appendChild(chip);
  });
}

// ============================================================================
// EVALUATION SECTION
// ============================================================================
let customWeights = null;

function setPiece(p) {
  selectedPiece = p;
  document.getElementById('sel-white').classList.toggle('sel', p === 0);
  document.getElementById('sel-black').classList.toggle('sel', p === 1);
  document.getElementById('sel-empty').classList.toggle('sel', p === null);
}

function clearEvalBoard() {
  evalBoard = new Array(24).fill(null);
  drawEvalBoard();
  runEval();
}

function drawEvalBoard() {
  drawBoard('eval-board', evalBoard, onEvalClick, {});
}

function onEvalClick(pos) {
  if (evalBoard[pos] === selectedPiece) {
    evalBoard[pos] = null;
  } else {
    evalBoard[pos] = selectedPiece;
  }
  drawEvalBoard();
  runEval();
}

async function runEval() {
  const mover = parseInt(document.getElementById('eval-mover').value);
  const weights = customWeights || CONSTS.weights;
  const resp = await fetch('/api/evaluate', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({board: evalBoard, mover, weights})
  });
  const data = await resp.json();
  renderEvalBars(data);
}

function renderEvalBars(data) {
  const container = document.getElementById('eval-bars');
  const comps = data.components;

  let html = '';

  // Check for terminal state
  if (comps.terminal) {
    const term = comps.terminal;
    const cls = term.result === 'WIN' ? 'positive' : 'negative';
    const color = term.result === 'WIN' ? 'var(--green)' : 'var(--red)';
    html += `<div style="background:${term.result === 'WIN' ? 'rgba(78, 204, 163, 0.1)' : 'rgba(233, 69, 96, 0.1)'};padding:20px;border-radius:8px;margin-bottom:16px;border:2px solid ${color}">
      <div style="font-size:1.5em;font-weight:bold;color:${color};margin-bottom:12px;">🏆 TERMINAL STATE: ${term.result}</div>
      <div style="color:var(--text);font-size:1.1em;">${term.desc}</div>
      <div style="color:var(--text-dim);font-size:0.9em;margin-top:8px;">Score: ${term.score.toLocaleString()}</div>
    </div>`;
    container.innerHTML = html;

    const totalEl = document.getElementById('eval-total');
    totalEl.textContent = `Terminal ${term.result}: ${term.score >= 0 ? '+' : ''}${term.score.toLocaleString()}`;
    totalEl.className = 'eval-total ' + cls;
    return;
  }

  // Normal evaluation
  const maxAbs = Math.max(1, ...Object.values(comps).map(c => Math.abs(c.score)));

  const order = ['pieces','mills','potential_mills','blocked_mills','unblocked_threats','double_mills','mobility','position'];
  const labels = {
    pieces: 'Piece Count',
    mills: 'Complete Mills',
    potential_mills: 'Potential Mills',
    blocked_mills: 'Blocked Mills',
    unblocked_threats: 'Unblocked Threats',
    double_mills: 'Double Mills',
    mobility: 'Mobility',
    position: 'Positional',
  };

  for (const key of order) {
    const c = comps[key];
    if (!c) continue;
    const pct = Math.abs(c.score) / maxAbs * 100;
    const cls = c.score >= 0 ? 'positive' : 'negative';
    html += `<div class="eval-row">
      <div class="eval-label">${labels[key]}</div>
      <div class="eval-bar-wrap"><div class="eval-bar ${cls}" style="width:${pct}%"></div></div>
      <div class="eval-value" style="color:${c.score >= 0 ? 'var(--green)' : 'var(--red)'}">${c.score >= 0 ? '+' : ''}${c.score.toLocaleString()}</div>
      <div class="eval-desc">${c.desc}</div>
    </div>`;
  }
  container.innerHTML = html;

  const totalEl = document.getElementById('eval-total');
  const t = data.total;
  totalEl.textContent = `Total Score: ${t >= 0 ? '+' : ''}${t.toLocaleString()}`;
  totalEl.className = 'eval-total ' + (t >= 0 ? 'positive' : 'negative');
}

function buildWeightsTable() {
  if (!CONSTS) return;
  const table = document.getElementById('weights-table');
  let html = '<tr><th>Weight</th><th>Value</th><th>Description</th></tr>';
  const descs = {
    win: 'Terminal win', loss: 'Terminal loss', draw: 'Draw penalty',
    mill: 'Per complete mill',
    potential_mill: '2 of 3 in mill line', double_mill: 'Piece in 2+ mills',
    blocked_mill: 'Blocking opponent mill', unblocked_threat: 'Opponent open threat',
    mobility: 'Available moves', position: 'Positional value',
  };
  for (const [key, val] of Object.entries(CONSTS.weights)) {
    html += `<tr><td><code>${key}</code></td><td><input type="number" value="${val}" data-key="${key}" onchange="onWeightChange()"></td><td style="color:var(--text-dim);font-size:0.85em;">${descs[key] || ''}</td></tr>`;
  }
  table.innerHTML = html;
}

function onWeightChange() {
  customWeights = {};
  document.querySelectorAll('#weights-table input').forEach(inp => {
    customWeights[inp.dataset.key] = parseFloat(inp.value) || 0;
  });
  runEval();
}

// ============================================================================
// REWARDS SECTION
// ============================================================================
function calcTerminalReward() {
  const outcome = document.getElementById('rw-outcome').value;
  const steps = parseInt(document.getElementById('rw-steps').value);
  document.getElementById('rw-steps-val').textContent = steps;
  const winBase = parseFloat(document.getElementById('rw-win-base').value);
  const winSpeed = parseFloat(document.getElementById('rw-win-speed').value);
  const lossRw = parseFloat(document.getElementById('rw-loss').value);
  const drawPen = parseFloat(document.getElementById('rw-draw').value);

  let reward;
  if (outcome === 'win') {
    reward = winBase + winSpeed * Math.max(0, 1 - steps / 200);
  } else if (outcome === 'loss') {
    reward = lossRw;
  } else {
    reward = drawPen;
  }

  const el = document.getElementById('rw-result');
  el.textContent = `Reward: ${reward.toFixed(3)}`;
  el.style.color = reward >= 0 ? 'var(--green)' : 'var(--red)';

  drawTerminalChart(winBase, winSpeed, lossRw, drawPen);
}

function drawTerminalChart(winBase, winSpeed, lossRw, drawPen) {
  const canvas = document.getElementById('rw-chart');
  const ctx = canvas.getContext('2d');
  const W = canvas.width = canvas.offsetWidth * (window.devicePixelRatio || 1);
  const H = canvas.height = 180 * (window.devicePixelRatio || 1);
  canvas.style.height = '180px';
  ctx.scale(window.devicePixelRatio || 1, window.devicePixelRatio || 1);
  const w = canvas.offsetWidth, h = 180;

  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#1a1a2e';
  ctx.fillRect(0, 0, w, h);

  const pad = {l: 50, r: 20, t: 20, b: 30};
  const gw = w - pad.l - pad.r, gh = h - pad.t - pad.b;

  // Y range
  const yMin = Math.min(lossRw, drawPen) - 0.2;
  const yMax = Math.max(winBase + winSpeed, 0) + 0.3;
  const toX = (s) => pad.l + (s / 300) * gw;
  const toY = (v) => pad.t + gh - ((v - yMin) / (yMax - yMin)) * gh;

  // Grid
  ctx.strokeStyle = '#333'; ctx.lineWidth = 1;
  for (let v = Math.ceil(yMin); v <= Math.floor(yMax); v += 0.5) {
    ctx.beginPath(); ctx.moveTo(pad.l, toY(v)); ctx.lineTo(pad.l + gw, toY(v)); ctx.stroke();
    ctx.fillStyle = '#666'; ctx.font = '11px monospace'; ctx.textAlign = 'right';
    ctx.fillText(v.toFixed(1), pad.l - 5, toY(v) + 4);
  }

  // Zero line
  ctx.strokeStyle = '#555'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad.l, toY(0)); ctx.lineTo(pad.l + gw, toY(0)); ctx.stroke();

  // Win line
  ctx.strokeStyle = '#4ecca3'; ctx.lineWidth = 2;
  ctx.beginPath();
  for (let s = 0; s <= 300; s += 2) {
    const r = winBase + winSpeed * Math.max(0, 1 - s / 200);
    const x = toX(s), y = toY(r);
    s === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.fillStyle = '#4ecca3'; ctx.font = '11px sans-serif'; ctx.textAlign = 'left';
  ctx.fillText('Win', toX(305), toY(winBase) + 4);

  // Loss line
  ctx.strokeStyle = '#e94560'; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(toX(0), toY(lossRw)); ctx.lineTo(toX(300), toY(lossRw)); ctx.stroke();
  ctx.fillStyle = '#e94560';
  ctx.fillText('Loss', toX(305), toY(lossRw) + 4);

  // Draw line
  ctx.strokeStyle = '#ffd54f'; ctx.lineWidth = 2; ctx.setLineDash([5,3]);
  ctx.beginPath(); ctx.moveTo(toX(0), toY(drawPen)); ctx.lineTo(toX(300), toY(drawPen)); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = '#ffd54f';
  ctx.fillText('Draw', toX(305), toY(drawPen) + 4);

  // X axis
  ctx.fillStyle = '#666'; ctx.font = '11px monospace'; ctx.textAlign = 'center';
  for (let s = 0; s <= 300; s += 50) {
    ctx.fillText(s, toX(s), h - 5);
  }
  ctx.fillText('Steps', toX(150), h - 5 + 15);
}

function calcShaping() {
  const phaseIdx = parseInt(document.getElementById('sh-phase').value);
  const progress = parseInt(document.getElementById('sh-progress').value) / 100;
  document.getElementById('sh-progress-val').textContent = Math.round(progress * 100) + '%';
  const captures = parseInt(document.getElementById('sh-captures').value);
  const pc = CONSTS.phase_configs[phaseIdx];

  let mult;
  if (pc.phase === 1) mult = 1.0;
  else if (pc.phase === 10) mult = 0.0;
  else {
    mult = progress >= 0.75 ? 0.0 : 1.0 - (progress / 0.75);
  }

  const millRw = pc.mill_rw * mult;
  const dblRw = pc.dbl_mill * mult;
  let reward = millRw * captures;
  if (captures >= 2) reward += dblRw;

  const el = document.getElementById('sh-result');
  el.innerHTML = `Shaping reward: <span style="color:var(--green)">${reward.toFixed(4)}</span> (multiplier: ${mult.toFixed(3)})`;

  drawShapingChart(pc);
}

function drawShapingChart(pc) {
  const canvas = document.getElementById('sh-chart');
  const ctx = canvas.getContext('2d');
  const W = canvas.width = canvas.offsetWidth * (window.devicePixelRatio || 1);
  const H = canvas.height = 180 * (window.devicePixelRatio || 1);
  canvas.style.height = '180px';
  ctx.scale(window.devicePixelRatio || 1, window.devicePixelRatio || 1);
  const w = canvas.offsetWidth, h = 180;

  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#1a1a2e';
  ctx.fillRect(0, 0, w, h);

  const pad = {l: 50, r: 20, t: 20, b: 30};
  const gw = w - pad.l - pad.r, gh = h - pad.t - pad.b;

  const toX = (p) => pad.l + p * gw;
  const toY = (v) => pad.t + gh - v * gh;

  // Grid
  ctx.strokeStyle = '#333'; ctx.lineWidth = 1;
  for (let v = 0; v <= 1; v += 0.25) {
    ctx.beginPath(); ctx.moveTo(pad.l, toY(v)); ctx.lineTo(pad.l + gw, toY(v)); ctx.stroke();
    ctx.fillStyle = '#666'; ctx.font = '11px monospace'; ctx.textAlign = 'right';
    ctx.fillText(v.toFixed(2), pad.l - 5, toY(v) + 4);
  }

  // Multiplier line
  ctx.strokeStyle = '#4fc3f7'; ctx.lineWidth = 2;
  ctx.beginPath();
  for (let p = 0; p <= 100; p++) {
    const prog = p / 100;
    let mult;
    if (pc.phase === 1) mult = 1.0;
    else if (pc.phase === 10) mult = 0.0;
    else mult = prog >= 0.75 ? 0.0 : 1.0 - (prog / 0.75);
    const x = toX(prog), y = toY(mult);
    p === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Effective mill reward line
  ctx.strokeStyle = '#4ecca3'; ctx.lineWidth = 2; ctx.setLineDash([4,2]);
  ctx.beginPath();
  for (let p = 0; p <= 100; p++) {
    const prog = p / 100;
    let mult;
    if (pc.phase === 1) mult = 1.0;
    else if (pc.phase === 10) mult = 0.0;
    else mult = prog >= 0.75 ? 0.0 : 1.0 - (prog / 0.75);
    const rw = pc.mill_rw * mult;
    const x = toX(prog), y = toY(rw / 0.3); // normalize to [0,1]
    p === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.setLineDash([]);

  // Labels
  ctx.fillStyle = '#4fc3f7'; ctx.font = '11px sans-serif'; ctx.textAlign = 'left';
  ctx.fillText('Multiplier', pad.l + 5, pad.t + 12);
  ctx.fillStyle = '#4ecca3';
  ctx.fillText('Mill Reward (normalized)', pad.l + 85, pad.t + 12);

  // X axis
  ctx.fillStyle = '#666'; ctx.font = '11px monospace'; ctx.textAlign = 'center';
  for (let p = 0; p <= 100; p += 25) {
    ctx.fillText(p + '%', toX(p/100), h - 5);
  }
  ctx.fillText('Phase Progress', toX(0.5), h + 8);
}

// ============================================================================
// CURRICULUM SECTION
// ============================================================================
function buildPhaseTable() {
  if (!CONSTS) return;
  const table = document.getElementById('phase-table');
  let html = '<tr><th>Phase</th><th>Description</th><th>Stones</th><th>Start</th><th>Opponent</th><th>Shaping</th><th>LR</th><th>Win</th><th>Loss</th><th>Draw</th><th>Mill Rw</th><th>Grad WR</th></tr>';
  CONSTS.phase_configs.forEach((pc, idx) => {
    html += `<tr onclick="this.classList.toggle('hl')" style="cursor:pointer;">
      <td><strong>${pc.phase}</strong></td>
      <td style="text-align:left;font-size:0.8em;">${pc.desc}</td>
      <td>${pc.stones}</td>
      <td>${pc.start}</td>
      <td>${pc.opponent}</td>
      <td>${pc.shaping_mult}</td>
      <td style="font-size:0.75em;">${pc.lr_start.toExponential(0)}&rarr;${pc.lr_end.toExponential(0)}</td>
      <td>${pc.win}</td>
      <td>${pc.loss}</td>
      <td>${pc.draw}</td>
      <td>${pc.mill_rw}</td>
      <td>${(pc.grad_wr * 100).toFixed(0)}%</td>
    </tr>`;
  });
  table.innerHTML = html;
}

function drawCurriculumChart() {
  if (!CONSTS) return;
  const canvas = document.getElementById('curriculum-chart');
  const ctx = canvas.getContext('2d');
  const W = canvas.width = canvas.offsetWidth * (window.devicePixelRatio || 1);
  const H = canvas.height = 220 * (window.devicePixelRatio || 1);
  canvas.style.height = '220px';
  ctx.scale(window.devicePixelRatio || 1, window.devicePixelRatio || 1);
  const w = canvas.offsetWidth, h = 220;

  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#1a1a2e';
  ctx.fillRect(0, 0, w, h);

  const pad = {l: 50, r: 20, t: 20, b: 35};
  const gw = w - pad.l - pad.r, gh = h - pad.t - pad.b;
  const phaseW = gw / 10;

  const toY = (v) => pad.t + gh - v * gh;

  // Grid
  ctx.strokeStyle = '#333'; ctx.lineWidth = 1;
  for (let v = 0; v <= 1; v += 0.25) {
    ctx.beginPath(); ctx.moveTo(pad.l, toY(v)); ctx.lineTo(pad.l + gw, toY(v)); ctx.stroke();
    ctx.fillStyle = '#666'; ctx.font = '11px monospace'; ctx.textAlign = 'right';
    ctx.fillText(v.toFixed(2), pad.l - 5, toY(v) + 4);
  }

  // Draw each phase
  const colors = ['#e94560','#ff9800','#ffd54f','#4ecca3','#4fc3f7','#533483','#e94560','#ff9800','#ffd54f','#4ecca3'];
  CONSTS.phase_configs.forEach((pc, idx) => {
    const x0 = pad.l + idx * phaseW;
    ctx.strokeStyle = colors[idx % colors.length]; ctx.lineWidth = 2;
    ctx.beginPath();
    for (let p = 0; p <= 100; p++) {
      const prog = p / 100;
      let mult;
      if (pc.phase === 1) mult = 1.0;
      else if (pc.phase === 10) mult = 0.0;
      else mult = prog >= 0.75 ? 0.0 : 1.0 - (prog / 0.75);
      const x = x0 + prog * phaseW;
      const y = toY(mult);
      p === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Phase label
    ctx.fillStyle = colors[idx % colors.length];
    ctx.font = '10px sans-serif'; ctx.textAlign = 'center';
    ctx.fillText('P' + pc.phase, x0 + phaseW / 2, h - 5);

    // Separator
    if (idx > 0) {
      ctx.strokeStyle = '#444'; ctx.lineWidth = 1; ctx.setLineDash([3,3]);
      ctx.beginPath(); ctx.moveTo(x0, pad.t); ctx.lineTo(x0, pad.t + gh); ctx.stroke();
      ctx.setLineDash([]);
    }
  });
}

function drawLRChart() {
  if (!CONSTS) return;
  const canvas = document.getElementById('lr-chart');
  const ctx = canvas.getContext('2d');
  const W = canvas.width = canvas.offsetWidth * (window.devicePixelRatio || 1);
  const H = canvas.height = 220 * (window.devicePixelRatio || 1);
  canvas.style.height = '220px';
  ctx.scale(window.devicePixelRatio || 1, window.devicePixelRatio || 1);
  const w = canvas.offsetWidth, h = 220;

  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#1a1a2e';
  ctx.fillRect(0, 0, w, h);

  const pad = {l: 60, r: 20, t: 20, b: 35};
  const gw = w - pad.l - pad.r, gh = h - pad.t - pad.b;

  // Collect all LR values
  const lrPairs = CONSTS.phase_configs.map(pc => [pc.lr_start, pc.lr_end]);
  const allLR = lrPairs.flat();
  const maxLR = Math.max(...allLR);
  const minLR = Math.min(...allLR.filter(v => v > 0));

  // Log scale
  const logMin = Math.log10(minLR) - 0.5;
  const logMax = Math.log10(maxLR) + 0.5;
  const toY = (lr) => {
    if (lr <= 0) return pad.t + gh;
    const l = Math.log10(lr);
    return pad.t + gh - ((l - logMin) / (logMax - logMin)) * gh;
  };

  // Grid
  ctx.strokeStyle = '#333'; ctx.lineWidth = 1;
  for (let exp = Math.ceil(logMin); exp <= Math.floor(logMax); exp++) {
    const y = toY(Math.pow(10, exp));
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + gw, y); ctx.stroke();
    ctx.fillStyle = '#666'; ctx.font = '11px monospace'; ctx.textAlign = 'right';
    ctx.fillText('1e' + exp, pad.l - 5, y + 4);
  }

  // Bars for each phase
  const barW = gw / 10 * 0.7;
  const gap = gw / 10 * 0.15;
  const colors = ['#e94560','#ff9800','#ffd54f','#4ecca3','#4fc3f7','#533483','#e94560','#ff9800','#ffd54f','#4ecca3'];

  CONSTS.phase_configs.forEach((pc, idx) => {
    const x = pad.l + idx * (gw / 10) + gap;
    const yStart = toY(pc.lr_start);
    const yEnd = toY(pc.lr_end);

    // Bar from start to end
    ctx.fillStyle = colors[idx % colors.length] + '40';
    ctx.fillRect(x, Math.min(yStart, yEnd), barW, Math.abs(yEnd - yStart) || 2);

    // Start dot
    ctx.fillStyle = colors[idx % colors.length];
    ctx.beginPath(); ctx.arc(x + barW/2, yStart, 4, 0, Math.PI*2); ctx.fill();
    // End dot
    ctx.beginPath(); ctx.arc(x + barW/2, yEnd, 3, 0, Math.PI*2); ctx.fill();

    // Arrow
    ctx.strokeStyle = colors[idx % colors.length]; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(x + barW/2, yStart); ctx.lineTo(x + barW/2, yEnd); ctx.stroke();

    // Label
    ctx.fillStyle = colors[idx % colors.length];
    ctx.font = '10px sans-serif'; ctx.textAlign = 'center';
    ctx.fillText('P' + pc.phase, x + barW/2, h - 5);
  });
}

// ============================================================================
// INIT
// ============================================================================
function populateShapingPhases() {
  if (!CONSTS) return;
  const sel = document.getElementById('sh-phase');
  sel.innerHTML = '';
  CONSTS.phase_configs.forEach((pc, idx) => {
    const opt = document.createElement('option');
    opt.value = idx;
    opt.textContent = `Phase ${pc.phase}: ${pc.desc.substring(0, 40)}`;
    sel.appendChild(opt);
  });
}

async function init() {
  const resp = await fetch('/api/constants');
  CONSTS = await resp.json();

  drawTopoBoard();
  buildMillList();
  drawEvalBoard();
  buildWeightsTable();
  runEval();
  populateShapingPhases();
  calcTerminalReward();
  calcShaping();
  buildPhaseTable();
  drawCurriculumChart();
  drawLRChart();
}

// Redraw charts on resize
let resizeTimer;
window.addEventListener('resize', () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    calcTerminalReward();
    calcShaping();
    drawCurriculumChart();
    drawLRChart();
  }, 200);
});

init();
</script>
</body>
</html>"""


@app.route('/')
def index():
    return render_template_string(HTML)


if __name__ == '__main__':
    print("Starting Minimax Analysis Server on http://0.0.0.0:7860")
    app.run(host='0.0.0.0', port=7860, debug=True)
