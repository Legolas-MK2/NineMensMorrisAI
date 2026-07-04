"""
Nine Men's Morris - Public Web App
A clean, mobile-friendly Flask app for playing against (or testing) trained
models. Self-contained: only scans `webapp/models/` for `.pt` files, so you
can ship this directory to other people without dragging along training
checkpoints.

Run:
    cd webapp
    python app.py
Then open http://localhost:7861 on any device on your network.
"""

from __future__ import annotations

import os

import random
import socket
import sys
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from flask import Flask, g, jsonify, render_template, request, send_from_directory

# The webapp depends on the training-side game/model code (model.py, utils.py,
# minimax.py, etc.) but ships with its OWN models/ directory.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import fastnmm  # noqa: E402

from board_utils import decode_action, parse_board_positions as _parse_board_positions  # noqa: E402
from config import Config  # noqa: E402
from minimax import MinimaxBot  # noqa: E402
from model import ActorCritic  # noqa: E402
from utils import build_token_obs, get_legal_mask  # noqa: E402


app = Flask(
    __name__,
    template_folder=str(_THIS_DIR / "templates"),
    static_folder=str(_THIS_DIR / "static"),
)

GAME = fastnmm.load_game("nine_mens_morris")
NUM_ACTIONS = GAME.num_distinct_actions()
OBS_SIZE = GAME.observation_tensor_size()
OBS_SHAPE = list(GAME.observation_tensor_shape())
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS_DIR = _THIS_DIR / "models"
MAX_MODEL_BYTES = Config.max_model_file_bytes

# Model cache is shared across sessions — models are immutable after load(),
# inference is read-only, and PyTorch handles concurrent forward() calls fine.
# A small lock protects the dict itself from a torn write during load().
_model_cache: Dict[str, ActorCritic] = {}
_model_cache_lock = threading.Lock()


@dataclass
class GameState:
    state: Any = None
    player_types: Dict[int, str] = field(default_factory=lambda: {0: "human", 1: "ai"})
    player_models: Dict[int, Any] = field(default_factory=lambda: {0: None, 1: None})
    player_minimax_depth: Dict[int, int] = field(default_factory=lambda: {0: 3, 1: 3})


@dataclass
class SessionContext:
    """Per-browser game state. Two visitors get two independent contexts so
    their games don't clobber each other. The `lock` serializes requests
    within a single session (the fastnmm state object is not re-entrant)
    while different sessions parallelize freely.
    """
    game_state: GameState = field(default_factory=GameState)
    minimax_bot: Optional[MinimaxBot] = None
    last_seen: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock)


# Cookie name + LRU bookkeeping. Capped at MAX_SESSIONS to keep memory bounded;
# anything idle longer than SESSION_TTL_S is evicted on the next request.
SESSION_COOKIE = "nmm_sid"
MAX_SESSIONS = 256
SESSION_TTL_S = 30 * 60

_sessions_lock = threading.Lock()
_sessions: "OrderedDict[str, SessionContext]" = OrderedDict()


def _resolve_session(sid: Optional[str]) -> Tuple[str, SessionContext]:
    """Look up the session for `sid` (or mint a new one), and prune stale
    entries. Returns (sid, ctx) — sid may differ from the input if it was
    missing or expired."""
    now = time.time()
    with _sessions_lock:
        # Drop expired sessions before any lookup.
        stale = [k for k, ctx in _sessions.items() if now - ctx.last_seen > SESSION_TTL_S]
        for k in stale:
            del _sessions[k]
        if sid and sid in _sessions:
            ctx = _sessions[sid]
            ctx.last_seen = now
            _sessions.move_to_end(sid)
            return sid, ctx
        new_sid = uuid.uuid4().hex
        ctx = SessionContext()
        _sessions[new_sid] = ctx
        # Bound memory: drop the oldest if we're over the cap.
        while len(_sessions) > MAX_SESSIONS:
            _sessions.popitem(last=False)
        return new_sid, ctx


@app.before_request
def _attach_session():
    if not request.path.startswith("/api/"):
        return
    sid_in = request.cookies.get(SESSION_COOKIE)
    sid, ctx = _resolve_session(sid_in)
    g.session_id = sid
    g.session_ctx = ctx
    g.session_is_new = (sid != sid_in)


@app.after_request
def _set_session_cookie(resp):
    if getattr(g, "session_is_new", False):
        resp.set_cookie(
            SESSION_COOKIE,
            g.session_id,
            max_age=60 * 60 * 24 * 7,  # 7 days
            samesite="Lax",
            httponly=True,
        )
    return resp


def list_models() -> List[Dict[str, str]]:
    """Scan webapp/models/ for .pt files. Skips empty / oversized files."""
    out: List[Dict[str, str]] = []
    if not MODELS_DIR.exists():
        return out
    for pt in sorted(MODELS_DIR.glob("*.pt"), reverse=True):
        try:
            size = pt.stat().st_size
        except OSError:
            continue
        if size == 0 or size > MAX_MODEL_BYTES:
            continue
        out.append({"name": pt.name, "path": str(pt)})
    return out


def is_allowed_model_path(path: str) -> bool:
    """Only allow models that live inside MODELS_DIR. The checkpoint loader
    uses torch.load(weights_only=False), which executes pickled code, so a
    client-supplied path outside our directory is arbitrary code execution."""
    try:
        return Path(path).resolve().is_relative_to(MODELS_DIR.resolve())
    except (OSError, ValueError):
        return False


def load_model(path: str) -> ActorCritic:
    # Fast path: serve a cached model without taking the lock — dict reads
    # are atomic under the GIL and the cached object is immutable after load.
    cached = _model_cache.get(path)
    if cached is not None:
        return cached
    with _model_cache_lock:
        cached = _model_cache.get(path)
        if cached is not None:
            return cached
        cfg = Config()
        model = ActorCritic(OBS_SIZE, NUM_ACTIONS, cfg).to(DEVICE)
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        model.eval()
        _model_cache[path] = model
        return model


def parse_board_positions(state) -> Dict[int, int]:
    return _parse_board_positions(state, OBS_SIZE, OBS_SHAPE)


def board_snapshot(ctx: SessionContext) -> Dict[str, Any]:
    if ctx.game_state.state is None:
        return {"positions": {}, "current_player": 0, "phase": "not_started"}
    state = ctx.game_state.state
    state_str = str(state)
    legal = state.legal_actions()

    phase = "placement"
    if "capture" in state_str.lower():
        phase = "capture"
    elif legal and all(a >= 24 for a in legal):
        phase = "movement"
    if state.is_terminal():
        phase = "terminal"

    return {
        "current_player": int(state.current_player()) if not state.is_terminal() else -1,
        "phase": phase,
        "is_terminal": state.is_terminal(),
        "legal_actions": [int(a) for a in legal],
        "p0_pieces": state.men_on_board(0),
        "p1_pieces": state.men_on_board(1),
        "p0_unplaced": int(state.men_to_deploy(0)),
        "p1_unplaced": int(state.men_to_deploy(1)),
        "returns": [float(r) for r in state.returns()] if state.is_terminal() else None,
        "positions": parse_board_positions(state),
    }


def ai_action(model: ActorCritic, state, player: int, temperature: float) -> Tuple[int, Dict[int, float]]:
    """Sample an action from the policy and return (action, per-legal-action probs).

    Temperature 0 = argmax (strongest, exploitable). 0.3-0.5 = balanced
    sharpness that still beats most humans but doesn't loop on the same line.
    """
    node_feats, global_feats = build_token_obs(state, player)
    node_t = torch.from_numpy(node_feats).unsqueeze(0).to(DEVICE)
    glob_t = torch.from_numpy(global_feats).unsqueeze(0).to(DEVICE)
    mask = torch.tensor(
        get_legal_mask(state, NUM_ACTIONS), dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits, _ = model(node_t, glob_t)
        masked = logits.float()
        masked[mask == 0] = -1e9

        if temperature < 0.01:
            action = int(masked.argmax(dim=-1).item())
        else:
            scaled = masked / temperature
            probs_sample = torch.softmax(scaled, dim=-1)
            action = int(torch.distributions.Categorical(probs_sample).sample().item())

        display = torch.softmax(masked, dim=-1)
        legal = state.legal_actions()
        per_action = {int(a): float(display[0, a].item()) for a in legal}
    return action, per_action


def minimax_action(ctx: SessionContext, state, depth: int) -> int:
    if ctx.minimax_bot is None or ctx.minimax_bot.max_depth != depth:
        ctx.minimax_bot = MinimaxBot(max_depth=depth)
    return ctx.minimax_bot.get_action(state)


def random_action(state) -> int:
    legal = state.legal_actions()
    return random.choice(legal) if legal else -1


def describe_action(action: int, is_capture: bool) -> str:
    info = decode_action(action, is_capture_phase=is_capture)
    if info["type"] == "place":
        return f"Placed at {info['position']}"
    if info["type"] == "move":
        return f"Moved {info['from']} → {info['to']}"
    return f"Captured {info['position']}"


# ---------------------------------------------------------------------------
# HTTP routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/favicon.ico")
def favicon():
    return send_from_directory(app.static_folder, "favicon.svg", mimetype="image/svg+xml")


@app.route("/api/models")
def api_models():
    return jsonify(list_models())


@app.route("/api/new_game", methods=["POST"])
def api_new_game():
    config = request.json or {}
    ctx: SessionContext = g.session_ctx
    with ctx.lock:
        # Reset minimax (transposition table is tied to the previous game tree)
        # and rebuild the per-session GameState from scratch.
        ctx.minimax_bot = None
        ctx.game_state = GameState()
        gs = ctx.game_state
        gs.state = GAME.new_initial_state(starting_stones=(9, 9))

        gs.player_types[0] = config.get("player0_type", "human")
        gs.player_types[1] = config.get("player1_type", "ai")
        gs.player_minimax_depth[0] = int(config.get("player0_depth", 3))
        gs.player_minimax_depth[1] = int(config.get("player1_depth", 3))

        for player in (0, 1):
            if gs.player_types[player] == "ai":
                model_path = config.get(f"player{player}_model", "")
                if model_path:
                    if not is_allowed_model_path(model_path):
                        return jsonify({"success": False, "error": "Invalid model path"}), 400
                    try:
                        gs.player_models[player] = load_model(model_path)
                    except Exception as exc:  # noqa: BLE001
                        app.logger.warning("model load failed: %s", exc)
                        return jsonify({"success": False, "error": f"Model load failed: {exc}"}), 400

        return jsonify({"success": True, "state": board_snapshot(ctx)})


@app.route("/api/move", methods=["POST"])
def api_move():
    ctx: SessionContext = g.session_ctx
    with ctx.lock:
        gs = ctx.game_state
        if gs.state is None or gs.state.is_terminal():
            return jsonify({"success": False, "error": "Game not active"}), 400
        data = request.json or {}
        action = data.get("action")
        if action is None:
            return jsonify({"success": False, "error": "No action provided"}), 400
        if action not in gs.state.legal_actions():
            return jsonify({"success": False, "error": "Illegal action"}), 400

        snap_before = board_snapshot(ctx)
        is_capture = snap_before["phase"] == "capture"
        player = gs.state.current_player()
        gs.state.apply_action(action)

        return jsonify({
            "success": True,
            "state": board_snapshot(ctx),
            "player": int(player),
            "move_description": describe_action(action, is_capture),
        })


@app.route("/api/ai_move", methods=["POST"])
def api_ai_move():
    ctx: SessionContext = g.session_ctx
    with ctx.lock:
        gs = ctx.game_state
        if gs.state is None or gs.state.is_terminal():
            return jsonify({"success": False, "error": "Game not active"}), 400

        data = request.json or {}
        player_type = data.get("player_type", "ai")
        temperature = float(data.get("temperature", 0.4))
        current_player = gs.state.current_player()
        probabilities: Optional[Dict[int, float]] = None

        # Defensive: if the seat whose turn it is was configured as human, refuse.
        # This catches a stale ai_move request that arrives after a new_game reset —
        # without this check the server would happily play for the human side using
        # whatever model the (now-orphaned) client request supplied.
        configured = gs.player_types.get(current_player, "human")
        if configured == "human":
            return jsonify({"success": False, "error": "Not the bot's turn"}), 409

        if player_type == "ai":
            model = gs.player_models.get(current_player)
            if model is None:
                model_path = data.get("model_path", "")
                if model_path:
                    if not is_allowed_model_path(model_path):
                        return jsonify({"success": False, "error": "Invalid model path"}), 400
                    try:
                        model = load_model(model_path)
                        gs.player_models[current_player] = model
                    except Exception as exc:  # noqa: BLE001
                        return jsonify({"success": False, "error": f"Model load failed: {exc}"}), 500
            if model is None:
                return jsonify({"success": False, "error": "No model available"}), 400
            action, probabilities = ai_action(model, gs.state, current_player, temperature)
        elif player_type == "minimax":
            depth = int(data.get("minimax_depth", gs.player_minimax_depth.get(current_player, 3)))
            action = minimax_action(ctx, gs.state, depth)
        else:
            action = random_action(gs.state)

        if action not in gs.state.legal_actions():
            return jsonify({"success": False, "error": "Could not produce a legal move"}), 500

        snap_before = board_snapshot(ctx)
        is_capture = snap_before["phase"] == "capture"
        gs.state.apply_action(action)

        return jsonify({
            "success": True,
            "state": board_snapshot(ctx),
            "player": int(current_player),
            "move_description": describe_action(action, is_capture),
            "probabilities": probabilities,
        })


@app.route("/api/state")
def api_state():
    ctx: SessionContext = g.session_ctx
    with ctx.lock:
        if ctx.game_state.state is None:
            return jsonify({"error": "No game in progress"}), 400
        return jsonify(board_snapshot(ctx))


def _print_banner(host: str, port: int) -> None:
    try:
        local_ip = socket.gethostbyname(socket.gethostname())
    except OSError:
        local_ip = "127.0.0.1"
    print("=" * 60)
    print(" Nine Men's Morris  —  Public Web App")
    print("=" * 60)
    print(f" Device       : {DEVICE}")
    print(f" Models dir   : {MODELS_DIR}")
    print(f" Models found : {len(list_models())}")
    print(f" Sessions     : up to {MAX_SESSIONS} concurrent, {SESSION_TTL_S // 60} min TTL")
    print(f" URL (local)  : http://localhost:{port}")
    if host == "0.0.0.0":
        print(f" URL (LAN)    : http://{local_ip}:{port}")
    print("=" * 60)


if __name__ == "__main__":
    host = os.environ.get("NMM_APP_HOST", "0.0.0.0")
    port = int(os.environ.get("NMM_APP_PORT", "7861"))
    _print_banner(host, port)
    # threaded=True so concurrent users don't queue behind each other.
    # Each session has its own lock; different sessions parallelize.
    app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)
