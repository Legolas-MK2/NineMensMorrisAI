"""
Nine Men's Morris evaluation dashboard.

Small standalone Flask app that mirrors the C++ heuristic in
`fastnmm/src/fastnmm/_core/nine_mens_morris.cpp` so you can click pieces
onto an arbitrary board, pick a phase, and see exactly which evaluation
component is doing what.

Run (from the repo root):
    python tools/eval_dashboard.py
    # then visit http://localhost:7860
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from flask import Flask, jsonify, render_template, request


# ---------------------------------------------------------------------------
# Board geometry (mirrors fastnmm + src/board_utils.py).
# ---------------------------------------------------------------------------

ADJACENCY: Tuple[Tuple[int, ...], ...] = (
    (1, 9),         (0, 2, 4),      (1, 14),
    (4, 10),        (1, 3, 5, 7),   (4, 13),
    (7, 11),        (4, 6, 8),      (7, 12),
    (0, 10, 21),    (3, 9, 11, 18), (6, 10, 15),
    (8, 13, 17),    (5, 12, 14, 20),(2, 13, 23),
    (11, 16),       (15, 17, 19),   (12, 16),
    (10, 19),       (16, 18, 20, 22),(13, 19),
    (9, 22),        (19, 21, 23),   (14, 22),
)

MILLS: Tuple[Tuple[int, int, int], ...] = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (9, 10, 11), (12, 13, 14),
    (15, 16, 17), (18, 19, 20), (21, 22, 23),
    (0, 9, 21), (3, 10, 18), (6, 11, 15),
    (1, 4, 7), (16, 19, 22),
    (8, 12, 17), (5, 13, 20), (2, 14, 23),
)

POINT_TO_COORD: Dict[int, Tuple[int, int]] = {
    0: (0, 0),  1: (0, 3),  2: (0, 6),
    3: (1, 1),  4: (1, 3),  5: (1, 5),
    6: (2, 2),  7: (2, 3),  8: (2, 4),
    9: (3, 0), 10: (3, 1), 11: (3, 2),
    12: (3, 4), 13: (3, 5), 14: (3, 6),
    15: (4, 2), 16: (4, 3), 17: (4, 4),
    18: (5, 1), 19: (5, 3), 20: (5, 5),
    21: (6, 0), 22: (6, 3), 23: (6, 6),
}


# ---------------------------------------------------------------------------
# Evaluation (Python port of EvaluateBreakdown in nine_mens_morris.cpp).
# ---------------------------------------------------------------------------


def _load_engine_weights() -> Dict[str, int]:
    """Read the evaluation weights from the installed fastnmm engine so the
    dashboard cannot drift when the C++ defaults change. Falls back to a
    static copy of the defaults if fastnmm is not importable (keeps the
    dashboard usable standalone).
    """
    try:
        from fastnmm import EvalWeights
        w = EvalWeights()
        return {
            "material_mid": w.w_material_mid, "material_end": w.w_material_end,
            "mill_mid": w.w_mill_mid,         "mill_end": w.w_mill_end,
            "open_mid": w.w_open_mid,         "open_end": w.w_open_end,
            "running": w.w_running,           "double": w.w_double,
            "mill_block": w.w_mill_block,     "blocked_mid": w.w_blocked_mid,
            "mobility_mid": w.w_mobility_mid,
        }
    except Exception:
        # Static fallback — mirrors EvalWeights defaults in
        # fastnmm/src/fastnmm/_core/nine_mens_morris.hpp.
        return {
            "material_mid": 8,  "material_end": 14,
            "mill_mid": 18,     "mill_end": 22,
            "open_mid": 14,     "open_end": 22,
            "running": 30,      "double": 28,
            "mill_block": 10,   "blocked_mid": 10,
            "mobility_mid": 1,
        }


_ENGINE_WEIGHTS = _load_engine_weights()


def _is_flying(on_board: int, unplaced: int) -> bool:
    return on_board == 3 and unplaced == 0


def _mill_count(board: List[int], player: int) -> int:
    return sum(
        1 for m in MILLS
        if board[m[0]] == player and board[m[1]] == player and board[m[2]] == player
    )


def _open_mills(board: List[int], player: int, placing: bool, flying: bool) -> int:
    threats = 0
    for m in MILLS:
        own_in = sum(1 for p in m if board[p] == player)
        opp_in = sum(1 for p in m if board[p] == 1 - player)
        if own_in != 2 or opp_in != 0:
            continue
        empty_slot = next(p for p in m if board[p] == -1)
        if placing or flying:
            threats += 1
        else:
            mill_set = set(m)
            movers = [
                a for a in ADJACENCY[empty_slot]
                if board[a] == player and a not in mill_set
            ]
            if movers:
                threats += 1
    return threats


def _mill_blocks(board: List[int], player: int) -> int:
    """Count mills where `player` has exactly 1 stone interrupting an
    otherwise complete opponent mill (2 opp + 1 own). That stone is the
    only thing keeping the opponent from forming the mill, so it earns
    defensive credit.
    """
    opp = 1 - player
    count = 0
    for m in MILLS:
        own_in = sum(1 for p in m if board[p] == player)
        opp_in = sum(1 for p in m if board[p] == opp)
        if own_in == 1 and opp_in == 2:
            count += 1
    return count


def _running_mills(board: List[int], player: int,
                   placing: bool, flying: bool) -> int:
    """Count "running mill" / swing-mill setups for `player`.

    A running mill is a piece that sits in a completed own mill and can
    be moved into an empty adjacent slot that forms a *different* own
    mill. Swinging the piece back and forth captures an opponent stone
    every turn, so the pattern is much stronger than a one-shot mill.
    """
    if placing or flying:
        return 0
    in_completed_mill = set()
    for m in MILLS:
        if all(board[p] == player for p in m):
            in_completed_mill.update(m)
    count = 0
    for pos in in_completed_mill:
        for a in ADJACENCY[pos]:
            if board[a] != -1:
                continue
            for m2 in MILLS:
                if a not in m2 or pos in m2:
                    continue
                others = [p for p in m2 if p != a]
                if all(board[p] == player for p in others):
                    count += 1
                    break
    return count


def _blocked_pieces(board: List[int], player: int, placing: bool, flying: bool) -> int:
    if placing or flying:
        return 0
    blocked = 0
    for pos, owner in enumerate(board):
        if owner != player:
            continue
        if not any(board[a] == -1 for a in ADJACENCY[pos]):
            blocked += 1
    return blocked


def _mobility(board: List[int], player: int,
              unplaced: int, on_board: int) -> int:
    empties = [p for p, o in enumerate(board) if o == -1]
    mob = 0
    if unplaced > 0:
        mob += len(empties)
    flying = _is_flying(on_board, unplaced)
    if flying:
        mob += on_board * len(empties)
    else:
        for pos, owner in enumerate(board):
            if owner == player:
                mob += sum(1 for a in ADJACENCY[pos] if board[a] == -1)
    return mob


def evaluate_breakdown(board: List[int],
                       current_player: int,
                       unplaced: Tuple[int, int]) -> Dict:
    """Return the full evaluation breakdown for an arbitrary position.

    `board` is a length-24 list with values in {-1, 0, 1} (-1 = empty,
    0 = white, 1 = black). `current_player` is whose turn it is. `unplaced`
    is (white_unplaced, black_unplaced) — pieces still to drop in the
    placing phase.
    """
    cp, opp = current_player, 1 - current_player
    on_board = (
        sum(1 for o in board if o == 0),
        sum(1 for o in board if o == 1),
    )
    own_mat = on_board[cp] + unplaced[cp]
    opp_mat = on_board[opp] + unplaced[opp]

    cp_placing = unplaced[cp] > 0
    opp_placing = unplaced[opp] > 0
    cp_flying = _is_flying(on_board[cp], unplaced[cp])
    opp_flying = _is_flying(on_board[opp], unplaced[opp])

    own_mills = _mill_count(board, cp)
    opp_mills = _mill_count(board, opp)
    own_open = _open_mills(board, cp, cp_placing, cp_flying)
    opp_open = _open_mills(board, opp, opp_placing, opp_flying)
    own_running = _running_mills(board, cp, cp_placing, cp_flying)
    opp_running = _running_mills(board, opp, opp_placing, opp_flying)
    own_blocks = _mill_blocks(board, cp)
    opp_blocks = _mill_blocks(board, opp)
    own_blocked = _blocked_pieces(board, cp, cp_placing, cp_flying)
    opp_blocked = _blocked_pieces(board, opp, opp_placing, opp_flying)
    own_mob = _mobility(board, cp, unplaced[cp], on_board[cp])
    opp_mob = _mobility(board, opp, unplaced[opp], on_board[opp])

    endgame = cp_flying or opp_flying

    # Weights come from the engine (see _load_engine_weights) so this
    # dashboard stays in lockstep with the C++ evaluation.
    _w = _ENGINE_WEIGHTS
    w_mat   = _w["material_end"] if endgame else _w["material_mid"]
    w_mill  = _w["mill_end"]     if endgame else _w["mill_mid"]
    w_open  = _w["open_end"]     if endgame else _w["open_mid"]
    w_dbl   = _w["double"]
    w_run   = _w["running"]
    w_mblk  = _w["mill_block"]
    w_blk   = 0 if endgame else _w["blocked_mid"]   # endgame blocked weight is fixed at 0
    w_mob   = 0 if endgame else _w["mobility_mid"]  # endgame mobility weight is fixed at 0

    # A running-mill move also shows up in open_mills, so the double-mill
    # bonus kicks in via running_mills + open_mills together.
    effective_own_open = own_open + own_running
    effective_opp_open = opp_open + opp_running
    own_dbl = (effective_own_open - 1) * w_dbl if effective_own_open >= 2 else 0
    opp_dbl = (effective_opp_open - 1) * w_dbl if effective_opp_open >= 2 else 0

    material_score    = w_mat  * (own_mat - opp_mat)
    mill_score        = w_mill * (own_mills - opp_mills)
    open_mill_score   = w_open * (own_open - opp_open)
    running_mill_score = w_run * (own_running - opp_running)
    double_mill_score = own_dbl - opp_dbl
    mill_block_score  = w_mblk * (own_blocks - opp_blocks)
    blocked_score     = -w_blk * (own_blocked - opp_blocked)
    mobility_score    = w_mob  * (own_mob - opp_mob)

    total = (material_score + mill_score + open_mill_score + running_mill_score
             + double_mill_score + mill_block_score
             + blocked_score + mobility_score)

    return {
        "current_player": cp,
        "endgame": endgame,
        "phase": {
            "current": "flying" if cp_flying else ("placing" if cp_placing else "moving"),
            "opponent": "flying" if opp_flying else ("placing" if opp_placing else "moving"),
        },
        "counts": {
            "own_material": own_mat, "opp_material": opp_mat,
            "own_mills": own_mills, "opp_mills": opp_mills,
            "own_open_mills": own_open, "opp_open_mills": opp_open,
            "own_running_mills": own_running, "opp_running_mills": opp_running,
            "own_mill_blocks": own_blocks, "opp_mill_blocks": opp_blocks,
            "own_blocked": own_blocked, "opp_blocked": opp_blocked,
            "own_mobility": own_mob, "opp_mobility": opp_mob,
            "own_on_board": on_board[cp], "opp_on_board": on_board[opp],
        },
        "weights": {
            "material": w_mat, "mill": w_mill, "open_mill": w_open,
            "running_mill": w_run, "double_mill": w_dbl,
            "mill_block": w_mblk,
            "blocked": w_blk, "mobility": w_mob,
        },
        "components": [
            {"name": "Material",     "diff": own_mat - opp_mat,
             "weight": w_mat,  "score": material_score,
             "desc": "(own_pieces - opp_pieces) × weight. Total stones on board + still to place."},
            {"name": "Mills",        "diff": own_mills - opp_mills,
             "weight": w_mill, "score": mill_score,
             "desc": "Completed 3-in-a-row lines."},
            {"name": "Open mills",   "diff": own_open - opp_open,
             "weight": w_open, "score": open_mill_score,
             "desc": "Mill lines with 2 own stones + a reachable empty third slot — one move from forming."},
            {"name": "Running mill", "diff": own_running - opp_running,
             "weight": w_run, "score": running_mill_score,
             "desc": "Piece in a completed mill that can swing into a different mill — captures every move."},
            {"name": "Double-mill",  "diff": max(0, effective_own_open - 1) - max(0, effective_opp_open - 1),
             "weight": w_dbl,  "score": double_mill_score,
             "desc": "Extra bonus when several simultaneous mill threats exist (open + running)."},
            {"name": "Mill block",   "diff": own_blocks - opp_blocks,
             "weight": w_mblk, "score": mill_block_score,
             "desc": "Own stones sitting in an opponent's 2-of-3 mill line — defensively interrupting a mill threat."},
            {"name": "Blocked",      "diff": own_blocked - opp_blocked,
             "weight": w_blk,  "score": blocked_score,
             "desc": "Pieces with no legal move (moving phase only). Score is negated."},
            {"name": "Mobility",     "diff": own_mob - opp_mob,
             "weight": w_mob,  "score": mobility_score,
             "desc": "Approx legal-action count (placing slots + adjacent moves)."},
        ],
        "total": total,
        "mills_highlight": {
            "white": [list(m) for m in MILLS
                      if all(board[p] == 0 for p in m)],
            "black": [list(m) for m in MILLS
                      if all(board[p] == 1 for p in m)],
        },
    }


# ---------------------------------------------------------------------------
# Presets to seed interesting board states.
# ---------------------------------------------------------------------------

def _empty() -> List[int]:
    return [-1] * 24


def _preset(pieces: Dict[int, int]) -> List[int]:
    b = _empty()
    for p, owner in pieces.items():
        b[p] = owner
    return b


PRESETS = {
    "empty":   {"board": _empty(), "unplaced": [9, 9], "cp": 0,
                "label": "Empty board, opening"},
    "open_mill": {
        "board": _preset({0: 0, 1: 0, 9: 0, 6: 1, 18: 1, 22: 1}),
        "unplaced": [6, 6], "cp": 0,
        "label": "White threatens a mill on the top edge",
    },
    "double_threat": {
        "board": _preset({0: 0, 9: 0, 21: 0, 3: 0, 18: 0, 6: 1, 11: 1, 22: 1, 7: 1}),
        "unplaced": [5, 5], "cp": 0,
        "label": "White: two simultaneous mill threats (10, 4)",
    },
    "blocked_piece": {
        "board": _preset({0: 0, 1: 1, 9: 1, 4: 0, 10: 0, 11: 0, 6: 0, 18: 1, 21: 1, 22: 1}),
        "unplaced": [0, 0], "cp": 0,
        "label": "Moving phase: white piece at 0 is blocked",
    },
    "mill_block": {
        "board": _preset({3: 0, 18: 0, 10: 1, 13: 1}),
        "unplaced": [7, 7], "cp": 1,
        "label": "Black at 10 blocks white's (3-10-18) mill",
    },
    "running_mill": {
        "board": _preset({3: 0, 6: 0, 10: 0, 15: 0, 18: 0}),
        "unplaced": [0, 4], "cp": 0,
        "label": "Running mill: 10 ↔ 11 swings between (3-10-18) and (6-11-15)",
    },
    "flying": {
        "board": _preset({0: 0, 11: 0, 23: 0, 1: 1, 4: 1, 7: 1, 22: 1, 19: 1, 13: 1}),
        "unplaced": [0, 0], "cp": 0,
        "label": "White flying (3 pieces); black has mill on 1-4-7",
    },
    "material_advantage": {
        "board": _preset({0: 0, 3: 0, 6: 0, 9: 0, 12: 0, 15: 0, 1: 1, 4: 1, 7: 1, 10: 1, 13: 1}),
        "unplaced": [3, 3], "cp": 0,
        "label": "Material-only edge to white; no mills",
    },
}


# ---------------------------------------------------------------------------
# Flask app.
# ---------------------------------------------------------------------------

app = Flask(__name__)




@app.route("/")
def index():
    coords = [POINT_TO_COORD[i] for i in range(24)]
    # Served from tools/templates/eval_dashboard.html (Flask's default
    # template folder next to this module).
    return render_template(
        "eval_dashboard.html",
        points=coords,
        adjacency=[list(a) for a in ADJACENCY],
        mills=[list(m) for m in MILLS],
        presets=PRESETS,
    )


@app.route("/api/evaluate", methods=["POST"])
def api_evaluate():
    payload = request.get_json(force=True)
    board = list(payload.get("board", [-1] * 24))
    if len(board) != 24:
        return jsonify({"error": "board must have 24 cells"}), 400
    cp = int(payload.get("current_player", 0))
    if cp not in (0, 1):
        return jsonify({"error": "current_player must be 0 or 1"}), 400
    unplaced = payload.get("unplaced", [9, 9])
    up = (int(unplaced[0]), int(unplaced[1]))
    if not (0 <= up[0] <= 9 and 0 <= up[1] <= 9):
        return jsonify({"error": "unplaced must be in [0, 9]"}), 400
    return jsonify(evaluate_breakdown(board, cp, up))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)
