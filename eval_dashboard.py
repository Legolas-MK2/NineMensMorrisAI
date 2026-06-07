"""
Nine Men's Morris evaluation dashboard.

Small standalone Flask app that mirrors the C++ heuristic in
`fastnmm/src/fastnmm/_core/nine_mens_morris.cpp` so you can click pieces
onto an arbitrary board, pick a phase, and see exactly which evaluation
component is doing what.

Run:
    python eval_dashboard.py
    # then visit http://localhost:5050
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from flask import Flask, jsonify, render_template_string, request


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

    w_mat   = 14 if endgame else 8
    w_mill  = 22 if endgame else 18
    w_open  = 22 if endgame else 14
    w_dbl   = 28
    w_run   = 30
    w_mblk  = 10
    w_blk   = 0  if endgame else 10
    w_mob   = 0  if endgame else 1

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


HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>NMM Evaluation Dashboard</title>
<style>
  body { font-family: -apple-system, Segoe UI, Roboto, sans-serif;
         background:#1e1e1e; color:#ddd; margin:0; padding:24px; }
  h1 { margin:0 0 8px 0; font-size:20px; }
  .sub { color:#888; margin-bottom:24px; font-size:13px; }
  .layout { display:grid; grid-template-columns: 540px 1fr; gap:32px; }
  .board-wrap { background:#252526; border-radius:8px; padding:20px; }
  svg { background:#2d2d30; border-radius:6px; display:block; }
  .controls { margin-top:16px; display:flex; flex-wrap:wrap; gap:8px; align-items:center; }
  .controls label { font-size:13px; color:#aaa; }
  select, button, input { background:#3c3c3c; color:#ddd; border:1px solid #555;
                          padding:6px 10px; border-radius:4px; font-size:13px; }
  button { cursor:pointer; }
  button:hover { background:#4c4c4c; }
  button.active { background:#0e639c; border-color:#1177bb; }
  .panel { background:#252526; border-radius:8px; padding:20px; }
  .total { font-size:32px; font-weight:bold; margin-bottom:8px; }
  .total.pos { color:#4ec9b0; }
  .total.neg { color:#f48771; }
  .total.zero { color:#888; }
  .phase { color:#aaa; font-size:13px; margin-bottom:16px; }
  table { width:100%; border-collapse: collapse; font-size:13px; }
  th, td { padding:8px 6px; text-align:right; border-bottom:1px solid #333; }
  th:first-child, td:first-child { text-align:left; }
  th { color:#888; font-weight:normal; font-size:12px; text-transform:uppercase; }
  td.score.pos { color:#4ec9b0; }
  td.score.neg { color:#f48771; }
  td.score.zero { color:#555; }
  .desc-row td { color:#888; font-size:11px; padding-top:0; padding-bottom:12px; }
  .legend { font-size:12px; color:#888; margin-top:12px; line-height:1.6; }
  .preset-row { display:flex; flex-wrap:wrap; gap:6px; margin-top:8px; }
  .preset-row button { font-size:12px; padding:4px 8px; }
  .hint { font-size:12px; color:#888; margin-top:6px; }
  .legend-dot { display:inline-block; width:10px; height:10px; border-radius:50%;
                margin-right:4px; vertical-align:middle; }
</style>
</head>
<body>
<h1>Nine Men's Morris — Evaluation Dashboard</h1>
<div class="sub">
  Click a square to cycle through empty → white → black. Adjust phase and
  current player to inspect how the heuristic weights each component.
</div>

<div class="layout">
  <div class="board-wrap">
    <svg id="board" viewBox="0 0 540 540" width="540" height="540"></svg>

    <div class="controls">
      <label>Brush:</label>
      <button id="brush-white" class="brush active" data-v="0">White</button>
      <button id="brush-black" class="brush" data-v="1">Black</button>
      <button id="brush-erase" class="brush" data-v="-1">Erase</button>

      <label style="margin-left:12px;">To move:</label>
      <select id="current-player">
        <option value="0">White</option>
        <option value="1">Black</option>
      </select>
    </div>

    <div class="controls">
      <label>White unplaced:</label>
      <input id="unplaced-white" type="number" min="0" max="9" value="9" style="width:60px;">
      <label>Black unplaced:</label>
      <input id="unplaced-black" type="number" min="0" max="9" value="9" style="width:60px;">
      <button id="reset">Reset</button>
    </div>

    <div class="preset-row" id="presets"></div>
    <div class="hint">
      Set unplaced to 0 and reduce a side to 3 stones to enter flying phase.
    </div>
  </div>

  <div class="panel">
    <div>Total evaluation (from current player's view)</div>
    <div id="total" class="total zero">0</div>
    <div id="phase" class="phase"></div>

    <table>
      <thead>
        <tr><th>Component</th><th>own−opp</th><th>weight</th><th>score</th></tr>
      </thead>
      <tbody id="components"></tbody>
    </table>

    <div class="legend">
      <div><span class="legend-dot" style="background:#dddddd"></span>White piece</div>
      <div><span class="legend-dot" style="background:#222222;border:1px solid #777"></span>Black piece</div>
      <div><span class="legend-dot" style="background:#4ec9b0"></span>Open mill threat (own)</div>
      <div><span class="legend-dot" style="background:#f48771"></span>Open mill threat (opponent)</div>
    </div>
    <div class="legend" style="margin-top:12px;">
      Score is reported from the side-to-move's perspective: positive = good for them.
      In the flying endgame, mobility/blocked weights drop to 0 and mill threats are amplified.
    </div>
  </div>
</div>

<script>
const POINTS = {{ points|tojson }};
const ADJ    = {{ adjacency|tojson }};
const PRESETS = {{ presets|tojson }};

const SIZE = 540;
const MARGIN = 40;
const CELL = (SIZE - 2 * MARGIN) / 6;

let board = Array(24).fill(-1);
let currentPlayer = 0;
let unplaced = [9, 9];
let brush = 0;
let highlight = { ownOpen: [], oppOpen: [] };

function pxOf(idx) {
  const [r, c] = POINTS[idx];
  return [MARGIN + c * CELL, MARGIN + r * CELL];
}

function buildBoard() {
  const svg = document.getElementById("board");
  svg.innerHTML = "";

  // Edges (each adjacency pair drawn once).
  const drawn = new Set();
  for (let i = 0; i < 24; i++) {
    for (const j of ADJ[i]) {
      const key = i < j ? i + "_" + j : j + "_" + i;
      if (drawn.has(key)) continue;
      drawn.add(key);
      const [x1, y1] = pxOf(i);
      const [x2, y2] = pxOf(j);
      const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
      line.setAttribute("x1", x1); line.setAttribute("y1", y1);
      line.setAttribute("x2", x2); line.setAttribute("y2", y2);
      line.setAttribute("stroke", "#666");
      line.setAttribute("stroke-width", "2");
      svg.appendChild(line);
    }
  }

  // Highlight rings for mill threats (drawn beneath stones).
  function ring(idx, color) {
    const [x, y] = pxOf(idx);
    const r = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    r.setAttribute("cx", x); r.setAttribute("cy", y);
    r.setAttribute("r", 24);
    r.setAttribute("fill", "none");
    r.setAttribute("stroke", color);
    r.setAttribute("stroke-width", "3");
    r.setAttribute("stroke-dasharray", "4 3");
    svg.appendChild(r);
  }
  highlight.ownOpen.forEach(i => ring(i, "#4ec9b0"));
  highlight.oppOpen.forEach(i => ring(i, "#f48771"));

  // Stones / click targets.
  for (let i = 0; i < 24; i++) {
    const [x, y] = pxOf(i);
    const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
    g.style.cursor = "pointer";
    g.addEventListener("click", () => onClick(i));

    const c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    c.setAttribute("cx", x); c.setAttribute("cy", y);
    c.setAttribute("r", 18);
    if (board[i] === 0) {
      c.setAttribute("fill", "#dddddd");
      c.setAttribute("stroke", "#000"); c.setAttribute("stroke-width", "1");
    } else if (board[i] === 1) {
      c.setAttribute("fill", "#222222");
      c.setAttribute("stroke", "#888"); c.setAttribute("stroke-width", "1");
    } else {
      c.setAttribute("fill", "#3a3a3a");
      c.setAttribute("stroke", "#555");
      c.setAttribute("stroke-width", "1");
    }
    g.appendChild(c);

    const t = document.createElementNS("http://www.w3.org/2000/svg", "text");
    t.setAttribute("x", x); t.setAttribute("y", y + 4);
    t.setAttribute("text-anchor", "middle");
    t.setAttribute("font-size", "10");
    t.setAttribute("fill", board[i] === -1 ? "#888" : (board[i] === 0 ? "#222" : "#ddd"));
    t.style.pointerEvents = "none";
    t.textContent = i;
    g.appendChild(t);

    svg.appendChild(g);
  }
}

function onClick(idx) {
  board[idx] = brush;
  refresh();
}

async function refresh() {
  const resp = await fetch("/api/evaluate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      board: board,
      current_player: currentPlayer,
      unplaced: unplaced,
    }),
  });
  const data = await resp.json();
  render(data);
}

function render(data) {
  const totalEl = document.getElementById("total");
  totalEl.textContent = (data.total >= 0 ? "+" : "") + data.total;
  totalEl.className = "total " +
      (data.total > 0 ? "pos" : data.total < 0 ? "neg" : "zero");

  const cpName = data.current_player === 0 ? "White" : "Black";
  const oppName = data.current_player === 0 ? "Black" : "White";
  document.getElementById("phase").textContent =
      `To move: ${cpName} (${data.phase.current}) — opponent ${oppName} (${data.phase.opponent})` +
      (data.endgame ? " — endgame weights active" : "");

  const tbody = document.getElementById("components");
  tbody.innerHTML = "";
  for (const c of data.components) {
    const tr = document.createElement("tr");
    const cls = c.score > 0 ? "pos" : c.score < 0 ? "neg" : "zero";
    tr.innerHTML = `
      <td>${c.name}</td>
      <td>${c.diff >= 0 ? '+' + c.diff : c.diff}</td>
      <td>${c.weight}</td>
      <td class="score ${cls}">${c.score >= 0 ? '+' + c.score : c.score}</td>
    `;
    tbody.appendChild(tr);
    const dr = document.createElement("tr");
    dr.className = "desc-row";
    dr.innerHTML = `<td colspan="4">${c.desc}</td>`;
    tbody.appendChild(dr);
  }

  // Compute highlight positions: open mill empty slots.
  const MILLS = {{ mills|tojson }};
  highlight = { ownOpen: [], oppOpen: [] };
  for (const m of MILLS) {
    const cp = data.current_player, opp = 1 - cp;
    const cntCp  = m.filter(p => board[p] === cp).length;
    const cntOpp = m.filter(p => board[p] === opp).length;
    const empty  = m.find(p => board[p] === -1);
    if (empty === undefined) continue;
    if (cntCp === 2 && cntOpp === 0) highlight.ownOpen.push(empty);
    else if (cntOpp === 2 && cntCp === 0) highlight.oppOpen.push(empty);
  }
  buildBoard();
}

// Wire up controls.
document.querySelectorAll(".brush").forEach(b => {
  b.addEventListener("click", () => {
    document.querySelectorAll(".brush").forEach(x => x.classList.remove("active"));
    b.classList.add("active");
    brush = parseInt(b.dataset.v, 10);
  });
});

document.getElementById("current-player").addEventListener("change", e => {
  currentPlayer = parseInt(e.target.value, 10);
  refresh();
});
document.getElementById("unplaced-white").addEventListener("input", e => {
  unplaced[0] = Math.max(0, Math.min(9, parseInt(e.target.value || "0", 10)));
  refresh();
});
document.getElementById("unplaced-black").addEventListener("input", e => {
  unplaced[1] = Math.max(0, Math.min(9, parseInt(e.target.value || "0", 10)));
  refresh();
});
document.getElementById("reset").addEventListener("click", () => {
  loadPreset("empty");
});

// Preset buttons.
const presetRow = document.getElementById("presets");
for (const [k, v] of Object.entries(PRESETS)) {
  const b = document.createElement("button");
  b.textContent = v.label;
  b.addEventListener("click", () => loadPreset(k));
  presetRow.appendChild(b);
}

function loadPreset(name) {
  const p = PRESETS[name];
  board = p.board.slice();
  unplaced = p.unplaced.slice();
  currentPlayer = p.cp;
  document.getElementById("current-player").value = String(currentPlayer);
  document.getElementById("unplaced-white").value = unplaced[0];
  document.getElementById("unplaced-black").value = unplaced[1];
  refresh();
}

// Initial render.
buildBoard();
refresh();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    coords = [POINT_TO_COORD[i] for i in range(24)]
    return render_template_string(
        HTML,
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
