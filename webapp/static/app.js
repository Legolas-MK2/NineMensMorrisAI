/* ----------------------------------------------------------------------
   Nine Men's Morris — public webapp
   Single-page client. Talks to the Flask backend at /api/*.

   Board action encoding (matches the engine):
       0..23      = place (or capture during capture phase)
       24 + f*24+t = move from point f to point t
   ---------------------------------------------------------------------- */

const BOARD_SIZE = 700;
const MARGIN = 50;
const CELL = (BOARD_SIZE - 2 * MARGIN) / 6;

// 24 board points -> (row, col) on a 7x7 grid (mirrors src/board_utils.py).
const POINT_TO_RC = {
     0:[0,0],  1:[0,3],  2:[0,6],
     3:[1,1],  4:[1,3],  5:[1,5],
     6:[2,2],  7:[2,3],  8:[2,4],
     9:[3,0], 10:[3,1], 11:[3,2],
    12:[3,4], 13:[3,5], 14:[3,6],
    15:[4,2], 16:[4,3], 17:[4,4],
    18:[5,1], 19:[5,3], 20:[5,5],
    21:[6,0], 22:[6,3], 23:[6,6],
};

// Connecting segments (drawn as one polyline each).
const BOARD_LINES = [
    [0,1,2], [3,4,5], [6,7,8],
    [9,10,11], [12,13,14],
    [15,16,17], [18,19,20], [21,22,23],
    [0,9,21], [3,10,18], [6,11,15],
    [1,4,7], [16,19,22],
    [8,12,17], [5,13,20], [2,14,23],
];

// ---------- State -----------------------------------------------------
const state = {
    boardState: null,            // last `state` payload from server
    selectedFrom: null,          // for the "select piece -> select dest" flow
    busy: false,                 // suppress double-clicks while a request is in flight
    gameStarted: false,
    // Bumped on every New Game. In-flight fetch handlers compare against this
    // before touching state.boardState — stale responses from a previous game
    // are dropped instead of clobbering the freshly initialised board.
    gameGen: 0,
    config: {                    // mirrors the form (refreshed on New Game)
        player0_type: 'ai',
        player1_type: 'human',
        player0_model: '',
        player1_model: '',
        player0_depth: 3,
        player1_depth: 3,
    },
    temperature: 0.4,
};

// ---------- DOM helpers ----------------------------------------------
const $ = (id) => document.getElementById(id);

function rcToXY(row, col) {
    return { x: MARGIN + col * CELL, y: MARGIN + row * CELL };
}

function pointXY(pos) {
    const [r, c] = POINT_TO_RC[pos];
    return rcToXY(r, c);
}

// ---------- Board drawing --------------------------------------------
function drawBoardLines() {
    const lines = $('board-lines');
    lines.innerHTML = '';
    for (const seg of BOARD_LINES) {
        for (let i = 0; i < seg.length - 1; i++) {
            const a = pointXY(seg[i]);
            const b = pointXY(seg[i + 1]);
            const ln = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            ln.setAttribute('x1', a.x);
            ln.setAttribute('y1', a.y);
            ln.setAttribute('x2', b.x);
            ln.setAttribute('y2', b.y);
            lines.appendChild(ln);
        }
    }
}

function drawBoardPoints() {
    const group = $('board-points');
    group.innerHTML = '';
    for (let pos = 0; pos < 24; pos++) {
        const { x, y } = pointXY(pos);
        const c = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        c.setAttribute('cx', x);
        c.setAttribute('cy', y);
        c.setAttribute('r', 16);
        c.dataset.pos = pos;
        c.addEventListener('click', () => onPointClick(pos));
        group.appendChild(c);
    }
}

function renderPieces(positions) {
    const group = $('board-pieces');
    group.innerHTML = '';
    for (const [posStr, player] of Object.entries(positions || {})) {
        const pos = parseInt(posStr, 10);
        const { x, y } = pointXY(pos);
        const c = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        c.setAttribute('cx', x);
        c.setAttribute('cy', y);
        c.setAttribute('r', 22);
        c.setAttribute('class', `piece-${player}`);
        group.appendChild(c);
    }
}

function renderHighlights(legalActions, snap) {
    const points = $('board-points').children;
    for (const el of points) {
        el.classList.remove('legal', 'selected');
    }
    const highlights = $('board-highlights');
    highlights.innerHTML = '';
    updateCaptureBanner(snap);

    if (!legalActions || !snap || snap.is_terminal) return;
    // Only highlight when it's the human's turn.
    const currentPlayer = snap.current_player;
    const type = state.config[`player${currentPlayer}_type`];
    if (type !== 'human') return;

    const legalSet = new Set(legalActions);

    if (snap.phase === 'capture') {
        // Capture phase: red ring on every capturable enemy piece (drawn on
        // the highlights group so it sits ABOVE the piece, not behind it).
        for (const a of legalActions) {
            if (a < 24) addPieceRing(a, 'capture');
        }
    } else if (snap.phase === 'placement') {
        for (const a of legalActions) {
            if (a < 24) markLegal(a);
        }
    } else if (snap.phase === 'movement') {
        if (state.selectedFrom === null) {
            // Show which of OUR pieces can be moved this turn — green ring
            // around each one (on top, so it's visible over the piece).
            const fromSet = new Set();
            for (const a of legalActions) {
                if (a >= 24) fromSet.add(Math.floor((a - 24) / 24));
            }
            for (const p of fromSet) addPieceRing(p, 'movable');
        } else {
            // Piece selected: yellow ring around it, blue dots on legal dests.
            addPieceRing(state.selectedFrom, 'selected');
            for (const a of legalActions) {
                if (a >= 24) {
                    const f = Math.floor((a - 24) / 24);
                    const t = (a - 24) % 24;
                    if (f === state.selectedFrom) markLegal(t);
                }
            }
        }
    }

    // Soft pulse on the most recent AI move target (if any).
    if (state.lastAiHighlight !== undefined && state.lastAiHighlight !== null) {
        const { x, y } = pointXY(state.lastAiHighlight);
        const c = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        c.setAttribute('cx', x);
        c.setAttribute('cy', y);
        c.setAttribute('r', 26);
        c.setAttribute('class', 'last-move');
        highlights.appendChild(c);
    }
}

function markLegal(pos) {
    const el = document.querySelector(`#board-points circle[data-pos="${pos}"]`);
    if (el) el.classList.add('legal');
}
function markSelected(pos) {
    const el = document.querySelector(`#board-points circle[data-pos="${pos}"]`);
    if (el) el.classList.add('selected');
}

/**
 * Draw a ring on the highlights group at `pos`. Kinds:
 *   'movable'  – green pulse: this stone is movable (movement phase).
 *   'selected' – yellow ring: the stone the human picked up.
 *   'capture'  – red pulse:   this enemy stone can be captured.
 * The ring sits above pieces (since highlights are rendered last) but is
 * pointer-events: none so taps still go through to the underlying point.
 */
function addPieceRing(pos, kind) {
    const { x, y } = pointXY(pos);
    const c = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    c.setAttribute('cx', x);
    c.setAttribute('cy', y);
    c.setAttribute('r', 26);
    c.setAttribute('class', `ring ring-${kind}`);
    $('board-highlights').appendChild(c);
}

function updateCaptureBanner(snap) {
    const banner = $('capture-banner');
    const text = $('capture-banner-text');
    if (!banner) return;
    if (!snap || snap.is_terminal || snap.phase !== 'capture') {
        banner.classList.add('hidden');
        return;
    }
    const cur = snap.current_player;
    const type = state.config[`player${cur}_type`];
    const colour = cur === 0 ? 'White' : 'Black';
    if (type === 'human') {
        text.textContent = `${colour} formed a mill — tap an opponent piece to capture!`;
    } else {
        text.textContent = `${colour} formed a mill — about to capture an opponent piece…`;
    }
    banner.classList.remove('hidden');
}

// ---------- Status display -------------------------------------------
function updateStatus(snap) {
    if (!snap) return;
    state.boardState = snap;

    $('p0-pieces').textContent = snap.p0_pieces ?? 0;
    $('p1-pieces').textContent = snap.p1_pieces ?? 0;
    $('p0-reserve').textContent = `${snap.p0_unplaced ?? 0} left`;
    $('p1-reserve').textContent = `${snap.p1_unplaced ?? 0} left`;

    $('phase-display').textContent = ({
        placement: 'Placement',
        movement: 'Movement',
        capture: 'Capture',
        terminal: 'Game over',
        not_started: 'Not started',
    }[snap.phase] || snap.phase);

    const cur = snap.current_player;
    $('player-card-0').classList.toggle('active', cur === 0 && !snap.is_terminal);
    $('player-card-1').classList.toggle('active', cur === 1 && !snap.is_terminal);

    const turn = $('turn-indicator');
    if (snap.is_terminal) {
        const returns = snap.returns || [0, 0];
        let msg;
        if (returns[0] > returns[1]) msg = 'White wins!';
        else if (returns[1] > returns[0]) msg = 'Black wins!';
        else msg = 'Draw.';
        turn.textContent = msg;
        $('winner-text').textContent = msg;
        $('game-over').classList.remove('hidden');
    } else {
        $('game-over').classList.add('hidden');
        const type = state.config[`player${cur}_type`];
        const colour = cur === 0 ? 'White' : 'Black';
        const verb = ({
            human: 'your move',
            ai: 'AI thinking…',
            minimax: 'Minimax thinking…',
            random: 'Random move…',
        }[type] || '');
        if (snap.phase === 'capture') {
            turn.textContent = `${colour} — capture an opponent piece`;
        } else {
            turn.textContent = `${colour}: ${verb}`;
        }
    }
}

function appendLog(player, description) {
    const log = $('move-log');
    const entry = document.createElement('div');
    entry.className = `log-entry player-${player}`;
    const who = player === 0 ? 'White' : 'Black';
    entry.innerHTML = `<span class="who">${who}:</span> ${description}`;
    log.appendChild(entry);
    log.scrollTop = log.scrollHeight;
}

function showAiThoughts(probs) {
    const box = $('ai-thoughts');
    if (!probs) {
        box.innerHTML = '<p class="hint">The AI\'s top-ranked moves will appear here.</p>';
        return;
    }
    const sorted = Object.entries(probs)
        .map(([a, p]) => [parseInt(a, 10), p])
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5);
    box.innerHTML = '';
    for (const [a, p] of sorted) {
        const desc = describeActionShort(a);
        const row = document.createElement('div');
        row.innerHTML = `
            <div class="move-prob"><span>${desc}</span><span>${(p * 100).toFixed(1)}%</span></div>
            <div class="bar"><span style="width:${(p * 100).toFixed(1)}%"></span></div>
        `;
        box.appendChild(row);
    }
}

function describeActionShort(action) {
    if (action < 24) return `point ${action}`;
    const f = Math.floor((action - 24) / 24);
    const t = (action - 24) % 24;
    return `${f} → ${t}`;
}

// ---------- Input flow -----------------------------------------------
function onPointClick(pos) {
    if (state.busy) return;
    const snap = state.boardState;
    if (!snap || snap.is_terminal) return;

    const cur = snap.current_player;
    const type = state.config[`player${cur}_type`];
    if (type !== 'human') return;

    const legalSet = new Set(snap.legal_actions);

    if (snap.phase === 'placement' || snap.phase === 'capture') {
        if (!legalSet.has(pos)) return;
        applyHumanMove(pos);
        return;
    }
    // Movement phase: two-step (select piece -> select dest)
    if (snap.phase === 'movement') {
        const piecePlayer = snap.positions[pos];
        if (state.selectedFrom === null) {
            // Need to select one of our pieces that has a legal move.
            if (piecePlayer !== cur) return;
            const hasMove = snap.legal_actions.some(a => a >= 24 && Math.floor((a - 24) / 24) === pos);
            if (!hasMove) return;
            state.selectedFrom = pos;
            renderHighlights(snap.legal_actions, snap);
            return;
        }
        // Second click: either confirm the destination or re-select.
        if (piecePlayer === cur) {
            const hasMove = snap.legal_actions.some(a => a >= 24 && Math.floor((a - 24) / 24) === pos);
            if (hasMove) {
                state.selectedFrom = pos;
                renderHighlights(snap.legal_actions, snap);
                return;
            }
        }
        const action = 24 + state.selectedFrom * 24 + pos;
        if (!legalSet.has(action)) {
            // Illegal destination — drop the selection.
            state.selectedFrom = null;
            renderHighlights(snap.legal_actions, snap);
            return;
        }
        state.selectedFrom = null;
        applyHumanMove(action);
    }
}

async function applyHumanMove(action) {
    const gen = state.gameGen;
    state.busy = true;
    try {
        const r = await fetch('/api/move', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action }),
        });
        const data = await r.json();
        if (gen !== state.gameGen) return;  // game restarted while in flight
        if (!data.success) {
            console.warn('Move rejected:', data.error);
            return;
        }
        appendLog(data.player, data.move_description);
        state.lastAiHighlight = null;
        updateStatus(data.state);
        renderPieces(data.state.positions);
        renderHighlights(data.state.legal_actions, data.state);
    } finally {
        if (gen === state.gameGen) state.busy = false;
    }
    if (gen !== state.gameGen) return;
    // If the next player is an AI/minimax/random, schedule their move.
    maybeAutoplay();
}

async function maybeAutoplay() {
    const snap = state.boardState;
    if (!snap || snap.is_terminal) return;
    const cur = snap.current_player;
    const type = state.config[`player${cur}_type`];
    if (type === 'human') return;
    // Small delay so the human sees what just happened.
    await sleep(250);
    await doAiMove();
}

async function doAiMove() {
    const snap = state.boardState;
    if (!snap || snap.is_terminal || state.busy) return;
    const cur = snap.current_player;
    const type = state.config[`player${cur}_type`];

    const gen = state.gameGen;
    state.busy = true;
    let failed = false;
    try {
        const payload = {
            player_type: type,
            model_path: state.config[`player${cur}_model`],
            minimax_depth: state.config[`player${cur}_depth`],
            temperature: state.temperature,
        };
        const r = await fetch('/api/ai_move', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        const data = await r.json();
        if (gen !== state.gameGen) return;  // game restarted while in flight
        if (!data.success) {
            failed = true;
            console.warn('AI move failed:', data.error);
            const turn = $('turn-indicator');
            if (turn) turn.textContent = `AI move failed: ${data.error || 'unknown error'}`;
            return;
        }
        appendLog(data.player, data.move_description);
        // Stash highlight for the move target (or for captures, the captured cell).
        state.lastAiHighlight = pickHighlightFromDescription(data.move_description);
        showAiThoughts(data.probabilities);
        updateStatus(data.state);
        renderPieces(data.state.positions);
        renderHighlights(data.state.legal_actions, data.state);
    } finally {
        if (gen === state.gameGen) state.busy = false;
    }
    if (gen !== state.gameGen || failed) return;
    // Chain if it's still a bot's turn (e.g. consecutive bot players, or post-capture).
    if (state.boardState && !state.boardState.is_terminal) {
        const nextType = state.config[`player${state.boardState.current_player}_type`];
        if (nextType !== 'human') {
            await sleep(250);
            await doAiMove();
        }
    }
}

function pickHighlightFromDescription(desc) {
    // "Placed at 3" / "Moved 3 → 4" / "Captured 7"
    const m = desc.match(/(\d+)\s*$/);
    return m ? parseInt(m[1], 10) : null;
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ---------- Setup form ------------------------------------------------
async function loadModels() {
    const r = await fetch('/api/models');
    const models = await r.json();
    for (const player of [0, 1]) {
        const sel = $(`player${player}-model`);
        sel.innerHTML = '';
        if (!models.length) {
            const opt = document.createElement('option');
            opt.value = '';
            opt.textContent = '(no models found — drop a .pt into webapp/models/)';
            sel.appendChild(opt);
            continue;
        }
        for (const m of models) {
            const opt = document.createElement('option');
            opt.value = m.path;
            opt.textContent = m.name;
            sel.appendChild(opt);
        }
    }
}

function syncPlayerTypeVisibility(player) {
    const type = $(`player${player}-type`).value;
    $(`player${player}-model-group`).classList.toggle('hidden', type !== 'ai');
    $(`player${player}-depth-group`).classList.toggle('hidden', type !== 'minimax');
}

async function newGame() {
    state.config.player0_type = $('player0-type').value;
    state.config.player1_type = $('player1-type').value;
    state.config.player0_model = $('player0-model').value;
    state.config.player1_model = $('player1-model').value;
    state.config.player0_depth = parseInt($('player0-depth').value, 10) || 3;
    state.config.player1_depth = parseInt($('player1-depth').value, 10) || 3;

    // Invalidate any in-flight requests from the previous game and wipe the
    // busy flag — otherwise a stale doAiMove finishing after this point would
    // overwrite state.boardState (and the freshly drawn highlights), and a
    // busy=true left over from a stalled request would block the new game's
    // first AI move from ever firing.
    state.gameGen++;
    state.busy = false;
    state.selectedFrom = null;
    state.lastAiHighlight = null;
    state.boardState = null;
    $('move-log').innerHTML = '';
    showAiThoughts(null);

    const gen = state.gameGen;
    const r = await fetch('/api/new_game', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(state.config),
    });
    const data = await r.json();
    // Bail if the user clicked New Game again while this one was in flight.
    if (gen !== state.gameGen) return;
    if (!data.success) {
        alert('Could not start game.');
        return;
    }
    state.gameStarted = true;
    updateStatus(data.state);
    renderPieces(data.state.positions);
    renderHighlights(data.state.legal_actions, data.state);
    maybeAutoplay();
}

// ---------- Init ------------------------------------------------------
function init() {
    drawBoardLines();
    drawBoardPoints();

    for (const player of [0, 1]) {
        $(`player${player}-type`).addEventListener('change', () => syncPlayerTypeVisibility(player));
        syncPlayerTypeVisibility(player);
    }

    $('temperature').addEventListener('input', (e) => {
        state.temperature = parseInt(e.target.value, 10) / 100;
        $('temperature-value').textContent = state.temperature.toFixed(2);
    });

    $('new-game-btn').addEventListener('click', newGame);
    $('play-again-btn').addEventListener('click', newGame);
    $('ai-move-btn').addEventListener('click', () => {
        if (!state.gameStarted) return;
        doAiMove();
    });

    // Load models, then auto-start a game with the default Human vs AI setup
    // so the user lands on a playable board (no "press New Game" friction).
    loadModels().then(() => newGame());
}

document.addEventListener('DOMContentLoaded', init);
