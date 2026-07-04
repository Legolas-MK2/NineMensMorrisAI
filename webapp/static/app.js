/* ----------------------------------------------------------------------
   Nine Men's Morris — public webapp (simplified UI)

   Layout: clickable player cards above the board pick what plays each
   colour (Human, AI <model>, Minimax, Random). The middle element shows
   the current phase during a game, or acts as a "Swap" button between
   games. A single Reset button under the board restarts.
   ---------------------------------------------------------------------- */

const BOARD_SIZE = 700;
const MARGIN = 50;
const CELL = (BOARD_SIZE - 2 * MARGIN) / 6;

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

const BOARD_LINES = [
    [0,1,2], [3,4,5], [6,7,8],
    [9,10,11], [12,13,14],
    [15,16,17], [18,19,20], [21,22,23],
    [0,9,21], [3,10,18], [6,11,15],
    [1,4,7], [16,19,22],
    [8,12,17], [5,13,20], [2,14,23],
];

const state = {
    boardState: null,
    selectedFrom: null,
    busy: false,
    gameStarted: false,
    gameGen: 0,
    models: [],
    config: {
        player0_type: 'ai',
        player1_type: 'human',
        player0_model: '',
        player1_model: '',
        player0_depth: 3,
        player1_depth: 3,
    },
    temperature: 0.4,
};

const $ = (id) => document.getElementById(id);

function rcToXY(row, col) { return { x: MARGIN + col * CELL, y: MARGIN + row * CELL }; }
function pointXY(pos) { const [r, c] = POINT_TO_RC[pos]; return rcToXY(r, c); }

// ---------- Board drawing --------------------------------------------
function drawBoardLines() {
    const lines = $('board-lines');
    lines.innerHTML = '';
    for (const seg of BOARD_LINES) {
        for (let i = 0; i < seg.length - 1; i++) {
            const a = pointXY(seg[i]);
            const b = pointXY(seg[i + 1]);
            const ln = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            ln.setAttribute('x1', a.x); ln.setAttribute('y1', a.y);
            ln.setAttribute('x2', b.x); ln.setAttribute('y2', b.y);
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
        c.setAttribute('cx', x); c.setAttribute('cy', y); c.setAttribute('r', 16);
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
        c.setAttribute('cx', x); c.setAttribute('cy', y); c.setAttribute('r', 22);
        c.setAttribute('class', `piece-${player}`);
        group.appendChild(c);
    }
}

function renderHighlights(legalActions, snap) {
    const points = $('board-points').children;
    for (const el of points) el.classList.remove('legal', 'selected');
    const highlights = $('board-highlights');
    highlights.innerHTML = '';
    updateCaptureBanner(snap);

    if (!legalActions || !snap || snap.is_terminal) return;
    const currentPlayer = snap.current_player;
    const type = state.config[`player${currentPlayer}_type`];
    if (type !== 'human') return;

    if (snap.phase === 'capture') {
        for (const a of legalActions) if (a < 24) addPieceRing(a, 'capture');
    } else if (snap.phase === 'placement') {
        for (const a of legalActions) if (a < 24) markLegal(a);
    } else if (snap.phase === 'movement') {
        if (state.selectedFrom === null) {
            const fromSet = new Set();
            for (const a of legalActions) if (a >= 24) fromSet.add(Math.floor((a - 24) / 24));
            for (const p of fromSet) addPieceRing(p, 'movable');
        } else {
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

    if (state.lastAiHighlight !== undefined && state.lastAiHighlight !== null) {
        const { x, y } = pointXY(state.lastAiHighlight);
        const c = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        c.setAttribute('cx', x); c.setAttribute('cy', y); c.setAttribute('r', 26);
        c.setAttribute('class', 'last-move');
        highlights.appendChild(c);
    }
}

function markLegal(pos) {
    const el = document.querySelector(`#board-points circle[data-pos="${pos}"]`);
    if (el) el.classList.add('legal');
}

function addPieceRing(pos, kind) {
    const { x, y } = pointXY(pos);
    const c = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    c.setAttribute('cx', x); c.setAttribute('cy', y); c.setAttribute('r', 26);
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
    text.textContent = type === 'human'
        ? `${colour} formed a mill — tap an opponent piece to capture!`
        : `${colour} formed a mill — about to capture an opponent piece…`;
    banner.classList.remove('hidden');
}

// ---------- Status / labels ------------------------------------------
function playerSummary(player) {
    const type = state.config[`player${player}_type`];
    if (type === 'human') return 'Human';
    if (type === 'ai') {
        const path = state.config[`player${player}_model`];
        const m = state.models.find(x => x.path === path);
        return m ? `AI: ${m.name}` : 'AI';
    }
    return type;
}

function refreshPlayerLabels() {
    $('p0-label').textContent = `White · ${playerSummary(0)}`;
    $('p1-label').textContent = `Black · ${playerSummary(1)}`;
}

function updatePhaseDisplay() {
    $('phase-display').textContent = '↔ Swap';
}

function updateStatus(snap) {
    if (!snap) return;
    state.boardState = snap;

    $('p0-pieces').textContent = snap.p0_pieces ?? 0;
    $('p1-pieces').textContent = snap.p1_pieces ?? 0;
    $('p0-reserve').textContent = `${snap.p0_unplaced ?? 0} left`;
    $('p1-reserve').textContent = `${snap.p1_unplaced ?? 0} left`;

    updatePhaseDisplay();

    const cur = snap.current_player;
    $('player-card-0').classList.toggle('active', cur === 0 && !snap.is_terminal);
    $('player-card-1').classList.toggle('active', cur === 1 && !snap.is_terminal);

    const turn = $('turn-indicator');
    if (snap.is_terminal) {
        const returns = snap.returns || [0, 0];
        if (returns[0] > returns[1]) turn.textContent = 'White wins!';
        else if (returns[1] > returns[0]) turn.textContent = 'Black wins!';
        else turn.textContent = 'Draw.';
    } else {
        const colour = cur === 0 ? 'White' : 'Black';
        const phaseLabel = ({
            placement: 'Placement',
            movement: 'Movement',
            capture: 'Capture',
        }[snap.phase] || snap.phase);
        const type = state.config[`player${cur}_type`];
        const verb = type === 'human' ? 'your move' : 'AI thinking…';
        turn.textContent = snap.phase === 'capture'
            ? `${colour} — ${phaseLabel} (capture an opponent piece)`
            : `${colour} — ${phaseLabel} · ${verb}`;
    }
}

// ---------- Input flow -----------------------------------------------
function onPointClick(pos) {
    if (state.busy) return;
    const snap = state.boardState;
    if (!snap || snap.is_terminal) return;

    const cur = snap.current_player;
    if (state.config[`player${cur}_type`] !== 'human') return;

    const legalSet = new Set(snap.legal_actions);

    if (snap.phase === 'placement' || snap.phase === 'capture') {
        if (!legalSet.has(pos)) return;
        applyHumanMove(pos);
        return;
    }
    if (snap.phase === 'movement') {
        const piecePlayer = snap.positions[pos];
        if (state.selectedFrom === null) {
            if (piecePlayer !== cur) return;
            const hasMove = snap.legal_actions.some(a => a >= 24 && Math.floor((a - 24) / 24) === pos);
            if (!hasMove) return;
            state.selectedFrom = pos;
            renderHighlights(snap.legal_actions, snap);
            return;
        }
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
        if (gen !== state.gameGen) return;
        if (!data.success) { console.warn('Move rejected:', data.error); return; }
        state.lastAiHighlight = null;
        updateStatus(data.state);
        renderPieces(data.state.positions);
        renderHighlights(data.state.legal_actions, data.state);
    } finally {
        if (gen === state.gameGen) state.busy = false;
    }
    if (gen !== state.gameGen) return;
    maybeAutoplay();
}

async function maybeAutoplay() {
    const snap = state.boardState;
    if (!snap || snap.is_terminal) return;
    const cur = snap.current_player;
    if (state.config[`player${cur}_type`] === 'human') return;
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
        if (gen !== state.gameGen) return;
        if (!data.success) {
            failed = true;
            const turn = $('turn-indicator');
            if (turn) turn.textContent = `AI move failed: ${data.error || 'unknown error'}`;
            return;
        }
        state.lastAiHighlight = pickHighlightFromDescription(data.move_description);
        updateStatus(data.state);
        renderPieces(data.state.positions);
        renderHighlights(data.state.legal_actions, data.state);
    } finally {
        if (gen === state.gameGen) state.busy = false;
    }
    if (gen !== state.gameGen || failed) return;
    if (state.boardState && !state.boardState.is_terminal) {
        const nextType = state.config[`player${state.boardState.current_player}_type`];
        if (nextType !== 'human') {
            await sleep(250);
            await doAiMove();
        }
    }
}

function pickHighlightFromDescription(desc) {
    const m = desc.match(/(\d+)\s*$/);
    return m ? parseInt(m[1], 10) : null;
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ---------- Player controls ------------------------------------------
function onPlayerCardClick(player) {
    // Only the AI side cycles to the next model. The Human card is inert.
    if (state.config[`player${player}_type`] !== 'ai') return;
    if (!state.models.length) return;
    const cur = state.models.findIndex(m => m.path === state.config[`player${player}_model`]);
    const next = state.models[(cur + 1) % state.models.length];
    state.config[`player${player}_model`] = next.path;
    refreshPlayerLabels();
    newGame();
}

function swapPlayers() {
    const c = state.config;
    [c.player0_type, c.player1_type] = [c.player1_type, c.player0_type];
    [c.player0_model, c.player1_model] = [c.player1_model, c.player0_model];
    refreshPlayerLabels();
    newGame();
}

// ---------- New game --------------------------------------------------
async function loadModels() {
    const r = await fetch('/api/models');
    state.models = await r.json();
    if (state.models.length) {
        if (!state.config.player0_model) state.config.player0_model = state.models[0].path;
        if (!state.config.player1_model) state.config.player1_model = state.models[0].path;
    }
    refreshPlayerLabels();
}

async function newGame() {
    state.gameGen++;
    state.busy = false;
    state.selectedFrom = null;
    state.lastAiHighlight = null;
    state.boardState = null;

    const gen = state.gameGen;
    const r = await fetch('/api/new_game', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(state.config),
    });
    const data = await r.json();
    if (gen !== state.gameGen) return;
    if (!data.success) { alert('Could not start game.'); return; }
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
        $(`player-card-${player}`).addEventListener('click', () => onPlayerCardClick(player));
    }

    $('phase-display').addEventListener('click', swapPlayers);
    $('reset-btn').addEventListener('click', newGame);

    loadModels().then(() => newGame());
}

document.addEventListener('DOMContentLoaded', init);
