"""
Minimax wrapper around the fastnmm C++ engine.

Public surface for the rest of the codebase:
    MinimaxBot(max_depth=…, random_move_prob=…) — exposes .step()/.get_action()
    evaluate_vs_minimax(model, device, num_actions, …)
    format_minimax_results(results)
"""

from __future__ import annotations

import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.amp import autocast

import fastnmm
from fastnmm import MinimaxBot as _FastMinimaxBot
from fastnmm import _core as _fastnmm_core


# Matches `kWinScore` in fastnmm/_core/nine_mens_morris.cpp:647. Used by the
# Python-side root splitter to reproduce the C++ engine's terminal scoring
# convention (`±kWinScore` minus a depth-tie-break) when we evaluate a root
# child that ends the game.
_K_WIN_SCORE = 100_000


class _FauxSearchResult:
    """Drop-in replacement for `_core.SearchResult` returned by the parallel
    root-splitter. Mirrors the same attributes so callers using
    `bot.last_search.nodes_visited` keep working."""
    __slots__ = ("action", "score", "nodes_visited")

    def __init__(self, action: int, score: int, nodes_visited: int) -> None:
        self.action = int(action)
        self.score = int(score)
        self.nodes_visited = int(nodes_visited)


class _ParallelRootMinimaxBot:
    """Root-splitting parallel minimax bot.

    Holds `num_threads` persistent `_core.MinimaxEngine` instances. On
    `step()`, the root legal actions are partitioned round-robin across
    threads; each thread plays its share by cloning the state, applying its
    root action, and running the existing single-threaded C++ alpha-beta on
    the child at `depth - 1`. The best per-thread (action, score) is
    reduced to a global best.

    Trade-offs vs a single serial search:
    - **No shared TT** across threads. Each engine builds its own TT as the
      game progresses; cross-thread transpositions don't help each other.
    - **No shared root alpha.** Each thread's search uses a full window, so
      we lose the alpha-narrowing that serial search gets from earlier root
      children. The C++ code already disables alpha-beta cuts at the root
      itself, but interior subtree searches would benefit from a tighter
      root alpha bound — we don't get that.
    - Expected speedup is sub-linear: ~1.5-2.5x for 4 threads.

    Strict-parity scoring is reproduced bit-exactly for terminal children
    (winner POV ± kWinScore, depth tie-break). Non-terminal children are
    scored by `engine.search(child, depth-1)` whose result is from the
    child's current-player POV; we negate when the side to move flips.
    """

    def __init__(self, player_id: int, depth: int, num_threads: int,
                 tt_bytes_per_thread: int,
                 strict_parity: bool = True) -> None:
        self.player_id = int(player_id)
        self.depth = int(depth)
        self.num_threads = max(1, int(num_threads))
        self.strict_parity = bool(strict_parity)
        self.engines = [
            _fastnmm_core.MinimaxEngine(int(tt_bytes_per_thread),
                                        bool(strict_parity))
            for _ in range(self.num_threads)
        ]
        self.executor = ThreadPoolExecutor(max_workers=self.num_threads)
        self.last_search: Optional[_FauxSearchResult] = None

    def _score_child(self, engine, state, action: int, cur_player: int,
                     depth: int) -> Tuple[int, int]:
        child = state.clone()
        child.apply_action(action)
        if child.is_terminal():
            r = child.returns()
            if r[cur_player] > r[1 - cur_player]:
                s = _K_WIN_SCORE - (100 - depth)
            elif r[cur_player] < r[1 - cur_player]:
                s = -_K_WIN_SCORE + (100 - depth)
            else:
                s = 0
            return s, 0
        result = engine.search(child, depth - 1)
        score = int(result.score)
        if child.current_player() != cur_player:
            score = -score
        return score, int(result.nodes_visited)

    def step(self, state) -> int:
        if state.is_terminal():
            return -1
        legal = state.legal_actions()
        if not legal:
            return -1
        if len(legal) == 1:
            self.last_search = _FauxSearchResult(legal[0], 0, 0)
            return legal[0]

        # Single-engine fast path: skips the executor round-trip for free.
        if self.num_threads == 1 or len(legal) == 1:
            res = self.engines[0].search(state, self.depth)
            self.last_search = _FauxSearchResult(
                int(res.action), int(res.score), int(res.nodes_visited))
            return int(res.action)

        cur_player = int(state.current_player())
        depth = self.depth
        engines = self.engines
        n_threads = min(self.num_threads, len(legal))

        def _worker(thread_idx: int):
            engine = engines[thread_idx]
            local_best_a = -1
            local_best_s = -_K_WIN_SCORE - 1
            local_nodes = 0
            for a in legal[thread_idx::n_threads]:
                s, n = self._score_child(engine, state, a, cur_player, depth)
                local_nodes += n
                if s > local_best_s:
                    local_best_s = s
                    local_best_a = a
            return local_best_a, local_best_s, local_nodes

        futures = [self.executor.submit(_worker, i) for i in range(n_threads)]

        best_action = legal[0]
        best_score = -_K_WIN_SCORE - 1
        total_nodes = 0
        for f in futures:
            a, s, n = f.result()
            total_nodes += n
            if s > best_score:
                best_score = s
                best_action = a

        self.last_search = _FauxSearchResult(best_action, best_score, total_nodes)
        return int(best_action)

    def close(self) -> None:
        self.executor.shutdown(wait=False)

    def __del__(self):
        try:
            self.executor.shutdown(wait=False)
        except Exception:
            pass


class MinimaxBot:
    """Thin wrapper around `fastnmm.MinimaxBot`."""

    def __init__(
        self,
        max_depth: int = 4,
        random_move_prob: float = 0.0,
        player_id: int = 0,
    ) -> None:
        self.max_depth = int(max_depth)
        self.random_move_prob = float(random_move_prob)
        self.player_id = int(player_id)
        self._bot = _FastMinimaxBot(player_id=self.player_id, depth=self.max_depth)
        self.last_search = None

    def step(self, state) -> int:
        """Pick an action for `state`, honouring `random_move_prob`."""
        if self.random_move_prob > 0.0 and random.random() < self.random_move_prob:
            return random.choice(state.legal_actions())
        action = self._bot.step(state)
        self.last_search = self._bot.last_search
        return action

    # Legacy alias used by existing call sites.
    get_action = step


# Match the worker bot pool's TT size so the eval doesn't pay first-search
# cost at every depth.
_EVAL_TT_BYTES = 64 * 1024 * 1024

# Single lock around GPU forward calls. fastnmm.MinimaxBot.step() releases the
# GIL during C++ search so multiple bot threads run concurrently on different
# cores; the model forward is small but cuDNN isn't reentrant from arbitrary
# Python threads, so we serialize it explicitly.
_gpu_lock = threading.Lock()


def _sample_stones(
    dist_counts: Optional[List[int]],
    dist_weights: Optional[List[float]],
    starting_stones: int,
) -> Tuple[int, int]:
    if dist_counts is not None:
        a = random.choices(dist_counts, weights=dist_weights)[0]
        b = random.choices(dist_counts, weights=dist_weights)[0]
        return (a, b)
    if starting_stones is None or starting_stones < 0:
        return (random.randint(3, 9), random.randint(3, 9))
    s = max(1, min(9, int(starting_stones)))
    return (s, s)


def _play_one_eval_game(
    model,
    device,
    num_actions: int,
    depth: int,
    ai_player: int,
    stones: Tuple[int, int],
    max_steps: int,
    use_mixed_precision: bool,
) -> Tuple[str, int]:
    """Play one model-vs-minimax game in a worker thread.

    Returns (outcome, nodes_visited). outcome is 'win', 'loss', or 'draw'.
    Each call owns its own MinimaxBot (TT is per-thread; the bot is not
    safe to share across threads).
    """
    from utils import get_legal_mask, build_token_obs

    game = fastnmm.load_game("nine_mens_morris")
    state = game.new_initial_state(starting_stones=stones)
    if state.is_terminal():
        return ('draw', 0)

    bot = _FastMinimaxBot(player_id=0, depth=depth, tt_bytes=_EVAL_TT_BYTES)
    nodes = 0
    steps = 0

    while not state.is_terminal() and steps < max_steps:
        current = state.current_player()
        if current == ai_player:
            node_feats, global_feats = build_token_obs(state, current)
            node_t = torch.from_numpy(node_feats).to(device).unsqueeze(0)
            glob_t = torch.from_numpy(global_feats).to(device).unsqueeze(0)
            mask = torch.from_numpy(
                get_legal_mask(state, num_actions)
            ).to(device).unsqueeze(0)
            with _gpu_lock, torch.no_grad():
                with autocast(
                    "cuda",
                    enabled=use_mixed_precision and device.type == "cuda",
                ):
                    logits, _ = model(node_t, glob_t)
                masked = logits.squeeze(0).float()
                masked[mask.squeeze(0) == 0] = -1e9
                action = int(masked.argmax().item())
        else:
            action = bot.step(state)
            if bot.last_search is not None:
                nodes += bot.last_search.nodes_visited
        state.apply_action(action)
        steps += 1

    if state.is_terminal():
        r = state.returns()
        if r[ai_player] > r[1 - ai_player]:
            return ('win', nodes)
        if r[ai_player] < r[1 - ai_player]:
            return ('loss', nodes)
        return ('draw', nodes)
    return ('draw', nodes)


def evaluate_vs_minimax(
    model,
    device,
    num_actions: int,
    max_depth: int = 6,
    games_per_depth: int = 2,
    max_steps: int = 200,
    use_mixed_precision: bool = True,
    unlimited: bool = True,
    config=None,
    starting_stones: int = 9,
    stone_distribution: Optional[List[Tuple[int, float]]] = None,
    time_budget_s: float = float('inf'),
    progress_callback: Optional[Callable[[int, Dict], None]] = None,
    max_threads: Optional[int] = None,
) -> Tuple[int, Dict]:
    """Play the model against progressively harder fastnmm minimax bots.

    If `stone_distribution` is provided, each player's stone count is sampled
    independently from it. Otherwise `starting_stones` is used as a fixed
    per-player count (pass -1 to randomize uniformly in [3, 9]).

    `games_per_depth=2` plays one game with the AI as player 0 and one with
    the AI as player 1, so each depth's win rate covers both colours.

    `unlimited=True` keeps the historical behaviour: depth climbs as long as
    the AI's win rate stays above 50%, and `max_depth` only matters when
    `unlimited=False`. There is no soft wall-clock budget by default
    (`time_budget_s=inf`).

    `progress_callback(depth, results_for_depth)` fires after each depth
    finishes — used by the trainer to print the accumulating
    `Minimax: D1:... | D2:... | ...` line as soon as each depth's W/D/L is
    known, instead of waiting for the whole eval to finish.

    Game-level parallelism: at each depth, all `games_per_depth` games run
    concurrently in a thread pool sized to `max_threads` (default
    os.cpu_count()). The fastnmm minimax search runs single-threaded in C++
    with the GIL released, so each game pins one core for its bot turns
    — using a large `games_per_depth` is the only way to saturate cores at
    the deepest, slowest depth (the engine has no parallel search).

    Climb is serial across depths: D{n+1} starts only after D{n} finishes.
    Pre-launching deeper depths in parallel was tried and reverted — even
    with cancel() on the executor, threads already executing a deep C++
    search cannot be killed, so wasted depths inflated wall time to T(D7)
    in every call even when the climb broke at D2.
    """
    game = fastnmm.load_game("nine_mens_morris")
    results: Dict[int, Dict] = {}
    max_depth_beaten = 0

    dist_counts = [c for c, _ in stone_distribution] if stone_distribution else None
    dist_weights = [w for _, w in stone_distribution] if stone_distribution else None

    deadline = (time.monotonic() + float(time_budget_s)
                if time_budget_s != float('inf') else float('inf'))

    cpu_count = os.cpu_count() or 4
    pool_size = max_threads if max_threads is not None else min(games_per_depth, cpu_count)
    pool_size = max(1, int(pool_size))

    model.eval()
    depth = 1
    while True:
        if not unlimited and depth > max_depth:
            break
        if time.monotonic() > deadline:
            break

        wins = draws = losses = 0
        total_nodes = 0

        # Build the games for this depth: each game uses ai_player = i % 2 so
        # both colours are represented evenly across `games_per_depth` games.
        game_specs = []
        for i in range(games_per_depth):
            ai_player = i % 2
            stones = _sample_stones(dist_counts, dist_weights, starting_stones)
            game_specs.append((ai_player, stones))

        with ThreadPoolExecutor(max_workers=pool_size) as executor:
            futures = [
                executor.submit(
                    _play_one_eval_game,
                    model, device, num_actions, depth,
                    ai_player, stones, max_steps, use_mixed_precision,
                )
                for ai_player, stones in game_specs
            ]
            for f in futures:
                outcome, nodes = f.result()
                total_nodes += nodes
                if outcome == 'win':
                    wins += 1
                elif outcome == 'loss':
                    losses += 1
                else:
                    draws += 1

        win_rate = (wins + 0.5 * draws) / games_per_depth
        results[depth] = {
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "win_rate": win_rate,
            "total_nodes": total_nodes,
        }

        if progress_callback is not None:
            try:
                progress_callback(depth, results[depth])
            except Exception:
                pass

        if win_rate > 0.5:
            max_depth_beaten = depth
            depth += 1
        else:
            break

    return max_depth_beaten, results


def format_minimax_results(results: Dict) -> str:
    parts = []
    for depth in sorted(results.keys()):
        r = results[depth]
        parts.append(f"D{depth}:{r['wins']}W/{r['draws']}D/{r['losses']}L")
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# C++-orchestrated progressive eval (multithreaded, root-splitting).
# ---------------------------------------------------------------------------


def evaluate_vs_minimax_cpp(
    model,
    device,
    num_actions: int,
    games_per_depth: int = 6,
    max_threads: int = 24,
    max_depth: int = 100,
    max_steps: int = 200,
    use_mixed_precision: bool = True,
    unlimited: bool = True,
    starting_stones: int = 9,
    stone_distribution: Optional[List[Tuple[int, float]]] = None,
    time_budget_s: float = float('inf'),
    progress_callback: Optional[Callable[[int, Dict], None]] = None,
    tt_bytes_per_thread: int = _EVAL_TT_BYTES,
) -> Tuple[int, Dict]:
    """C++-orchestrated version of `evaluate_vs_minimax`.

    Runs the whole progressive model-vs-minimax climb inside the fastnmm
    C++ engine. At each depth, `games_per_depth` games run concurrently on
    their own threads; each game's bot uses root-splitting across
    `max(1, max_threads // games_per_depth)` worker threads. With the
    defaults (6 games x 4 workers/game = 24) a 24-core Threadripper is
    saturated during the bot search, which is where >95% of wall time is
    spent at depth >= 4.

    Returns the same `(max_depth_beaten, results)` tuple shape as
    `evaluate_vs_minimax` so call sites can swap implementations without
    code changes. The per-depth dict has an extra `wall_seconds` key.
    """
    from utils import get_legal_mask, build_token_obs

    dist_counts = [c for c, _ in stone_distribution] if stone_distribution else None
    dist_weights = [w for _, w in stone_distribution] if stone_distribution else None

    model.eval()

    def _policy_fn(state, current_player, game_idx, depth):
        # Called from C++ worker threads with the GIL re-acquired. cuDNN
        # is not reentrant from arbitrary Python threads, so serialise
        # all model forwards behind the existing _gpu_lock.
        node_feats, global_feats = build_token_obs(state, current_player)
        node_t = torch.from_numpy(node_feats).to(device).unsqueeze(0)
        glob_t = torch.from_numpy(global_feats).to(device).unsqueeze(0)
        mask = torch.from_numpy(
            get_legal_mask(state, num_actions)
        ).to(device).unsqueeze(0)
        with _gpu_lock, torch.no_grad():
            with autocast(
                "cuda",
                enabled=use_mixed_precision and device.type == "cuda",
            ):
                logits, _ = model(node_t, glob_t)
            masked = logits.squeeze(0).float()
            masked[mask.squeeze(0) == 0] = -1e9
            return int(masked.argmax().item())

    def _spec_fn(depth, game_idx):
        if dist_counts is not None:
            a = random.choices(dist_counts, weights=dist_weights)[0]
            b = random.choices(dist_counts, weights=dist_weights)[0]
        elif starting_stones is None or starting_stones < 0:
            a = random.randint(3, 9)
            b = random.randint(3, 9)
        else:
            s = max(1, min(9, int(starting_stones)))
            a = b = s
        return (int(a), int(b), int(game_idx) % 2)

    progress_cb = None
    if progress_callback is not None:
        def progress_cb(depth, dr):
            try:
                progress_callback(int(depth), {
                    "wins":         int(dr.wins),
                    "draws":        int(dr.draws),
                    "losses":       int(dr.losses),
                    "total_nodes":  int(dr.total_nodes),
                    "win_rate":     float(dr.win_rate),
                    "wall_seconds": float(dr.wall_seconds),
                })
            except Exception:
                pass

    budget = -1.0 if time_budget_s == float('inf') else float(time_budget_s)

    raw = _fastnmm_core.progressive_minimax_eval(
        _policy_fn,
        max_depth=int(max_depth),
        games_per_depth=int(games_per_depth),
        max_threads=int(max_threads),
        max_steps=int(max_steps),
        tt_bytes_per_thread=int(tt_bytes_per_thread),
        unlimited=bool(unlimited),
        strict_parity=True,
        time_budget_s=float(budget),
        spec_fn=_spec_fn,
        progress_fn=progress_cb,
    )

    results: Dict[int, Dict] = {}
    for depth, blk in raw["per_depth"].items():
        results[int(depth)] = {
            "wins":         int(blk["wins"]),
            "draws":        int(blk["draws"]),
            "losses":       int(blk["losses"]),
            "total_nodes":  int(blk["total_nodes"]),
            "win_rate":     float(blk["win_rate"]),
            "wall_seconds": float(blk["wall_seconds"]),
        }
    return int(raw["max_depth_beaten"]), results
