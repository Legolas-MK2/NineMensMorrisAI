"""
Minimax wrapper around the fastnmm C++ engine.

Public surface for the rest of the codebase:
    MinimaxBot(max_depth=…, random_move_prob=…) — exposes .step()/.get_action()
    evaluate_vs_minimax_cpp(model, device, num_actions, …)
    format_minimax_results(results)
"""

from __future__ import annotations

import random
import threading
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.amp import autocast

from fastnmm import MinimaxBot as _FastMinimaxBot
from fastnmm import _core as _fastnmm_core


class MinimaxBot:
    """Thin wrapper around `fastnmm.MinimaxBot`."""

    def __init__(
        self,
        max_depth: int = 4,
        random_move_prob: float = 0.0,
        player_id: int = 0,
        weight_jitter: float = 0.0,
        root_epsilon: int = 0,
        seed: Optional[int] = None,
    ) -> None:
        self.max_depth = int(max_depth)
        self.random_move_prob = float(random_move_prob)
        self.player_id = int(player_id)
        self._bot = _FastMinimaxBot(
            player_id=self.player_id,
            depth=self.max_depth,
            weight_jitter=float(weight_jitter),
            root_epsilon=int(root_epsilon),
            tiebreak_seed=seed,
        )
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

    def reroll_weights(self, jitter: Optional[float] = None) -> None:
        """Resample the underlying engine's evaluation weights."""
        self._bot.reroll_weights(jitter)


# Per-thread TT size for the progressive eval. Smaller than the worker
# pool's 128 MiB (config.minimax_tt_bytes_per_bot) because the eval spawns
# many short-lived engines (games_per_depth x threads_per_game).
_EVAL_TT_BYTES = 64 * 1024 * 1024

# Single lock around GPU forward calls. fastnmm.MinimaxBot.step() releases the
# GIL during C++ search so multiple bot threads run concurrently on different
# cores; the model forward is small but cuDNN isn't reentrant from arbitrary
# Python threads, so we serialize it explicitly.
_gpu_lock = threading.Lock()


def format_minimax_results(results: Dict) -> str:
    parts = []
    for depth in sorted(results.keys()):
        r = results[depth]
        parts.append(f"D{depth}:{r['wins']}W/{r['draws']}D/{r['losses']}L")
    return " | ".join(parts)


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
    """Progressive model-vs-minimax climb, orchestrated by the fastnmm C++ engine.

    At each depth, `games_per_depth` games run concurrently on their own
    threads; each game's bot uses root-splitting across
    `max(1, max_threads // games_per_depth)` worker threads. With the
    defaults (6 games x 4 workers/game = 24) a 24-core Threadripper is
    saturated during the bot search, which is where >95% of wall time is
    spent at depth >= 4.

    If `stone_distribution` is provided, each player's stone count is sampled
    independently from it. Otherwise `starting_stones` is used as a fixed
    per-player count (pass -1 to randomize uniformly in [3, 9]).

    `unlimited=True` keeps climbing as long as the AI's win rate stays above
    50%; `max_depth` only matters when `unlimited=False`.

    `progress_callback(depth, results_for_depth)` fires after each depth
    finishes — used by the trainer to print the accumulating
    `Minimax: D1:... | D2:... | ...` line live.

    Returns `(max_depth_beaten, results)`; each per-depth dict includes a
    `wall_seconds` key.
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

    if progress_callback is None:
        progress_cb = None
    else:
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
