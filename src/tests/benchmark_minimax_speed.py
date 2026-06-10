#!/usr/bin/env python3
"""Benchmark minimax search speed at each depth.

For each depth in --depths, play --games full games of D-vs-D and report
the average wall time per `bot.step()` call. Each player owns its own
MinimaxBot (own transposition table); the bots are reused across all
games at that depth so the TT warms up like it does in the worker pool.

NOTE: D6/D7 are SLOW. 10 full games at D7 can take many minutes — the
script prints per-game progress so you can Ctrl-C and keep partial
results.

Usage (from src/):
    python tests/benchmark_minimax_speed.py
    python tests/benchmark_minimax_speed.py --depths 1 2 3 --games 5
    python tests/benchmark_minimax_speed.py --depths 6 7 --games 3 --stones-min 3 --stones-max 9
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from typing import List, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_HERE)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import fastnmm
from fastnmm import MinimaxBot

DEFAULT_TT_BYTES = 128 * 1024 * 1024  # matches Config.minimax_tt_bytes_per_bot


def benchmark_depth(
    depth: int,
    n_games: int,
    tt_bytes: int,
    stones_range: Tuple[int, int],
    max_moves: int,
    seed: int,
) -> dict:
    """Play `n_games` games of D-vs-D, return per-move timing stats."""
    rng = random.Random(seed)
    game = fastnmm.load_game("nine_mens_morris")
    bots = [
        MinimaxBot(player_id=0, depth=depth, tt_bytes=tt_bytes),
        MinimaxBot(player_id=1, depth=depth, tt_bytes=tt_bytes),
    ]

    total_moves = 0
    total_seconds = 0.0
    longest_move_s = 0.0
    games_completed = 0
    s_lo, s_hi = stones_range

    for g in range(n_games):
        a = rng.randint(s_lo, s_hi)
        b = rng.randint(s_lo, s_hi)
        state = game.new_initial_state(starting_stones=(a, b))
        if state.is_terminal():
            print(f"  D{depth} game {g+1}/{n_games}: stones=({a},{b}) terminal at init, skipping",
                  flush=True)
            continue

        moves_this_game = 0
        game_seconds = 0.0
        while not state.is_terminal() and moves_this_game < max_moves:
            pid = state.current_player()
            t0 = time.perf_counter()
            action = bots[pid].step(state)
            dt = time.perf_counter() - t0

            total_seconds += dt
            game_seconds += dt
            total_moves += 1
            if dt > longest_move_s:
                longest_move_s = dt

            state.apply_action(action)
            moves_this_game += 1

        games_completed += 1
        avg_so_far_ms = (total_seconds / total_moves * 1000) if total_moves else 0.0
        print(
            f"  D{depth} game {g+1}/{n_games}: stones=({a},{b}) moves={moves_this_game} "
            f"game_wall={game_seconds:.2f}s avg_so_far={avg_so_far_ms:.2f} ms/move",
            flush=True,
        )

    avg_ms = (total_seconds / total_moves * 1000) if total_moves else 0.0
    return {
        "depth": depth,
        "games": games_completed,
        "moves": total_moves,
        "total_seconds": total_seconds,
        "avg_ms_per_move": avg_ms,
        "max_ms_per_move": longest_move_s * 1000,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark per-depth minimax move time (D-vs-D self-play).",
    )
    parser.add_argument(
        "--depths", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7],
        help="Depths to benchmark (default: 1..7)",
    )
    parser.add_argument(
        "--games", type=int, default=10,
        help="Games per depth (default: 10)",
    )
    parser.add_argument(
        "--tt-bytes", type=int, default=DEFAULT_TT_BYTES,
        help="Transposition table size per bot, in bytes (default: 128 MiB)",
    )
    parser.add_argument(
        "--stones-min", type=int, default=9,
        help="Min starting stones per player (default: 9 = full game)",
    )
    parser.add_argument(
        "--stones-max", type=int, default=9,
        help="Max starting stones per player (default: 9)",
    )
    parser.add_argument(
        "--max-moves", type=int, default=300,
        help="Cap on moves per game (default: 300)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for starting-stone sampling (default: 42)",
    )
    args = parser.parse_args()

    print("Minimax speed benchmark")
    print(f"  depths:          {args.depths}")
    print(f"  games per depth: {args.games}")
    print(f"  starting stones: random in [{args.stones_min}, {args.stones_max}] per player")
    print(f"  tt_bytes:        {args.tt_bytes // (1024*1024)} MiB per bot")
    print(f"  max moves/game:  {args.max_moves}")
    print()

    results: List[dict] = []
    for d in args.depths:
        print(f"==== D{d} vs D{d}, {args.games} games ====")
        t0 = time.perf_counter()
        r = benchmark_depth(
            depth=d,
            n_games=args.games,
            tt_bytes=args.tt_bytes,
            stones_range=(args.stones_min, args.stones_max),
            max_moves=args.max_moves,
            seed=args.seed + d,
        )
        wall = time.perf_counter() - t0
        r["wall_seconds"] = wall
        results.append(r)
        print(
            f"  D{d} done: {r['moves']} moves over {r['games']} games | "
            f"avg {r['avg_ms_per_move']:.2f} ms/move | "
            f"max {r['max_ms_per_move']:.0f} ms | wall {wall:.1f} s"
        )
        print()

    print("=" * 72)
    print(f"{'Depth':<7}{'Games':<7}{'Moves':<7}{'Avg ms/move':<14}{'Max ms/move':<14}{'Wall s':<10}")
    print("-" * 72)
    for r in results:
        print(
            f"D{r['depth']:<6}{r['games']:<7}{r['moves']:<7}"
            f"{r['avg_ms_per_move']:<14.2f}{r['max_ms_per_move']:<14.0f}{r['wall_seconds']:<10.1f}"
        )


if __name__ == "__main__":
    main()
