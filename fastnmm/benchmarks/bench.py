"""Benchmark fastnmm against OpenSpiel for random self-play.

Runs N random games with each engine and reports:
- total wall time
- games / second
- actions / second
- relative speedup (fastnmm vs OpenSpiel)

Usage:
    python benchmarks/bench.py                 # default 2000 games
    python benchmarks/bench.py --games 10000   # more games
"""
from __future__ import annotations

import argparse
import random
import time


def bench_fastnmm(num_games: int, seed0: int = 0):
    import fastnmm
    g = fastnmm.load_game("nine_mens_morris")
    total_actions = 0
    t0 = time.perf_counter()
    for seed in range(seed0, seed0 + num_games):
        rng = random.Random(seed)
        s = g.new_initial_state()
        while not s.is_terminal():
            s.apply_action(rng.choice(s.legal_actions()))
            total_actions += 1
    return time.perf_counter() - t0, total_actions


def bench_openspiel(num_games: int, seed0: int = 0):
    try:
        import pyspiel
    except ImportError:
        return None
    g = pyspiel.load_game("nine_mens_morris")
    total_actions = 0
    t0 = time.perf_counter()
    for seed in range(seed0, seed0 + num_games):
        rng = random.Random(seed)
        s = g.new_initial_state()
        while not s.is_terminal():
            s.apply_action(rng.choice(s.legal_actions()))
            total_actions += 1
    return time.perf_counter() - t0, total_actions


def bench_fastnmm_hot_path(num_games: int, seed0: int = 0):
    """Same as bench_fastnmm but without Python's ``random.choice`` overhead --
    picks the first legal action. This isolates engine cost from RNG cost and
    shows the raw C++ throughput."""
    import fastnmm
    g = fastnmm.load_game("nine_mens_morris")
    total_actions = 0
    t0 = time.perf_counter()
    for seed in range(seed0, seed0 + num_games):
        rng = random.Random(seed)
        s = g.new_initial_state()
        while not s.is_terminal():
            la = s.legal_actions()
            s.apply_action(la[rng.randrange(len(la))])
            total_actions += 1
    return time.perf_counter() - t0, total_actions


def fmt(num_games: int, total_time: float, total_actions: int, label: str):
    gps = num_games / total_time
    aps = total_actions / total_time
    print(f"  {label:22s} {total_time:7.3f} s   "
          f"{gps:8.0f} games/s   {aps:10.0f} actions/s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=100)
    args = ap.parse_args()

    # Warmup (import + cache)
    print(f"Warmup: {args.warmup} games …")
    bench_fastnmm(args.warmup)
    osp = bench_openspiel(args.warmup)

    print(f"\nBenchmark: {args.games} random games per engine\n")

    ft, fa = bench_fastnmm(args.games)
    fmt(args.games, ft, fa, "fastnmm")

    fht, fha = bench_fastnmm_hot_path(args.games)
    fmt(args.games, fht, fha, "fastnmm (hot path)")

    if osp is None:
        print("\n  OpenSpiel: not installed; skipping comparison.")
        return

    ot, oa = bench_openspiel(args.games)
    fmt(args.games, ot, oa, "openspiel")

    speedup_actions = (oa / ot) / (fa / ft) * (fa / ft) / (oa / ot)  # placeholder
    print()
    print(f"  Speedup (actions/s): fastnmm is "
          f"{(fa / ft) / (oa / ot):.2f}x openspiel")


if __name__ == "__main__":
    main()
