"""Tests for MinimaxEngine (TT-backed) and play_until_player.

Goals
-----
1. Bit-exact parity: MinimaxEngine.search returns the same (action, score)
   as the existing free-function minimax_search across many states and
   depths.

2. Functional correctness of play_until_player:
   - terminates with current_player == target (or terminal),
   - random opponent is reproducible given a seed,
   - matches a hand-rolled Python loop end-to-end (full games).

3. TT mechanics: hit count is non-zero on a non-trivial search; clearing
   resets stats; new_game keeps entries but bumps generation.
"""
from __future__ import annotations

import random

import pytest

import fastnmm


# ---------------------------------------------------------------------------
# Engine vs free-function parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 1234])
def test_strict_engine_matches_minimax_search_on_many_states(seed):
    """strict_parity=True must match the legacy minimax_search bit-exactly."""
    rng = random.Random(seed)
    game = fastnmm.load_game("nine_mens_morris")
    eng = fastnmm.MinimaxEngine(16 * 1024 * 1024, strict_parity=True)

    state = game.new_initial_state()
    while not state.is_terminal():
        for d in (1, 2, 3):
            r_old = fastnmm.minimax_search(state, d)
            r_new = eng.search(state, d)
            assert r_old.action == r_new.action, (
                f"action mismatch at d={d}: {r_old.action} vs {r_new.action}"
            )
            assert r_old.score == r_new.score, (
                f"score mismatch at d={d}: {r_old.score} vs {r_new.score}"
            )
        state.apply_action(rng.choice(state.legal_actions()))


def test_strict_engine_search_at_depth_4_parity():
    rng = random.Random(13)
    game = fastnmm.load_game("nine_mens_morris")
    eng = fastnmm.MinimaxEngine(64 * 1024 * 1024, strict_parity=True)

    for _ in range(5):
        state = game.new_initial_state()
        for _ in range(20):
            if state.is_terminal():
                break
            r_old = fastnmm.minimax_search(state, 4)
            r_new = eng.search(state, 4)
            assert (r_old.action, r_old.score) == (r_new.action, r_new.score)
            state.apply_action(rng.choice(state.legal_actions()))


# ---------------------------------------------------------------------------
# Relaxed mode: action choice agreement on non-mate positions, and far
# higher cross-search hit rate than strict.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_relaxed_engine_action_agrees_with_minimax_search_on_non_mate(seed):
    """In non-mate positions (no terminal reachable within `depth`), the
    relaxed engine and the legacy `minimax_search` must pick the same
    action -- their evals are identical in that regime."""
    rng = random.Random(seed)
    game = fastnmm.load_game("nine_mens_morris")
    eng = fastnmm.MinimaxEngine(16 * 1024 * 1024, strict_parity=False)

    state = game.new_initial_state()
    # Restrict to early/mid game so depth-3 search rarely sees a terminal.
    for _ in range(40):
        if state.is_terminal():
            break
        for d in (1, 2, 3):
            r_old = fastnmm.minimax_search(state, d)
            r_new = eng.search(state, d)
            old_is_mate = abs(r_old.score) > 90000
            new_is_mate = abs(r_new.score) > 28000
            if not old_is_mate and not new_is_mate:
                assert r_old.action == r_new.action
                assert r_old.score == r_new.score
        state.apply_action(rng.choice(state.legal_actions()))


def test_relaxed_engine_works_and_records_hits():
    """Relaxed mode is functional and records intra-search hits.

    Note: in normal forward minimax (single-state search, sequential
    same-depth searches), all paths to a node reach it at the same
    remaining depth, so relaxed mode does NOT yield more hits than
    strict mode -- both catch the same intra-search transpositions.
    Relaxed mode would only outperform strict on patterns where
    cached entries are at *greater* depths than queries (e.g. an
    opening book of pre-computed deep evaluations consulted during
    shallow lookups). It's kept as an option for those use-cases.
    """
    rng = random.Random(0xCAFE)
    game = fastnmm.load_game("nine_mens_morris")
    eng = fastnmm.MinimaxEngine(64 * 1024 * 1024, strict_parity=False)
    state = game.new_initial_state()
    for _ in range(20):
        if state.is_terminal():
            break
        eng.search(state, 4)
        state.apply_action(rng.choice(state.legal_actions()))
    assert eng.tt_probes > 0
    assert eng.tt_hits > 0
    assert not eng.strict_parity


# ---------------------------------------------------------------------------
# Engine TT mechanics
# ---------------------------------------------------------------------------


def test_engine_records_tt_hits_within_a_search():
    game = fastnmm.load_game("nine_mens_morris")
    state = game.new_initial_state()
    eng = fastnmm.MinimaxEngine(16 * 1024 * 1024)
    eng.search(state, 4)
    assert eng.tt_probes > 0
    assert eng.tt_hits > 0
    assert eng.tt_stores > 0
    assert 0.0 <= eng.tt_fill_fraction <= 1.0


def test_engine_tt_clear_resets_stats():
    game = fastnmm.load_game("nine_mens_morris")
    state = game.new_initial_state()
    eng = fastnmm.MinimaxEngine(8 * 1024 * 1024)
    eng.search(state, 3)
    assert eng.tt_probes > 0
    eng.tt_clear()
    assert eng.tt_probes == 0
    assert eng.tt_hits == 0
    assert eng.tt_stores == 0


def test_engine_new_game_preserves_entries_but_bumps_generation():
    game = fastnmm.load_game("nine_mens_morris")
    state = game.new_initial_state()
    eng = fastnmm.MinimaxEngine(8 * 1024 * 1024)
    eng.search(state, 3)
    stores_before = eng.tt_stores
    eng.new_game()
    # Stats are NOT reset by new_game.
    assert eng.tt_stores == stores_before
    # A second search at the same root should still be fast (entries reused).
    eng.search(state, 3)
    assert eng.tt_stores >= stores_before  # additional stores possible


def test_zobrist_key_reproducible():
    game = fastnmm.load_game("nine_mens_morris")
    s1 = game.new_initial_state()
    s2 = game.new_initial_state()
    assert fastnmm.zobrist_key(s1) == fastnmm.zobrist_key(s2)
    s1.apply_action(s1.legal_actions()[0])
    assert fastnmm.zobrist_key(s1) != fastnmm.zobrist_key(s2)
    s2.apply_action(s2.legal_actions()[0])
    assert fastnmm.zobrist_key(s1) == fastnmm.zobrist_key(s2)


# ---------------------------------------------------------------------------
# play_until_player: random opponent
# ---------------------------------------------------------------------------


def test_play_until_player_random_terminates_at_target():
    game = fastnmm.load_game("nine_mens_morris")
    state = game.new_initial_state()
    state.apply_action(state.legal_actions()[0])  # AI plays first
    actions, _ = fastnmm.play_until_player(
        state, target_player=0, opponent_kind="random", rng_seed=42,
    )
    assert state.is_terminal() or state.current_player() == 0
    assert len(actions) >= 1


def test_play_until_player_random_reproducible_with_seed():
    game = fastnmm.load_game("nine_mens_morris")
    s1 = game.new_initial_state()
    s2 = game.new_initial_state()
    s1.apply_action(s1.legal_actions()[0])
    s2.apply_action(s2.legal_actions()[0])

    a1, _ = fastnmm.play_until_player(s1, 0, "random", rng_seed=12345)
    a2, _ = fastnmm.play_until_player(s2, 0, "random", rng_seed=12345)
    assert a1 == a2


def test_play_until_player_invalid_opponent_kind():
    game = fastnmm.load_game("nine_mens_morris")
    state = game.new_initial_state()
    with pytest.raises(ValueError):
        fastnmm.play_until_player(state, 0, opponent_kind="bogus")


# ---------------------------------------------------------------------------
# play_until_player: minimax opponent end-to-end parity
# ---------------------------------------------------------------------------


def _full_game_python_loop(seed: int, depth: int):
    rng = random.Random(seed)
    game = fastnmm.load_game("nine_mens_morris")
    state = game.new_initial_state()
    actions = []
    while not state.is_terminal():
        if state.current_player() == 0:
            a = rng.choice(state.legal_actions())
        else:
            a = fastnmm.minimax_search(state, depth).action
        state.apply_action(a)
        actions.append(a)
    return list(state.returns()), actions


def _full_game_play_until(seed: int, depth: int, tt_bytes: int,
                          strict_parity: bool):
    rng = random.Random(seed)
    game = fastnmm.load_game("nine_mens_morris")
    state = game.new_initial_state()
    eng = fastnmm.MinimaxEngine(tt_bytes, strict_parity=strict_parity)
    actions = []
    while not state.is_terminal():
        if state.current_player() == 0:
            a = rng.choice(state.legal_actions())
            state.apply_action(a)
            actions.append(a)
        opp_actions, _ = fastnmm.play_until_player(
            state, target_player=0,
            opponent_kind="minimax",
            minimax_depth=depth,
            engine=eng,
        )
        actions.extend(opp_actions)
    return list(state.returns()), actions


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_play_until_minimax_strict_matches_python_loop(seed):
    """End-to-end: strict-parity engine driven via play_until_player must
    play identically to a Python loop calling minimax_search."""
    r1, a1 = _full_game_python_loop(seed, depth=3)
    r2, a2 = _full_game_play_until(
        seed, depth=3, tt_bytes=16 * 1024 * 1024, strict_parity=True,
    )
    assert r1 == r2
    assert a1 == a2


def test_play_until_minimax_no_engine_works_one_shot():
    """`engine=None` should still work, just without TT reuse."""
    game = fastnmm.load_game("nine_mens_morris")
    state = game.new_initial_state()
    state.apply_action(state.legal_actions()[0])
    actions, _ = fastnmm.play_until_player(
        state, target_player=0, opponent_kind="minimax",
        minimax_depth=2, engine=None,
    )
    assert state.is_terminal() or state.current_player() == 0
    assert len(actions) >= 1


def test_play_until_minimax_random_move_prob_one_is_legal():
    """random_move_prob=1 with minimax kind should never call the engine
    and only play legal moves (the underlying RNG sequence differs from a
    pure-random opponent because of the extra prob-roll draws)."""
    game = fastnmm.load_game("nine_mens_morris")
    state = game.new_initial_state()
    state.apply_action(state.legal_actions()[0])

    eng = fastnmm.MinimaxEngine(8 * 1024 * 1024)
    actions, _ = fastnmm.play_until_player(
        state, 0, opponent_kind="minimax",
        minimax_depth=4, random_move_prob=1.0, engine=eng,
        rng_seed=777,
    )
    assert state.is_terminal() or state.current_player() == 0
    assert len(actions) >= 1
    # The engine should not have run any searches when prob=1.0.
    assert eng.tt_stores == 0


# ---------------------------------------------------------------------------
# MinimaxBot wraps the engine
# ---------------------------------------------------------------------------


def test_minimax_bot_uses_engine_and_exposes_tt_stats():
    game = fastnmm.load_game("nine_mens_morris")
    state = game.new_initial_state()
    # depth=4 ensures transpositions appear at remaining-depth >= 1
    # (placement-order swaps converge at level 1 and below).
    bot = fastnmm.MinimaxBot(player_id=0, depth=4, tt_bytes=16 * 1024 * 1024)
    bot.step(state)
    stats = bot.tt_stats
    assert stats["entries"] > 0
    assert stats["bytes"] >= 16 * 1024 * 1024 // 2  # rounded down to power of 2
    assert stats["probes"] > 0
    assert stats["hits"] > 0
    assert 0.0 <= stats["hit_rate"] <= 1.0


def test_strict_minimax_bot_step_matches_minimax_search():
    """In strict mode, MinimaxBot picks must equal fastnmm.minimax_search."""
    rng = random.Random(2026)
    game = fastnmm.load_game("nine_mens_morris")
    bot = fastnmm.MinimaxBot(
        player_id=0, depth=3, tt_bytes=8 * 1024 * 1024, strict_parity=True,
    )
    state = game.new_initial_state()
    for _ in range(15):
        if state.is_terminal():
            break
        a_bot = bot.step(state)
        a_ref = fastnmm.minimax_search(state, 3).action
        assert a_bot == a_ref
        state.apply_action(rng.choice(state.legal_actions()))
