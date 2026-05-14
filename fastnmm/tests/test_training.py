"""Tests for training-oriented features: custom starting stones and bots."""
import pytest

import fastnmm
from fastnmm import (
    RandomBot,
    MinimaxBot,
    make_uniform_random_bot,
    make_minimax_bot,
    play_match,
    play_matches,
)


# ---------------------------------------------------------------------------
# Custom starting stones
# ---------------------------------------------------------------------------


def test_load_game_default_stones():
    g = fastnmm.load_game("nine_mens_morris")
    assert g.starting_stones == (9, 9)
    s = g.new_initial_state()
    assert s.men_to_deploy(0) == 9
    assert s.men_to_deploy(1) == 9


def test_load_game_with_starting_stones_tuple():
    g = fastnmm.load_game("nine_mens_morris", starting_stones=(5, 6))
    assert g.starting_stones == (5, 6)
    s = g.new_initial_state()
    assert s.men_to_deploy(0) == 5
    assert s.men_to_deploy(1) == 6


def test_load_game_with_starting_stones_list():
    g = fastnmm.load_game("nine_mens_morris", starting_stones=[3, 7])
    s = g.new_initial_state()
    assert s.men_to_deploy(0) == 3
    assert s.men_to_deploy(1) == 7


def test_new_initial_state_override():
    g = fastnmm.load_game("nine_mens_morris", starting_stones=(9, 9))
    s = g.new_initial_state(starting_stones=(4, 4))
    assert s.men_to_deploy(0) == 4
    assert s.men_to_deploy(1) == 4


@pytest.mark.parametrize("bad", [(0, 5), (5, 0), (10, 5), (5, 10), (-1, 5)])
def test_invalid_stones_rejected(bad):
    with pytest.raises(ValueError):
        fastnmm.load_game("nine_mens_morris", starting_stones=bad)


@pytest.mark.parametrize("bad", [(5,), (5, 6, 7), [], [5]])
def test_wrong_length_stones_rejected(bad):
    with pytest.raises(ValueError):
        fastnmm.load_game("nine_mens_morris", starting_stones=bad)


def test_custom_stones_random_game_terminates():
    """Games with any valid starting_stones always terminate."""
    import random
    for stones in [(5, 6), (3, 3), (9, 3), (1, 9), (4, 7)]:
        g = fastnmm.load_game(starting_stones=stones)
        rng = random.Random(0)
        for seed in range(10):
            rng.seed(seed)
            s = g.new_initial_state()
            steps = 0
            while not s.is_terminal():
                s.apply_action(rng.choice(s.legal_actions()))
                steps += 1
                assert steps < 500, f"stones {stones}, seed {seed}: too long"
            assert s.is_terminal()


def test_asymmetric_placing_transitions_per_player():
    """With (3, 5) stones, W finishes placing first and starts moving while
    B is still placing."""
    s = fastnmm.load_game(starting_stones=(3, 5)).new_initial_state()
    # Interleaved W, B, W, B, W, B --> 3 W pieces + 3 B pieces.
    for a in [0, 3, 6, 9, 15, 12]:
        assert a in s.legal_actions()
        s.apply_action(a)
    # Now: W has 0 to deploy, B has 2. It's W's turn.
    assert s.men_to_deploy(0) == 0
    assert s.men_to_deploy(1) == 2
    assert s.current_player() == 0
    assert s.current_phase() == fastnmm.Phase.MOVING
    # All legal actions must be MOVE actions.
    assert all(a >= 24 for a in s.legal_actions())
    # And B, when it's their turn, must still be placing.
    s.apply_action(s.legal_actions()[0])   # W moves somewhere
    if not s.is_terminal() and s.current_player() == 1:
        assert s.current_phase() == fastnmm.Phase.PLACING
        assert all(a < 24 for a in s.legal_actions())


# ---------------------------------------------------------------------------
# RandomBot
# ---------------------------------------------------------------------------


def test_random_bot_returns_legal_actions():
    s = fastnmm.load_game().new_initial_state()
    bot = RandomBot(player_id=0, seed=0)
    for _ in range(30):
        if s.is_terminal():
            break
        a = bot.step(s)
        assert a in s.legal_actions()
        s.apply_action(a)


def test_random_bot_deterministic_with_seed():
    bot_a = RandomBot(player_id=0, seed=42)
    bot_b = RandomBot(player_id=0, seed=42)
    s_a = fastnmm.load_game().new_initial_state()
    s_b = fastnmm.load_game().new_initial_state()
    for _ in range(20):
        if s_a.is_terminal():
            break
        a = bot_a.step(s_a)
        b = bot_b.step(s_b)
        assert a == b
        s_a.apply_action(a)
        s_b.apply_action(b)


def test_random_bot_restart_restores_rng():
    bot = RandomBot(player_id=0, seed=123)
    s1 = fastnmm.load_game().new_initial_state()
    actions_1 = [bot.step(s1) for _ in range(5)
                 if s1.apply_action(bot.step(s1)) is None and not s1.is_terminal()]

    # Simpler and more correct: take two sequences, reset in between.
    bot = RandomBot(player_id=0, seed=123)
    state1 = fastnmm.load_game().new_initial_state()
    seq1 = []
    for _ in range(5):
        a = bot.step(state1)
        seq1.append(a)
        state1.apply_action(a)

    bot.restart()
    state2 = fastnmm.load_game().new_initial_state()
    seq2 = []
    for _ in range(5):
        a = bot.step(state2)
        seq2.append(a)
        state2.apply_action(a)

    assert seq1 == seq2


# ---------------------------------------------------------------------------
# MinimaxBot
# ---------------------------------------------------------------------------


def test_minimax_bot_returns_legal_action():
    s = fastnmm.load_game().new_initial_state()
    bot = MinimaxBot(player_id=0, depth=2)
    a = bot.step(s)
    assert a in s.legal_actions()


def test_minimax_bot_records_search_result():
    s = fastnmm.load_game().new_initial_state()
    bot = MinimaxBot(player_id=0, depth=3)
    bot.step(s)
    assert bot.last_search is not None
    assert bot.last_search.action in s.legal_actions()
    assert bot.last_search.nodes_visited > 0


def test_minimax_depth_zero_rejected():
    with pytest.raises(ValueError):
        MinimaxBot(player_id=0, depth=0)


def test_minimax_search_free_functions():
    s = fastnmm.load_game().new_initial_state()
    r = fastnmm.minimax_search(s, depth=2)
    assert r.action in s.legal_actions()
    a = fastnmm.minimax_action(s, depth=2)
    assert a in s.legal_actions()


def test_evaluate_returns_int():
    s = fastnmm.load_game().new_initial_state()
    v = fastnmm.evaluate(s)
    assert isinstance(v, int)
    # Initial position is symmetric -> 0 (modulo small eval tiebreaks).
    assert v == 0


def test_minimax_beats_random_majority():
    """Play 10 games MinimaxBot(d=3) vs RandomBot at (5,6) stones.  Minimax
    should win the majority (allowing for some random upsets)."""
    stats = play_matches(
        [MinimaxBot(0, depth=3), RandomBot(1, seed=0)],
        num_games=10,
        starting_stones=(5, 6),
        swap_colors=True,
    )
    mm_wins, rand_wins = stats["wins"]
    # MinimaxBot should outperform random by a clear margin.
    assert mm_wins >= rand_wins + 2, f"minimax {mm_wins} vs random {rand_wins}"


# ---------------------------------------------------------------------------
# play_match / play_matches
# ---------------------------------------------------------------------------


def test_play_match_returns_valid_zero_sum():
    r = play_match([RandomBot(0, seed=0), RandomBot(1, seed=1)])
    assert len(r) == 2
    assert sum(r) == 0.0
    assert tuple(r) in {(1.0, -1.0), (-1.0, 1.0), (0.0, 0.0)}


def test_play_match_with_custom_stones():
    r = play_match(
        [RandomBot(0, seed=0), RandomBot(1, seed=1)],
        starting_stones=(4, 6),
    )
    assert sum(r) == 0.0


def test_play_match_wrong_number_of_bots():
    with pytest.raises(ValueError):
        play_match([RandomBot(0)])


def test_play_match_with_custom_max_turns():
    r = play_match(
        [RandomBot(0, seed=0), RandomBot(1, seed=1)],
        max_turns=5,
    )
    assert sum(r) == 0.0


def test_play_matches_structure():
    stats = play_matches(
        [RandomBot(0, seed=0), RandomBot(1, seed=1)],
        num_games=6,
        swap_colors=False,
    )
    assert set(stats) == {"wins", "draws", "returns"}
    assert len(stats["returns"]) == 6
    total = sum(stats["wins"]) + stats["draws"]
    assert total == 6


def test_duck_typed_bot_works():
    """Any object with step(state) -> int works; no subclassing needed."""

    class FirstLegalBot:
        player_id = 0
        def step(self, state):
            return state.legal_actions()[0]

    r = play_match([FirstLegalBot(), RandomBot(1, seed=0)])
    assert sum(r) == 0.0


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def test_make_uniform_random_bot():
    bot = make_uniform_random_bot(player_id=1, seed=7)
    assert isinstance(bot, RandomBot)
    assert bot.player_id == 1


def test_make_minimax_bot():
    bot = make_minimax_bot(player_id=0, depth=3)
    assert isinstance(bot, MinimaxBot)
    assert bot.depth == 3
