"""Parity tests against OpenSpiel's reference implementation of Nine Men's
Morris.

These tests are skipped if :mod:`pyspiel` is not importable.

We drive both engines with identical action sequences and require their
legal-action sets, terminal/non-terminal status, current player, and
terminal returns to agree.

Two small divergences from OpenSpiel are intentional and documented --
they are actually OpenSpiel bugs that ``fastnmm`` fixes:

1. OpenSpiel's ``legal_actions`` omits the moves ``1 -> 0`` (action 48)
   and ``9 -> 0`` (action 240), even though its own transition function
   accepts them and mills involving point 0 fire correctly.  This is a
   missing back-edge in OpenSpiel's neighbour list.  ``fastnmm`` lists
   these moves, and the parity test forgives the difference.

2. As a consequence of (1), a handful of games reach a position where
   OpenSpiel declares the current player stalemated (because their only
   legal move is 1 -> 0 or 9 -> 0), while ``fastnmm`` correctly finds
   the move.  When this happens the parity test stops comparing
   (OpenSpiel has diverged into a bug-induced terminal).
"""
import random

import pytest

pyspiel = pytest.importorskip("pyspiel")
import fastnmm


# OpenSpiel's broken-adjacency edges.
OSP_MISSING_ACTIONS = {
    fastnmm.encode_move(1, 0),   # action 48
    fastnmm.encode_move(9, 0),   # action 240
}


@pytest.fixture(scope="module")
def osg():
    return pyspiel.load_game("nine_mens_morris")


@pytest.fixture(scope="module")
def fng():
    return fastnmm.load_game("nine_mens_morris")


# ---------------------------------------------------------------------------
# Metadata parity
# ---------------------------------------------------------------------------

def test_metadata_matches(osg, fng):
    assert osg.num_distinct_actions() == fng.num_distinct_actions()
    assert osg.num_players() == fng.num_players()
    assert osg.min_utility() == fng.min_utility()
    assert osg.max_utility() == fng.max_utility()
    assert list(osg.observation_tensor_shape()) == list(fng.observation_tensor_shape())
    assert osg.observation_tensor_size() == fng.observation_tensor_size()


def test_action_strings_match(osg, fng):
    # Spot-check across the full range.
    for a in [0, 1, 23, 24, 25, 48, 240, 300, 599]:
        assert osg.action_to_string(0, a) == fng.action_to_string(0, a)


# ---------------------------------------------------------------------------
# Initial-state parity
# ---------------------------------------------------------------------------

def test_initial_legal_actions_match(osg, fng):
    os_s = osg.new_initial_state()
    fn_s = fng.new_initial_state()
    assert os_s.legal_actions() == fn_s.legal_actions()
    assert os_s.current_player() == fn_s.current_player()


def test_initial_str_matches(osg, fng):
    os_s = osg.new_initial_state()
    fn_s = fng.new_initial_state()
    assert str(os_s).strip() == str(fn_s).strip()


def test_initial_observation_tensor_matches(osg, fng):
    import numpy as np

    os_s = osg.new_initial_state()
    fn_s = fng.new_initial_state()
    os_t = np.array(os_s.observation_tensor(0), dtype=np.float32)
    fn_t = np.array(fn_s.observation_tensor(0), dtype=np.float32)
    assert os_t.shape == fn_t.shape
    np.testing.assert_array_equal(os_t, fn_t)


# ---------------------------------------------------------------------------
# Random-game parity
# ---------------------------------------------------------------------------

def _compare_nonterminal(os_s, fn_s):
    os_la = set(os_s.legal_actions())
    fn_la = set(fn_s.legal_actions())
    # fastnmm may legally list 1->0 and 9->0; OpenSpiel omits them.
    extra_in_fn = (fn_la - os_la) - OSP_MISSING_ACTIONS
    assert not extra_in_fn, f"fastnmm has unexpected extra actions: {sorted(extra_in_fn)}"
    assert not (os_la - fn_la), f"OpenSpiel has actions fastnmm doesn't: {sorted(os_la - fn_la)}"
    assert os_s.current_player() == fn_s.current_player()


@pytest.mark.parametrize("seed", range(50))
def test_parity_random_games(osg, fng, seed):
    rng = random.Random(seed)
    os_s = osg.new_initial_state()
    fn_s = fng.new_initial_state()

    while not os_s.is_terminal():
        _compare_nonterminal(os_s, fn_s)
        # Choose an action legal in BOTH engines.
        common = list(set(os_s.legal_actions()) & set(fn_s.legal_actions()))
        a = rng.choice(common)
        os_s.apply_action(a)
        fn_s.apply_action(a)

    # OpenSpiel has terminated. fastnmm may also be terminal (exact match)
    # or still running (the OpenSpiel-adjacency-bug case).
    if fn_s.is_terminal():
        assert os_s.returns() == fn_s.returns()
    else:
        # fastnmm is alive; verify its remaining legal actions are a subset of
        # the two quirk-moves (i.e. OpenSpiel falsely stalemated the player).
        leftover = set(fn_s.legal_actions()) - OSP_MISSING_ACTIONS
        assert not leftover, (
            "fastnmm is non-terminal with real moves available while "
            f"OpenSpiel terminated. leftover actions: {sorted(leftover)}"
        )


def test_parity_observation_tensor_during_game(osg, fng):
    import numpy as np
    rng = random.Random(0)
    os_s = osg.new_initial_state()
    fn_s = fng.new_initial_state()
    for _ in range(30):
        if os_s.is_terminal():
            break
        os_t = np.array(os_s.observation_tensor(0), dtype=np.float32)
        fn_t = np.array(fn_s.observation_tensor(0), dtype=np.float32)
        np.testing.assert_array_equal(os_t, fn_t)
        common = list(set(os_s.legal_actions()) & set(fn_s.legal_actions()))
        a = rng.choice(common)
        os_s.apply_action(a)
        fn_s.apply_action(a)


def test_parity_str_during_game(osg, fng):
    rng = random.Random(1)
    os_s = osg.new_initial_state()
    fn_s = fng.new_initial_state()
    for _ in range(40):
        if os_s.is_terminal():
            break
        assert str(os_s).strip() == str(fn_s).strip()
        common = list(set(os_s.legal_actions()) & set(fn_s.legal_actions()))
        a = rng.choice(common)
        os_s.apply_action(a)
        fn_s.apply_action(a)
