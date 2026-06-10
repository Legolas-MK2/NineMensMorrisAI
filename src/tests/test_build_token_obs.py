"""Tests for `build_token_obs` in src/utils.py."""

from __future__ import annotations

import sys
import os

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import numpy as np
import pytest

import fastnmm

from utils import (
    build_token_obs, parse_board_from_state,
    NODE_FEAT_DIM, GLOBAL_FEAT_DIM, NUM_POSITIONS,
)


def test_shapes_and_dtypes():
    game = fastnmm.load_game("nine_mens_morris")
    state = game.new_initial_state()
    nf, gf = build_token_obs(state, 0)
    assert nf.shape == (NUM_POSITIONS, NODE_FEAT_DIM)
    assert gf.shape == (GLOBAL_FEAT_DIM,)
    assert nf.dtype == np.float32
    assert gf.dtype == np.float32


def test_initial_state_global_feats():
    """At the empty initial state: placing phase, full unplaced, zero onboard."""
    game = fastnmm.load_game("nine_mens_morris")
    state = game.new_initial_state()
    _, gf = build_token_obs(state, 0)
    # phase one-hot starts with placing.
    assert gf[0] == 1.0 and gf[1] == 0.0 and gf[2] == 0.0
    # unplaced = 9/9 = 1.0 both, onboard = 0.
    assert gf[3] == 1.0 and gf[4] == 1.0
    assert gf[5] == 0.0 and gf[6] == 0.0
    # no flying.
    assert gf[7] == 0.0 and gf[8] == 0.0
    # turn 0.
    assert gf[9] == 0.0


def test_initial_state_node_feats_all_empty():
    game = fastnmm.load_game("nine_mens_morris")
    state = game.new_initial_state()
    nf, _ = build_token_obs(state, 0)
    # All positions empty -> channel 0 == 1.0 for every row, others zero.
    assert (nf[:, 0] == 1.0).all()
    assert (nf[:, 1] == 0.0).all()
    assert (nf[:, 2] == 0.0).all()


def test_node_feats_match_parse_board():
    """node_feats must agree with parse_board_from_state after a few moves."""
    import random
    rng = random.Random(0)
    game = fastnmm.load_game("nine_mens_morris")
    state = game.new_initial_state()
    for _ in range(6):
        if state.is_terminal():
            break
        legal = state.legal_actions()
        if not legal:
            break
        state.apply_action(rng.choice(legal))

    board = parse_board_from_state(state)
    for player in (0, 1):
        nf, _ = build_token_obs(state, player)
        for p in range(NUM_POSITIONS):
            v = board[p]
            if v is None:
                assert nf[p, 0] == 1.0 and nf[p, 1] == 0.0 and nf[p, 2] == 0.0, (player, p)
            elif v == player:
                assert nf[p, 0] == 0.0 and nf[p, 1] == 1.0 and nf[p, 2] == 0.0, (player, p)
            else:
                assert nf[p, 0] == 0.0 and nf[p, 1] == 0.0 and nf[p, 2] == 1.0, (player, p)


def test_relativization_swap():
    """node_feats viewed by player 0 vs 1 should swap channels 1 and 2."""
    import random
    rng = random.Random(1)
    game = fastnmm.load_game("nine_mens_morris")
    state = game.new_initial_state()
    for _ in range(6):
        legal = state.legal_actions()
        if not legal:
            break
        state.apply_action(rng.choice(legal))

    nf0, _ = build_token_obs(state, 0)
    nf1, _ = build_token_obs(state, 1)
    np.testing.assert_array_equal(nf0[:, 0], nf1[:, 0])
    np.testing.assert_array_equal(nf0[:, 1], nf1[:, 2])
    np.testing.assert_array_equal(nf0[:, 2], nf1[:, 1])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
