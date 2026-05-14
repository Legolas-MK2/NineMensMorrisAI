"""Basic tests for the fastnmm package."""
import copy
import pytest

import fastnmm


@pytest.fixture
def game():
    return fastnmm.load_game("nine_mens_morris")


@pytest.fixture
def init(game):
    return game.new_initial_state()


# ---------------------------------------------------------------------------
# Package surface
# ---------------------------------------------------------------------------

def test_load_game_returns_game(game):
    assert isinstance(game, fastnmm.Game)
    assert game.name == "nine_mens_morris"


def test_load_game_aliases():
    # Any of these should work.
    for alias in ("nine_mens_morris", "Nine_Mens_Morris", "morris", "mühle", "mill"):
        assert isinstance(fastnmm.load_game(alias), fastnmm.Game)


def test_load_game_unknown_raises():
    with pytest.raises(ValueError):
        fastnmm.load_game("chess")


def test_game_metadata(game):
    assert game.num_distinct_actions() == 600
    assert game.num_players() == 2
    assert game.min_utility() == -1.0
    assert game.max_utility() == 1.0
    assert game.utility_sum() == 0.0
    assert game.observation_tensor_shape() == [5, 7, 7]
    assert game.observation_tensor_size() == 5 * 7 * 7


def test_public_constants():
    assert fastnmm.NUM_POSITIONS == 24
    assert fastnmm.NUM_PLAYERS == 2
    assert fastnmm.PIECES_PER_PLAYER == 9
    assert fastnmm.NUM_DISTINCT_ACTIONS == 600
    assert fastnmm.MOVE_ACTION_OFFSET == 24
    assert fastnmm.OBSERVATION_PLANES == 5
    assert fastnmm.OBSERVATION_ROWS == 7
    assert fastnmm.OBSERVATION_COLS == 7


# ---------------------------------------------------------------------------
# Action encoding
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("p", range(24))
def test_place_action_encoding(p):
    assert not fastnmm.is_move_action(p)


def test_move_action_encoding_round_trip():
    for f in range(24):
        for t in range(24):
            a = fastnmm.encode_move(f, t)
            assert a == 24 + f * 24 + t
            assert fastnmm.is_move_action(a)
            assert fastnmm.move_from(a) == f
            assert fastnmm.move_to(a) == t


def test_action_to_string(game):
    assert game.action_to_string(0, 0) == "Point 0"
    assert game.action_to_string(0, 23) == "Point 23"
    assert game.action_to_string(0, 24) == "Move 0 -> 0"
    assert game.action_to_string(0, 25) == "Move 0 -> 1"
    assert game.action_to_string(1, 599) == "Move 23 -> 23"


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

def test_initial_state(init):
    assert not init.is_terminal()
    assert init.current_player() == 0
    assert init.move_number() == 0
    assert init.history() == []
    assert init.men_to_deploy(0) == 9
    assert init.men_to_deploy(1) == 9
    assert init.men_on_board(0) == 0
    assert init.men_on_board(1) == 0
    assert init.current_phase() == fastnmm.Phase.PLACING
    assert init.turn() == 0
    assert init.winner() == -1


def test_initial_legal_actions_are_all_points(init):
    assert init.legal_actions() == list(range(24))


def test_initial_returns_and_rewards(init):
    assert init.returns() == [0.0, 0.0]
    assert init.rewards() == [0.0, 0.0]


# ---------------------------------------------------------------------------
# Apply single action
# ---------------------------------------------------------------------------

def test_apply_place_action_updates_state(init):
    init.apply_action(0)
    assert init.men_to_deploy(0) == 8
    assert init.men_on_board(0) == 1
    assert init.current_player() == 1
    assert init.history() == [0]
    assert init.move_number() == 1
    assert init.turn() == 1
    # Bitboard has bit 0 set.
    assert init.board_bitboard(0) == 1
    assert init.board_bitboard(1) == 0


def test_place_on_occupied_raises(init):
    init.apply_action(0)
    with pytest.raises(Exception):
        init.apply_action(0)


def test_apply_action_on_terminal_raises(game):
    """Forcibly drive to terminal with a mill + repeated captures."""
    s = game.new_initial_state()
    # W forms mill [0,1,2], then captures B's piece.
    seq = [1, 5, 2, 13, 0, 5]  # last: W captures B's 5... actually need valid moves
    # Easier: just run random until terminal.
    import random
    rng = random.Random(0)
    while not s.is_terminal():
        s.apply_action(rng.choice(s.legal_actions()))
    with pytest.raises(Exception):
        s.apply_action(0)


# ---------------------------------------------------------------------------
# Clone / deepcopy / serialization
# ---------------------------------------------------------------------------

def test_clone_is_independent(init):
    init.apply_action(0)
    init.apply_action(5)
    cloned = init.clone()
    cloned.apply_action(10)
    assert init.history() == [0, 5]
    assert cloned.history() == [0, 5, 10]


def test_deepcopy_is_independent(init):
    init.apply_action(0)
    c = copy.deepcopy(init)
    c.apply_action(5)
    assert init.history() == [0]
    assert c.history() == [0, 5]


def test_serialize_roundtrip(game, init):
    init.apply_action(0)
    init.apply_action(2)
    init.apply_action(5)
    data = init.serialize()
    restored = game.deserialize_state(data)
    assert restored.history() == init.history()
    assert str(restored) == str(init)


# ---------------------------------------------------------------------------
# String rendering
# ---------------------------------------------------------------------------

def test_initial_str_layout_shape(init):
    lines = str(init).split("\n")
    # First 13 lines are the board (13 visual rows), followed by status block.
    assert lines[0] == ".------.------."
    assert lines[12] == ".------.------."
    assert any("Current player: W" in l for l in lines)
    assert any("Turn number: 0" in l for l in lines)
    assert any("Men to deploy: 9 9" in l for l in lines)
    assert any("Num men: 9 9" in l for l in lines)


def test_str_reflects_white_placement(init):
    init.apply_action(0)
    assert str(init).startswith("W------.------.")


def test_str_reflects_black_placement(init):
    init.apply_action(0)
    init.apply_action(5)
    lines = str(init).split("\n")
    assert "B" in lines[2]  # row where point 5 is drawn


# ---------------------------------------------------------------------------
# Observation tensor
# ---------------------------------------------------------------------------

def test_observation_tensor_size_and_values(init):
    tensor = init.observation_tensor(0)
    assert len(tensor) == 5 * 7 * 7
    assert all(v in (0.0, 1.0) for v in tensor)


def test_observation_tensor_numpy_shape(init):
    import numpy as np
    arr = init.observation_tensor_numpy(0)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (5, 7, 7)
    assert arr.dtype == np.float32


def test_observation_tensor_pieces_planes(init):
    import numpy as np
    arr = init.observation_tensor_numpy(0)
    # Planes 0 (white) and 1 (black) start empty.
    assert arr[0].sum() == 0
    assert arr[1].sum() == 0
    init.apply_action(0)  # W places at 0
    arr = init.observation_tensor_numpy(0)
    assert arr[0].sum() == 1
    assert arr[1].sum() == 0
    init.apply_action(5)  # B places at 5
    arr = init.observation_tensor_numpy(0)
    assert arr[0].sum() == 1
    assert arr[1].sum() == 1


def test_observation_tensor_structural_planes_constant(init):
    """Planes 3 and 4 are constant structural masks (edges of the board)."""
    import numpy as np
    arr1 = init.observation_tensor_numpy(0)
    init.apply_action(0)
    init.apply_action(5)
    init.apply_action(2)
    arr2 = init.observation_tensor_numpy(0)
    np.testing.assert_array_equal(arr1[3], arr2[3])
    np.testing.assert_array_equal(arr1[4], arr2[4])
