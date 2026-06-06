"""fastnmm -- an extremely fast Nine Men's Morris engine.

Public surface mirrors OpenSpiel's `pyspiel` API for the
``nine_mens_morris`` game, plus training-oriented extensions:

* configurable starting-stone counts (``starting_stones=(5, 6)``), and
* built-in :class:`RandomBot` and :class:`MinimaxBot` opponents.

Quick start::

    import fastnmm
    game  = fastnmm.load_game("nine_mens_morris")
    state = game.new_initial_state()

    while not state.is_terminal():
        state.apply_action(state.legal_actions()[0])
    print(state.returns())

Training usage::

    import fastnmm
    from fastnmm.bots import RandomBot, MinimaxBot, play_match

    # Pit minimax (depth 4) vs random, with reduced starting stones:
    returns = play_match(
        [MinimaxBot(player_id=0, depth=4), RandomBot(player_id=1, seed=42)],
        starting_stones=(5, 6),
    )

Bots are also usable stand-alone in your own training loop::

    minimax = MinimaxBot(player_id=0, depth=4)
    state = fastnmm.load_game("nine_mens_morris",
                              starting_stones=(5, 6)).new_initial_state()
    while not state.is_terminal():
        if state.current_player() == 0:
            action = minimax.step(state)       # engine picks
        else:
            action = your_agent.act(state)     # your model picks
        state.apply_action(action)
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

from . import _core
from ._core import (
    NUM_POSITIONS,
    NUM_PLAYERS,
    PIECES_PER_PLAYER,
    NUM_DISTINCT_ACTIONS,
    MOVE_ACTION_OFFSET,
    OBSERVATION_PLANES,
    OBSERVATION_ROWS,
    OBSERVATION_COLS,
    OBSERVATION_SIZE,
    MAX_GAME_LENGTH,
    DEFAULT_MAX_TURNS,
    Phase,
    State,
    SearchResult,
    MinimaxEngine,
    OpponentKind,
    encode_move,
    move_from,
    move_to,
    is_move_action,
    random_playouts,
    evaluate,
    minimax_action,
    minimax_search,
    zobrist_key,
    play_until_player,
)
from .bots import (
    Bot,
    RandomBot,
    MinimaxBot,
    make_uniform_random_bot,
    make_minimax_bot,
    play_match,
    play_matches,
    StartingStones,
)

__all__ = [
    "load_game",
    "Game",
    "State",
    "Phase",
    "SearchResult",
    "MinimaxEngine",
    "OpponentKind",
    "encode_move",
    "move_from",
    "move_to",
    "is_move_action",
    "random_playouts",
    "evaluate",
    "minimax_action",
    "minimax_search",
    "zobrist_key",
    "play_until_player",
    "Bot",
    "RandomBot",
    "MinimaxBot",
    "make_uniform_random_bot",
    "make_minimax_bot",
    "play_match",
    "play_matches",
    "StartingStones",
    "NUM_POSITIONS",
    "NUM_PLAYERS",
    "PIECES_PER_PLAYER",
    "NUM_DISTINCT_ACTIONS",
    "MOVE_ACTION_OFFSET",
    "OBSERVATION_PLANES",
    "OBSERVATION_ROWS",
    "OBSERVATION_COLS",
    "OBSERVATION_SIZE",
    "MAX_GAME_LENGTH",
    "DEFAULT_MAX_TURNS",
]

__version__ = "0.2.0"


# ---------------------------------------------------------------------------
# Game wrapper with training-oriented extensions
# ---------------------------------------------------------------------------


def _coerce_stones(
    starting_stones: Union[Tuple[int, int], List[int], None],
) -> Optional[Tuple[int, int]]:
    if starting_stones is None:
        return None
    if len(starting_stones) != 2:
        raise ValueError(
            "starting_stones must be a 2-element tuple/list like (5, 6)"
        )
    a, b = int(starting_stones[0]), int(starting_stones[1])
    if not (1 <= a <= 9 and 1 <= b <= 9):
        raise ValueError(
            f"starting_stones values must be in [1, 9]; got {starting_stones!r}"
        )
    return (a, b)


class Game:
    """OpenSpiel-compatible game object with a training-friendly constructor.

    Parameters
    ----------
    starting_stones:
        Optional ``(white, black)`` tuple / list giving the number of
        stones each player begins with.  Both values must be in
        ``[1, 9]``; default is ``(9, 9)``.
    """

    name = "nine_mens_morris"

    def __init__(
        self,
        starting_stones: Union[Tuple[int, int], List[int], None] = None,
    ) -> None:
        self._starting_stones = _coerce_stones(starting_stones)

    # --- state factory --------------------------------------------------
    def new_initial_state(
        self,
        starting_stones: Union[Tuple[int, int], List[int], None] = None,
    ) -> State:
        """Create a fresh initial state.

        ``starting_stones`` overrides the game-level default for this
        call only.  Pass ``None`` to use the default configured on the
        :class:`Game` (or ``(9, 9)`` if none was set).
        """
        override = _coerce_stones(starting_stones)
        stones = override if override is not None else self._starting_stones
        if stones is None:
            return _core.new_initial_state()
        return _core.new_initial_state_with_stones(*stones)

    # --- OpenSpiel-style game metadata ---------------------------------
    def num_distinct_actions(self) -> int:
        return _core.Game.num_distinct_actions()

    def num_players(self) -> int:
        return _core.Game.num_players()

    def min_utility(self) -> float:
        return _core.Game.min_utility()

    def max_utility(self) -> float:
        return _core.Game.max_utility()

    def utility_sum(self) -> float:
        return _core.Game.utility_sum()

    def max_game_length(self) -> int:
        return _core.Game.max_game_length()

    def observation_tensor_shape(self) -> List[int]:
        return _core.Game.observation_tensor_shape()

    def observation_tensor_size(self) -> int:
        return _core.Game.observation_tensor_size()

    def action_to_string(self, player: int, action: int) -> str:
        return _core.Game.action_to_string(player, action)

    def deserialize_state(self, data: str) -> State:
        return _core.Game.deserialize_state(data)

    @property
    def starting_stones(self) -> Tuple[int, int]:
        """The ``(white, black)`` starting-stone pair used by this Game."""
        return self._starting_stones if self._starting_stones is not None else (9, 9)

    def __repr__(self) -> str:
        if self._starting_stones is None:
            return "<fastnmm.Game 'nine_mens_morris'>"
        return (
            f"<fastnmm.Game 'nine_mens_morris' "
            f"starting_stones={self._starting_stones}>"
        )


def load_game(
    name: str = "nine_mens_morris",
    starting_stones: Union[Tuple[int, int], List[int], None] = None,
) -> Game:
    """OpenSpiel-compatible loader with an optional ``starting_stones`` kwarg.

    Only the ``nine_mens_morris`` game (with several friendly aliases)
    is supported; anything else raises :class:`ValueError`.
    """
    canonical = name.strip().lower()
    aliases = {
        "nine_mens_morris",
        "nine_men_morris",
        "ninemensmorris",
        "morris",
        "muehle",
        "mühle",
        "mill",
    }
    if canonical not in aliases:
        raise ValueError(
            f"fastnmm only supports 'nine_mens_morris' (got {name!r})."
        )
    return Game(starting_stones=starting_stones)


def play_random_game(seed: int = 0) -> Sequence[float]:
    """Play one full random game and return the terminal returns."""
    import random as _random
    rng = _random.Random(seed)
    state = _core.new_initial_state()
    while not state.is_terminal():
        state.apply_action(rng.choice(state.legal_actions()))
    return state.returns()
