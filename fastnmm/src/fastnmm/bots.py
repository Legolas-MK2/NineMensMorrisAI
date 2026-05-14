"""Bots for AI training and evaluation.

Every bot is a lightweight object with a single required method::

    bot.step(state) -> int

which returns a legal action for the given state. Bots may hold internal
state (RNG, evaluator caches, …) and expose it; bots are *owned by a
player*, stored in ``bot.player_id``.

A bot can be used two ways:

1. **Standalone**, by calling ``bot.step(state)`` directly from your own
   game loop.  This is the intended style when you want full control
   (e.g. training an RL agent that calls a minimax opponent once per
   turn).

2. **Auto-play**, by handing two bots to :func:`play_match`, which
   runs a complete game between them and returns the terminal returns.

The two built-in bots are :class:`RandomBot` and :class:`MinimaxBot`.
Any object with a ``step(state) -> int`` method also works -- just
implement your own.
"""
from __future__ import annotations

import random
from typing import List, Optional, Sequence, Tuple, Union

from . import _core


StartingStones = Union[Tuple[int, int], List[int], None]


# ---------------------------------------------------------------------------
# Bot protocol
# ---------------------------------------------------------------------------


class Bot:
    """Abstract base.  Override :meth:`step`.

    Any duck-typed object with a ``step(state) -> int`` attribute works
    in :func:`play_match` -- subclassing is not required.
    """

    player_id: int = -1

    def step(self, state: "_core.State") -> int:   # pragma: no cover
        raise NotImplementedError

    def restart(self) -> None:
        """Reset per-game state (e.g. search caches). Default: no-op."""

    def restart_at(self, state: "_core.State") -> None:  # pragma: no cover
        """OpenSpiel-compat alias for :meth:`restart`."""
        self.restart()


# ---------------------------------------------------------------------------
# Random bot
# ---------------------------------------------------------------------------


class RandomBot(Bot):
    """Picks a legal action uniformly at random.

    Parameters
    ----------
    player_id:
        Which player (0 = White, 1 = Black) this bot plays.  Needed
        only to disambiguate when used inside :func:`play_match`; the
        bot itself simply samples from ``state.legal_actions()``.
    seed:
        Optional integer seed for reproducibility.  If ``None``, the
        default Python ``random`` RNG is used.
    """

    def __init__(self, player_id: int, seed: Optional[int] = None):
        self.player_id = int(player_id)
        self._seed = seed
        self._rng = random.Random(seed)

    def step(self, state: "_core.State") -> int:
        return self._rng.choice(state.legal_actions())

    def restart(self) -> None:
        self._rng = random.Random(self._seed)

    def __repr__(self) -> str:
        return f"RandomBot(player_id={self.player_id}, seed={self._seed})"


# ---------------------------------------------------------------------------
# Minimax bot
# ---------------------------------------------------------------------------


_DEFAULT_TT_BYTES = 256 * 1024 * 1024  # 256 MiB


class MinimaxBot(Bot):
    """Alpha-beta negamax with a material/mobility/mill heuristic.

    The search runs in C++ and is backed by a persistent transposition
    table (Zobrist hashing). The TT lives on the bot, so consecutive
    ``step()`` calls reuse cached sub-results from previous searches.

    Parameters
    ----------
    player_id:
        0 = White, 1 = Black. Used only by :func:`play_match`.
    depth:
        Ply depth for the search.  ``4`` is a reasonable default; ``6``
        is strong but noticeably slower in endgames.  Note: in
        Nine Men's Morris a "ply" is a single engine-level action, and
        *captures are additional plies* -- a mill-forming move plus the
        capture counts as 2 plies, so effective "move depth" is roughly
        ``depth / 1.2``.
    tt_bytes:
        Approximate transposition-table size in bytes. Rounded down to
        the largest power-of-two count of 16-byte entries. Defaults to
        256 MiB. With plenty of RAM you can pass much larger values
        (e.g. ``2 * 1024**3`` for 2 GiB; the engine supports up to
        ~128 GiB).
    strict_parity:
        ``True`` (default): bit-exact match with
        :func:`fastnmm.minimax_search`. ``False``: position-relative
        mate-distance scoring (kept as an opt-in; gives the same hit
        rate as strict in normal forward minimax).
    tiebreak_seed:
        Reserved for future use; kept for API compatibility.
    """

    def __init__(
        self,
        player_id: int,
        depth: int = 4,
        tt_bytes: int = _DEFAULT_TT_BYTES,
        strict_parity: bool = True,
        root_cache: Optional["_core.SharedMoveCache"] = None,
        tiebreak_seed: Optional[int] = None,
    ):
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.player_id = int(player_id)
        self.depth = int(depth)
        self._tiebreak_seed = tiebreak_seed
        self._rng = random.Random(tiebreak_seed)
        self.engine = _core.MinimaxEngine(int(tt_bytes), bool(strict_parity))
        if root_cache is not None:
            self.engine.set_root_cache(root_cache)
        # Hold a Python reference so the cache outlives the engine.
        self._root_cache = root_cache
        self.last_search: Optional["_core.SearchResult"] = None

    def step(self, state: "_core.State") -> int:
        result = self.engine.search(state, self.depth)
        self.last_search = result
        return result.action

    def evaluate(self, state: "_core.State") -> int:
        """Run a search and return its score from the state-player's view."""
        return self.engine.search(state, self.depth).score

    def restart(self) -> None:
        # Bump generation so old entries become preferred for replacement,
        # but keep useful sub-trees that may still apply.
        self.engine.new_game()
        self._rng = random.Random(self._tiebreak_seed)
        self.last_search = None

    @property
    def tt_stats(self) -> dict:
        e = self.engine
        probes = e.tt_probes
        return {
            "entries":    e.tt_num_entries,
            "bytes":      e.tt_bytes,
            "probes":     probes,
            "hits":       e.tt_hits,
            "stores":     e.tt_stores,
            "collisions": e.tt_collisions,
            "hit_rate":   (e.tt_hits / probes) if probes > 0 else 0.0,
            "fill":       e.tt_fill_fraction,
        }

    def __repr__(self) -> str:
        return (
            f"MinimaxBot(player_id={self.player_id}, "
            f"depth={self.depth}, "
            f"tt_bytes={self.engine.tt_bytes})"
        )


# ---------------------------------------------------------------------------
# Factory helpers matching OpenSpiel's naming
# ---------------------------------------------------------------------------


def make_uniform_random_bot(player_id: int, seed: Optional[int] = None) -> RandomBot:
    """OpenSpiel-compatible factory for :class:`RandomBot`."""
    return RandomBot(player_id=player_id, seed=seed)


def make_minimax_bot(player_id: int, depth: int = 4) -> MinimaxBot:
    """Factory for :class:`MinimaxBot`."""
    return MinimaxBot(player_id=player_id, depth=depth)


# ---------------------------------------------------------------------------
# Match runner
# ---------------------------------------------------------------------------


def _validate_stones(starting_stones: StartingStones) -> Optional[Tuple[int, int]]:
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


def play_match(
    bots: Sequence[Bot],
    starting_stones: StartingStones = None,
    max_turns: Optional[int] = None,
    verbose: bool = False,
) -> List[float]:
    """Play one full game between two bots and return the terminal returns.

    Parameters
    ----------
    bots:
        Two-element sequence of objects exposing ``step(state) -> int``.
        The first plays White (player 0), the second plays Black
        (player 1).  You can freely mix -- e.g.
        ``[RandomBot(0), MinimaxBot(1, depth=4)]`` or even two custom
        objects.
    starting_stones:
        Optional ``(white, black)`` pair overriding the default
        ``(9, 9)``.  Both values must be in ``[1, 9]``.
    max_turns:
        Optional override for the turn-based draw threshold.  Default
        is :data:`~fastnmm._core.DEFAULT_MAX_TURNS` (200), matching
        OpenSpiel's behaviour.
    verbose:
        If true, print each action and the final state.

    Returns
    -------
    A 2-element list of floats: ``[return_for_white, return_for_black]``.
    Each value is ``+1.0`` for a win, ``-1.0`` for a loss, or ``0.0``
    for a draw.
    """
    if len(bots) != 2:
        raise ValueError(f"expected 2 bots; got {len(bots)}")
    stones = _validate_stones(starting_stones)
    state = (
        _core.new_initial_state_with_stones(*stones)
        if stones is not None
        else _core.new_initial_state()
    )
    if max_turns is not None:
        state.set_max_turns(int(max_turns))

    for bot in bots:
        if hasattr(bot, "restart"):
            bot.restart()

    while not state.is_terminal():
        cp = state.current_player()
        action = bots[cp].step(state)
        if verbose:
            print(f"player {cp} plays {_core.Game.action_to_string(cp, action)}")
        state.apply_action(action)

    if verbose:
        print(state)
        print("returns:", state.returns())
    return list(state.returns())


def play_matches(
    bots: Sequence[Bot],
    num_games: int,
    starting_stones: StartingStones = None,
    max_turns: Optional[int] = None,
    swap_colors: bool = True,
) -> dict:
    """Play ``num_games`` games between two bots and return summary stats.

    When ``swap_colors`` is True (default), games alternate which bot
    plays White so colour advantage is averaged out.

    Returns a dict with keys:
      ``wins`` (list of 2 ints: wins for bots[0] and bots[1]),
      ``draws`` (int),
      ``returns`` (list of per-game [r0, r1] pairs).
    """
    wins = [0, 0]
    draws = 0
    returns = []
    for g in range(num_games):
        pair = list(bots)
        flipped = swap_colors and (g % 2 == 1)
        if flipped:
            pair = [bots[1], bots[0]]
        r = play_match(pair, starting_stones=starting_stones, max_turns=max_turns)
        if flipped:
            r = [r[1], r[0]]  # map back to original-bot indexing
        returns.append(r)
        if r[0] > 0:
            wins[0] += 1
        elif r[1] > 0:
            wins[1] += 1
        else:
            draws += 1
    return {"wins": wins, "draws": draws, "returns": returns}
