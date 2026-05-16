"""
Shared board geometry, action encoding, and observation parsing for the
Nine Men's Morris webui / claude_plays entry points.

Pulled out of webui.py and claude_plays.py — both files previously had their
own (slightly divergent) copies of the constants below.
"""

from __future__ import annotations

from typing import Dict, Any, Sequence

import numpy as np


# 24 board positions → (row, col) on the 7×7 observation grid.
#     0-----------1-----------2
#     |           |           |
#     |     3-----4-----5     |
#     |     |     |     |     |
#     |     |  6--7--8  |     |
#     |     |  |     |  |     |
#     9----10-11    12-13----14
#     |     |  |     |  |     |
#     |     | 15-16-17  |     |
#     |     |     |     |     |
#     |    18----19----20     |
#     |           |           |
#    21----------22----------23
POINT_TO_COORD: Dict[int, tuple] = {
    0: (0, 0),  1: (0, 3),  2: (0, 6),
    3: (1, 1),  4: (1, 3),  5: (1, 5),
    6: (2, 2),  7: (2, 3),  8: (2, 4),
    9: (3, 0), 10: (3, 1), 11: (3, 2),
    12: (3, 4), 13: (3, 5), 14: (3, 6),
    15: (4, 2), 16: (4, 3), 17: (4, 4),
    18: (5, 1), 19: (5, 3), 20: (5, 5),
    21: (6, 0), 22: (6, 3), 23: (6, 6),
}

POSITION_NAMES: Dict[int, str] = {
     0: "outer top-left",   1: "outer top-mid",    2: "outer top-right",
     3: "mid top-left",     4: "mid top-mid",      5: "mid top-right",
     6: "inner top-left",   7: "inner top-mid",    8: "inner top-right",
     9: "outer left-mid",  10: "mid left-mid",    11: "inner left-mid",
    12: "inner right-mid", 13: "mid right-mid",   14: "outer right-mid",
    15: "inner bot-left",  16: "inner bot-mid",   17: "inner bot-right",
    18: "mid bot-left",    19: "mid bot-mid",     20: "mid bot-right",
    21: "outer bot-left",  22: "outer bot-mid",   23: "outer bot-right",
}

MILLS: tuple = (
    (0, 1, 2),   (0, 9, 21),  (2, 14, 23), (21, 22, 23),
    (3, 4, 5),   (3, 10, 18), (5, 13, 20), (18, 19, 20),
    (6, 7, 8),   (6, 11, 15), (8, 12, 17), (15, 16, 17),
    (1, 4, 7),   (9, 10, 11), (12, 13, 14),(16, 19, 22),
)


def parse_board_positions(state, obs_size: int, obs_shape: Sequence[int]) -> Dict[int, int]:
    """Return {board_position: player_id} from the flat observation tensor.

    Observation layout: [ch0 (n_cells), ch1 (n_cells), ...] where ch0 is
    player-0 piece presence on a `grid_rows × grid_cols` grid and ch1 is
    player-1. `obs_size` is the flat length and `obs_shape` is
    `[channels, grid_rows, grid_cols]` (e.g. `[5, 7, 7]` for nine_mens_morris).
    """
    obs = np.array(state.observation_tensor(0))
    n_cells = obs_size // obs_shape[0]
    cols = obs_shape[2]
    positions: Dict[int, int] = {}
    for pos, (row, col) in POINT_TO_COORD.items():
        idx = row * cols + col
        if obs[idx] == 1:
            positions[pos] = 0
        elif obs[n_cells + idx] == 1:
            positions[pos] = 1
    return positions


def decode_action(action: int, is_capture_phase: bool = False) -> Dict[str, Any]:
    """Decode an action ID into a {"type": ..., ...} descriptor.

    Action layout for nine_mens_morris:
        0-23  → placement (or capture during capture phase, same encoding)
        24+   → movement: 24 + from*24 + to
    """
    if action < 24:
        kind = "capture" if is_capture_phase else "place"
        return {"type": kind, "position": int(action)}
    offset = action - 24
    return {"type": "move", "from": int(offset // 24), "to": int(offset % 24)}


def encode_action(action_info: Dict[str, Any]) -> int:
    """Inverse of decode_action — collapses {"type": ..., ...} → action ID."""
    t = action_info["type"]
    if t == "place" or t == "capture":
        return int(action_info["position"])
    if t == "move":
        return 24 + int(action_info["from"]) * 24 + int(action_info["to"])
    raise ValueError(f"unknown action type: {t!r}")
