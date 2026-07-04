"""
Shared board geometry, action encoding, and observation parsing for the
Nine Men's Morris webui / claude_plays entry points.

Pulled out of webui.py and claude_plays.py — both files previously had their
own (slightly divergent) copies of the constants below.
"""

from __future__ import annotations

from typing import Dict, Any, Sequence

import numpy as np

# Canonical geometry lives in utils.py (single source of truth, shared with
# the training pipeline); this module only re-shapes it for the web/agent
# entry points. MILLS is re-exported for those entry points.
from utils import MILLS, BOARD_POS_TO_GRID

__all__ = [
    "MILLS", "POINT_TO_COORD", "POSITION_NAMES",
    "parse_board_positions", "decode_action",
]


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
POINT_TO_COORD: Dict[int, tuple] = dict(enumerate(BOARD_POS_TO_GRID))

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
