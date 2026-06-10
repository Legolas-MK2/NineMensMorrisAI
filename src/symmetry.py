"""
Board-symmetry automorphism group for Nine Men's Morris.

The 24-point board graph has an order-16 automorphism group:
    D4 (rotations + reflections of the 7x7 grid, 8 elements)
  x  Z2 (inner-ring <-> outer-ring swap)

A position and its image under any of these 16 permutations are
strategically identical, so applying a random sigma to (obs, mask, action)
during PPO rollout gives 16x effective position diversity without changing
the model or the loss.

Public surface (all cached at import time):
    AUTOMORPHISMS:        List[np.ndarray[int64, 24]]   -- 16 board permutations
    INV_AUTOMORPHISMS:    List[np.ndarray[int64, 24]]   -- their inverses
    ACT_PERM:             List[np.ndarray[int64, 600]]  -- forward action perm
    ACT_PERM_INV:         List[np.ndarray[int64, 600]]  -- inverse action perm
    CELL_SRC:             List[np.ndarray[int64, 49]]   -- dest->source gather

    permute_obs(obs_flat, sigma_idx)   -> np.ndarray (245,)
    permute_mask(mask, sigma_idx)      -> np.ndarray (600,)
    apply_act_perm(action, sigma_idx)  -> int  (canonical -> frame sigma)
    apply_act_perm_inv(action, sigma_idx) -> int  (frame sigma -> canonical)

Convention: in frame sigma, the content at canonical board position p
appears at position sigma[p]. This matches the action permutation
(ACT_PERM[sigma_idx][place_p] = place_{sigma[p]}).
"""

from __future__ import annotations

from typing import List, Tuple
import numpy as np

from utils import ADJACENCY, MILLS, BOARD_POS_TO_GRID


N_POSITIONS = 24
N_GRID_CELLS = 49        # 7 x 7
N_ACTIONS = 600          # 24 places/captures + 24*24 moves
GRID_ROWS = 7
GRID_COLS = 7

# Canonical edge / mill sets used by validation.
_CANON_EDGES = frozenset(
    frozenset((u, v)) for u, nbrs in enumerate(ADJACENCY) for v in nbrs
)
_CANON_MILLS = frozenset(frozenset(m) for m in MILLS)


# ----- Generator construction ------------------------------------------------

def _coord_to_pos_map() -> dict:
    return {coord: p for p, coord in enumerate(BOARD_POS_TO_GRID)}


def _grid_action_lift(grid_perm) -> np.ndarray:
    """Lift a 7x7 grid permutation (function on (r,c)) to a 24-point board permutation.

    grid_perm: callable (r, c) -> (r', c').
    Returns array sigma where sigma[p] is the image of board position p.
    """
    coord_to_pos = _coord_to_pos_map()
    sigma = np.empty(N_POSITIONS, dtype=np.int64)
    for p, (r, c) in enumerate(BOARD_POS_TO_GRID):
        r2, c2 = grid_perm(r, c)
        if (r2, c2) not in coord_to_pos:
            raise RuntimeError(
                f"grid permutation sends board position {p} at {(r, c)} to "
                f"non-board cell {(r2, c2)}"
            )
        sigma[p] = coord_to_pos[(r2, c2)]
    return sigma


def _build_rot90() -> np.ndarray:
    """90 deg CCW rotation about grid center (3, 3): (r, c) -> (c, 6 - r)."""
    return _grid_action_lift(lambda r, c: (c, GRID_COLS - 1 - r))


def _build_reflect_v() -> np.ndarray:
    """Vertical reflection about column 3: (r, c) -> (r, 6 - c)."""
    return _grid_action_lift(lambda r, c: (r, GRID_COLS - 1 - c))


def _build_ring_swap() -> np.ndarray:
    """Inner ring <-> outer ring swap; mid-ring fixed.

    Outer-ring positions have max(|dr|, |dc|) == 3 from center (3, 3); inner-ring
    have max == 1; mid-ring have max == 2. Pairs are matched by direction-signum,
    so a position at offset (sign_r*1, sign_c*1) swaps with (sign_r*3, sign_c*3).
    """
    coord_to_pos = _coord_to_pos_map()
    sigma = np.arange(N_POSITIONS, dtype=np.int64)
    cr, cc = 3, 3
    for p, (r, c) in enumerate(BOARD_POS_TO_GRID):
        dr, dc = r - cr, c - cc
        ring = max(abs(dr), abs(dc))
        if ring == 2:
            continue  # mid-ring fixed
        if ring not in (1, 3):
            raise RuntimeError(f"unexpected ring for position {p}: max-offset {ring}")
        sign_r = (dr > 0) - (dr < 0)
        sign_c = (dc > 0) - (dc < 0)
        new_ring = 1 if ring == 3 else 3
        target = (cr + new_ring * sign_r, cc + new_ring * sign_c)
        if target not in coord_to_pos:
            raise RuntimeError(
                f"ring swap of position {p} at {(r, c)} -> {target} is not a board cell"
            )
        sigma[p] = coord_to_pos[target]
    return sigma


# ----- Validation and closure ------------------------------------------------

def _is_permutation_of_range(arr: np.ndarray, n: int) -> bool:
    return arr.shape == (n,) and set(arr.tolist()) == set(range(n))


def _preserves_adjacency(sigma: np.ndarray) -> bool:
    image = frozenset(
        frozenset((int(sigma[u]), int(sigma[v])))
        for u, nbrs in enumerate(ADJACENCY)
        for v in nbrs
    )
    return image == _CANON_EDGES


def _preserves_mills(sigma: np.ndarray) -> bool:
    image = frozenset(frozenset(int(sigma[p]) for p in m) for m in MILLS)
    return image == _CANON_MILLS


def _validate_generator(sigma: np.ndarray, name: str) -> None:
    if not _is_permutation_of_range(sigma, N_POSITIONS):
        raise RuntimeError(f"generator {name!r} is not a permutation of 0..23: {sigma}")
    if not _preserves_adjacency(sigma):
        raise RuntimeError(f"generator {name!r} does not preserve ADJACENCY")
    if not _preserves_mills(sigma):
        raise RuntimeError(f"generator {name!r} does not preserve MILLS")


def _compose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return permutation c with c[i] = a[b[i]]."""
    return a[b]


def _invert(sigma: np.ndarray) -> np.ndarray:
    inv = np.empty_like(sigma)
    inv[sigma] = np.arange(sigma.shape[0], dtype=sigma.dtype)
    return inv


def _close_group(generators: List[np.ndarray]) -> List[np.ndarray]:
    """BFS closure: keep composing generators until no new permutations appear."""
    identity = np.arange(N_POSITIONS, dtype=np.int64)
    seen = {identity.tobytes(): identity}
    frontier = [identity]
    while frontier:
        next_frontier = []
        for g in frontier:
            for h in generators:
                prod = _compose(h, g)
                key = prod.tobytes()
                if key not in seen:
                    seen[key] = prod
                    next_frontier.append(prod)
        frontier = next_frontier
    return list(seen.values())


# ----- Build everything at import time --------------------------------------

_GENERATORS = [
    ("rot90", _build_rot90()),
    ("reflect_v", _build_reflect_v()),
    ("ring_swap", _build_ring_swap()),
]
for _name, _g in _GENERATORS:
    _validate_generator(_g, _name)

AUTOMORPHISMS: List[np.ndarray] = _close_group([g for _, g in _GENERATORS])

# Order-16 expected for the standard board (no diagonals). If the engine ever
# grows diagonal mills, ring_swap stops preserving adjacency and validation
# above fails first; this assert is a safety net for the no-diagonal case.
assert len(AUTOMORPHISMS) == 16, (
    f"expected automorphism group of order 16, got {len(AUTOMORPHISMS)}. "
    "If the board has diagonals, expected size is 8 and generators must change."
)

# Re-validate every closed element (cheap and catches BFS bugs).
for _idx, _sigma in enumerate(AUTOMORPHISMS):
    assert _is_permutation_of_range(_sigma, N_POSITIONS), f"element {_idx} not a permutation"
    assert _preserves_adjacency(_sigma), f"element {_idx} breaks adjacency"
    assert _preserves_mills(_sigma), f"element {_idx} breaks mills"

# Identity must be present (so sigma=0 in eval is exact pass-through).
_IDENTITY = np.arange(N_POSITIONS, dtype=np.int64)
assert any(np.array_equal(s, _IDENTITY) for s in AUTOMORPHISMS), "identity missing from group"

INV_AUTOMORPHISMS: List[np.ndarray] = [_invert(s) for s in AUTOMORPHISMS]


def _build_act_perm(sigma: np.ndarray) -> np.ndarray:
    """Forward action permutation: ACT_PERM[sigma][a_canon] = a in frame sigma."""
    out = np.empty(N_ACTIONS, dtype=np.int64)
    # Places / captures: 0..23
    for a in range(N_POSITIONS):
        out[a] = int(sigma[a])
    # Moves: 24 + f*24 + t
    for f in range(N_POSITIONS):
        sf = int(sigma[f])
        base = 24 + f * 24
        for t in range(N_POSITIONS):
            out[base + t] = 24 + sf * 24 + int(sigma[t])
    return out


ACT_PERM: List[np.ndarray] = [_build_act_perm(s) for s in AUTOMORPHISMS]
ACT_PERM_INV: List[np.ndarray] = [_invert(p) for p in ACT_PERM]

for _idx, (_p, _pi) in enumerate(zip(ACT_PERM, ACT_PERM_INV)):
    assert _p.shape == (N_ACTIONS,) and _pi.shape == (N_ACTIONS,)
    assert np.array_equal(_p[_pi], np.arange(N_ACTIONS)), f"act_perm {_idx} round-trip failed"


def _build_cell_src(sigma: np.ndarray) -> np.ndarray:
    """Dest->source gather index for the 49-cell grid.

    In frame sigma the content at canonical board position p appears at sigma[p],
    so out[grid_of(sigma[p])] = src[grid_of(p)]. Non-board cells map to themselves.
    """
    cell_src = np.arange(N_GRID_CELLS, dtype=np.int64)
    for p in range(N_POSITIONS):
        sp = int(sigma[p])
        r_dst, c_dst = BOARD_POS_TO_GRID[sp]
        r_src, c_src = BOARD_POS_TO_GRID[p]
        cell_src[r_dst * GRID_COLS + c_dst] = r_src * GRID_COLS + c_src
    return cell_src


CELL_SRC: List[np.ndarray] = [_build_cell_src(s) for s in AUTOMORPHISMS]


# ----- Public helpers --------------------------------------------------------

def permute_token_obs(node_feats: np.ndarray, sigma_idx: int) -> np.ndarray:
    """Permute a [24, F] token observation into frame sigma.

    Content at canonical board position p must appear at row sigma[p] of the
    output. With dest->source gather: out[r] = node_feats[INV[sigma][r]], so
    we index rows by the inverse permutation.

    `global_feats` is symmetry-invariant -- do NOT pass it through this function.
    """
    if sigma_idx == 0:
        return node_feats
    return node_feats[INV_AUTOMORPHISMS[sigma_idx]]


def permute_obs(obs_flat: np.ndarray, sigma_idx: int) -> np.ndarray:
    """Permute a flat (5,7,7) observation into frame sigma.

    All 5 channels are permuted by the same cell-level gather. Channels 0/1
    (piece presence) are the only ones with board-cell content; channels 2-4
    are either structural masks on non-board cells (which the gather leaves
    fixed) or repeated scalars, so they pass through invariant either way.
    """
    cell_src = CELL_SRC[sigma_idx]
    return obs_flat.reshape(5, N_GRID_CELLS)[:, cell_src].reshape(-1)


def permute_mask(mask: np.ndarray, sigma_idx: int) -> np.ndarray:
    """Permute a (600,) legal-action mask into frame sigma.

    mask_aug[ACT_PERM[a]] = mask[a]  <=>  mask_aug = mask[ACT_PERM_INV].
    """
    return mask[ACT_PERM_INV[sigma_idx]]


def apply_act_perm(action: int, sigma_idx: int) -> int:
    """Canonical action id -> frame-sigma action id."""
    return int(ACT_PERM[sigma_idx][action])


def apply_act_perm_inv(action: int, sigma_idx: int) -> int:
    """Frame-sigma action id -> canonical action id."""
    return int(ACT_PERM_INV[sigma_idx][action])


__all__ = [
    "AUTOMORPHISMS",
    "INV_AUTOMORPHISMS",
    "ACT_PERM",
    "ACT_PERM_INV",
    "CELL_SRC",
    "permute_obs",
    "permute_token_obs",
    "permute_mask",
    "apply_act_perm",
    "apply_act_perm_inv",
    "N_POSITIONS",
    "N_ACTIONS",
    "N_GRID_CELLS",
]
