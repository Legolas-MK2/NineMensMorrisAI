"""Tests for src/symmetry.py — automorphism group and frame consistency.

Run from src/:
    python -m pytest tests/test_symmetry.py -v
"""

from __future__ import annotations

import sys
import os

# Allow `python -m pytest tests/...` from the src/ directory.
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import numpy as np
import pytest

import fastnmm

from utils import (
    ADJACENCY, MILLS, BOARD_POS_TO_GRID,
    ADJ_MATRIX, MILL_MATRIX,
    get_legal_mask, relativize_obs, build_token_obs,
)
from symmetry import (
    AUTOMORPHISMS,
    INV_AUTOMORPHISMS,
    ACT_PERM,
    ACT_PERM_INV,
    CELL_SRC,
    permute_obs,
    permute_token_obs,
    permute_mask,
    apply_act_perm,
    apply_act_perm_inv,
    N_POSITIONS,
    N_ACTIONS,
    N_GRID_CELLS,
)


N_OBS = 5 * N_GRID_CELLS  # 245


def test_group_size():
    assert len(AUTOMORPHISMS) == 16
    assert len(INV_AUTOMORPHISMS) == 16
    assert len(ACT_PERM) == 16
    assert len(ACT_PERM_INV) == 16
    assert len(CELL_SRC) == 16


def test_identity_in_group():
    identity = np.arange(N_POSITIONS, dtype=np.int64)
    assert any(np.array_equal(s, identity) for s in AUTOMORPHISMS)


def test_each_element_is_permutation():
    for idx, sigma in enumerate(AUTOMORPHISMS):
        assert sigma.shape == (N_POSITIONS,), idx
        assert sorted(sigma.tolist()) == list(range(N_POSITIONS)), idx


def test_inverses_round_trip():
    for sigma, sigma_inv in zip(AUTOMORPHISMS, INV_AUTOMORPHISMS):
        assert np.array_equal(sigma[sigma_inv], np.arange(N_POSITIONS))
        assert np.array_equal(sigma_inv[sigma], np.arange(N_POSITIONS))


def test_preserves_adjacency():
    canon = frozenset(
        frozenset((u, v)) for u, nbrs in enumerate(ADJACENCY) for v in nbrs
    )
    for idx, sigma in enumerate(AUTOMORPHISMS):
        image = frozenset(
            frozenset((int(sigma[u]), int(sigma[v])))
            for u, nbrs in enumerate(ADJACENCY)
            for v in nbrs
        )
        assert image == canon, f"sigma {idx} breaks adjacency"


def test_preserves_mills():
    canon = frozenset(frozenset(m) for m in MILLS)
    for idx, sigma in enumerate(AUTOMORPHISMS):
        image = frozenset(frozenset(int(sigma[p]) for p in m) for m in MILLS)
        assert image == canon, f"sigma {idx} breaks mills"


def test_act_perm_roundtrip():
    arange = np.arange(N_ACTIONS, dtype=np.int64)
    for idx, (p, pi) in enumerate(zip(ACT_PERM, ACT_PERM_INV)):
        assert np.array_equal(p[pi], arange), idx
        assert np.array_equal(pi[p], arange), idx


def test_act_perm_consistent_with_sigma():
    """act_perm should agree with sigma on places, and lift to (sigma[f], sigma[t]) on moves."""
    for idx, sigma in enumerate(AUTOMORPHISMS):
        for p in range(N_POSITIONS):
            assert ACT_PERM[idx][p] == sigma[p], (idx, p)
        for f in range(N_POSITIONS):
            for t in range(N_POSITIONS):
                a = 24 + f * 24 + t
                expected = 24 + int(sigma[f]) * 24 + int(sigma[t])
                assert ACT_PERM[idx][a] == expected, (idx, f, t)


def test_obs_roundtrip():
    """Necessary but not sufficient — passes even if direction is inverted."""
    rng = np.random.default_rng(0)
    for idx in range(16):
        obs = rng.random(N_OBS, dtype=np.float32)
        sigma_inv_idx = _inverse_element_index(idx)
        out = permute_obs(permute_obs(obs, idx), sigma_inv_idx)
        np.testing.assert_allclose(out, obs)


def _inverse_element_index(idx: int) -> int:
    """Return the index j such that AUTOMORPHISMS[j] is the inverse of AUTOMORPHISMS[idx]."""
    target = INV_AUTOMORPHISMS[idx]
    for j, sigma in enumerate(AUTOMORPHISMS):
        if np.array_equal(sigma, target):
            return j
    raise AssertionError(f"inverse of element {idx} not present in group")


def _make_obs_with_pieces(p0_positions, p1_positions=()):
    """Build a (5,7,7) obs flat array with the given board positions occupied.

    Channel 0 = player 0 pieces, channel 1 = player 1 pieces, others zero.
    Cell index = row * 7 + col, using BOARD_POS_TO_GRID.
    """
    obs = np.zeros((5, N_GRID_CELLS), dtype=np.float32)
    for p in p0_positions:
        r, c = BOARD_POS_TO_GRID[p]
        obs[0, r * 7 + c] = 1.0
    for p in p1_positions:
        r, c = BOARD_POS_TO_GRID[p]
        obs[1, r * 7 + c] = 1.0
    return obs.reshape(-1)


def test_obs_action_frame_consistency_places():
    """The pinned test: obs, mask, action all reference the same sigma.

    For each sigma and each position p:
      - A single piece placed at p in canonical frame should, after permute_obs,
        sit at grid cell BOARD_POS_TO_GRID[sigma[p]].
      - ACT_PERM[sigma][place_p] must equal sigma[p].
      - permute_mask of a mask with only place_p legal should mark exactly sigma[p].
    A direction-inverted permute_obs would fail this for the order-4 rotations.
    """
    for idx, sigma in enumerate(AUTOMORPHISMS):
        for p in range(N_POSITIONS):
            obs = _make_obs_with_pieces([p])
            aug = permute_obs(obs, idx).reshape(5, N_GRID_CELLS)

            target_p = int(sigma[p])
            r, c = BOARD_POS_TO_GRID[target_p]
            target_cell = r * 7 + c

            # Only target_cell on channel 0 should be hot.
            assert aug[0, target_cell] == 1.0, (idx, p)
            assert aug[0].sum() == 1.0, (idx, p)
            # Channel 1 (and the others) untouched.
            assert aug[1].sum() == 0.0, (idx, p)

            # Action permutation matches.
            assert ACT_PERM[idx][p] == target_p, (idx, p)

            # Mask permutation marks exactly target_p.
            mask = np.zeros(N_ACTIONS, dtype=np.float32)
            mask[p] = 1.0
            aug_mask = permute_mask(mask, idx)
            assert aug_mask.sum() == 1.0, (idx, p)
            assert aug_mask[target_p] == 1.0, (idx, p)


def test_obs_action_frame_consistency_moves():
    """Same pinning test, for move actions (24 + f*24 + t)."""
    rng = np.random.default_rng(42)
    # Sample a handful of (f, t) pairs (full 576 grid is overkill).
    pairs = [(rng.integers(N_POSITIONS), rng.integers(N_POSITIONS)) for _ in range(20)]
    for idx, sigma in enumerate(AUTOMORPHISMS):
        for f, t in pairs:
            f, t = int(f), int(t)
            # Build obs with p0 piece at f, p1 piece at t.
            obs = _make_obs_with_pieces([f], [t]).copy()
            aug = permute_obs(obs, idx).reshape(5, N_GRID_CELLS)

            sf, st = int(sigma[f]), int(sigma[t])
            fr, fc = BOARD_POS_TO_GRID[sf]
            tr, tc = BOARD_POS_TO_GRID[st]

            if f == t:
                # Both channels hot at the same cell -- channel-1 wasn't overwritten.
                pass
            assert aug[0, fr * 7 + fc] == 1.0, (idx, f, t)
            assert aug[1, tr * 7 + tc] == 1.0, (idx, f, t)
            assert aug[0].sum() == 1.0, (idx, f, t)
            assert aug[1].sum() == 1.0, (idx, f, t)

            a_canon = 24 + f * 24 + t
            a_aug = 24 + sf * 24 + st
            assert ACT_PERM[idx][a_canon] == a_aug, (idx, f, t)

            mask = np.zeros(N_ACTIONS, dtype=np.float32)
            mask[a_canon] = 1.0
            aug_mask = permute_mask(mask, idx)
            assert aug_mask.sum() == 1.0, (idx, f, t)
            assert aug_mask[a_aug] == 1.0, (idx, f, t)


def test_action_legality_under_aug():
    """For a fresh state, image of every legal canonical action is legal in the permuted mask,
    and the inverse round-trips back to canonical."""
    game = fastnmm.load_game("nine_mens_morris")
    state = game.new_initial_state()
    mask = get_legal_mask(state, N_ACTIONS)
    legal_canon = np.flatnonzero(mask)

    for idx in range(16):
        aug_mask = permute_mask(mask, idx)
        # Same count of legal actions.
        assert aug_mask.sum() == mask.sum(), idx
        # Every canonical legal action's image is legal in the augmented mask.
        for a in legal_canon:
            a_aug = apply_act_perm(int(a), idx)
            assert aug_mask[a_aug] == 1.0, (idx, a)
            # And round-trips.
            assert apply_act_perm_inv(a_aug, idx) == int(a), (idx, a)


def test_obs_permutation_matches_real_state():
    """A real-state obs from fastnmm, permuted by sigma, should match what we'd
    get if we manually moved each piece on the board from p to sigma[p]."""
    game = fastnmm.load_game("nine_mens_morris")
    state = game.new_initial_state()
    # Play a few legal moves to populate the board.
    rng = np.random.default_rng(123)
    for _ in range(8):
        if state.is_terminal():
            break
        legal = state.legal_actions()
        if not legal:
            break
        state.apply_action(int(rng.choice(legal)))

    obs = relativize_obs(state, 0)  # flat (245,)

    for idx, sigma in enumerate(AUTOMORPHISMS):
        aug = permute_obs(obs, idx).reshape(5, N_GRID_CELLS)
        canon = obs.reshape(5, N_GRID_CELLS)
        # For each board position p, the content at sigma[p]'s grid cell in aug
        # should equal the content at p's grid cell in canon, for every channel.
        for p in range(N_POSITIONS):
            sp = int(sigma[p])
            r_src, c_src = BOARD_POS_TO_GRID[p]
            r_dst, c_dst = BOARD_POS_TO_GRID[sp]
            src_cell = r_src * 7 + c_src
            dst_cell = r_dst * 7 + c_dst
            for ch in range(5):
                assert aug[ch, dst_cell] == canon[ch, src_cell], (idx, p, ch)


def test_token_obs_roundtrip():
    """permute_token_obs round-trips for every (g, g^-1). Necessary but not sufficient."""
    rng = np.random.default_rng(7)
    for idx in range(16):
        nf = rng.random((N_POSITIONS, 3), dtype=np.float32)
        inv_idx = _inverse_element_index(idx)
        out = permute_token_obs(permute_token_obs(nf, idx), inv_idx)
        np.testing.assert_allclose(out, nf)


def test_token_obs_frame_consistency_places():
    """In frame sigma the row for canonical position p must sit at row sigma[p],
    and act_perm/permute_mask must agree on the same sigma[p]."""
    for idx, sigma in enumerate(AUTOMORPHISMS):
        for p in range(N_POSITIONS):
            nf = np.zeros((N_POSITIONS, 3), dtype=np.float32)
            nf[p, 1] = 1.0  # "mine" at p
            aug = permute_token_obs(nf, idx)

            sp = int(sigma[p])
            # Only row sp should be hot.
            assert aug[sp, 1] == 1.0, (idx, p)
            assert aug[:, 1].sum() == 1.0, (idx, p)

            assert ACT_PERM[idx][p] == sp, (idx, p)

            mask = np.zeros(N_ACTIONS, dtype=np.float32)
            mask[p] = 1.0
            aug_mask = permute_mask(mask, idx)
            assert aug_mask.sum() == 1.0, (idx, p)
            assert aug_mask[sp] == 1.0, (idx, p)


def test_token_obs_frame_consistency_moves():
    """For a (f, t) move: rows sigma[f]/sigma[t] occupied; mask marks 24+sigma[f]*24+sigma[t]."""
    rng = np.random.default_rng(99)
    pairs = [(int(rng.integers(N_POSITIONS)), int(rng.integers(N_POSITIONS))) for _ in range(20)]
    for idx, sigma in enumerate(AUTOMORPHISMS):
        for f, t in pairs:
            nf = np.zeros((N_POSITIONS, 3), dtype=np.float32)
            nf[f, 1] = 1.0  # "mine" at f
            nf[t, 2] = 1.0  # "opp"  at t (may collide if f==t, fine)
            aug = permute_token_obs(nf, idx)

            sf, st = int(sigma[f]), int(sigma[t])
            assert aug[sf, 1] == 1.0, (idx, f, t)
            assert aug[st, 2] == 1.0, (idx, f, t)

            a_canon = 24 + f * 24 + t
            a_aug = 24 + sf * 24 + st
            assert ACT_PERM[idx][a_canon] == a_aug, (idx, f, t)

            mask = np.zeros(N_ACTIONS, dtype=np.float32)
            mask[a_canon] = 1.0
            aug_mask = permute_mask(mask, idx)
            assert aug_mask.sum() == 1.0, (idx, f, t)
            assert aug_mask[a_aug] == 1.0, (idx, f, t)


def test_attention_bias_invariance():
    """ADJ_MATRIX and MILL_MATRIX must be sigma-invariant for every automorphism.

    This is what makes the structural attention bias correct in every augmented
    frame without permuting the bias matrices.
    """
    for idx, sigma in enumerate(AUTOMORPHISMS):
        s = sigma.astype(np.intp)
        permuted_adj = ADJ_MATRIX[np.ix_(s, s)]
        permuted_mill = MILL_MATRIX[np.ix_(s, s)]
        assert np.array_equal(permuted_adj, ADJ_MATRIX), f"sigma {idx} breaks ADJ_MATRIX"
        assert np.array_equal(permuted_mill, MILL_MATRIX), f"sigma {idx} breaks MILL_MATRIX"


def test_token_obs_real_state_consistent():
    """For a real fastnmm state, content at canonical p must appear at sigma[p]
    in the augmented token obs (for every sigma, every channel)."""
    game = fastnmm.load_game("nine_mens_morris")
    state = game.new_initial_state()
    rng = np.random.default_rng(321)
    for _ in range(8):
        if state.is_terminal():
            break
        legal = state.legal_actions()
        if not legal:
            break
        state.apply_action(int(rng.choice(legal)))

    nf, _ = build_token_obs(state, 0)
    for idx, sigma in enumerate(AUTOMORPHISMS):
        aug = permute_token_obs(nf, idx)
        for p in range(N_POSITIONS):
            sp = int(sigma[p])
            np.testing.assert_array_equal(aug[sp], nf[p], err_msg=f"sigma={idx} p={p}")


def test_group_closure_under_composition():
    """For any two elements a, b in the group, a o b is also in the group."""
    keys = {s.tobytes() for s in AUTOMORPHISMS}
    for a in AUTOMORPHISMS:
        for b in AUTOMORPHISMS:
            prod = a[b]
            assert prod.tobytes() in keys


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
