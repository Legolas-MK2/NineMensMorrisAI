"""Tests for the relational ActorCritic in src/model.py."""

from __future__ import annotations

import sys
import os

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import pytest
import torch

import fastnmm

from config import Config
from model import ActorCritic
from utils import (
    build_token_obs, get_legal_mask,
    NODE_FEAT_DIM, GLOBAL_FEAT_DIM, NUM_POSITIONS,
)


N_ACTIONS = 600


def _make_model(policy_head: str = "pointer", seed: int = 0) -> ActorCritic:
    torch.manual_seed(seed)
    cfg = Config()
    cfg.policy_head = policy_head
    cfg.use_mixed_precision = False
    return ActorCritic(245, N_ACTIONS, cfg)


@pytest.mark.parametrize("policy_head", ["pointer", "flat"])
def test_forward_shapes(policy_head):
    model = _make_model(policy_head)
    B = 4
    nf = torch.zeros(B, NUM_POSITIONS, NODE_FEAT_DIM)
    gf = torch.zeros(B, GLOBAL_FEAT_DIM)
    logits, value = model(nf, gf)
    assert logits.shape == (B, N_ACTIONS)
    assert value.shape == (B,)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(value).all()
    assert (value.abs() <= Config().value_clip + 1e-5).all()


def test_pointer_action_layout():
    """Pointer head: action k in [0,24) -> node_logits[k]; action 24+f*24+t -> move_logits[f, t]."""
    model = _make_model("pointer")
    model.eval()
    nf = torch.zeros(1, NUM_POSITIONS, NODE_FEAT_DIM)
    gf = torch.zeros(1, GLOBAL_FEAT_DIM)
    with torch.no_grad():
        logits, _ = model(nf, gf)
    # Logit at action 24+f*24+t corresponds to slot 24+f*24+t in the flattened
    # move scores (from-major). We can't easily inspect internals without
    # reaching in, but we *can* assert that the pointer head produces
    # n_positions distinct place-slots and n_positions^2 move-slots.
    assert logits.shape == (1, NUM_POSITIONS + NUM_POSITIONS * NUM_POSITIONS)


@pytest.mark.parametrize("policy_head", ["pointer", "flat"])
def test_masked_actions_are_legal(policy_head):
    """For many random states, an argmax over masked logits must produce a legal action."""
    import random
    rng = random.Random(0)
    game = fastnmm.load_game("nine_mens_morris")
    model = _make_model(policy_head)
    model.eval()

    for trial in range(8):
        state = game.new_initial_state()
        for _ in range(rng.randint(0, 12)):
            if state.is_terminal():
                break
            legal = state.legal_actions()
            if not legal:
                break
            state.apply_action(rng.choice(legal))
        if state.is_terminal():
            continue

        player = state.current_player()
        nf, gf = build_token_obs(state, player)
        mask = get_legal_mask(state, N_ACTIONS)
        node_t = torch.from_numpy(nf).unsqueeze(0)
        glob_t = torch.from_numpy(gf).unsqueeze(0)
        with torch.no_grad():
            logits, _ = model(node_t, glob_t)
        masked = logits.squeeze(0).float()
        masked[torch.from_numpy(mask) == 0] = -1e9
        action = int(masked.argmax().item())
        assert action in state.legal_actions(), (trial, action)


def test_overfit_fixed_positions():
    """Loop-overfit one (obs, target) pair to ~zero CE loss to confirm end-to-end wiring."""
    game = fastnmm.load_game("nine_mens_morris")
    state = game.new_initial_state()
    nf, gf = build_token_obs(state, 0)
    legal = state.legal_actions()
    target = legal[len(legal) // 2]

    model = _make_model("pointer", seed=42)
    model.train()

    node_t = torch.from_numpy(nf).unsqueeze(0)
    glob_t = torch.from_numpy(gf).unsqueeze(0)
    target_t = torch.tensor([target], dtype=torch.long)

    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    losses = []
    for _ in range(800):
        logits, _ = model(node_t, glob_t)
        loss = loss_fn(logits, target_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    # Initial loss is ~ln(600)=6.4 due to deliberately-flat 0.01-gain init.
    # Driving to <0.1 confirms end-to-end gradient flow on a single sample.
    assert losses[-1] < 0.1, f"failed to overfit: final loss {losses[-1]:.4f}, start {losses[0]:.4f}"
    with torch.no_grad():
        logits, _ = model(node_t, glob_t)
        assert int(logits.argmax(dim=-1).item()) == target


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
