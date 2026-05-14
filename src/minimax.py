"""
Minimax wrapper around the fastnmm C++ engine.

Public surface for the rest of the codebase:
    MinimaxBot(max_depth=…, random_move_prob=…) — exposes .step()/.get_action()
    evaluate_vs_minimax(model, device, num_actions, …)
    format_minimax_results(results)
"""

from __future__ import annotations

import random
from typing import Dict, Tuple

import torch
from torch.amp import autocast

import fastnmm
from fastnmm import MinimaxBot as _FastMinimaxBot


class MinimaxBot:
    """Thin wrapper around `fastnmm.MinimaxBot`."""

    def __init__(
        self,
        max_depth: int = 4,
        random_move_prob: float = 0.0,
        player_id: int = 0,
    ) -> None:
        self.max_depth = int(max_depth)
        self.random_move_prob = float(random_move_prob)
        self.player_id = int(player_id)
        self._bot = _FastMinimaxBot(player_id=self.player_id, depth=self.max_depth)
        self.last_search = None

    def step(self, state) -> int:
        """Pick an action for `state`, honouring `random_move_prob`."""
        if self.random_move_prob > 0.0 and random.random() < self.random_move_prob:
            return random.choice(state.legal_actions())
        action = self._bot.step(state)
        self.last_search = self._bot.last_search
        return action

    # Legacy alias used by existing call sites.
    get_action = step


def evaluate_vs_minimax(
    model,
    device,
    num_actions: int,
    max_depth: int = 6,
    games_per_depth: int = 10,
    max_steps: int = 200,
    use_mixed_precision: bool = True,
    unlimited: bool = True,
    config=None,
    starting_stones: int = 9,
) -> Tuple[int, Dict]:
    """Play the model against progressively harder fastnmm minimax bots.

    `starting_stones` is the per-player stone count to seed each game with;
    pass -1 to randomize each player's count in [3, 9] per game.
    """
    # Local import to avoid a circular dependency at module load time.
    from utils import get_legal_mask

    game = fastnmm.load_game("nine_mens_morris")
    results: Dict[int, Dict] = {}
    max_depth_beaten = 0

    model.eval()
    depth = 1
    while True:
        if not unlimited and depth > max_depth:
            break

        bot = _FastMinimaxBot(player_id=0, depth=depth)
        wins = draws = losses = 0
        total_nodes = 0

        with torch.no_grad():
            for game_idx in range(games_per_depth):
                if starting_stones is None or starting_stones < 0:
                    stones = (random.randint(3, 9), random.randint(3, 9))
                else:
                    s = max(1, min(9, int(starting_stones)))
                    stones = (s, s)
                state = game.new_initial_state(starting_stones=stones)
                if state.is_terminal():
                    draws += 1
                    continue

                ai_player = game_idx % 2
                steps = 0
                while not state.is_terminal() and steps < max_steps:
                    current = state.current_player()
                    if current == ai_player:
                        # Use the (5,7,7) numpy tensor and flatten — matches
                        # the model's expected obs_size of 245.
                        obs_arr = state.observation_tensor_numpy(current).reshape(-1)
                        obs = torch.from_numpy(obs_arr).to(device).unsqueeze(0)
                        mask = torch.tensor(
                            get_legal_mask(state, num_actions),
                            dtype=torch.float32, device=device,
                        ).unsqueeze(0)
                        with autocast(
                            "cuda",
                            enabled=use_mixed_precision and device.type == "cuda",
                        ):
                            logits, _ = model(obs)
                        masked = logits.squeeze(0).float()
                        masked[mask.squeeze(0) == 0] = -1e9
                        action = masked.argmax().item()
                    else:
                        action = bot.step(state)
                        if bot.last_search is not None:
                            total_nodes += bot.last_search.nodes_visited
                    state.apply_action(action)
                    steps += 1

                if state.is_terminal():
                    r = state.returns()
                    if r[ai_player] > r[1 - ai_player]:
                        wins += 1
                    elif r[ai_player] < r[1 - ai_player]:
                        losses += 1
                    else:
                        draws += 1
                else:
                    draws += 1

        win_rate = (wins + 0.5 * draws) / games_per_depth
        results[depth] = {
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "win_rate": win_rate,
            "total_nodes": total_nodes,
        }
        if win_rate > 0.5:
            max_depth_beaten = depth
            depth += 1
        else:
            break

    return max_depth_beaten, results


def format_minimax_results(results: Dict) -> str:
    parts = []
    for depth in sorted(results.keys()):
        r = results[depth]
        parts.append(f"D{depth}:{r['wins']}W/{r['draws']}D/{r['losses']}L")
    return " | ".join(parts)
