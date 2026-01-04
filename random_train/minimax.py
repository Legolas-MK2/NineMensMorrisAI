"""
Nine Men's Morris - Minimax Bot
Alpha-beta pruning implementation for training opponent and evaluation
"""

from typing import Tuple, Dict
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast

from utils import get_legal_mask


class MinimaxBot:
    """Minimax bot with alpha-beta pruning for Nine Men's Morris."""
    
    def __init__(self, max_depth: int = 3):
        self.max_depth = max_depth
        self.nodes_evaluated = 0
    
    def evaluate(self, state, player: int) -> float:
        """
        Heuristic evaluation of board position.
        Positive = good for player, negative = bad.
        """
        if state.is_terminal():
            returns = state.returns()
            if returns[player] > returns[1 - player]:
                return 10000  # Win
            elif returns[player] < returns[1 - player]:
                return -10000  # Loss
            else:
                return 0  # Draw
        
        # Count pieces from state string
        state_str = str(state)
        p0_pieces = state_str.count('o') + state_str.count('O')
        p1_pieces = state_str.count('x') + state_str.count('X')
        
        if player == 0:
            my_pieces, opp_pieces = p0_pieces, p1_pieces
        else:
            my_pieces, opp_pieces = p1_pieces, p0_pieces
        
        # Piece advantage is the main heuristic
        piece_score = (my_pieces - opp_pieces) * 100
        
        # Mobility bonus
        if state.current_player() == player:
            mobility = len(state.legal_actions())
        else:
            mobility = 0
        
        return piece_score + mobility
    
    def minimax(self, state, depth: int, alpha: float, beta: float,
                maximizing: bool, player: int) -> Tuple[float, int]:
        """Minimax with alpha-beta pruning. Returns (score, best_action)."""
        self.nodes_evaluated += 1
        
        if depth == 0 or state.is_terminal():
            return self.evaluate(state, player), -1
        
        legal_actions = state.legal_actions()
        if not legal_actions:
            return self.evaluate(state, player), -1
        
        best_action = legal_actions[0]
        
        if maximizing:
            max_eval = float('-inf')
            for action in legal_actions:
                child = state.clone()
                child.apply_action(action)
                eval_score, _ = self.minimax(child, depth - 1, alpha, beta, False, player)
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_action = action
                
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            
            return max_eval, best_action
        else:
            min_eval = float('inf')
            for action in legal_actions:
                child = state.clone()
                child.apply_action(action)
                eval_score, _ = self.minimax(child, depth - 1, alpha, beta, True, player)
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_action = action
                
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            
            return min_eval, best_action
    
    def get_action(self, state) -> int:
        """Get best action for current player."""
        self.nodes_evaluated = 0
        player = state.current_player()
        _, action = self.minimax(
            state, self.max_depth,
            float('-inf'), float('inf'),
            True, player
        )
        return action


def evaluate_vs_minimax(
    model,
    device,
    num_actions: int,
    max_depth: int = 6,
    games_per_depth: int = 10,
    max_steps: int = 200,
    use_mixed_precision: bool = True
) -> Tuple[int, Dict]:
    """
    Progressive evaluation against minimax bots.
    Returns (max_depth_beaten, detailed_results).
    """
    import pyspiel
    game = pyspiel.load_game("nine_mens_morris")
    results = {}
    max_depth_beaten = 0
    
    model.eval()
    
    for depth in range(1, max_depth + 1):
        bot = MinimaxBot(max_depth=depth)
        wins, draws, losses = 0, 0, 0
        
        with torch.no_grad():
            for game_idx in range(games_per_depth):
                state = game.new_initial_state()
                ai_player = game_idx % 2
                steps = 0
                
                while not state.is_terminal() and steps < max_steps:
                    current = state.current_player()
                    
                    if current == ai_player:
                        obs = torch.tensor(
                            state.observation_tensor(current),
                            dtype=torch.float32, device=device
                        ).unsqueeze(0)
                        mask = torch.tensor(
                            get_legal_mask(state, num_actions),
                            dtype=torch.float32, device=device
                        ).unsqueeze(0)
                        
                        with autocast('cuda', enabled=use_mixed_precision and device.type == 'cuda'):
                            logits, _ = model(obs)
                        
                        masked = logits.squeeze(0).float()
                        masked[mask.squeeze(0) == 0] = -1e9
                        action = masked.argmax().item()
                    else:
                        action = bot.get_action(state)
                    
                    state.apply_action(action)
                    steps += 1
                
                if state.is_terminal():
                    returns = state.returns()
                    if returns[ai_player] > returns[1 - ai_player]:
                        wins += 1
                    elif returns[ai_player] < returns[1 - ai_player]:
                        losses += 1
                    else:
                        draws += 1
                else:
                    draws += 1  # Timeout = draw
        
        win_rate = (wins + 0.5 * draws) / games_per_depth
        results[depth] = {
            'wins': wins,
            'draws': draws,
            'losses': losses,
            'win_rate': win_rate
        }
        
        if win_rate > 0.5:
            max_depth_beaten = depth
        else:
            break  # Stop if we're losing
    
    return max_depth_beaten, results


def format_minimax_results(results: Dict) -> str:
    """Format minimax evaluation results as a string."""
    parts = []
    for depth in sorted(results.keys()):
        r = results[depth]
        parts.append(f"D{depth}:{r['wins']}W/{r['draws']}D/{r['losses']}L")
    return " | ".join(parts)
