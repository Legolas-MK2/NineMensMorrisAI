#!/usr/bin/env python3
"""
Nine Men's Morris - Main Entry Point
Curriculum-based training with phased progression
"""

import os
import sys
import glob
import argparse
import time
import random
import multiprocessing as mp

import torch

# Import pyspiel with bug fix wrapper
from game_wrapper import load_game as load_game_fixed
import pyspiel

from config import Config
from model import ActorCritic
from trainer import PPOTrainer
from minimax import MinimaxBot, evaluate_vs_minimax
from utils import get_legal_mask
from curriculum import CurriculumManager, Phase, PHASE_CONFIGS


def get_game():
    """Get game instance using pyspiel with position 0 bug fix."""
    return load_game_fixed("nine_mens_morris")


def prepare_game_state(state, random_moves: int):
    """Play random moves to prepare the board state (not recorded)."""
    moves_made = 0
    while moves_made < random_moves and not state.is_terminal():
        # Check if either player has only 3 stones - stop early
        try:
            obs = state.observation_tensor(0)
            p0_pieces = sum(1 for i in range(24) if obs[i] == 1)
            p1_pieces = sum(1 for i in range(24) if obs[i + 24] == 1)
            if p0_pieces <= 3 or p1_pieces <= 3:
                break
        except:
            pass

        legal_actions = state.legal_actions()
        if not legal_actions:
            break

        action = random.choice(legal_actions)
        state.apply_action(action)
        moves_made += 1


def play_interactive(config: Config = None):
    """Interactive play mode."""
    if config is None:
        config = Config()

    # Find available models
    files = []
    for d in ["models", "checkpoints"]:
        if os.path.exists(d):
            files.extend(glob.glob(f"{d}/*.pt"))

    if not files:
        print("No models found!")
        return

    files.sort(key=os.path.getmtime, reverse=True)
    print("\nAvailable models:")
    for i, f in enumerate(files[:10], 1):
        print(f"  {i}. {os.path.basename(f)}")

    try:
        idx = int(input("\nSelect model (number): ")) - 1
        path = files[idx]
    except (ValueError, IndexError):
        print("Invalid selection")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    game = get_game()

    # Load model
    model = ActorCritic(
        game.observation_tensor_size(),
        game.num_distinct_actions(),
        config
    ).to(device)
    
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get('model_state_dict', ckpt))
    model.eval()
    
    print(f"\n✓ Loaded: {os.path.basename(path)}")
    
    # Mode selection
    print("\nModes:")
    print("  1. Play against AI")
    print("  2. Watch AI vs Minimax")
    print("  3. Test AI vs all Minimax depths")
    
    mode = input("\nSelect mode (1/2/3): ").strip()
    
    if mode == "3":
        # Progressive minimax test
        print("\nTesting AI against Minimax depths 1-6...")
        random_moves = int(input("Number of random moves to prepare board (0-150): ") or "75")
        max_beaten, results = evaluate_vs_minimax(
            model, device, game.num_distinct_actions(),
            max_depth=6, games_per_depth=20, max_steps=200,
            random_moves=random_moves
        )

        print(f"\n{'=' * 50}")
        print(f"Results: AI beats Minimax up to depth {max_beaten}")
        print(f"{'=' * 50}")
        for depth, r in results.items():
            status = "✓" if r['win_rate'] > 0.5 else "✗"
            print(f"  {status} Depth {depth}: {r['wins']}W / {r['draws']}D / {r['losses']}L ({r['win_rate']:.0%})")
        return
    
    if mode == "2":
        # AI vs Minimax
        try:
            depth = int(input("Minimax depth (1-6): "))
            ai_player = int(input("AI plays as (0/1): "))
            random_moves = int(input("Random moves to prepare board (0-150): ") or "75")
        except ValueError:
            print("Invalid input")
            return

        bot = MinimaxBot(max_depth=depth)
        state = game.new_initial_state()

        # Prepare board with random moves
        prepare_game_state(state, random_moves)

        move_num = 0

        print(f"\n{'=' * 50}")
        print(f"AI (Player {ai_player}) vs Minimax Depth {depth}")
        print(f"Board prepared with {random_moves} random moves")
        print(f"{'=' * 50}")
        
        while not state.is_terminal():
            print(f"\n--- Move {move_num} ---")
            print(state)
            
            current = state.current_player()
            
            if current == ai_player:
                obs = torch.tensor(
                    state.observation_tensor(current),
                    dtype=torch.float32, device=device
                ).unsqueeze(0)
                mask = torch.tensor(
                    get_legal_mask(state, game.num_distinct_actions()),
                    dtype=torch.float32, device=device
                ).unsqueeze(0)
                
                with torch.no_grad():
                    logits, v = model(obs)
                    masked = logits.float()
                    masked[mask == 0] = -1e9
                    a = masked.argmax().item()
                
                print(f"AI plays: {a} (V={v.item():.2f})")
            else:
                a = bot.get_action(state)
                print(f"Minimax (depth {depth}) plays: {a}")
            
            state.apply_action(a)
            move_num += 1
            time.sleep(0.3)
        
        print(f"\n{'=' * 50}")
        print(state)
        r = state.returns()
        if r[ai_player] > r[1 - ai_player]:
            print("🎉 AI WINS!")
        elif r[ai_player] < r[1 - ai_player]:
            print(f"🤖 Minimax (depth {depth}) WINS!")
        else:
            print("🤝 DRAW!")
        return
    
    # Mode 1: Human vs AI
    try:
        human = int(input("Play as (0/1): "))
        random_moves = int(input("Random moves to prepare board (0-150): ") or "75")
    except ValueError:
        human = 0
        random_moves = 75

    state = game.new_initial_state()

    # Prepare board with random moves
    prepare_game_state(state, random_moves)

    print(f"\n{'=' * 50}")
    print(f"You are Player {human}")
    print(f"Board prepared with {random_moves} random moves")
    print(f"{'=' * 50}")
    
    while not state.is_terminal():
        print(f"\n{state}")
        
        if state.current_player() == human:
            legal = state.legal_actions()
            print(f"Legal moves: {legal[:20]}{'...' if len(legal) > 20 else ''}")
            
            try:
                a = int(input("Your move: "))
                if a not in legal:
                    print("Illegal move!")
                    continue
            except ValueError:
                print("Invalid input!")
                continue
        else:
            obs = torch.tensor(
                state.observation_tensor(state.current_player()),
                dtype=torch.float32, device=device
            ).unsqueeze(0)
            mask = torch.tensor(
                get_legal_mask(state, game.num_distinct_actions()),
                dtype=torch.float32, device=device
            ).unsqueeze(0)
            
            with torch.no_grad():
                logits, v = model(obs)
                masked = logits.float()
                masked[mask == 0] = -1e9
                a = masked.argmax().item()
            
            print(f"AI plays: {a} (V={v.item():.2f})")
        
        state.apply_action(a)
    
    print(f"\n{state}")
    r = state.returns()
    if r[human] > r[1 - human]:
        print("🎉 YOU WIN!")
    elif r[human] < r[1 - human]:
        print("🤖 AI WINS!")
    else:
        print("🤝 DRAW!")


def show_curriculum_info():
    """Show information about curriculum phases."""
    print("\n" + "=" * 70)
    print("CURRICULUM TRAINING PHASES")
    print("=" * 70)
    print("\nNote: pyspiel Nine Men's Morris starts with full game.")
    print("Random moves are used to prepare board positions for training.\n")

    # Random moves per phase
    random_moves_info = {
        1: "150", 2: "150",
        3: "~129", 4: "~107", 5: "~86", 6: "~64", 7: "~43", 8: "~21",
        9: "0",
        10: "0-150 (random)"
    }

    for phase, cfg in PHASE_CONFIGS.items():
        phase_num = int(phase)
        print(f"\nPhase {phase_num}: {cfg.description}")
        print("-" * 50)
        print(f"  Random Prep:     {random_moves_info.get(phase_num, 'N/A')} moves")
        print(f"  Opponent:        {cfg.opponent_type}")
        print(f"  Learning Rate:   {cfg.lr_start:.0e} → {cfg.lr_end:.0e}")
        print(f"  Shaping Mult:    {cfg.shaping_multiplier:.1f}x")
        print(f"  Win Threshold:   {cfg.win_rate_threshold:.0%}")
        print(f"  Min Games:       {cfg.min_games_for_graduation}")
        if cfg.max_episodes > 0:
            print(f"  Max Episodes:    {cfg.max_episodes:,}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Nine Men's Morris - Curriculum PPO Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start curriculum training from Phase 1
  python main.py train --workers 16 --envs 32

  # Resume from checkpoint
  python main.py resume --checkpoint checkpoints/best_*.pt

  # Start from a specific phase
  python main.py train --workers 16 --envs 32 --phase 2

  # Play against trained model
  python main.py play

  # Show curriculum information
  python main.py info
        """
    )
    
    parser.add_argument(
        'mode',
        choices=['train', 'play', 'resume', 'info'],
        help='Mode: train, play, resume, or info'
    )
    parser.add_argument(
        '--episodes', type=int,
        help='Maximum total episodes'
    )
    parser.add_argument(
        '--checkpoint', type=str,
        help='Checkpoint path for resume'
    )
    parser.add_argument(
        '--workers', type=int, default=16,
        help='Number of worker processes (default: 16)'
    )
    parser.add_argument(
        '--envs', type=int, default=32,
        help='Environments per worker (default: 32)'
    )
    parser.add_argument(
        '--phase', type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        help='Start from specific phase (1-10)'
    )

    args = parser.parse_args()
    
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    if args.mode == 'info':
        show_curriculum_info()
        return
    
    if args.mode == 'train':
        config = Config()
        if args.episodes:
            config.total_episodes = args.episodes
        config.num_workers = args.workers
        config.envs_per_worker = args.envs

        trainer = PPOTrainer(config)

        # Start from specific phase if requested
        if args.phase:
            trainer.curriculum.current_phase = Phase(args.phase)
            trainer.curriculum.stats.phase = Phase(args.phase)
            phase_cfg = PHASE_CONFIGS[Phase(args.phase)]
            print(f"Starting from Phase {args.phase}: {phase_cfg.description}")

        trainer.train()
        
    elif args.mode == 'resume':
        config = Config()
        if args.episodes:
            config.total_episodes = args.episodes
        config.num_workers = args.workers
        config.envs_per_worker = args.envs

        trainer = PPOTrainer(config)

        if args.checkpoint:
            trainer.load_checkpoint(args.checkpoint)
        else:
            # Find latest checkpoint
            ckpts = glob.glob("checkpoints/*.pt")
            if ckpts:
                latest = max(ckpts, key=os.path.getmtime)
                trainer.load_checkpoint(latest)
            else:
                print("No checkpoints found! Use 'train' mode to start fresh.")
                return

        trainer.train()

    elif args.mode == 'play':
        config = Config()
        play_interactive(config)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("=" * 70)
        print("Nine Men's Morris - Curriculum PPO Training (pyspiel)")
        print("=" * 70)
        print()
        print("This training system uses a 10-phase curriculum with random board prep:")
        print()
        print("  Phase 1-2:  150 random moves prep, vs RANDOM (warmup)")
        print("  Phase 3-8:  Linear decrease (129->21 moves), mixed opponents")
        print("  Phase 9:    No prep moves, full game, mixed opponents")
        print("  Phase 10:   0-150 random prep, final phase with D1-D6 minimax")
        print()
        print("Random moves prepare diverse board positions without custom engine.")
        print("Training starts after board prep - prep moves aren't learned.")
        print()
        print("Usage:")
        print("  python main.py train [options]    # Start training")
        print("  python main.py resume [options]   # Resume from checkpoint")
        print("  python main.py play               # Play against model")
        print("  python main.py info               # Show phase details")
        print()
        print("Options:")
        print("  --workers N         Number of worker processes (default: 16)")
        print("  --envs N            Environments per worker (default: 32)")
        print("  --episodes N        Maximum total episodes")
        print("  --phase N           Start from curriculum phase N (1-10)")
        print()
        print("Examples:")
        print("  # Standard training")
        print("  python main.py train --workers 16 --envs 32")
        print()
        print("  # Start from phase 5")
        print("  python main.py train --phase 5")
        print()
    else:
        main()
