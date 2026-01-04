#!/usr/bin/env python3
"""
Nine Men's Morris - Random Training Entry Point
Trains for 10M epochs against random opponent, then evaluates with minimax
"""

import os
import sys
import glob
import argparse
import time
import random
import multiprocessing as mp

import torch
import pyspiel

from config import Config
from model import ActorCritic
from trainer import PPOTrainer
from minimax import MinimaxBot, evaluate_vs_minimax
from utils import get_legal_mask


def play_interactive():
    """Interactive play mode."""
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
    game = pyspiel.load_game("nine_mens_morris")

    # Load model
    config = Config()
    model = ActorCritic(
        game.observation_tensor_size(),
        game.num_distinct_actions(),
        config
    ).to(device)

    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get('model_state_dict', ckpt))
    model.eval()

    print(f"\nLoaded: {os.path.basename(path)}")

    # Mode selection
    print("\nModes:")
    print("  1. Play against AI")
    print("  2. Watch AI vs Minimax")
    print("  3. Test AI vs all Minimax depths")

    mode = input("\nSelect mode (1/2/3): ").strip()

    if mode == "3":
        # Progressive minimax test
        print("\nTesting AI against Minimax depths 1-6...")
        max_beaten, results = evaluate_vs_minimax(
            model, device, game.num_distinct_actions(),
            max_depth=6, games_per_depth=20, max_steps=200
        )

        print(f"\n{'=' * 50}")
        print(f"Results: AI beats Minimax up to depth {max_beaten}")
        print(f"{'=' * 50}")
        for depth, r in results.items():
            status = "PASS" if r['win_rate'] > 0.5 else "FAIL"
            print(f"  {status} Depth {depth}: {r['wins']}W / {r['draws']}D / {r['losses']}L ({r['win_rate']:.0%})")
        return

    if mode == "2":
        # AI vs Minimax
        try:
            depth = int(input("Minimax depth (1-6): "))
            ai_player = int(input("AI plays as (0/1): "))
        except ValueError:
            print("Invalid input")
            return

        bot = MinimaxBot(max_depth=depth)
        state = game.new_initial_state()
        move_num = 0

        print(f"\n{'=' * 50}")
        print(f"AI (Player {ai_player}) vs Minimax Depth {depth}")
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
            print("AI WINS!")
        elif r[ai_player] < r[1 - ai_player]:
            print(f"Minimax (depth {depth}) WINS!")
        else:
            print("DRAW!")
        return

    # Mode 1: Human vs AI
    try:
        human = int(input("Play as (0/1): "))
    except ValueError:
        human = 0

    state = game.new_initial_state()

    print(f"\n{'=' * 50}")
    print(f"You are Player {human}")
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
        print("YOU WIN!")
    elif r[human] < r[1 - human]:
        print("AI WINS!")
    else:
        print("DRAW!")


def final_evaluation(model_path: str):
    """Run comprehensive minimax evaluation on a trained model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    game = pyspiel.load_game("nine_mens_morris")

    config = Config()
    model = ActorCritic(
        game.observation_tensor_size(),
        game.num_distinct_actions(),
        config
    ).to(device)

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get('model_state_dict', ckpt))
    model.eval()

    print(f"\nLoaded model: {model_path}")
    print("\n" + "=" * 70)
    print("COMPREHENSIVE MINIMAX EVALUATION")
    print("=" * 70)

    num_actions = game.num_distinct_actions()

    for depth in range(1, 7):
        bot = MinimaxBot(max_depth=depth)
        wins, losses, draws = 0, 0, 0
        games_to_play = 100  # Comprehensive evaluation

        with torch.no_grad():
            for game_idx in range(games_to_play):
                state = game.new_initial_state()
                ai_player = game_idx % 2
                steps = 0

                while not state.is_terminal() and steps < config.max_game_steps:
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
                    draws += 1

        win_rate = (wins + 0.5 * draws) / games_to_play
        status = "PASS" if win_rate >= 0.5 else "FAIL"
        print(f"  Depth {depth}: {wins:3d}W / {draws:3d}D / {losses:3d}L  |  Win Rate: {win_rate:6.1%}  |  {status}")

    print("=" * 70)
    print("Evaluation complete!")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Nine Men's Morris - Random Opponent Training (10M epochs)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start training (10M epochs vs random)
  python main.py train --workers 16 --envs 32

  # Resume from checkpoint
  python main.py resume --checkpoint checkpoints/checkpoint_*.pt

  # Play against trained model
  python main.py play

  # Evaluate model with minimax
  python main.py eval --model models/final_*.pt
        """
    )

    parser.add_argument(
        'mode',
        choices=['train', 'play', 'resume', 'eval'],
        help='Mode: train, play, resume, or eval'
    )
    parser.add_argument(
        '--episodes', type=int,
        help='Maximum total episodes (default: 10M)'
    )
    parser.add_argument(
        '--checkpoint', type=str,
        help='Checkpoint path for resume'
    )
    parser.add_argument(
        '--model', type=str,
        help='Model path for evaluation'
    )
    parser.add_argument(
        '--workers', type=int, default=16,
        help='Number of worker processes (default: 16)'
    )
    parser.add_argument(
        '--envs', type=int, default=32,
        help='Environments per worker (default: 32)'
    )

    args = parser.parse_args()

    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)

    if args.mode == 'train':
        config = Config()
        if args.episodes:
            config.total_episodes = args.episodes
        config.num_workers = args.workers
        config.envs_per_worker = args.envs

        trainer = PPOTrainer(config)
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

    elif args.mode == 'eval':
        if args.model:
            final_evaluation(args.model)
        else:
            # Find latest model
            models = glob.glob("models/*.pt")
            if models:
                latest = max(models, key=os.path.getmtime)
                final_evaluation(latest)
            else:
                print("No models found! Specify --model or train first.")

    elif args.mode == 'play':
        play_interactive()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("=" * 70)
        print("Nine Men's Morris - Random Opponent Training")
        print("=" * 70)
        print()
        print("This trains a model for 10 MILLION epochs against random opponents,")
        print("then evaluates performance against Minimax bots (depths 1-6).")
        print()
        print("Usage:")
        print("  python main.py train [options]    # Start training")
        print("  python main.py resume [options]   # Resume from checkpoint")
        print("  python main.py play               # Play against model")
        print("  python main.py eval [--model X]   # Evaluate with minimax")
        print()
        print("Options:")
        print("  --workers N      Number of worker processes (default: 16)")
        print("  --envs N         Environments per worker (default: 32)")
        print("  --episodes N     Override episode count (default: 10M)")
        print()
        print("Quick start:")
        print("  python main.py train --workers 16 --envs 32")
        print()
    else:
        main()
