#!/usr/bin/env python3
"""
Quick method to train on jumping phase WITHOUT wasting time on random play.

Strategy:
1. Generate 3v3 states ONCE and save them
2. Load instantly for training (no random games needed)
"""

import pyspiel
import pickle
import sys
sys.path.append('/Project/Nine Mens Morris/claude')
from minimax import MinimaxBot

def count_pieces(state):
    s = str(state)
    return s.count('W'), s.count('B')

# ============================================================================
# STEP 1: Generate and save 3v3 states (do this ONCE)
# ============================================================================
def generate_and_save_jumping_states(num_states=50, filename='jumping_states.pkl'):
    """Generate 3v3 states once and save them."""
    game = pyspiel.load_game("nine_mens_morris")
    bot = MinimaxBot(max_depth=2, random_move_prob=0.2)

    saved_states = []
    print(f"Generating {num_states} jumping phase states...")

    attempts = 0
    max_attempts = num_states * 20

    while len(saved_states) < num_states and attempts < max_attempts:
        state = game.new_initial_state()
        steps = 0

        while not state.is_terminal() and steps < 400:
            white, black = count_pieces(state)

            # Accept any state with 3-5 pieces each (good for jumping practice)
            if 3 <= white <= 5 and 3 <= black <= 5:
                # SERIALIZE IT!
                serialized = state.serialize()
                saved_states.append(serialized)
                print(f"  Found {white}v{black} state (#{len(saved_states)})")
                break

            action = bot.get_action(state)
            state.apply_action(action)
            steps += 1

        attempts += 1

    # Save to file
    with open(filename, 'wb') as f:
        pickle.dump(saved_states, f)

    print(f"\n✓ Saved {len(saved_states)} states to {filename}")
    return saved_states


# ============================================================================
# STEP 2: Load states instantly (use this for training)
# ============================================================================
def load_jumping_states(filename='jumping_states.pkl'):
    """Load pre-generated states instantly - NO random play needed!"""
    with open(filename, 'rb') as f:
        serialized_states = pickle.load(f)

    print(f"✓ Loaded {len(serialized_states)} pre-generated states")
    return serialized_states


def deserialize_states(game, serialized_states):
    """Convert serialized strings back to game states."""
    states = []
    for s in serialized_states:
        state = game.deserialize_state(s)
        states.append(state)
    return states


# ============================================================================
# EXAMPLE USAGE
# ============================================================================
if __name__ == "__main__":
    import os

    filename = 'jumping_states.pkl'

    # Check if we already have saved states
    if os.path.exists(filename):
        print(f"Found existing {filename}")
        print("Loading instantly...\n")

        serialized = load_jumping_states(filename)

        # Deserialize to use
        game = pyspiel.load_game("nine_mens_morris")
        states = deserialize_states(game, serialized)

        print(f"Example state:")
        print(states[0])
        white, black = count_pieces(states[0])
        print(f"Pieces: {white}v{black}")

    else:
        print(f"No saved states found. Generating...")
        print("(This only needs to be done ONCE)\n")

        serialized = generate_and_save_jumping_states(num_states=50)

        print("\nNow you can load these instantly for training!")
        print("No more wasting time on random vs random games!")


# ============================================================================
# HOW TO USE IN YOUR TRAINING:
# ============================================================================
"""
# In your training code:

# 1. Load once at startup (instant, no random games!)
serialized = load_jumping_states('jumping_states.pkl')
game = pyspiel.load_game("nine_mens_morris")

# 2. For each training episode:
for episode in range(num_episodes):
    # Pick a random pre-generated state
    random_serialized = random.choice(serialized)

    # Deserialize it (fast!)
    state = game.deserialize_state(random_serialized)

    # Train from this state!
    # ... your training code ...
"""
