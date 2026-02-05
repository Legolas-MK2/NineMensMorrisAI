"""
Nine Men's Morris - Worker Process
Experience collection with curriculum-based opponents and game settings
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import time
import random
import queue
import numpy as np
import torch
import torch.nn.functional as F
from multiprocessing import Queue, Event
from typing import Dict, Any, Optional, Tuple

from config import Config
from model import ActorCritic
from utils import (
    get_legal_mask, count_pieces_from_state,
    compute_gae, ExperienceBatch, RewardCalculator
)
from minimax import MinimaxBot

# Import pyspiel with bug fix wrapper
import pyspiel
from game_wrapper import load_game as load_game_fixed


class EnvState:
    """State for a single environment."""

    def __init__(self, game, reward_calculator: RewardCalculator):
        self.game = game
        self.reward_calculator = reward_calculator
        self.reset()

    def reset(self):
        self.state = self.game.new_initial_state()
        self.step_count = 0
        self.experiences = {0: [], 1: []}
        self.pieces = {0: 9, 1: 9}

        # Opponent settings (set by worker based on curriculum)
        self.opponent_type = 'random'
        self.ai_player = random.randint(0, 1)
        self.minimax_bot = None
        self.minimax_depth = 0
        self.clone_player = 1 - self.ai_player

    def setup_opponent(self, opponent_type: str, minimax_depth: int = 1):
        """Configure opponent for this game."""
        self.opponent_type = opponent_type
        self.minimax_depth = minimax_depth if opponent_type == 'minimax' else 0

        if opponent_type == 'minimax':
            self.minimax_bot = MinimaxBot(max_depth=minimax_depth, random_move_prob=0.3)
        else:
            self.minimax_bot = None


def get_opponent_action(env: EnvState, state, num_actions: int, clone_model: Optional[ActorCritic] = None) -> int:
    """Get action for opponent based on opponent type."""
    if env.opponent_type == 'random':
        return random.choice(state.legal_actions())

    elif env.opponent_type == 'minimax':
        if env.minimax_bot is None:
            env.minimax_bot = MinimaxBot(max_depth=env.minimax_depth, random_move_prob=0.3)
        return env.minimax_bot.get_action(state)

    elif env.opponent_type == 'self' and clone_model is not None:
        return get_clone_action(state, num_actions, clone_model)

    else:
        return random.choice(state.legal_actions())


def get_clone_action(state, num_actions: int, clone_model: ActorCritic) -> int:
    """Get action from clone model."""
    legal_actions = state.legal_actions()
    current_player = state.current_player()

    obs = torch.tensor(
        state.observation_tensor(current_player),
        dtype=torch.float32
    ).unsqueeze(0)
    mask = torch.tensor(
        get_legal_mask(state, num_actions),
        dtype=torch.float32
    ).unsqueeze(0)

    with torch.no_grad():
        logits, _ = clone_model(obs)
        masked = logits.squeeze(0).float()
        masked[mask.squeeze(0) == 0] = -1e9
        action = masked.argmax().item()

    return action


def worker_process(
    worker_id: int,
    config: Config,
    obs_size: int,
    num_actions: int,
    request_queue: Queue,
    response_queue: Queue,
    experience_queue: Queue,
    control_queue: Queue,
    ready_event: Event,
    pause_event: Event,
    resume_event: Event,
    shared_state: Dict[str, Any],
):
    """
    Worker process that collects experiences with curriculum-based opponents.

    Game settings (random_moves for board preparation) are received via control queue.
    """
    np.random.seed(worker_id + int(time.time() * 1000) % 2**31)
    random.seed(worker_id + int(time.time() * 1000) % 2**31)

    num_envs = config.envs_per_worker

    # Current game settings (updated via control queue)
    current_random_moves = 150  # Number of random moves to prepare board

    # Create game using pyspiel with position 0 bug fix
    game = load_game_fixed("nine_mens_morris")

    # Initialize reward calculator
    default_reward_config = {
        'win_reward_base': 1.0,
        'win_reward_speed_bonus': 1.0,
        'loss_reward': -1.5,
        'draw_penalty': -0.5,
        'mill_reward': 0.3,
        'enemy_mill_penalty': -0.3,
        'block_mill_reward': 0.2,
        'double_mill_reward': 0.5,
        'double_mill_extra_reward': 0.8,
        'setup_capture_reward': 0.2,
    }
    reward_calculator = RewardCalculator(default_reward_config)

    envs = [EnvState(game, reward_calculator) for _ in range(num_envs)]

    # Current curriculum settings
    current_opponent_type = 'random'
    current_minimax_depth = 1
    current_reward_config = default_reward_config.copy()

    # For mixed mode: opponent selection per game
    current_opponent_mix = None  # Dict with 'minimax', 'self', 'random' probabilities
    current_minimax_min_depth = 1
    current_minimax_max_depth = 4

    # Clone model for self-play
    clone_model: Optional[ActorCritic] = None

    ready_event.set()
    running = True
    request_counter = 0

    def count_stones(state) -> Tuple[int, int]:
        """Count stones for each player from the state."""
        try:
            # Use observation tensor to count pieces
            obs = state.observation_tensor(0)
            player0_pieces = sum(1 for i in range(24) if obs[i] == 1)
            player1_pieces = sum(1 for i in range(24) if obs[i + 24] == 1)
            return player0_pieces, player1_pieces
        except:
            return 9, 9  # Default if parsing fails

    def play_random_moves(state, num_moves: int):
        """
        Play random moves to prepare the board for training.
        Stops early if a player reaches 3 stones (about to lose).

        Args:
            state: Initial game state
            num_moves: Target number of random moves

        Returns:
            The state after random moves (modified in place)
        """
        moves_made = 0
        while moves_made < num_moves and not state.is_terminal():
            # Check if either player has only 3 stones - stop early
            p0_stones, p1_stones = count_stones(state)
            if p0_stones <= 3 or p1_stones <= 3:
                # One player is about to lose, stop random moves
                break

            legal_actions = state.legal_actions()
            if not legal_actions:
                break

            action = random.choice(legal_actions)
            state.apply_action(action)
            moves_made += 1

        return state

    def recreate_game():
        """Recreate game with current settings."""
        nonlocal game, envs
        game = load_game_fixed("nine_mens_morris")
        # Reinitialize all environments with new game
        envs = [EnvState(game, reward_calculator) for _ in range(num_envs)]
        for env in envs:
            setup_new_game(env)

    def select_mixed_opponent() -> Tuple[str, int]:
        """Select opponent for mixed training mode with random minimax depth."""
        if current_opponent_mix is None:
            return ('random', 0)

        roll = random.random()
        minimax_prob = current_opponent_mix.get('minimax', 0.35)
        self_prob = current_opponent_mix.get('self', 0.60)

        if roll < minimax_prob:
            # Random depth selection (no progressive rounds)
            depth = random.randint(current_minimax_min_depth, current_minimax_max_depth)
            return ('minimax', depth)
        elif roll < minimax_prob + self_prob:
            return ('self', 0)
        else:
            return ('random', 0)

    def setup_new_game(env: EnvState):
        """Set up a new game with current curriculum settings."""
        nonlocal game

        env.game = game
        env.reset()

        # Determine number of random moves to prepare the board
        num_random = current_random_moves
        if num_random < 0:
            # Phase 10: random between 0 and 150
            num_random = random.randint(0, 150)

        # Play random moves to prepare the board (not recorded in training)
        if num_random > 0 and not env.state.is_terminal():
            play_random_moves(env.state, num_random)

        # If the random moves resulted in terminal state, reset and try again
        if env.state.is_terminal():
            env.reset()
            # Try with fewer random moves
            if num_random > 50:
                play_random_moves(env.state, num_random // 2)

        if current_opponent_type == 'mixed':
            opp_type, depth = select_mixed_opponent()
            env.setup_opponent(opp_type, depth)
        else:
            env.setup_opponent(current_opponent_type, current_minimax_depth)

        env.reward_calculator.update_config(current_reward_config)

    # Initialize all environments
    for env in envs:
        setup_new_game(env)

    while running:
        # Check for pause signal
        if pause_event.is_set():
            resume_event.wait()
            resume_event.clear()
            while True:
                try:
                    response_queue.get_nowait()
                except queue.Empty:
                    break
            request_counter += 1000
            continue

        # Check control messages
        try:
            msg = control_queue.get_nowait()
            if msg['type'] == 'stop':
                running = False
                break
            elif msg['type'] == 'update_curriculum':
                current_opponent_type = msg.get('opponent_type', current_opponent_type)
                current_minimax_depth = msg.get('minimax_depth', current_minimax_depth)
                if 'reward_config' in msg:
                    current_reward_config = msg['reward_config']
                    reward_calculator.update_config(current_reward_config)
                if 'opponent_mix' in msg:
                    current_opponent_mix = msg['opponent_mix']

            elif msg['type'] == 'update_game_settings':
                # New game settings (random moves for board preparation)
                new_random_moves = msg.get('random_moves', current_random_moves)
                if new_random_moves != current_random_moves:
                    current_random_moves = new_random_moves
                    recreate_game()

            elif msg['type'] == 'update_clone':
                if 'clone_state_dict' in msg:
                    if clone_model is None:
                        clone_model = ActorCritic(obs_size, num_actions, config)
                    clone_model.load_state_dict(msg['clone_state_dict'])
                    clone_model.eval()

            elif msg['type'] == 'update_minimax_range':
                # Update minimax depth range (for Phase 10)
                current_minimax_min_depth = msg.get('min_depth', 1)
                current_minimax_max_depth = msg.get('max_depth', 4)

        except queue.Empty:
            pass

        # Collect observations needing inference
        inference_requests = []
        opponent_actions = []

        for env_idx, env in enumerate(envs):
            state = env.state

            # Check terminal
            if state.is_terminal() or env.step_count >= config.max_game_steps:
                for player in [0, 1]:
                    if env.experiences[player]:
                        if state.is_terminal():
                            final_reward = env.reward_calculator.calculate_terminal_reward(
                                state.returns(), player, env.step_count, config.max_game_steps
                            )
                        else:
                            final_reward = env.reward_calculator.calculate_timeout_penalty()

                        env.experiences[player][-1]['reward'] += final_reward
                        env.experiences[player][-1]['done'] = 1.0

                        rewards = np.array([e['reward'] for e in env.experiences[player]], dtype=np.float32)
                        values = np.array([e['value'] for e in env.experiences[player]], dtype=np.float32)

                        advantages, returns = compute_gae(
                            rewards, values,
                            gamma=config.gamma,
                            gae_lambda=config.gae_lambda
                        )

                        batch = ExperienceBatch(
                            obs=np.stack([e['obs'] for e in env.experiences[player]]),
                            actions=np.array([e['action'] for e in env.experiences[player]], dtype=np.int64),
                            logprobs=np.array([e['logprob'] for e in env.experiences[player]], dtype=np.float32),
                            values=values,
                            rewards=rewards,
                            dones=np.array([e['done'] for e in env.experiences[player]], dtype=np.float32),
                            masks=np.stack([e['mask'] for e in env.experiences[player]]),
                            advantages=advantages,
                            returns=returns,
                            game_result=final_reward,
                            game_steps=env.step_count,
                            opponent_type=env.opponent_type,
                            minimax_depth=env.minimax_depth
                        )
                        experience_queue.put(batch)

                setup_new_game(env)
                continue

            current_player = state.current_player()

            if env.opponent_type == 'self' and clone_model is not None:
                is_ai_turn = (current_player == env.ai_player)
                if is_ai_turn:
                    obs = np.array(state.observation_tensor(current_player), dtype=np.float32)
                    mask = get_legal_mask(state, num_actions)
                    inference_requests.append({
                        'env_idx': env_idx,
                        'player': current_player,
                        'obs': obs,
                        'mask': mask,
                        'is_ai_player': True
                    })
                else:
                    action = get_clone_action(state, num_actions, clone_model)
                    opponent_actions.append({
                        'env_idx': env_idx,
                        'player': current_player,
                        'action': action
                    })
            elif env.opponent_type == 'self':
                obs = np.array(state.observation_tensor(current_player), dtype=np.float32)
                mask = get_legal_mask(state, num_actions)
                inference_requests.append({
                    'env_idx': env_idx,
                    'player': current_player,
                    'obs': obs,
                    'mask': mask,
                    'is_ai_player': current_player == env.ai_player
                })
            elif current_player == env.ai_player:
                obs = np.array(state.observation_tensor(current_player), dtype=np.float32)
                mask = get_legal_mask(state, num_actions)
                inference_requests.append({
                    'env_idx': env_idx,
                    'player': current_player,
                    'obs': obs,
                    'mask': mask,
                    'is_ai_player': True
                })
            else:
                action = get_opponent_action(env, state, num_actions, clone_model)
                opponent_actions.append({
                    'env_idx': env_idx,
                    'player': current_player,
                    'action': action
                })

        # Apply opponent actions immediately
        for opp in opponent_actions:
            env = envs[opp['env_idx']]
            state = env.state
            player = opp['player']
            action = opp['action']

            # Validate action is legal (state may have changed)
            legal_actions = state.legal_actions()
            if not legal_actions:
                continue
            if action not in legal_actions:
                action = random.choice(legal_actions)

            ai_player = env.ai_player
            prev_ai_pieces = env.pieces[ai_player]

            state.apply_action(action)
            env.step_count += 1

            if not state.is_terminal():
                my_pieces, opp_pieces = count_pieces_from_state(state, player)
                env.pieces[player] = my_pieces
                env.pieces[1 - player] = opp_pieces

                new_ai_pieces = env.pieces[ai_player]
                if new_ai_pieces < prev_ai_pieces:
                    pieces_lost = prev_ai_pieces - new_ai_pieces
                    if env.experiences[ai_player]:
                        penalty = env.reward_calculator.config.get('enemy_mill_penalty', -0.3)
                        env.experiences[ai_player][-1]['reward'] += penalty * pieces_lost

        if not inference_requests:
            time.sleep(0.001)
            continue

        # Send batch request
        request_counter += 1
        current_request_id = request_counter

        request_queue.put({
            'worker_id': worker_id,
            'request_id': current_request_id,
            'num_requests': len(inference_requests),
            'requests': inference_requests
        })

        # Wait for response
        response = None
        attempts = 0
        while attempts < 10:
            try:
                resp = response_queue.get(timeout=0.5)
                if resp.get('request_id') == current_request_id:
                    response = resp
                    break
                attempts += 1
            except queue.Empty:
                if pause_event.is_set():
                    break
                attempts += 1

        if response is None:
            continue

        # Apply actions from inference
        for i, req in enumerate(inference_requests):
            env_idx = req['env_idx']
            player = req['player']
            obs = req['obs']
            mask = req['mask']
            is_ai_player = req['is_ai_player']

            action = response['actions'][i]
            logprob = response['logprobs'][i]
            value = response['values'][i]

            env = envs[env_idx]
            state = env.state

            legal_actions = state.legal_actions()
            if not legal_actions:
                # State became terminal between request and response, skip
                continue
            if action not in legal_actions:
                action = random.choice(legal_actions)
                logprob = -np.log(len(legal_actions))

            prev_my_pieces, prev_opp_pieces = count_pieces_from_state(state, player)
            prev_state_info = {
                'my_pieces': prev_my_pieces,
                'opp_pieces': prev_opp_pieces,
            }

            state.apply_action(action)
            env.step_count += 1

            shaping_reward = 0.0
            if not state.is_terminal():
                new_my_pieces, new_opp_pieces = count_pieces_from_state(state, player)
                env.pieces[player] = new_my_pieces
                env.pieces[1 - player] = new_opp_pieces

                new_state_info = {
                    'my_pieces': new_my_pieces,
                    'opp_pieces': new_opp_pieces,
                }

                shaping_reward = env.reward_calculator.calculate_shaping_reward(
                    prev_state_info, new_state_info, player
                )

            if env.opponent_type == 'self' or is_ai_player:
                env.experiences[player].append({
                    'obs': obs,
                    'action': action,
                    'logprob': logprob,
                    'value': value,
                    'reward': shaping_reward,
                    'done': 0.0,
                    'mask': mask
                })

    print(f"Worker {worker_id} finished")
