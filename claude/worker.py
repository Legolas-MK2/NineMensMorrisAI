"""
Nine Men's Morris - Worker Process
Experience collection with curriculum-based opponents
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
import pyspiel
from multiprocessing import Queue, Event
from typing import Dict, Any, Optional, Tuple

from config import Config
from model import ActorCritic
from utils import (
    get_legal_mask, count_pieces_from_state,
    compute_gae, ExperienceBatch, RewardCalculator
)
from minimax import MinimaxBot


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

        # These will be set by the worker based on curriculum
        self.opponent_type = 'random'
        self.ai_player = random.randint(0, 1)
        self.minimax_bot = None
        self.minimax_depth = 0
        # For Phase 5: which player is the clone (opponent) in self-play
        self.clone_player = 1 - self.ai_player

    def setup_opponent(self, opponent_type: str, minimax_depth: int = 1):
        """Configure opponent for this game."""
        self.opponent_type = opponent_type
        self.minimax_depth = minimax_depth if opponent_type == 'minimax' else 0

        if opponent_type == 'minimax':
            # Use random_move_prob=0.3 during training to prevent overfitting
            self.minimax_bot = MinimaxBot(max_depth=minimax_depth, random_move_prob=0.3)
        else:
            self.minimax_bot = None


def get_opponent_action(env: EnvState, state, num_actions: int, clone_model: Optional[ActorCritic] = None) -> int:
    """Get action for opponent based on opponent type."""
    if env.opponent_type == 'random':
        return random.choice(state.legal_actions())

    elif env.opponent_type == 'minimax':
        if env.minimax_bot is None:
            # Use random_move_prob=0.3 during training to prevent overfitting
            env.minimax_bot = MinimaxBot(max_depth=env.minimax_depth, random_move_prob=0.3)
        return env.minimax_bot.get_action(state)

    elif env.opponent_type == 'self' and clone_model is not None:
        # Use clone model for opponent in Phase 5
        return get_clone_action(state, num_actions, clone_model)

    else:  # 'self' without clone - use random fallback
        return random.choice(state.legal_actions())


def get_clone_action(state, num_actions: int, clone_model: ActorCritic) -> int:
    """Get action from clone model."""
    # 5% chance to make a random move instead
    legal_actions = state.legal_actions()
    #if random.random() < 0.05:
    #    return random.choice(legal_actions)

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
    shared_state: Dict[str, Any],  # Shared curriculum state
):
    """
    Worker process that collects experiences with curriculum-based opponents.
    
    shared_state contains:
    - opponent_type: str ('random', 'minimax', 'self')
    - minimax_depth: int
    - reward_config: Dict[str, float]
    """
    np.random.seed(worker_id + int(time.time() * 1000) % 2**31)
    random.seed(worker_id + int(time.time() * 1000) % 2**31)

    game = pyspiel.load_game("nine_mens_morris")
    num_envs = config.envs_per_worker

    # Initialize reward calculator with default config
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

    # Current curriculum settings (will be updated via control queue)
    current_opponent_type = 'random'
    current_minimax_depth = 1
    current_reward_config = default_reward_config.copy()
    # For mixed mode (Phase 5)
    current_opponent_mix = None  # Dict with 'self', 'minimax', 'random' probabilities

    # Clone model for Phase 5 self-play evolution
    clone_model: Optional[ActorCritic] = None

    ready_event.set()
    running = True
    request_counter = 0

    def select_mixed_opponent() -> Tuple[str, int]:
        """Select opponent type for mixed training mode."""
        if current_opponent_mix is None:
            return ('random', 0)

        roll = random.random()
        self_prob = current_opponent_mix.get('self', 0.5)
        minimax_prob = current_opponent_mix.get('minimax', 0.35)

        if roll < self_prob:
            return ('self', 0)
        elif roll < self_prob + minimax_prob:
            return ('minimax', current_minimax_depth)
        else:
            return ('random', 0)

    def setup_new_game(env: EnvState):
        """Set up a new game with current curriculum settings."""
        env.reset()

        # For mixed mode, randomly select opponent each game
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
            # Clear stale responses
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
                # Update curriculum settings
                current_opponent_type = msg.get('opponent_type', current_opponent_type)
                current_minimax_depth = msg.get('minimax_depth', current_minimax_depth)
                if 'reward_config' in msg:
                    current_reward_config = msg['reward_config']
                    reward_calculator.update_config(current_reward_config)
                # For Phase 5 mixed training
                if 'opponent_mix' in msg:
                    current_opponent_mix = msg['opponent_mix']
            elif msg['type'] == 'update_clone':
                # Update clone model for Phase 5 self-play
                if 'clone_state_dict' in msg:
                    if clone_model is None:
                        clone_model = ActorCritic(obs_size, num_actions, config)
                    clone_model.load_state_dict(msg['clone_state_dict'])
                    clone_model.eval()
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

                        # Build arrays for this episode
                        rewards = np.array([e['reward'] for e in env.experiences[player]], dtype=np.float32)
                        values = np.array([e['value'] for e in env.experiences[player]], dtype=np.float32)

                        # Compute GAE per-episode
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

            # Determine if this player needs neural network inference
            # In Phase 5 self-play with clone: only AI player uses main NN
            # Clone player uses the local clone model
            if env.opponent_type == 'self' and clone_model is not None:
                # Phase 5: AI vs Clone
                is_ai_turn = (current_player == env.ai_player)
                if is_ai_turn:
                    # AI player needs inference from main model
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
                    # Clone player uses local clone model
                    action = get_clone_action(state, num_actions, clone_model)
                    opponent_actions.append({
                        'env_idx': env_idx,
                        'player': current_player,
                        'action': action
                    })
            elif env.opponent_type == 'self':
                # Self-play without clone (shouldn't happen in Phase 5, but fallback)
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
                # AI player needs inference
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
                # Opponent (random or minimax)
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
            
            # Track AI's pieces BEFORE opponent moves (to detect if opponent captures)
            ai_player = env.ai_player
            prev_ai_pieces = env.pieces[ai_player]

            state.apply_action(action)
            env.step_count += 1

            # Update piece counts
            if not state.is_terminal():
                my_pieces, opp_pieces = count_pieces_from_state(state, player)
                env.pieces[player] = my_pieces
                env.pieces[1 - player] = opp_pieces
                
                # Check if opponent captured one of AI's pieces
                new_ai_pieces = env.pieces[ai_player]
                if new_ai_pieces < prev_ai_pieces:
                    # Opponent made a mill and captured! Add penalty to AI's last experience
                    pieces_lost = prev_ai_pieces - new_ai_pieces
                    if env.experiences[ai_player]:
                        penalty = env.reward_calculator.config.get('enemy_mill_penalty', -0.3)
                        env.experiences[ai_player][-1]['reward'] += penalty * pieces_lost
        
        if not inference_requests:
            time.sleep(0.001)
            continue
        
        # Send batch request for neural network actions
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

            # Verify action is legal
            legal_actions = state.legal_actions()
            
            if action not in legal_actions:
                action = random.choice(legal_actions)
                logprob = -np.log(len(legal_actions))

            # 10% chance to make a random move instead
            #if random.random() < 0.10:
            #    action = random.choice(legal_actions)
            #    logprob = -np.log(len(legal_actions))
            
            # Track state before action for reward shaping
            prev_my_pieces, prev_opp_pieces = count_pieces_from_state(state, player)
            prev_state_info = {
                'my_pieces': prev_my_pieces,
                'opp_pieces': prev_opp_pieces,
            }

            # Apply action
            state.apply_action(action)
            env.step_count += 1

            # Calculate shaping reward
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
            
            # Store experience (for AI player, or both in self-play)
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
