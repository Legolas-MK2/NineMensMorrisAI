"""
Nine Men's Morris - Worker Process for Random Opponent Training
Simplified worker that only plays against random opponents
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import time
import random
import queue
import numpy as np
import pyspiel
from multiprocessing import Queue, Event
from typing import Dict, Tuple

from config import Config
from utils import get_legal_mask, count_pieces_from_state, compute_gae, ExperienceBatch


class RewardCalculator:
    """Calculates rewards for random opponent training."""

    def __init__(self, config: Config):
        self.config = config

    def calculate_terminal_reward(
        self,
        returns,
        player: int,
        steps: int,
        max_steps: int = 200
    ) -> float:
        """Calculate reward for terminal game state."""
        my_return = returns[player]
        opp_return = returns[1 - player]

        if my_return > opp_return:
            # Win - bonus for fast wins
            base = self.config.win_reward_base
            bonus = self.config.win_reward_speed_bonus
            speed_bonus = bonus * max(0, 1.0 - (steps / max_steps))
            return base + speed_bonus

        elif my_return < opp_return:
            return self.config.loss_reward
        else:
            return self.config.draw_penalty

    def calculate_shaping_reward(
        self,
        prev_state_info: Dict,
        new_state_info: Dict,
        player: int
    ) -> float:
        """Calculate reward shaping for piece captures."""
        reward = 0.0

        prev_opp = prev_state_info.get('opp_pieces', 0)
        new_opp = new_state_info.get('opp_pieces', 0)

        # Did we capture? (opponent lost a piece = we formed a mill)
        pieces_captured = prev_opp - new_opp
        if pieces_captured > 0:
            reward += self.config.mill_reward * pieces_captured
            if pieces_captured >= 2:
                reward += self.config.double_mill_reward

        return reward

    def calculate_timeout_penalty(self) -> float:
        """Penalty for game timing out."""
        return self.config.draw_penalty * 0.8


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
        self.ai_player = random.randint(0, 1)


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
    resume_event: Event
):
    """
    Worker process that collects experiences against random opponents.
    """
    np.random.seed(worker_id + int(time.time() * 1000) % 2**31)
    random.seed(worker_id + int(time.time() * 1000) % 2**31)

    game = pyspiel.load_game("nine_mens_morris")
    num_envs = config.envs_per_worker

    reward_calculator = RewardCalculator(config)
    envs = [EnvState(game, reward_calculator) for _ in range(num_envs)]

    ready_event.set()
    running = True
    request_counter = 0

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
                            opponent_type='random',
                            minimax_depth=0
                        )
                        experience_queue.put(batch)

                env.reset()
                continue

            current_player = state.current_player()

            if current_player == env.ai_player:
                # AI player needs inference
                obs = np.array(state.observation_tensor(current_player), dtype=np.float32)
                mask = get_legal_mask(state, num_actions)

                inference_requests.append({
                    'env_idx': env_idx,
                    'player': current_player,
                    'obs': obs,
                    'mask': mask,
                })
            else:
                # Random opponent
                action = random.choice(state.legal_actions())
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

            # Track AI's pieces BEFORE opponent moves
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
                    pieces_lost = prev_ai_pieces - new_ai_pieces
                    if env.experiences[ai_player]:
                        penalty = config.enemy_mill_penalty
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

            # Store experience
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
