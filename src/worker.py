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
from concurrent.futures import ThreadPoolExecutor, Future
import numpy as np
import torch
import torch.nn.functional as F
from multiprocessing import Queue, Event
from typing import Dict, Any, Optional, Tuple

from config import Config
from model import ActorCritic
from utils import (
    get_legal_mask, count_pieces_from_state, extract_state_features,
    compute_gae, ExperienceBatch, RewardCalculator, relativize_obs
)

import fastnmm
from fastnmm import MinimaxBot

_DEFAULT_TT_BYTES_PER_BOT = 128 * 1024 * 1024  # matches Config default


class _RandomizedMinimaxBot:
    """fastnmm.MinimaxBot with a per-call `random_move_prob` exploration roll.

    Wraps the C++ bot so we can keep the curriculum's "30% random override"
    behaviour without modifying fastnmm. The wrapped bot owns a persistent
    per-process TT.
    """

    def __init__(self, depth: int, random_move_prob: float = 0.0,
                 player_id: int = 0,
                 tt_bytes: int = _DEFAULT_TT_BYTES_PER_BOT):
        self.depth = int(depth)
        self.random_move_prob = float(random_move_prob)
        self._bot = MinimaxBot(
            player_id=player_id, depth=self.depth, tt_bytes=int(tt_bytes),
        )

    def step(self, state) -> int:
        if self.random_move_prob > 0.0 and random.random() < self.random_move_prob:
            return random.choice(state.legal_actions())
        return self._bot.step(state)

    # Legacy alias used elsewhere in the codebase.
    get_action = step


class MinimaxBotPool:
    """Pool of persistent fastnmm MinimaxBots keyed by depth.

    `tt_bytes` is forwarded to each lazily-created bot.
    """

    def __init__(self, tt_bytes: int = _DEFAULT_TT_BYTES_PER_BOT):
        self._bots: Dict[int, _RandomizedMinimaxBot] = {}
        self._tt_bytes = int(tt_bytes)

    def get(self, depth: int) -> _RandomizedMinimaxBot:
        if depth not in self._bots:
            self._bots[depth] = _RandomizedMinimaxBot(
                depth=depth, random_move_prob=0.0,
                tt_bytes=self._tt_bytes,
            )
        return self._bots[depth]


class EnvState:
    """State for a single environment."""

    def __init__(self, game, reward_calculator: RewardCalculator):
        self.game = game
        self.reward_calculator = reward_calculator
        self.reset()

    def reset(self, starting_stones: Optional[Tuple[int, int]] = None):
        if starting_stones is None:
            self.state = self.game.new_initial_state()
            init_pieces = (9, 9)
        else:
            self.state = self.game.new_initial_state(starting_stones=starting_stones)
            init_pieces = starting_stones
        self.step_count = 0
        self.experiences = {0: [], 1: []}
        self.pieces = {0: init_pieces[0], 1: init_pieces[1]}

        # Opponent settings (set by worker based on curriculum)
        self.opponent_type = 'random'
        self.ai_player = random.randint(0, 1)
        self.minimax_bot = None
        self.minimax_depth = 0
        self.clone_player = 1 - self.ai_player

        # Async minimax state
        self.pending_minimax: Optional[Future] = None

    def setup_opponent(self, opponent_type: str, minimax_depth: int = 1,
                       bot_pool: 'MinimaxBotPool' = None):
        """Configure opponent for this game."""
        self.opponent_type = opponent_type
        self.minimax_depth = minimax_depth if opponent_type == 'minimax' else 0

        if opponent_type == 'minimax' and bot_pool is not None:
            self.minimax_bot = bot_pool.get(minimax_depth)
        elif opponent_type == 'minimax':
            self.minimax_bot = _RandomizedMinimaxBot(
                depth=minimax_depth, random_move_prob=0.0,
            )
        else:
            self.minimax_bot = None


def get_opponent_action(env: EnvState, state, num_actions: int, clone_model: Optional[ActorCritic] = None) -> int:
    """Get action for opponent based on opponent type."""
    if env.opponent_type == 'random':
        return random.choice(state.legal_actions())

    elif env.opponent_type == 'minimax':
        if env.minimax_bot is None:
            return random.choice(state.legal_actions())
        return env.minimax_bot.step(state)

    elif env.opponent_type == 'self' and clone_model is not None:
        return get_clone_action(state, num_actions, clone_model)

    else:
        return random.choice(state.legal_actions())


def get_clone_action(state, num_actions: int, clone_model: ActorCritic) -> int:
    """Get action from clone model."""
    legal_actions = state.legal_actions()
    current_player = state.current_player()

    obs_arr = relativize_obs(state, current_player)
    obs = torch.from_numpy(obs_arr).unsqueeze(0)
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

    Game settings (per-player stone distribution for engine board init) are
    received via control queue.
    """
    np.random.seed(worker_id + int(time.time() * 1000) % 2**31)
    random.seed(worker_id + int(time.time() * 1000) % 2**31)

    # Restrict workers to a sub-range of cores so the X/VNC server keeps
    # CPU available and the desktop stays responsive. Without this, 16+
    # workers each running C++ minimax on a multi-thread executor saturate
    # all cores and the VNC framebuffer stops updating.
    reserved = int(getattr(config, "reserved_display_cores", 0))
    if reserved > 0:
        try:
            available = sorted(os.sched_getaffinity(0))
            worker_cores = [c for c in available if c < (len(available) - reserved)]
            if worker_cores:
                os.sched_setaffinity(0, set(worker_cores))
        except (AttributeError, OSError):
            pass

    nice_level = int(getattr(config, "worker_nice", 0))
    if nice_level > 0:
        try:
            os.nice(nice_level)
        except OSError:
            pass

    num_envs = config.envs_per_worker

    # Bootstrap settings from trainer so workers start in the correct mode
    # even before first control-queue updates are processed.
    initial_stone_distribution_raw = shared_state.get('initial_stone_distribution')
    initial_opponent_type = str(shared_state.get('initial_opponent_type', 'random'))
    initial_minimax_depth = int(shared_state.get('initial_minimax_depth', 1))
    initial_opponent_distribution_raw = shared_state.get('initial_opponent_distribution')
    initial_reward_config_raw = shared_state.get('initial_reward_config')

    # Per-player starting-stone distribution. Each env reset draws two
    # independent samples from this distribution → asymmetric games are common
    # and adjacent phases overlap. Updated via control queue on phase change.
    if isinstance(initial_stone_distribution_raw, (list, tuple)) and initial_stone_distribution_raw:
        current_stone_distribution = [(int(c), float(w)) for c, w in initial_stone_distribution_raw]
    else:
        current_stone_distribution = [(9, 1.0)]

    # Create game using the fastnmm C++ engine
    game = fastnmm.load_game("nine_mens_morris")

    # Fallback reward config used if no initial config is provided.
    default_reward_config = {
        'win_reward_base': 2.0,
        'win_reward_speed_bonus': 1.0,
        'loss_reward': -2.0,
        'draw_penalty': -1.5,
        'mill_reward': 0.3,
        'enemy_mill_penalty': -0.3,
        'block_mill_reward': 0.2,
        'double_mill_reward': 0.5,
        'setup_capture_reward': 0.2,
        'step_penalty': -0.003,
        'piece_advantage_reward': 0.02,
        'mobility_reward': 0.05,
        'max_shaping_abs': 0.20,
        'gamma': 0.99,
    }

    # Current curriculum settings
    current_opponent_type = initial_opponent_type
    current_minimax_depth = initial_minimax_depth
    if isinstance(initial_reward_config_raw, dict):
        current_reward_config = dict(initial_reward_config_raw)
    else:
        current_reward_config = default_reward_config.copy()

    # For mixed mode: full per-opponent sampling distribution sent by the
    # trainer. Keys: 'self', 'random', 'minimax_d{d}'. Each value is the
    # probability of selecting that opponent for the next game. The trainer
    # computes this from the active phase, unlocked depths, and current
    # dampened-set, so the worker just samples a flat categorical.
    current_opponent_distribution: Optional[Dict[str, float]] = (
        dict(initial_opponent_distribution_raw)
        if isinstance(initial_opponent_distribution_raw, dict)
        else None
    )

    reward_calculator = RewardCalculator(current_reward_config)
    envs = [EnvState(game, reward_calculator) for _ in range(num_envs)]

    # Clone model for self-play
    clone_model: Optional[ActorCritic] = None

    # Persistent fastnmm minimax bot pool (one C++ bot per depth).
    # Each bot owns its own transposition table sized via Config.
    bot_pool = MinimaxBotPool(
        tt_bytes=int(getattr(config, "minimax_tt_bytes_per_bot",
                             _DEFAULT_TT_BYTES_PER_BOT)),
    )

    ready_event.set()
    running = True
    request_counter = 0

    def sample_starting_stones() -> Tuple[int, int]:
        """Draw two independent per-player stone counts from the current
        phase's distribution. Asymmetric pairs are common by design — the
        distribution overlaps with adjacent phases to smooth transitions.
        """
        counts = [c for c, _ in current_stone_distribution]
        weights = [w for _, w in current_stone_distribution]
        a = random.choices(counts, weights=weights)[0]
        b = random.choices(counts, weights=weights)[0]
        return (a, b)

    def recreate_game():
        """Recreate game with current settings."""
        nonlocal game, envs
        game = fastnmm.load_game("nine_mens_morris")
        # Reinitialize all environments with new game
        envs = [EnvState(game, reward_calculator) for _ in range(num_envs)]
        for env in envs:
            setup_new_game(env)

    def select_mixed_opponent() -> Tuple[str, int]:
        """Sample one opponent slot from the trainer-supplied distribution.

        `current_opponent_distribution` maps slot keys ('self', 'random',
        'minimax_d{d}') to probabilities that sum to 1.0. The trainer
        recomputes it after every PPO update so dampened slots collapse to
        1% and the remaining mass redistributes by the equal-with-selfplay×3
        rule. The worker just samples a flat categorical.
        """
        dist = current_opponent_distribution
        if not dist:
            # Bootstrap fallback before the first control message arrives.
            return ('self', 0)
        pick = random.choices(list(dist.keys()), weights=list(dist.values()))[0]
        if pick == 'self':
            return ('self', 0)
        if pick == 'random':
            return ('random', 0)
        # 'minimax_d{n}'
        return ('minimax', int(pick.split('_d')[1]))

    def setup_new_game(env: EnvState):
        """Set up a new game with current curriculum settings."""
        nonlocal game

        env.game = game
        env.reset(starting_stones=sample_starting_stones())

        if current_opponent_type == 'mixed':
            opp_type, depth = select_mixed_opponent()
            env.setup_opponent(opp_type, depth, bot_pool=bot_pool)
        else:
            env.setup_opponent(current_opponent_type, current_minimax_depth, bot_pool=bot_pool)

        env.reward_calculator.update_config(current_reward_config)

    # Initialize all environments
    for env in envs:
        setup_new_game(env)

    # Thread pool for async minimax. Sized via Config: too many threads
    # across all workers saturates the box and starves the display server.
    minimax_thread_count = max(1, int(getattr(config, "minimax_threads_per_worker", 2)))
    minimax_executor = ThreadPoolExecutor(max_workers=minimax_thread_count)

    def apply_opponent_action(env: EnvState, player: int, action: int):
        """Apply an opponent action and track shaping penalties."""
        state = env.state
        legal_actions = state.legal_actions()
        if not legal_actions:
            return
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

    def finalize_game(env: EnvState):
        """Finalize a completed game and submit experience batches."""
        state = env.state
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

        # Check and drain all pending control messages to avoid stale settings.
        recreate_after_control = False
        while True:
            try:
                msg = control_queue.get_nowait()
            except queue.Empty:
                break

            if msg['type'] == 'stop':
                running = False
                break

            if msg['type'] == 'update_curriculum':
                old_opp_type = current_opponent_type
                old_dist = current_opponent_distribution

                current_opponent_type = msg.get('opponent_type', current_opponent_type)
                current_minimax_depth = msg.get('minimax_depth', current_minimax_depth)
                if 'reward_config' in msg:
                    current_reward_config = msg['reward_config']
                    reward_calculator.update_config(current_reward_config)
                if 'opponent_distribution' in msg:
                    current_opponent_distribution = dict(msg['opponent_distribution'])

                if (current_opponent_type != old_opp_type
                        or current_opponent_distribution != old_dist):
                    recreate_after_control = True
                continue

            if msg['type'] == 'update_game_settings':
                new_dist_raw = msg.get('stone_distribution')
                if new_dist_raw is not None:
                    new_dist = [(int(c), float(w)) for c, w in new_dist_raw]
                    if new_dist != current_stone_distribution:
                        current_stone_distribution = new_dist
                        recreate_after_control = True
                continue

            if msg['type'] == 'update_clone':
                if 'clone_state_dict' in msg:
                    if clone_model is None:
                        clone_model = ActorCritic(obs_size, num_actions, config)
                    clone_model.load_state_dict(msg['clone_state_dict'])
                    clone_model.eval()
                continue

            if msg['type'] == 'update_opponent_distribution':
                raw = msg.get('distribution') or {}
                current_opponent_distribution = {
                    str(k): max(0.0, float(v)) for k, v in raw.items()
                }
                continue

        if not running:
            break

        if recreate_after_control:
            recreate_game()

        # --- Phase 1: Collect completed async minimax results ---
        for env in envs:
            if env.pending_minimax is not None and env.pending_minimax.done():
                action = env.pending_minimax.result()
                player = env.state.current_player()
                apply_opponent_action(env, player, action)
                env.pending_minimax = None

        # --- Phase 2: Collect observations / submit opponent actions ---
        inference_requests = []

        for env_idx, env in enumerate(envs):
            # Skip envs waiting for async minimax
            if env.pending_minimax is not None:
                continue

            state = env.state

            # Check terminal
            if state.is_terminal() or env.step_count >= config.max_game_steps:
                finalize_game(env)
                setup_new_game(env)
                continue

            current_player = state.current_player()

            if env.opponent_type == 'self' and clone_model is not None:
                is_ai_turn = (current_player == env.ai_player)
                if is_ai_turn:
                    obs = relativize_obs(state, current_player)
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
                    apply_opponent_action(env, current_player, action)
            elif env.opponent_type == 'self':
                obs = relativize_obs(state, current_player)
                mask = get_legal_mask(state, num_actions)
                inference_requests.append({
                    'env_idx': env_idx,
                    'player': current_player,
                    'obs': obs,
                    'mask': mask,
                    'is_ai_player': current_player == env.ai_player
                })
            elif current_player == env.ai_player:
                obs = relativize_obs(state, current_player)
                mask = get_legal_mask(state, num_actions)
                inference_requests.append({
                    'env_idx': env_idx,
                    'player': current_player,
                    'obs': obs,
                    'mask': mask,
                    'is_ai_player': True
                })
            else:
                # Opponent turn (non-AI)
                if env.opponent_type == 'minimax' and env.minimax_bot is not None:
                    # Submit minimax to thread pool — don't block
                    state_clone = state.clone()
                    bot = env.minimax_bot
                    env.pending_minimax = minimax_executor.submit(bot.step, state_clone)
                else:
                    action = get_opponent_action(env, state, num_actions, clone_model)
                    apply_opponent_action(env, current_player, action)

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
                continue
            if action not in legal_actions:
                action = random.choice(legal_actions)
                logprob = -np.log(len(legal_actions))

            prev_state_info = extract_state_features(state, player)

            state.apply_action(action)
            env.step_count += 1

            shaping_reward = 0.0
            if not state.is_terminal():
                new_state_info = extract_state_features(state, player)
                env.pieces[player] = int(new_state_info.get('my_pieces', env.pieces[player]))
                env.pieces[1 - player] = int(new_state_info.get('opp_pieces', env.pieces[1 - player]))

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

    minimax_executor.shutdown(wait=False)
    print(f"Worker {worker_id} finished")
