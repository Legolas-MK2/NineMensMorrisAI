"""
Nine Men's Morris - PPO Trainer with Curriculum Learning
Main training loop with phased curriculum progression

Phase Structure:
- Phase 1: 3 stones jumping, vs random (warmup)
- Phase 2-9: 3-9 stones, mixed opponents (30% minimax, 60% self, 10% random)
- Phase 10: Full game, harder minimax (35% minimax, 55% self, 10% random)
"""

import os
import time
import csv
import glob
import random
import queue
from collections import deque
from typing import List, Tuple, Dict, Optional
import multiprocessing as mp
from multiprocessing import Process, Queue, Event

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

from game_wrapper import load_game as load_game_fixed

from config import Config
from model import ActorCritic
from utils import get_legal_mask, ExperienceBatch, prepare_game_state
from minimax import evaluate_vs_minimax, format_minimax_results
from worker import worker_process
from curriculum import (
    CurriculumManager, Phase, PHASE_CONFIGS, MIXED_CONFIG,
)
from shared_cache import SharedMoveCache
from lr_scheduler import RLPlateauScheduler

# Once WR vs random exceeds this, random-game experiences are dropped from PPO training
# (games still run for stats/logging purposes)
RANDOM_TRAIN_CUTOFF = 0.95


class PPOTrainer:
    """PPO trainer with curriculum-based training."""

    def __init__(self, config: Config, resume_mode: bool = False):
        self.config = config
        self.device = torch.device(config.device)
        self.resume_mode = resume_mode  # Track if we're resuming

        # Initialize game engine using pyspiel with position 0 bug fix
        game = load_game_fixed("nine_mens_morris")

        self.obs_size = game.observation_tensor_size()
        self.num_actions = game.num_distinct_actions()

        # Store obs shape so the model can encode observations correctly
        config.obs_shape = list(game.observation_tensor_shape())

        # Initialize model
        self.model = ActorCritic(self.obs_size, self.num_actions, config).to(self.device)
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # Optimizer with separate param groups
        # AdamW is superior to SGD for board games with sparse rewards
        policy_params = []
        value_params = []
        for name, param in self.model.named_parameters():
            if 'value' in name:
                value_params.append(param)
            else:
                policy_params.append(param)

        initial_lr = PHASE_CONFIGS[Phase.PHASE_1].lr_start
        self.optimizer = torch.optim.AdamW([
            {'params': policy_params, 'lr': initial_lr},
            {'params': value_params, 'lr': initial_lr}
        ], weight_decay=0.01, eps=1e-5)

        # RL Plateau Scheduler - tracks win rate against hardest opponent (Minimax-2)
        # This prevents premature LR drop when agent beats easy opponents but not hard ones
        self.lr_scheduler = RLPlateauScheduler(
            self.optimizer,
            factor=config.scheduler_factor,       # LR multiplier when reducing
            patience=config.scheduler_patience,   # Evaluation steps before reducing
            min_lr=config.scheduler_min_lr,       # Minimum LR
            threshold=config.scheduler_threshold, # Minimum improvement to reset patience
            target_win_rate=config.scheduler_target_wr, # Track WR target (no hard stop)
            metric_name="wr_vs_mm_d2",
            allow_early_stop=False
        )

        # Mixed precision
        self.scaler = GradScaler('cuda') if config.use_mixed_precision and self.device.type == 'cuda' else None

        # Curriculum manager
        self.curriculum = CurriculumManager(save_dir=config.curriculum_dir)
        self.curriculum.on_phase_change_callbacks.append(self._on_phase_change)
        self.curriculum.on_clone_update_callbacks.append(self._on_clone_update)

        # Multiprocessing components
        self.workers: List[Process] = []
        self.request_queue: Queue = None
        self.response_queues: List[Queue] = []
        self.experience_queue: Queue = None
        self.control_queues: List[Queue] = []
        self.ready_events: List[Event] = []
        self.pause_events: List[Event] = []
        self.resume_events: List[Event] = []

        # Statistics
        self.episode_count = 0
        self.update_count = 0
        self.total_steps = 0
        self.ema_return = None
        self.best_win_rate = 0.0
        self.recent_returns = deque(maxlen=5000)

        self.start_time = None

        # Clone model for self-play
        self.clone_model = None

        # Create directories
        os.makedirs(config.model_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.curriculum_dir, exist_ok=True)

        self.log_file = None
        self.log_writer = None
        self.existing_log_path = None  # Track existing log path for resume mode

    def _on_phase_change(self, old_phase: Phase, new_phase: Phase):
        """Callback when curriculum phase changes."""
        # Save checkpoint at phase transition
        self.save_checkpoint(f"phase{int(old_phase)}_complete")

        # Reset plateau scheduler so prior-phase reductions and best_score
        # (achieved under different game conditions) don't carry forward.
        self.lr_scheduler.reset()

        # Apply new phase's cosine LR with a clean plateau multiplier.
        self._update_learning_rate()

        # Broadcast new game settings and curriculum to workers
        self._broadcast_game_settings()
        self._broadcast_curriculum_update()

        # Initialize clone for mixed phases (Phase 2+)
        if new_phase != Phase.COMPLETED:
            new_config = PHASE_CONFIGS.get(new_phase)
            if new_config and new_config.opponent_type == 'mixed':
                self._update_clone_model()
                # New mixed phase starts at D1 only; D2-D7 unlock progressively via win rate
                self._broadcast_minimax_range(min_depth=1, max_depth=1)

    def _broadcast_minimax_range(self, min_depth: int, max_depth: int):
        """Send minimax depth range to all workers."""
        msg = {
            'type': 'update_minimax_range',
            'min_depth': min_depth,
            'max_depth': max_depth,
        }
        for q in self.control_queues:
            try:
                q.put(msg)
            except:
                pass

    def _update_clone_model(self):
        """Update the clone model with current model weights."""
        if self.clone_model is None:
            self.clone_model = ActorCritic(self.obs_size, self.num_actions, self.config).to(self.device)

        self.clone_model.load_state_dict(self.model.state_dict())
        self.clone_model.eval()

        print(f"  Clone model updated")

        self._broadcast_clone_update()

    def _on_clone_update(self):
        """Callback when clone should be updated (85% WR over 1000 games, cooldown-gated)."""
        self._update_clone_model()
        # Refresh effective LR. Cosine does NOT restart here; the old behavior
        # (lr_start spike on every clone) was the main trigger of policy collapse.
        self._update_learning_rate()

    def _broadcast_clone_update(self):
        """Send updated clone weights to workers."""
        if self.clone_model is None:
            return

        clone_state = self.clone_model.state_dict()
        msg = {
            'type': 'update_clone',
            'clone_state_dict': {k: v.cpu() for k, v in clone_state.items()},
        }

        for q in self.control_queues:
            try:
                q.put(msg)
            except:
                pass

    def _broadcast_game_settings(self):
        """Send game settings to all workers."""
        random_moves = self.curriculum.get_random_moves_for_phase()
        msg = {
            'type': 'update_game_settings',
            'random_moves': random_moves,
        }

        for q in self.control_queues:
            try:
                q.put(msg)
            except:
                pass

    def _update_learning_rate(self):
        """Apply the effective LR: cosine base * plateau multiplier.

        The curriculum provides a cosine schedule (monotonic across a phase).
        The RLPlateauScheduler contributes a multiplicative reduction every
        time the hard-opponent WR plateaus. Composing both avoids the old
        bug where cosine silently clobbered plateau reductions on every loop.
        """
        base_lr = self.curriculum.get_learning_rate()
        config = self.curriculum.get_config()
        base_lr = max(min(base_lr, config.lr_start), config.lr_end)

        plateau_mult = self.lr_scheduler.get_plateau_multiplier()
        lr = max(base_lr * plateau_mult, self.lr_scheduler.min_lr)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.lr_scheduler.notify_current_lr(lr)

    def _broadcast_curriculum_update(self):
        """Send curriculum update to all workers."""
        config = self.curriculum.get_config()
        reward_config = self.curriculum.get_reward_config()

        msg = {
            'type': 'update_curriculum',
            'opponent_type': config.opponent_type,
            'minimax_depth': 1,  # Will be managed per-round
            'reward_config': reward_config,
        }

        # For mixed mode, include opponent mix
        if config.opponent_type == 'mixed':
            msg['opponent_mix'] = MIXED_CONFIG['opponent_mix']

        for q in self.control_queues:
            try:
                q.put(msg)
            except:
                pass

    def get_entropy_coef(self) -> float:
        """Get current entropy coefficient with gradual decay."""
        cfg = self.config
        progress = min(1.0, self.episode_count / cfg.entropy_decay_episodes)
        return cfg.entropy_coef_start + progress * (cfg.entropy_coef_end - cfg.entropy_coef_start)

    def start_workers(self):
        """Start worker processes."""
        print(f"Starting {self.config.num_workers} workers...")

        self.request_queue = mp.Queue()
        self.experience_queue = mp.Queue()

        # Shared move cache: 20 GB lock-free shared memory hash table (D1/D2 only)
        # TT budget: 130 GB split across workers for alpha-beta search (D1-D4)
        MOVE_CACHE_GB = 20.0
        TT_BUDGET_MB = 130 * 1024
        self.shared_move_cache = SharedMoveCache(size_gb=MOVE_CACHE_GB, create=True)
        tt_budget_per_worker = TT_BUDGET_MB // max(1, self.config.num_workers)
        curr_cfg = self.curriculum.get_config()
        curr_reward_cfg = self.curriculum.get_reward_config()
        curr_random_moves = self.curriculum.get_random_moves_for_phase()
        curr_active_mm_max = self.curriculum.get_active_minimax_max_depth()

        shared_state = {
            'shm_cache_size_gb': MOVE_CACHE_GB,
            'tt_budget_mb': tt_budget_per_worker,
            # Seed workers with current curriculum at process start to avoid
            # spending whole batches in fallback random mode if first control
            # messages are delayed or dropped.
            'initial_random_moves': curr_random_moves,
            'initial_opponent_type': curr_cfg.opponent_type,
            'initial_minimax_depth': 1,
            'initial_reward_config': curr_reward_cfg,
            'initial_opponent_mix': dict(MIXED_CONFIG['opponent_mix']) if curr_cfg.opponent_type == 'mixed' else None,
            'initial_minimax_min_depth': 1,
            'initial_minimax_max_depth': curr_active_mm_max,
        }

        for i in range(self.config.num_workers):
            response_q = mp.Queue()
            control_q = mp.Queue()
            ready_evt = mp.Event()
            pause_evt = mp.Event()
            resume_evt = mp.Event()

            p = Process(
                target=worker_process,
                args=(
                    i, self.config, self.obs_size, self.num_actions,
                    self.request_queue, response_q, self.experience_queue,
                    control_q, ready_evt, pause_evt, resume_evt,
                    shared_state
                ),
                daemon=True
            )
            p.start()

            self.workers.append(p)
            self.response_queues.append(response_q)
            self.control_queues.append(control_q)
            self.ready_events.append(ready_evt)
            self.pause_events.append(pause_evt)
            self.resume_events.append(resume_evt)

        for evt in self.ready_events:
            evt.wait(timeout=30)

        print(f"All {self.config.num_workers} workers ready!")

        # Send initial game settings and curriculum
        self._broadcast_game_settings()
        self._broadcast_curriculum_update()

        # Set initial minimax range from curriculum state (D1 on fresh start, unlocked depth on resume)
        active_max = self.curriculum.get_active_minimax_max_depth()
        self._broadcast_minimax_range(min_depth=1, max_depth=max(1, active_max))

        # Initialize clone for mixed phases
        config = self.curriculum.get_config()
        if config.opponent_type == 'mixed':
            self._update_clone_model()

    def stop_workers(self):
        """Stop all worker processes."""
        for q in self.control_queues:
            try:
                q.put({'type': 'stop'})
            except:
                pass

        for p in self.workers:
            p.join(timeout=2)
            if p.is_alive():
                p.terminate()

        # Clean up shared memory move cache
        if hasattr(self, 'shared_move_cache'):
            try:
                self.shared_move_cache.close()
                self.shared_move_cache.unlink()
            except Exception:
                pass

    def pause_workers(self):
        """Pause all workers for PPO update."""
        for evt in self.pause_events:
            evt.set()
        time.sleep(0.1)

        while True:
            try:
                self.request_queue.get_nowait()
            except queue.Empty:
                break

        for resp_q in self.response_queues:
            while True:
                try:
                    resp_q.get_nowait()
                except queue.Empty:
                    break

    def resume_workers(self):
        """Resume all workers after PPO update."""
        for evt in self.pause_events:
            evt.clear()
        for evt in self.resume_events:
            evt.set()

    def process_inference_requests(self, timeout: float = 0.01) -> int:
        """Process batched inference requests from workers."""
        all_requests = []
        worker_indices = {}
        worker_request_ids = {}

        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                request = self.request_queue.get_nowait()
                worker_id = request['worker_id']
                request_id = request.get('request_id', 0)

                if worker_id not in worker_indices:
                    worker_indices[worker_id] = []
                    worker_request_ids[worker_id] = request_id

                for req in request['requests']:
                    worker_indices[worker_id].append(len(all_requests))
                    all_requests.append(req)
            except queue.Empty:
                if all_requests:
                    break
                time.sleep(0.001)

        if not all_requests:
            return 0

        obs_batch = torch.from_numpy(np.stack([r['obs'] for r in all_requests])).to(self.device)
        mask_batch = torch.from_numpy(np.stack([r['mask'] for r in all_requests])).to(self.device)

        with torch.no_grad():
            with autocast('cuda', enabled=self.config.use_mixed_precision):
                logits, values = self.model(obs_batch)

            masked_logits = logits.float()
            masked_logits[mask_batch == 0] = -1e9

            probs = F.softmax(masked_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            actions = dist.sample()
            logprobs = dist.log_prob(actions)

        actions_np = actions.cpu().numpy()
        logprobs_np = logprobs.cpu().numpy()
        values_np = values.cpu().numpy()

        for worker_id, indices in worker_indices.items():
            response = {
                'request_id': worker_request_ids.get(worker_id, 0),
                'actions': [int(actions_np[i]) for i in indices],
                'logprobs': [float(logprobs_np[i]) for i in indices],
                'values': [float(values_np[i]) for i in indices]
            }
            self.response_queues[worker_id].put(response)

        return len(all_requests)

    def collect_experiences(self, target_episodes: int) -> Tuple[List[ExperienceBatch], List[float]]:
        """Collect experiences from workers."""
        all_experiences = []
        all_returns = []

        self.model.eval()

        while len(all_experiences) < target_episodes:
            self.process_inference_requests(timeout=0.005)

            try:
                while True:
                    batch = self.experience_queue.get_nowait()
                    all_returns.append(batch.game_result)
                    self.recent_returns.append(batch.game_result)

                    # Update curriculum with game result (always — for stats/logging)
                    minimax_depth = getattr(batch, 'minimax_depth', 0)
                    self.curriculum.add_game_result(
                        batch.game_result,
                        opponent_type=batch.opponent_type,
                        minimax_depth=minimax_depth
                    )

                    # Drop random-game experiences from PPO training once WR vs random
                    # is at or above the cutoff (games still run for stats above).
                    # In Phase 1 (random-only) we always train; filter only in mixed phases.
                    curr_cfg = self.curriculum.get_config()
                    if (batch.opponent_type == 'random'
                            and curr_cfg.opponent_type == 'mixed'):
                        wr_random = self.curriculum.mixed_state.get_win_rate_vs_opponent('random')
                        if wr_random >= RANDOM_TRAIN_CUTOFF:
                            continue  # play counted for stats, but not for training

                    all_experiences.append(batch)

            except queue.Empty:
                pass

        # Guardrail: mixed phases should include minimax/self games in each batch.
        # If we somehow collect only random games, immediately rebroadcast current
        # curriculum settings so workers can't remain in fallback random-only mode.
        curr_cfg = self.curriculum.get_config()
        if curr_cfg.opponent_type == 'mixed' and all_experiences:
            opp_counts: Dict[str, int] = {}
            for batch in all_experiences:
                opp = getattr(batch, 'opponent_type', 'random')
                opp_counts[opp] = opp_counts.get(opp, 0) + 1

            non_random_games = opp_counts.get('minimax', 0) + opp_counts.get('self', 0)
            if non_random_games == 0:
                print("  Warning: mixed phase batch had 0 minimax/self games; rebroadcasting curriculum.")
                self._broadcast_curriculum_update()
                active_max = self.curriculum.get_active_minimax_max_depth()
                self._broadcast_minimax_range(min_depth=1, max_depth=active_max)

        self.episode_count += len(all_experiences)
        return all_experiences, all_returns

    def update_policy(self, experiences: List[ExperienceBatch]) -> Dict:
        """Perform PPO update."""
        if not experiences:
            return {}

        all_obs = torch.from_numpy(np.concatenate([e.obs for e in experiences])).to(self.device)
        all_actions = torch.from_numpy(np.concatenate([e.actions for e in experiences])).to(self.device)
        all_old_logprobs = torch.from_numpy(np.concatenate([e.logprobs for e in experiences])).to(self.device)
        all_old_values = torch.from_numpy(np.concatenate([e.values for e in experiences])).to(self.device)
        all_masks = torch.from_numpy(np.concatenate([e.masks for e in experiences])).to(self.device)

        advantages = torch.from_numpy(np.concatenate([e.advantages for e in experiences])).to(self.device)
        returns = torch.from_numpy(np.concatenate([e.returns for e in experiences])).to(self.device)

        self.total_steps += len(all_obs)

        with torch.no_grad():
            adv_mean, adv_std = advantages.mean(), advantages.std()
            if adv_std > 1e-8:
                advantages = (advantages - adv_mean) / (adv_std + 1e-8)
            advantages = torch.clamp(advantages, -self.config.advantage_clip, self.config.advantage_clip)

        metrics = {'policy_loss': 0, 'value_loss': 0, 'entropy': 0, 'kl_div': 0, 'clip_frac': 0}
        num_updates = 0
        entropy_coef = self.get_entropy_coef()

        self.model.train()

        for epoch in range(self.config.ppo_epochs):
            indices = torch.randperm(len(all_obs), device=self.device)

            for start in range(0, len(all_obs), self.config.mini_batch_size):
                end = min(start + self.config.mini_batch_size, len(all_obs))
                idx = indices[start:end]

                with autocast('cuda', enabled=self.config.use_mixed_precision and self.device.type == 'cuda'):
                    logits, values = self.model(all_obs[idx])

                    masked_logits = logits.float()
                    masked_logits[all_masks[idx] == 0] = -1e4

                    log_probs = F.log_softmax(masked_logits, dim=-1)
                    new_logprobs = log_probs.gather(-1, all_actions[idx].unsqueeze(-1)).squeeze(-1)

                    probs = F.softmax(masked_logits, dim=-1)
                    entropy = -(probs * log_probs).sum(dim=-1).mean()

                    log_ratio = torch.clamp(
                        new_logprobs - all_old_logprobs[idx],
                        -self.config.log_prob_clip, self.config.log_prob_clip
                    )
                    ratio = torch.clamp(
                        torch.exp(log_ratio),
                        1.0 / self.config.ratio_clip, self.config.ratio_clip
                    )

                    surr1 = ratio * advantages[idx]
                    surr2 = torch.clamp(
                        ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon
                    ) * advantages[idx]
                    policy_loss = -torch.min(surr1, surr2).mean()

                    values_clipped = all_old_values[idx] + torch.clamp(
                        values - all_old_values[idx],
                        -self.config.value_clip, self.config.value_clip
                    )

                    vf_loss1 = (values - returns[idx]) ** 2
                    vf_loss2 = (values_clipped - returns[idx]) ** 2
                    vf_loss_unclipped = torch.max(vf_loss1, vf_loss2)
                    vf_loss_clamped = torch.clamp(vf_loss_unclipped, 0, self.config.value_loss_clamp)
                    value_loss = 0.5 * vf_loss_clamped.mean()

                    loss = policy_loss + self.config.value_coef * value_loss - entropy_coef * entropy

                self.optimizer.zero_grad()
                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    self.optimizer.step()

                with torch.no_grad():
                    kl = (all_old_logprobs[idx] - new_logprobs).mean()
                    clip_frac = ((ratio - 1.0).abs() > self.config.clip_epsilon).float().mean()

                metrics['policy_loss'] += policy_loss.item()
                metrics['value_loss'] += value_loss.item()
                metrics['entropy'] += entropy.item()
                metrics['kl_div'] += kl.item()
                metrics['clip_frac'] += clip_frac.item()
                num_updates += 1

        self.update_count += 1

        for k in metrics:
            metrics[k] /= max(1, num_updates)
        metrics['lr'] = self.optimizer.param_groups[0]['lr']
        metrics['entropy_coef'] = entropy_coef
        metrics['grad_norm'] = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm

        return metrics

    def evaluate_vs_random(self, num_games: int = 200) -> float:
        """Evaluate model against random opponent."""
        game = load_game_fixed("nine_mens_morris")
        random_moves = self.curriculum.get_random_moves_for_phase()
        if random_moves < 0:
            random_moves = 75  # Use average for evaluation
        wins, draws = 0, 0

        self.model.eval()
        with torch.no_grad():
            for i in range(num_games):
                state = game.new_initial_state()

                # Prepare board with random moves
                prepare_game_state(state, random_moves)

                # Skip if game ended during preparation
                if state.is_terminal():
                    continue

                our_player = i % 2
                steps = 0

                while not state.is_terminal() and steps < self.config.max_game_steps:
                    pid = state.current_player()

                    if pid == our_player:
                        obs = torch.tensor(
                            state.observation_tensor(pid),
                            dtype=torch.float32, device=self.device
                        ).unsqueeze(0)
                        mask = torch.tensor(
                            get_legal_mask(state, self.num_actions),
                            dtype=torch.float32, device=self.device
                        ).unsqueeze(0)

                        with autocast('cuda', enabled=self.config.use_mixed_precision):
                            logits, _ = self.model(obs)

                        masked = logits.squeeze(0).float()
                        masked[mask.squeeze(0) == 0] = -1e4
                        action = masked.argmax().item()
                    else:
                        action = random.choice(state.legal_actions())

                    state.apply_action(action)
                    steps += 1

                if state.is_terminal():
                    r = state.returns()
                    if r[our_player] > r[1 - our_player]:
                        wins += 1
                    elif r[our_player] == r[1 - our_player]:
                        draws += 1

        return (wins + 0.5 * draws) / num_games

    def evaluate_vs_minimax_progressive(self) -> Tuple[int, str]:
        """Test AI against progressively harder minimax bots."""
        random_moves = self.curriculum.get_random_moves_for_phase()
        if random_moves < 0:
            random_moves = 75  # Use average for evaluation
        max_depth_beaten, results = evaluate_vs_minimax(
            self.model, self.device, self.num_actions,
            max_depth=6, games_per_depth=10, max_steps=150,
            use_mixed_precision=self.config.use_mixed_precision,
            random_moves=random_moves,
        )
        result_str = format_minimax_results(results)
        return max_depth_beaten, result_str

    def log_progress(self, returns: List[float], metrics: Dict):
        """Log training progress."""
        if not returns:
            return

        avg_return = np.mean(returns)
        self.ema_return = avg_return if self.ema_return is None else 0.95 * self.ema_return + 0.05 * avg_return

        curr_stats = self.curriculum.stats
        win_rate = curr_stats.get_win_rate()
        draw_rate = curr_stats.get_draw_rate()

        elapsed = time.time() - self.start_time
        eps_per_sec = (self.episode_count - self.start_episode_count) / elapsed if elapsed > 0 else 0

        max_depth_beaten, minimax_str = self.evaluate_vs_minimax_progressive()
        curriculum_status = self.curriculum.get_status_string()
        config = self.curriculum.get_config()

        # Get per-opponent win rates (last 500 games each).
        # Phase 1 uses random-only training and does not populate mixed buffers.
        opp_wr = self.curriculum.get_opponent_win_rates()
        if config.opponent_type != 'mixed':
            opp_wr = {
                'wr_vs_mm_d1': 0.0, 'wr_vs_mm_d2': 0.0,
                'wr_vs_mm_d3': 0.0, 'wr_vs_mm_d4': 0.0,
                'wr_vs_mm_d5': 0.0, 'wr_vs_mm_d6': 0.0, 'wr_vs_mm_d7': 0.0,
                'wr_vs_random': win_rate,
                'wr_vs_self': 0.0,
                'active_mm_max_depth': 0,
            }

        # Get scheduler status
        scheduler_best = self.lr_scheduler.get_best_score() if config.opponent_type == 'mixed' else 0.0
        scheduler_lr = self.lr_scheduler.get_current_lr() if config.opponent_type == 'mixed' else metrics.get('lr', 0)
        
        # Main progress line - matches original format
        print(
            f"[Phase {int(self.curriculum.current_phase)}] Ep {self.episode_count:,} | {curriculum_status} | "
            f"Ret: {avg_return:+.3f} | PL: {metrics.get('policy_loss', 0):+.4f} | VL: {metrics.get('value_loss', 0):.3f} | "
            f"LR: {metrics.get('lr', 0):.1e} | {eps_per_sec:.0f}/s"
        )
        
        # Print scheduler status in mixed phases (Phase 2+)
        if config.opponent_type == 'mixed':
            print(
                f"  [Scheduler] Best WR(D2): {scheduler_best:.1%} | Current LR: {scheduler_lr:.1e} | "
                f"Reductions: {self.lr_scheduler.reductions} | Wait: {self.lr_scheduler.wait}/{self.lr_scheduler.patience}"
            )
        
        # Minimax evaluation results
        active_max = opp_wr['active_mm_max_depth']

        def _d_str(d: int) -> str:
            key = f'wr_vs_mm_d{d}'
            if active_max >= d:
                return f" D{d}:{opp_wr.get(key, 0.0):.0%}"
            return f" D{d}:locked"

        depth_str = "".join(_d_str(d) for d in range(1, 8))
        print(f"  Minimax: {minimax_str}")
        if config.opponent_type == 'mixed':
            wr_random = opp_wr['wr_vs_random']
            rnd_train_note = " (no-train)" if wr_random >= RANDOM_TRAIN_CUTOFF else ""
            print(f"  WR(500):{depth_str} [MaxD:{active_max}] "
                  f"Rnd:{wr_random:.0%}{rnd_train_note} Self:{opp_wr['wr_vs_self']:.0%}")
        else:
            print(f"  WR({len(curr_stats.recent_results)}): Rnd:{win_rate:.0%} (mixed opponent WR tracking starts in Phase 2)")

        # CSV logging
        if self.log_file is None:
            # In resume mode, find and append to existing log file
            if self.resume_mode and self.existing_log_path:
                path = self.existing_log_path
                file_mode = 'a'  # Append mode
                write_header = False
            else:
                # Find existing log file to append to (for resume mode without existing_log_path set)
                path = None
                if self.resume_mode:
                    # Find the most recent curriculum log file
                    log_files = glob.glob(os.path.join(self.config.log_dir, "*_curriculum.csv"))
                    if log_files:
                        path = max(log_files, key=os.path.getmtime)
                        self.existing_log_path = path
                        file_mode = 'a'
                        write_header = False
                
                if path is None:
                    path = os.path.join(self.config.log_dir, f"{time.strftime('%Y%m%d_%H%M%S')}_curriculum.csv")
                    file_mode = 'w'
                    write_header = True
            
            self.log_file = open(path, file_mode, newline='')
            self.log_writer = csv.DictWriter(self.log_file, fieldnames=[
                'episode', 'phase', 'random_moves', 'steps', 'avg_return', 'ema_return',
                'win_rate', 'draw_rate', 'policy_loss', 'value_loss', 'entropy', 'kl_div',
                'clip_frac', 'grad_norm', 'lr', 'minimax_depth_beaten', 'clone_gen',
                'active_mm_max_depth', 'shaping_mult',
                'wr_vs_mm_d1', 'wr_vs_mm_d2', 'wr_vs_mm_d3', 'wr_vs_mm_d4',
                'wr_vs_mm_d5', 'wr_vs_mm_d6', 'wr_vs_mm_d7',
                'wr_vs_random', 'wr_vs_self',
                # RL Plateau Scheduler metrics
                'scheduler_best_wr_d2', 'scheduler_current_lr', 'scheduler_reductions', 'scheduler_wait'
            ])
            if write_header:
                self.log_writer.writeheader()

        self.log_writer.writerow({
            'episode': self.episode_count,
            'phase': int(self.curriculum.current_phase),
            'random_moves': self.curriculum.get_random_moves_for_phase(),
            'steps': self.total_steps,
            'avg_return': avg_return,
            'ema_return': self.ema_return,
            'win_rate': win_rate,
            'draw_rate': draw_rate,
            'policy_loss': metrics.get('policy_loss', 0),
            'value_loss': metrics.get('value_loss', 0),
            'entropy': metrics.get('entropy', 0),
            'kl_div': metrics.get('kl_div', 0),
            'clip_frac': metrics.get('clip_frac', 0),
            'grad_norm': metrics.get('grad_norm', 0),
            'lr': metrics.get('lr', 0),
            'minimax_depth_beaten': max_depth_beaten,
            'clone_gen': self.curriculum.mixed_state.clone_generation,
            'active_mm_max_depth': opp_wr['active_mm_max_depth'],
            'shaping_mult': self.curriculum.get_shaping_multiplier(),
            'wr_vs_mm_d1': opp_wr['wr_vs_mm_d1'],
            'wr_vs_mm_d2': opp_wr['wr_vs_mm_d2'],
            'wr_vs_mm_d3': opp_wr['wr_vs_mm_d3'],
            'wr_vs_mm_d4': opp_wr['wr_vs_mm_d4'],
            'wr_vs_mm_d5': opp_wr.get('wr_vs_mm_d5', 0.0),
            'wr_vs_mm_d6': opp_wr.get('wr_vs_mm_d6', 0.0),
            'wr_vs_mm_d7': opp_wr.get('wr_vs_mm_d7', 0.0),
            'wr_vs_random': opp_wr['wr_vs_random'],
            'wr_vs_self': opp_wr['wr_vs_self'],
            # RL Plateau Scheduler metrics
            'scheduler_best_wr_d2': scheduler_best,
            'scheduler_current_lr': scheduler_lr,
            'scheduler_reductions': self.lr_scheduler.reductions,
            'scheduler_wait': self.lr_scheduler.wait,
        })
        self.log_file.flush()

    def save_checkpoint(self, prefix="checkpoint"):
        """Save model checkpoint."""
        path = os.path.join(
            self.config.checkpoint_dir,
            f"{time.strftime('%Y%m%d_%H%M%S')}_{prefix}_ep{self.episode_count}.pt"
        )
        curriculum_state = self.curriculum.to_state_dict()
        lr_scheduler_state = self.lr_scheduler.get_state_dict()
        torch.save({
            'episode': self.episode_count,
            'total_steps': self.total_steps,
            'update_count': self.update_count,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'ema_return': self.ema_return,
            'best_win_rate': self.best_win_rate,
            'curriculum_phase': int(self.curriculum.current_phase),
            'curriculum_state': curriculum_state,
            'lr_scheduler_state': lr_scheduler_state,
        }, path)
        print(f"  Saved: {path}")

        # Keep per-checkpoint sidecar state for backward compatibility/debugging.
        self.curriculum.save_state(path=f"{path}.curriculum.json")
        self.curriculum.save_state()

    def load_weights_only(self, path: str):
        """Load only model weights. Accepts both checkpoint dicts and raw state_dicts."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        # Checkpoints wrap weights in 'model_state_dict'; final models are raw state_dicts.
        state_dict = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
        self.model.load_state_dict(state_dict)
        print(f"  Loaded model weights from: {path}")
        print(f"  Training state reset to beginning (episode 0, Phase 1)")

    def load_checkpoint(self, path: str):
        """Load model from checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        if self.scaler and ckpt.get('scaler_state_dict'):
            try:
                self.scaler.load_state_dict(ckpt['scaler_state_dict'])
            except:
                pass

        self.episode_count = ckpt['episode']
        self.total_steps = ckpt.get('total_steps', 0)
        self.update_count = ckpt.get('update_count', 0)
        self.ema_return = ckpt.get('ema_return')
        self.best_win_rate = ckpt.get('best_win_rate', 0.0)

        # Load LR scheduler state
        if 'lr_scheduler_state' in ckpt:
            self.lr_scheduler.load_state_dict(ckpt['lr_scheduler_state'])
            print(f"  LR scheduler restored: best_score={self.lr_scheduler.get_best_score():.1%}, reductions={self.lr_scheduler.reductions}")

        curriculum_loaded = False
        if ckpt.get('curriculum_state') is not None:
            curriculum_loaded = self.curriculum.load_state_dict(ckpt['curriculum_state'])
        else:
            sidecar_path = f"{path}.curriculum.json"
            if os.path.exists(sidecar_path):
                curriculum_loaded = self.curriculum.load_state(sidecar_path)

        if not curriculum_loaded:
            self.curriculum.load_state()

        print(f"  Loaded from episode {self.episode_count}, Phase {int(self.curriculum.current_phase)}")

    def train(self):
        """Main training loop with curriculum."""
        cfg = self.config

        print("=" * 70)
        print("Nine Men's Morris - Curriculum PPO Training")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Workers: {cfg.num_workers} x {cfg.envs_per_worker} envs = {cfg.num_workers * cfg.envs_per_worker} parallel")
        print()

        print("Training Phases (with random board preparation):")
        for phase, phase_cfg in PHASE_CONFIGS.items():
            print(f"  Phase {int(phase)}: "
                  f"LR {phase_cfg.lr_start:.0e}→{phase_cfg.lr_end:.0e} cosine | {phase_cfg.description[:30]}")
        print()

        print(f"LR schedule: RL Plateau Scheduler based on Win Rate vs Minimax-2")
        print(f"  Cosine annealing per phase, restarts on clone update")
        print(f"  No scheduler auto-stop (phase completion drives termination)")
        print()

        print(f"Mixed Training (Phase 2-9):")
        mix = MIXED_CONFIG['opponent_mix']
        print(f"  Opponent mix: {mix['minimax']*100:.0f}% minimax, {mix['self']*100:.0f}% self-play, {mix['random']*100:.0f}% random")
        print(f"  Minimax: Random D{MIXED_CONFIG['minimax_min_depth']}-D{MIXED_CONFIG['minimax_max_depth']}")
        print(f"  Self-play: Clone update at {MIXED_CONFIG['selfplay_winrate_threshold']*100:.0f}% WR over {MIXED_CONFIG['selfplay_winrate_games']} games")
        print(f"  Graduation: Trend-based (plateau detection < 1° angle over 1M episodes)")
        print()

        print(f"Phase 10 (Final):")
        print(f"  Minimax: Random D1-D4, no shaping, trend-based graduation")
        print()

        config = self.curriculum.get_config()
        random_moves = self.curriculum.get_random_moves_for_phase()
        print(f"Starting Phase {int(self.curriculum.current_phase)}: {config.description}")
        print(f"  Random moves for board prep: {random_moves if random_moves >= 0 else '0-150 (random)'}")
        print("=" * 70)

        self.start_time = time.time()
        self.start_episode_count = self.episode_count

        self.start_workers()

        try:
            while self.curriculum.current_phase != Phase.COMPLETED:
                # Collect experiences (workers run, main serves inference)
                experiences, returns = self.collect_experiences(cfg.episodes_per_update)

                # PPO update — workers are NOT paused.
                # They keep playing (minimax, random, self-play turns) and
                # queue up inference requests. Requests pile up during PPO
                # and get served on the next collect_experiences call.
                metrics = self.update_policy(experiences)

                # Apply cosine * plateau multiplier every loop.
                self._update_learning_rate()

                # Check for clone update (85% WR over 1000 self-play games)
                if self.curriculum.should_update_clone():
                    self.curriculum.do_clone_update()

                # Check for minimax depth unlock (D3 when D1 WR>80%, D4 when D2 WR>80%)
                if self.curriculum.check_and_unlock_minimax_depth():
                    new_max = self.curriculum.get_active_minimax_max_depth()
                    self._broadcast_minimax_range(min_depth=1, max_depth=new_max)
                    self.save_checkpoint(f"depth{new_max}_unlocked")

                # Logging + plateau scheduler step.
                # The scheduler is intentionally stepped here (every log_interval
                # episodes, ~25k) rather than every PPO loop (~8k). patience=10
                # therefore means 10 × 25k = 250k episodes without improvement
                # before a reduction fires — matching the original design intent.
                # Calling it every loop caused 90+ spurious reductions in Phase 2-9.
                if self.episode_count % cfg.log_interval < cfg.episodes_per_update:
                    if config.opponent_type == 'mixed':
                        opp_wr = self.curriculum.get_opponent_win_rates()
                        hard_opponent_wr = opp_wr.get('wr_vs_mm_d2', 0.0)
                        self.lr_scheduler.step(hard_opponent_wr, episode=self.episode_count)
                        self._update_learning_rate()  # re-apply if a reduction just fired

                    self.log_progress(returns, metrics)
                    # Sample minimax winrate for trend-based graduation (phases 2+)
                    self.curriculum.sample_minimax_winrate()

                # Check graduation
                if self.episode_count % cfg.graduation_check_interval < cfg.episodes_per_update:
                    self.curriculum.check_and_graduate()

                # Checkpointing
                if self.episode_count % cfg.save_interval < cfg.episodes_per_update:
                    self.save_checkpoint()

        except KeyboardInterrupt:
            print("\n  Interrupted")
        finally:
            self.stop_workers()

            torch.save(
                self.model.state_dict(),
                os.path.join(cfg.model_dir, f"{time.strftime('%Y%m%d_%H%M%S')}_final.pt")
            )

            self.curriculum.save_state()

            if self.log_file:
                self.log_file.close()

            elapsed = time.time() - self.start_time
            session_eps = self.episode_count - self.start_episode_count
            print(f"\nDone: {session_eps:,} episodes in {elapsed / 3600:.1f}h ({session_eps / elapsed:.0f}/s)")

            self.curriculum.print_summary()

