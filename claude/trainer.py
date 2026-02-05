"""
Nine Men's Morris - PPO Trainer with Curriculum Learning
Main training loop with phased curriculum progression

Phase Structure:
- Phase 1: 3 stones jumping, vs random (warmup)
- Phase 2-8: 3-9 stones, mixed opponents (30% minimax, 50% self, 20% random)
- Phase 9: Full game (placing phase), mixed opponents
"""

import os
import math
import time
import csv
import random
import queue
from collections import deque
from typing import List, Tuple, Dict, Optional
import multiprocessing as mp
from multiprocessing import Process, Queue, Event, Value

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

import pyspiel
from game_wrapper import load_game as load_game_fixed

from config import Config
from model import ActorCritic
from utils import get_legal_mask, ExperienceBatch
from minimax import evaluate_vs_minimax, format_minimax_results
from worker import worker_process
from curriculum import (
    CurriculumManager, Phase, PHASE_CONFIGS, MIXED_CONFIG,
    calculate_win_reward, calculate_loss_penalty
)
from minimax import MinimaxBot


class PPOTrainer:
    """PPO trainer with curriculum-based training."""

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)

        # Initialize game engine using pyspiel with position 0 bug fix
        game = load_game_fixed("nine_mens_morris")

        self.obs_size = game.observation_tensor_size()
        self.num_actions = game.num_distinct_actions()

        # Initialize model
        self.model = ActorCritic(self.obs_size, self.num_actions, config).to(self.device)
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # Optimizer with separate param groups
        policy_params = []
        value_params = []
        for name, param in self.model.named_parameters():
            if 'value' in name:
                value_params.append(param)
            else:
                policy_params.append(param)

        self.optimizer = torch.optim.AdamW([
            {'params': policy_params, 'lr': config.lr_policy},
            {'params': value_params, 'lr': config.lr_value}
        ], weight_decay=0.01, eps=1e-5)

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

    def _on_phase_change(self, old_phase: Phase, new_phase: Phase):
        """Callback when curriculum phase changes."""
        # Save checkpoint at phase transition
        self.save_checkpoint(f"phase{int(old_phase)}_complete")

        # Update learning rate for new phase
        self._update_learning_rate()

        # Broadcast new game settings and curriculum to workers
        self._broadcast_game_settings()
        self._broadcast_curriculum_update()

        # Initialize clone for mixed phases (Phase 2+)
        if new_phase != Phase.COMPLETED:
            new_config = PHASE_CONFIGS.get(new_phase)
            if new_config and new_config.opponent_type == 'mixed':
                self._update_clone_model()

        # Phase 10: Update minimax depth range to D1-D6
        if new_phase == Phase.PHASE_10:
            self._broadcast_minimax_range(min_depth=1, max_depth=6)

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
        """Callback when clone should be updated (90% WR over 5000 games)."""
        self._update_clone_model()

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
        """Update optimizer learning rate based on curriculum."""
        lr = self.curriculum.get_learning_rate()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

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

        shared_state = {}

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
                    all_experiences.append(batch)
                    all_returns.append(batch.game_result)
                    self.recent_returns.append(batch.game_result)

                    # Update curriculum with game result
                    minimax_depth = getattr(batch, 'minimax_depth', 0)
                    self.curriculum.add_game_result(
                        batch.game_result,
                        opponent_type=batch.opponent_type,
                        minimax_depth=minimax_depth
                    )

            except queue.Empty:
                pass

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

    def _prepare_game_state(self, state, random_moves: int):
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
                self._prepare_game_state(state, random_moves)

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
        eps_per_sec = self.episode_count / elapsed if elapsed > 0 else 0

        # max_depth_beaten, minimax_str = self.evaluate_vs_minimax_progressive()
        max_depth_beaten, minimax_str = -1, -1
        curriculum_status = self.curriculum.get_status_string()
        config = self.curriculum.get_config()

        # Get per-opponent win rates (last 500 games each)
        opp_wr = self.curriculum.get_opponent_win_rates()

        print(
            f"[Phase {int(self.curriculum.current_phase)}] "
            f"Ep {self.episode_count:,} | {curriculum_status} | "
            f"Ret: {avg_return:+.3f} | "
            f"PL: {metrics.get('policy_loss', 0):+.4f} | VL: {metrics.get('value_loss', 0):.3f} | "
            f"LR: {metrics.get('lr', 0):.1e} | "
            f"{eps_per_sec:.0f}/s"
        )
        print(f"  Minimax: {minimax_str}")
        print(f"  WR(500): D1:{opp_wr['wr_vs_mm_d1']:.0%} D2:{opp_wr['wr_vs_mm_d2']:.0%} Rnd:{opp_wr['wr_vs_random']:.0%} Self:{opp_wr['wr_vs_self']:.0%}")

        # CSV logging
        if self.log_file is None:
            path = os.path.join(self.config.log_dir, f"{time.strftime('%Y%m%d_%H%M%S')}_curriculum.csv")
            self.log_file = open(path, 'w', newline='')
            self.log_writer = csv.DictWriter(self.log_file, fieldnames=[
                'episode', 'phase', 'random_moves', 'steps', 'avg_return', 'ema_return',
                'win_rate', 'draw_rate', 'policy_loss', 'value_loss', 'entropy', 'kl_div',
                'clip_frac', 'grad_norm', 'lr', 'minimax_depth_beaten', 'clone_gen',
                'wr_vs_mm_d1', 'wr_vs_mm_d2', 'wr_vs_random', 'wr_vs_self'
            ])
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
            'wr_vs_mm_d1': opp_wr['wr_vs_mm_d1'],
            'wr_vs_mm_d2': opp_wr['wr_vs_mm_d2'],
            'wr_vs_random': opp_wr['wr_vs_random'],
            'wr_vs_self': opp_wr['wr_vs_self'],
        })
        self.log_file.flush()

    def save_checkpoint(self, prefix="checkpoint"):
        """Save model checkpoint."""
        path = os.path.join(
            self.config.checkpoint_dir,
            f"{time.strftime('%Y%m%d_%H%M%S')}_{prefix}_ep{self.episode_count}.pt"
        )
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
        }, path)
        print(f"  Saved: {path}")

        self.curriculum.save_state()

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
            if phase_cfg.opponent_type == 'random':
                grad_str = f"{phase_cfg.win_rate_threshold:.0%} WR vs random"
            else:
                grad_str = f"{MIXED_CONFIG['graduation_episodes']/1e6:.1f}M episodes"
            print(f"  Phase {int(phase)}: "
                  f"LR {phase_cfg.lr_start:.0e}->{phase_cfg.lr_end:.0e} | {phase_cfg.description[:30]}")
        print()

        print(f"Mixed Training (Phase 2-9):")
        mix = MIXED_CONFIG['opponent_mix']
        print(f"  Opponent mix: {mix['minimax']*100:.0f}% minimax, {mix['self']*100:.0f}% self-play, {mix['random']*100:.0f}% random")
        print(f"  Minimax: Random D{MIXED_CONFIG['minimax_min_depth']}-D{MIXED_CONFIG['minimax_max_depth']}")
        print(f"  Self-play: Clone update at {MIXED_CONFIG['selfplay_winrate_threshold']*100:.0f}% WR over {MIXED_CONFIG['selfplay_winrate_games']} games")
        print(f"  Graduation: {MIXED_CONFIG['graduation_episodes']/1e6:.1f}M episodes per phase")
        print()

        print(f"Phase 10 (Final):")
        print(f"  Minimax: Random D1-D6, no shaping, 1M episodes")
        print()

        config = self.curriculum.get_config()
        random_moves = self.curriculum.get_random_moves_for_phase()
        print(f"Starting Phase {int(self.curriculum.current_phase)}: {config.description}")
        print(f"  Random moves for board prep: {random_moves if random_moves >= 0 else '0-150 (random)'}")
        print("=" * 70)

        self.start_time = time.time()

        self.start_workers()

        try:
            while self.curriculum.current_phase != Phase.COMPLETED:
                if self.episode_count >= cfg.total_episodes:
                    print(f"\nReached max episodes ({cfg.total_episodes:,})")
                    break

                # Collect experiences
                experiences, returns = self.collect_experiences(cfg.episodes_per_update)

                # PPO update
                self.pause_workers()
                metrics = self.update_policy(experiences)

                self._update_learning_rate()

                self.resume_workers()

                # Check for clone update (90% WR over 5000 self-play games)
                if self.curriculum.should_update_clone():
                    self.pause_workers()
                    self.curriculum.do_clone_update()
                    self.resume_workers()

                # Logging
                if self.episode_count % cfg.log_interval < cfg.episodes_per_update:
                    self.pause_workers()
                    self.log_progress(returns, metrics)
                    self.resume_workers()

                # Check graduation
                if self.episode_count % cfg.graduation_check_interval < cfg.episodes_per_update:
                    self.pause_workers()
                    self.curriculum.check_and_graduate()
                    self.resume_workers()

                # Checkpointing
                if self.episode_count % cfg.save_interval < cfg.episodes_per_update:
                    self.pause_workers()
                    self.save_checkpoint()
                    self.resume_workers()

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
            print(f"\nDone: {self.episode_count:,} episodes in {elapsed / 3600:.1f}h ({self.episode_count / elapsed:.0f}/s)")

            self.curriculum.print_summary()
