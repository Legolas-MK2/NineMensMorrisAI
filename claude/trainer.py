"""
Nine Men's Morris - PPO Trainer with Curriculum Learning
Main training loop with phased curriculum progression
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

from config import Config
from model import ActorCritic
from utils import get_legal_mask, ExperienceBatch
from minimax import evaluate_vs_minimax, format_minimax_results
from worker import worker_process
from curriculum import (
    CurriculumManager, Phase, PHASE_CONFIGS, PHASE5_CONFIG,
    calculate_win_reward, calculate_loss_penalty, MixedTrainingState
)
from minimax import MinimaxBot


class PPOTrainer:
    """PPO trainer with curriculum-based training."""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize game
        game = pyspiel.load_game("nine_mens_morris")
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
        self.curriculum.on_depth_change_callbacks.append(self._on_depth_change)
        
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

        # Phase 5: Clone model for self-play evolution
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

        # If entering Phase 5, initialize clone model
        if new_phase == Phase.PHASE_5_SELF_PLAY:
            self._update_clone_model()

        # Notify workers of new curriculum settings
        self._broadcast_curriculum_update()

    def _update_clone_model(self):
        """Update the clone model with current model weights."""
        if self.clone_model is None:
            self.clone_model = ActorCritic(self.obs_size, self.num_actions, self.config).to(self.device)

        # Copy weights from current model to clone
        self.clone_model.load_state_dict(self.model.state_dict())
        self.clone_model.eval()

        print(f"  Updated clone model for self-play")

        # Send to workers
        self._broadcast_clone_update()

    def _on_clone_update(self):
        """Callback when clone should be updated (every 100k epochs in Phase 5)."""
        self._update_clone_model()

    def _on_depth_change(self, old_depth: int, new_depth: int):
        """Callback when minimax depth changes in Phase 5."""
        # Broadcast to workers so they use the new depth for minimax games
        self._broadcast_curriculum_update()

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

    def do_full_minimax_eval(self):
        """
        Full minimax evaluation for Phase 5.
        Tests against all depths D1-D6 and prints results.
        """
        print(f"\n{'='*60}")
        print(f"  FULL MINIMAX EVALUATION (every {PHASE5_CONFIG['full_eval_interval']:,} epochs)")
        print(f"  Testing against Minimax D1-D6...")
        print(f"{'='*60}")

        game = pyspiel.load_game("nine_mens_morris")
        self.model.eval()
        results = {}

        for depth in PHASE5_CONFIG['full_eval_depths']:
            bot = MinimaxBot(max_depth=depth)
            wins, losses, draws = 0, 0, 0
            games_to_play = 20  # Play 20 games per depth

            with torch.no_grad():
                for game_idx in range(games_to_play):
                    state = game.new_initial_state()
                    ai_player = game_idx % 2
                    steps = 0

                    while not state.is_terminal() and steps < self.config.max_game_steps:
                        current = state.current_player()

                        if current == ai_player:
                            obs = torch.tensor(
                                state.observation_tensor(current),
                                dtype=torch.float32, device=self.device
                            ).unsqueeze(0)
                            mask = torch.tensor(
                                get_legal_mask(state, self.num_actions),
                                dtype=torch.float32, device=self.device
                            ).unsqueeze(0)

                            with autocast('cuda', enabled=self.config.use_mixed_precision and self.device.type == 'cuda'):
                                logits, _ = self.model(obs)

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
            results[depth] = {'wins': wins, 'draws': draws, 'losses': losses, 'win_rate': win_rate}
            status = "✓" if win_rate >= 0.5 else "✗"
            print(f"  {status} D{depth}: {wins}W/{draws}D/{losses}L (WR: {win_rate:.0%})")

        print(f"{'='*60}\n")
        self.curriculum.reset_full_eval_counter()
        return results

    def _handle_phase5_logic(self):
        """Handle Phase 5 mixed training logic."""
        curriculum = self.curriculum

        # Check if should update clone (every 100k epochs)
        if curriculum.should_update_clone():
            curriculum.do_clone_update()
            # do_clone_update triggers _on_clone_update callback
            return

        # Check if should do full minimax evaluation (every 250k epochs)
        if curriculum.should_do_full_eval():
            self.do_full_minimax_eval()
            return

        # Check if current minimax depth is mastered and should advance
        curriculum.check_minimax_mastery()

    def _update_learning_rate(self):
        """Update optimizer learning rate based on curriculum."""
        lr = self.curriculum.get_learning_rate()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def _broadcast_curriculum_update(self):
        """Send curriculum update to all workers."""
        config = self.curriculum.get_config()
        reward_config = self.curriculum.get_reward_config()
        minimax_depth = self.curriculum.get_minimax_depth()

        # For Phase 5, send mixed training config
        if self.curriculum.current_phase == Phase.PHASE_5_SELF_PLAY:
            msg = {
                'type': 'update_curriculum',
                'opponent_type': 'mixed',  # Special mode for Phase 5
                'minimax_depth': self.curriculum.get_current_minimax_depth(),
                'reward_config': reward_config,
                'opponent_mix': PHASE5_CONFIG['opponent_mix'],
            }
        else:
            msg = {
                'type': 'update_curriculum',
                'opponent_type': config.opponent_type,
                'minimax_depth': minimax_depth,
                'reward_config': reward_config,
            }

        for q in self.control_queues:
            try:
                q.put(msg)
            except:
                pass
    
    def get_entropy_coef(self) -> float:
        """Get current entropy coefficient with gradual decay (like random_train)."""
        cfg = self.config
        # Decay entropy over entropy_decay_episodes (not tied to phase)
        # This matches random_train's approach for consistent exploration
        progress = min(1.0, self.episode_count / cfg.entropy_decay_episodes)
        return cfg.entropy_coef_start + progress * (cfg.entropy_coef_end - cfg.entropy_coef_start)
    
    def start_workers(self):
        """Start worker processes."""
        print(f"Starting {self.config.num_workers} workers...")
        
        self.request_queue = mp.Queue()
        self.experience_queue = mp.Queue()
        
        # Shared curriculum state (simplified - full sync via control queue)
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
        
        # Send initial curriculum settings
        self._broadcast_curriculum_update()
    
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
        
        # Drain queues
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
        
        # Batch inference
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
        
        # Send responses
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
            processed = self.process_inference_requests(timeout=0.005)

            try:
                while True:
                    batch = self.experience_queue.get_nowait()
                    all_experiences.append(batch)
                    all_returns.append(batch.game_result)
                    self.recent_returns.append(batch.game_result)

                    # Update curriculum with game result
                    # batch.opponent_type is 'random', 'minimax', or 'self'
                    # batch.minimax_depth is the depth if opponent is minimax
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
        """Perform PPO update using pre-computed per-episode GAE."""
        if not experiences:
            return {}

        # Stack all experiences
        all_obs = torch.from_numpy(np.concatenate([e.obs for e in experiences])).to(self.device)
        all_actions = torch.from_numpy(np.concatenate([e.actions for e in experiences])).to(self.device)
        all_old_logprobs = torch.from_numpy(np.concatenate([e.logprobs for e in experiences])).to(self.device)
        all_old_values = torch.from_numpy(np.concatenate([e.values for e in experiences])).to(self.device)
        all_masks = torch.from_numpy(np.concatenate([e.masks for e in experiences])).to(self.device)
        
        advantages = torch.from_numpy(np.concatenate([e.advantages for e in experiences])).to(self.device)
        returns = torch.from_numpy(np.concatenate([e.returns for e in experiences])).to(self.device)

        self.total_steps += len(all_obs)

        # Normalize advantages only
        with torch.no_grad():
            adv_mean, adv_std = advantages.mean(), advantages.std()
            if adv_std > 1e-8:
                advantages = (advantages - adv_mean) / (adv_std + 1e-8)
            advantages = torch.clamp(advantages, -self.config.advantage_clip, self.config.advantage_clip)
        
        # PPO epochs
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
                    
                    # Clipped ratio
                    log_ratio = torch.clamp(
                        new_logprobs - all_old_logprobs[idx],
                        -self.config.log_prob_clip, self.config.log_prob_clip
                    )
                    ratio = torch.clamp(
                        torch.exp(log_ratio),
                        1.0 / self.config.ratio_clip, self.config.ratio_clip
                    )
                    
                    # Policy loss
                    surr1 = ratio * advantages[idx]
                    surr2 = torch.clamp(
                        ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon
                    ) * advantages[idx]
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # Value loss with clipping
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
        game = pyspiel.load_game("nine_mens_morris")
        wins, draws = 0, 0
        
        self.model.eval()
        with torch.no_grad():
            for i in range(num_games):
                state = game.new_initial_state()
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
        max_depth_beaten, results = evaluate_vs_minimax(
            self.model, self.device, self.num_actions,
            max_depth=6, games_per_depth=10, max_steps=150,
            use_mixed_precision=self.config.use_mixed_precision
        )
        result_str = format_minimax_results(results)
        return max_depth_beaten, result_str
    
    def log_progress(self, returns: List[float], metrics: Dict):
        """Log training progress with minimax evaluation."""
        if not returns:
            return
        
        avg_return = np.mean(returns)
        self.ema_return = avg_return if self.ema_return is None else 0.95 * self.ema_return + 0.05 * avg_return
        
        # Get curriculum stats
        curr_stats = self.curriculum.stats
        win_rate = curr_stats.get_win_rate()
        draw_rate = curr_stats.get_draw_rate()
        
        elapsed = time.time() - self.start_time
        eps_per_sec = self.episode_count / elapsed if elapsed > 0 else 0
        
        # Always do minimax evaluation
        max_depth_beaten, minimax_str = self.evaluate_vs_minimax_progressive()
        
        # Curriculum status
        curriculum_status = self.curriculum.get_status_string()
        
        print(
            f"[{self.curriculum.current_phase.name}] "
            f"Ep {self.episode_count:,} | {curriculum_status} | "
            f"Ret: {avg_return:+.3f} | "
            f"PL: {metrics.get('policy_loss', 0):+.4f} | VL: {metrics.get('value_loss', 0):.3f} | "
            f"LR: {metrics.get('lr', 0):.1e} | "
            f"{eps_per_sec:.0f}/s"
        )
        print(f"  ⚔ Minimax: {minimax_str}")
        
        # CSV logging
        if self.log_file is None:
            path = os.path.join(self.config.log_dir, f"{time.strftime('%Y%m%d_%H%M%S')}_curriculum.csv")
            self.log_file = open(path, 'w', newline='')
            self.log_writer = csv.DictWriter(self.log_file, fieldnames=[
                'episode', 'phase', 'steps', 'avg_return', 'ema_return', 'win_rate', 'draw_rate',
                'policy_loss', 'value_loss', 'entropy', 'kl_div', 'clip_frac', 'grad_norm', 'lr',
                'minimax_depth', 'minimax_depth_beaten'
            ])
            self.log_writer.writeheader()
        
        self.log_writer.writerow({
            'episode': self.episode_count,
            'phase': int(self.curriculum.current_phase),
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
            'minimax_depth': curr_stats.current_minimax_depth,
            'minimax_depth_beaten': max_depth_beaten,
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
        print(f"✓ Saved: {path}")
        
        # Also save curriculum state
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
        
        # Try to load curriculum state
        self.curriculum.load_state()
        
        print(f"✓ Loaded from episode {self.episode_count}, Phase {int(self.curriculum.current_phase)}")
    
    def train(self):
        """Main training loop with curriculum."""
        cfg = self.config
        
        print("=" * 70)
        print("Nine Men's Morris - Curriculum PPO Training")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Workers: {cfg.num_workers} x {cfg.envs_per_worker} envs = {cfg.num_workers * cfg.envs_per_worker} parallel")
        print()
        
        print("Training Phases:")
        for phase, phase_cfg in PHASE_CONFIGS.items():
            if phase == Phase.PHASE_5_SELF_PLAY:
                grad_str = f"master D{PHASE5_CONFIG['graduation_minimax_depth']} or {PHASE5_CONFIG['graduation_total_episodes']/1e6:.0f}M ep"
            elif phase_cfg.minimax_depth_to_beat > 0:
                grad_str = f"{phase_cfg.win_rate_threshold:.0%} WR vs D{phase_cfg.minimax_depth_to_beat}"
            elif phase_cfg.max_episodes > 0:
                grad_str = f"{phase_cfg.max_episodes//1000}k episodes"
            else:
                grad_str = f"{phase_cfg.win_rate_threshold:.0%} WR"
            print(f"  Phase {int(phase)}: {phase_cfg.opponent_type:10s} | LR {phase_cfg.lr_start:.0e}→{phase_cfg.lr_end:.0e} | Graduate: {grad_str}")
        print()
        print(f"Phase 5 Mixed Training Config:")
        mix = PHASE5_CONFIG['opponent_mix']
        print(f"  - Opponent mix: {mix['self']*100:.0f}% self-play, {mix['minimax']*100:.0f}% minimax, {mix['random']*100:.0f}% random")
        print(f"  - Minimax cycling: D1 → D{PHASE5_CONFIG['minimax_max_depth']} (90% WR to advance)")
        print(f"  - Clone update: every {PHASE5_CONFIG['clone_update_interval']:,} epochs")
        print(f"  - Full eval: every {PHASE5_CONFIG['full_eval_interval']:,} epochs")
        print(f"  - Graduate: master D{PHASE5_CONFIG['graduation_minimax_depth']} or {PHASE5_CONFIG['graduation_total_episodes']/1e6:.0f}M episodes")
        print()
        
        print(f"▶ Starting Phase {int(self.curriculum.current_phase)}: {self.curriculum.get_config().description}")
        print("=" * 70)

        self.start_time = time.time()

        self.start_workers()

        try:
            while self.curriculum.current_phase != Phase.COMPLETED:
                # Check if we've hit max episodes
                if self.episode_count >= cfg.total_episodes:
                    print(f"\nReached max episodes ({cfg.total_episodes:,})")
                    break
                
                # Collect experiences
                experiences, returns = self.collect_experiences(cfg.episodes_per_update)
                
                # PPO update
                self.pause_workers()
                metrics = self.update_policy(experiences)
                
                # Update learning rate based on curriculum
                self._update_learning_rate()
                
                self.resume_workers()
                
                # Logging (includes minimax eval)
                if self.episode_count % cfg.log_interval < cfg.episodes_per_update:
                    self.pause_workers()
                    self.log_progress(returns, metrics)
                    self.resume_workers()
                
                # Check graduation (and depth promotion)
                if self.episode_count % cfg.graduation_check_interval < cfg.episodes_per_update:
                    self.pause_workers()

                    old_depth = self.curriculum.stats.current_minimax_depth
                    graduated = self.curriculum.check_and_graduate()
                    new_depth = self.curriculum.stats.current_minimax_depth

                    # Broadcast if phase changed OR depth changed
                    if graduated or new_depth != old_depth:
                        self._broadcast_curriculum_update()

                    self.resume_workers()

                # Phase 5: Self-play evolution special handling
                if self.curriculum.current_phase == Phase.PHASE_5_SELF_PLAY:
                    self.pause_workers()
                    self._handle_phase5_logic()
                    self.resume_workers()

                # Checkpointing
                if self.episode_count % cfg.save_interval < cfg.episodes_per_update:
                    self.pause_workers()
                    self.save_checkpoint()
                    self.resume_workers()
                    
        except KeyboardInterrupt:
            print("\n✗ Interrupted")
        finally:
            self.stop_workers()
            
            # Save final model
            torch.save(
                self.model.state_dict(),
                os.path.join(cfg.model_dir, f"{time.strftime('%Y%m%d_%H%M%S')}_final.pt")
            )
            
            # Save curriculum state
            self.curriculum.save_state()
            
            if self.log_file:
                self.log_file.close()
            
            elapsed = time.time() - self.start_time
            print(f"\nDone: {self.episode_count:,} episodes in {elapsed / 3600:.1f}h ({self.episode_count / elapsed:.0f}/s)")
            
            # Print curriculum summary
            self.curriculum.print_summary()
