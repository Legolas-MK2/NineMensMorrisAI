"""
Nine Men's Morris - PPO Trainer with Curriculum Learning
Main training loop with phased curriculum progression.

See the module docstring of curriculum.py for the full phase structure.
Training runs until the operator stops Phase 11 (which is infinite).
"""

import os
import json
import time
import csv
import glob
import queue
from typing import List, Tuple, Dict, Optional
import multiprocessing as mp
from multiprocessing import Process, Queue

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

import fastnmm

from config import Config
from model import ActorCritic
from utils import ExperienceBatch
from minimax import evaluate_vs_minimax_cpp, format_minimax_results
from worker import worker_process
from curriculum import (
    CurriculumManager, Phase, PHASE_CONFIGS, MIXED_CONFIG, GRADUATION_CONFIG,
)
from lr_scheduler import WarmRestartLRScheduler
from logging_setup import get_logger

logger = get_logger(__name__)


class PPOTrainer:
    """PPO trainer with curriculum-based training."""

    def __init__(self, config: Config, resume_mode: bool = False):
        self.config = config
        self.device = torch.device(config.device)
        self.resume_mode = resume_mode  # Track if we're resuming

        # Initialize game engine using the fastnmm C++ engine
        game = fastnmm.load_game("nine_mens_morris")

        self.obs_size = game.observation_tensor_size()
        self.num_actions = game.num_distinct_actions()

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

        # Initial optimizer LR will be overwritten immediately by the scheduler
        # (warmup starts at LR ~ 0). Set to lr_peak just for parameter-group setup.
        self.optimizer = torch.optim.AdamW([
            {'params': policy_params, 'lr': config.lr_peak},
            {'params': value_params, 'lr': config.lr_peak}
        ], weight_decay=0.01, eps=1e-5)

        # Warmup + warm-restart cosine, driven by PPO updates.
        # Convert the episode-based config knobs to updates via episodes_per_update.
        eps_per_update = max(1, config.episodes_per_update)
        warmup_updates = max(1, config.lr_warmup_episodes // eps_per_update)
        cycle_t_max_updates = max(1, config.lr_cycle_t_max_episodes // eps_per_update)
        self.lr_scheduler = WarmRestartLRScheduler(
            self.optimizer,
            lr_peak=config.lr_peak,
            lr_min=config.lr_min,
            warmup_updates=warmup_updates,
            cycle_t_max_updates=cycle_t_max_updates,
            phase_reset_factor=config.lr_phase_reset_factor,
            clone_bump_factor=config.lr_clone_bump_factor,
        )

        # Mixed precision
        self.scaler = GradScaler('cuda') if config.use_mixed_precision and self.device.type == 'cuda' else None

        # Curriculum manager. wr_sample_interval keeps the per-depth WR slope
        # math aligned with the actual logging cadence.
        self.curriculum = CurriculumManager(
            save_dir=config.curriculum_dir,
            wr_sample_interval=config.log_interval,
            graduation_min_episodes=config.graduation_min_episodes,
        )
        self.curriculum.on_phase_change_callbacks.append(self._on_phase_change)
        self.curriculum.on_clone_update_callbacks.append(self._on_clone_update)

        # Multiprocessing components
        self.workers: List[Process] = []
        self.request_queue: Queue = None
        self.response_queues: List[Queue] = []
        self.experience_queue: Queue = None
        self.control_queues: List[Queue] = []
        self.ready_events: List = []

        # Statistics
        self.episode_count = 0
        self.update_count = 0
        self.update_count_at_phase_start = 0  # for updates_in_phase logging
        self.total_steps = 0
        self.ema_return = None

        # Cache for the progressive minimax eval. Refreshed only at phase
        # graduations (see `_on_phase_change`) and once at training start
        # (see `train()`). `log_progress` never triggers a fresh eval --
        # it reads from this cache so the CSV column stays populated.
        self._last_minimax_eval: Optional[Tuple[int, str]] = None

        # Phase 11 sub-phase tracker. Phase 11 toggles between 'full' (9
        # stones / player) and 'mix' (uniform {3..9}) sub-phases every
        # PHASE_11_FULL_GAME_EPISODES / PHASE_11_MIX_EPISODES episodes.
        # We rebroadcast game settings (stone distribution) to workers on
        # every sub-phase flip; the trainer detects the flip by comparing
        # the curriculum's reported sub-phase against this cache at each
        # log tick. Initialized lazily on first phase-11 log tick.
        self._last_phase11_subphase: Optional[str] = None

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

        # Heartbeat file — atomic write of a single JSON line consumed by the
        # system monitor sidecar so we can detect "main thread blocked in C++
        # for >N seconds" without attaching py-spy.
        self.heartbeat_path = os.path.join(config.log_dir, "trainer_heartbeat.json")
        self.last_heartbeat_stage = "init"
        self._last_heartbeat_ts = 0.0
        # Minimum interval (s) between throttled in-loop heartbeat refreshes
        # called from `_heartbeat_tick`. Stage-transition `_heartbeat()` calls
        # always write unconditionally; this only rate-limits the per-batch
        # "I'm still serving inference" refresh.
        self._heartbeat_min_interval = 2.0

    def _heartbeat(self, stage: str):
        """Touch the heartbeat file with the current stage and PID."""
        try:
            payload = {
                "ts": time.time(),
                "pid": os.getpid(),
                "stage": stage,
                "episode": self.episode_count,
                "update": self.update_count,
                "phase": int(self.curriculum.current_phase),
            }
            tmp = self.heartbeat_path + ".tmp"
            with open(tmp, "w") as f:
                f.write(json.dumps(payload))
            os.replace(tmp, self.heartbeat_path)
            self.last_heartbeat_stage = stage
            self._last_heartbeat_ts = payload["ts"]
        except Exception:
            pass

    def _heartbeat_tick(self):
        """Refresh the heartbeat from inside long-running loops so the
        sidecar doesn't false-fire STALE on a legitimately slow collect
        (e.g. workers cold-cache D7 minimax). Throttled to
        `_heartbeat_min_interval` seconds.
        """
        now = time.time()
        if now - self._last_heartbeat_ts >= self._heartbeat_min_interval:
            self._heartbeat(self.last_heartbeat_stage)

    def _on_phase_change(self, old_phase: Phase, new_phase: Phase):
        """Callback when curriculum phase changes."""
        # Log per-depth graduation snapshot (captured by curriculum before reset).
        self._log_phase_graduation(old_phase, new_phase)

        # Notify LR scheduler: shrink peak by phase_reset_factor and restart cycle.
        self.lr_scheduler.notify_phase_graduated()
        self.update_count_at_phase_start = self.update_count

        # End-of-phase: run the progressive minimax eval so the depth
        # ladder reflects the freshly-graduated model. Result is cached
        # in `_last_minimax_eval` so every subsequent log_progress line
        # under the new phase reuses it until the next graduation.
        print(f"\n=== End-of-phase eval (Phase {int(old_phase)} -> {int(new_phase)}) ===",
              flush=True)
        try:
            max_depth_beaten, minimax_str = self.evaluate_vs_minimax_progressive()
            self._last_minimax_eval = (max_depth_beaten, minimax_str)
        except Exception as e:
            logger.warning("End-of-phase minimax eval failed: %r", e)

        # Save checkpoint at phase transition
        self.save_checkpoint(f"phase{int(old_phase)}_complete")

        # Broadcast new game settings and curriculum to workers
        self._broadcast_game_settings()
        self._broadcast_curriculum_update()

        # Refresh clone weights for the new phase. Every phase (including
        # Phase 1 warmup) samples self-play, so the clone needs to track the
        # latest weights at every phase boundary.
        if new_phase != Phase.COMPLETED:
            new_config = PHASE_CONFIGS.get(new_phase)
            if new_config and new_config.opponent_type == 'mixed':
                self._update_clone_model()
                # active_minimax_max_depth carries over across graduations
                # (Phase 1 has no minimax in its distribution; Phase 2 starts
                # at D1 and progressively unlocks). The full distribution
                # encodes both the unlocked range and the (post-reset,
                # all-False) dampened set.
                self._broadcast_opponent_distribution()

    def _broadcast_opponent_distribution(self):
        """Send the full per-opponent sampling distribution to all workers.

        Slot keys: 'self', 'random', 'minimax_d1' .. 'minimax_d{active_max}'.
        Dampened slots (WR-based for minimax/random, timed for self-play) are
        pinned at the configured cap and the remaining mass is redistributed
        via the equal-share-with-selfplay×3 rule (see
        `compute_opponent_distribution`). Called after each PPO update so the
        dampener tracks current WR snapshots and the timed self-play cap can
        expire promptly.
        """
        if self.curriculum.get_config().opponent_type != 'mixed':
            return
        self._broadcast_to_workers({
            'type': 'update_opponent_distribution',
            'distribution': self.curriculum.get_opponent_distribution(),
        })

    def _broadcast_to_workers(self, msg: Dict):
        """Put `msg` on every worker's control queue.

        Queue failures are logged but non-fatal: a worker that missed a
        broadcast gets refreshed on the next log tick's rebroadcast.
        """
        for i, q in enumerate(self.control_queues):
            try:
                q.put(msg)
            except Exception as e:
                logger.warning("control-queue put to worker %d failed: %r", i, e)

    def _update_clone_model(self):
        """Update the clone model with current model weights."""
        if self.clone_model is None:
            self.clone_model = ActorCritic(self.obs_size, self.num_actions, self.config).to(self.device)

        self.clone_model.load_state_dict(self.model.state_dict())
        self.clone_model.eval()

        logger.info("Clone model updated")

        self._broadcast_clone_update()

    def _on_clone_update(self):
        """Callback when the clone should be updated.

        Trigger: wr_vs_self over the rolling 500-game self-play window
        crosses MIXED_CONFIG['selfplay_winrate_threshold'], checked at
        each log tick.
        """
        self._update_clone_model()
        # Bump LR slightly to help adapt to the harder snapshot, then restart cycle.
        self.lr_scheduler.notify_clone_replaced()

    def _broadcast_clone_update(self):
        """Send updated clone weights to workers."""
        if self.clone_model is None:
            return

        clone_state = self.clone_model.state_dict()
        self._broadcast_to_workers({
            'type': 'update_clone',
            'clone_state_dict': {k: v.cpu() for k, v in clone_state.items()},
        })

    def _broadcast_game_settings(self):
        """Send game settings (stone distribution + AI-disadvantage flag) to workers.

        The `ai_disadvantage` flag is True during Phase 11's 'mix' sub-phase:
        the worker draws two stone counts independently and then ensures the AI
        player (the one producing gradients) always receives the smaller count,
        training it to win from a disadvantaged position.
        """
        self._broadcast_to_workers({
            'type': 'update_game_settings',
            'stone_distribution': self.curriculum.get_stone_distribution_for_phase(),
            'ai_disadvantage': self.curriculum.get_ai_disadvantage(),
        })

    def _broadcast_curriculum_update(self):
        """Send curriculum update to all workers."""
        config = self.curriculum.get_config()
        reward_config = self.curriculum.get_reward_config()

        msg = {
            'type': 'update_curriculum',
            'opponent_type': config.opponent_type,
            'reward_config': reward_config,
        }

        # For mixed mode, include the full per-opponent distribution.
        if config.opponent_type == 'mixed':
            msg['opponent_distribution'] = self.curriculum.get_opponent_distribution()

        self._broadcast_to_workers(msg)

    def _maybe_broadcast_phase11_subphase_change(self):
        """Detect a Phase 11 sub-phase flip and rebroadcast game settings.

        Phase 11 alternates infinitely between two stone-distribution regimes
        (full game vs uniform mix). The distribution lives in worker memory
        and is only re-sent on explicit broadcasts, so the trainer must push
        a refreshed `update_game_settings` message whenever the curriculum
        reports a different sub-phase than it did at the previous log tick.
        No-op outside Phase 11.
        """
        if self.curriculum.current_phase != Phase.PHASE_11:
            self._last_phase11_subphase = None
            return
        current = self.curriculum.get_phase11_subphase()
        if self._last_phase11_subphase is None:
            # First log tick under phase 11 — sync the cache; the initial
            # game-settings broadcast was already sent on phase entry by
            # `_on_phase_change`.
            self._last_phase11_subphase = current
            return
        if current != self._last_phase11_subphase:
            prev = self._last_phase11_subphase
            self._last_phase11_subphase = current
            sub_pos, sub_len = self.curriculum.get_phase11_subphase_progress()
            adv = self.curriculum.get_ai_disadvantage()
            print(
                f"  [Phase 11] Sub-phase flip: {prev} -> {current} "
                f"(next {sub_len:,} eps; cycle {sub_pos:,}/{sub_len:,})"
                + (" [AI disadvantage ON: AI gets fewer stones]" if adv else ""),
                flush=True,
            )
            # Stone-distribution and ai_disadvantage flag both changed;
            # reward config (flat draw penalty in phase 11) is unaffected.
            self._broadcast_game_settings()

    def _broadcast_reward_config(self):
        """Send only the reward config to all workers (no opp-type / dist).

        Used at the log tick during phase 10 to propagate the slowly-decaying
        draw_penalty. Kept separate from `_broadcast_curriculum_update` so it
        cannot accidentally trigger env recreation in workers — the worker
        handler updates the reward_calculator and nothing else.
        """
        self._broadcast_to_workers({
            'type': 'update_reward_config',
            'reward_config': self.curriculum.get_reward_config(),
        })

    def get_entropy_coef(self) -> float:
        """Get current entropy coefficient with gradual decay."""
        cfg = self.config
        progress = min(1.0, self.episode_count / cfg.entropy_decay_episodes)
        return cfg.entropy_coef_start + progress * (cfg.entropy_coef_end - cfg.entropy_coef_start)

    def _build_worker_shared_state(self) -> Dict:
        """Construct the `shared_state` dict workers bootstrap from."""
        curr_cfg = self.curriculum.get_config()
        return {
            'initial_stone_distribution': self.curriculum.get_stone_distribution_for_phase(),
            'initial_ai_disadvantage': self.curriculum.get_ai_disadvantage(),
            'initial_opponent_type': curr_cfg.opponent_type,
            'initial_reward_config': self.curriculum.get_reward_config(),
            'initial_opponent_distribution': (
                self.curriculum.get_opponent_distribution()
                if curr_cfg.opponent_type == 'mixed'
                else None
            ),
        }

    def start_workers(self):
        """Start worker processes."""
        print(f"Starting {self.config.num_workers} workers...")

        self.request_queue = mp.Queue()
        self.experience_queue = mp.Queue()

        shared_state = self._build_worker_shared_state()

        for i in range(self.config.num_workers):
            response_q = mp.Queue()
            control_q = mp.Queue()
            ready_evt = mp.Event()

            p = Process(
                target=worker_process,
                args=(
                    i, self.config, self.obs_size, self.num_actions,
                    self.request_queue, response_q, self.experience_queue,
                    control_q, ready_evt,
                    shared_state
                ),
                daemon=True
            )
            p.start()

            self.workers.append(p)
            self.response_queues.append(response_q)
            self.control_queues.append(control_q)
            self.ready_events.append(ready_evt)

        for evt in self.ready_events:
            evt.wait(timeout=30)

        print(f"All {self.config.num_workers} workers ready!")

        # Send initial game settings and curriculum
        self._broadcast_game_settings()
        self._broadcast_curriculum_update()

        # On resume, active_minimax_max_depth and dominated flags carry over
        # from the checkpoint and are reflected in the distribution.
        self._broadcast_opponent_distribution()

        # Initialize clone for mixed phases
        config = self.curriculum.get_config()
        if config.opponent_type == 'mixed':
            self._update_clone_model()

    def stop_workers(self):
        """Stop all worker processes."""
        self._broadcast_to_workers({'type': 'stop'})

        for p in self.workers:
            p.join(timeout=2)
            if p.is_alive():
                p.terminate()

    def process_inference_requests(self, timeout: float = 0.01) -> int:
        """Process batched inference requests from workers."""
        # Throttled refresh — keeps `heartbeat_age` low during long collects
        # so monitor doesn't falsely flag STALE while the main thread is
        # actively serving GPU forward calls.
        self._heartbeat_tick()
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

        node_batch = torch.from_numpy(np.stack([r['node_feats'] for r in all_requests])).to(self.device)
        global_batch = torch.from_numpy(np.stack([r['global_feats'] for r in all_requests])).to(self.device)
        mask_batch = torch.from_numpy(np.stack([r['mask'] for r in all_requests])).to(self.device)

        with torch.no_grad():
            with autocast('cuda', enabled=self.config.use_mixed_precision):
                logits, values = self.model(node_batch, global_batch)

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

                    # Update curriculum with game result (always — for stats/logging)
                    minimax_depth = getattr(batch, 'minimax_depth', 0)
                    self.curriculum.add_game_result(
                        batch.outcome,
                        opponent_type=batch.opponent_type,
                        minimax_depth=minimax_depth
                    )

                    # Drop dampened-opponent experiences from PPO training.
                    # Sampling and training share the same dampened-set so a
                    # game played against a 1%-pinned opponent never feeds
                    # the gradient. The `opponent_type != 'mixed'` branch is
                    # an unreachable safety net (all phases are mixed today).
                    curr_cfg = self.curriculum.get_config()
                    if curr_cfg.opponent_type == 'mixed':
                        dampened = self.curriculum.mixed_state.get_dampened_set()
                        if batch.opponent_type == 'random':
                            if 'random' in dampened:
                                continue
                        elif batch.opponent_type == 'minimax':
                            if f'minimax_d{minimax_depth}' in dampened:
                                continue
                        elif batch.opponent_type == 'self':
                            if 'self' in dampened:
                                continue

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
                logger.warning("Mixed phase batch had 0 minimax/self games; rebroadcasting curriculum.")
                self._broadcast_curriculum_update()
                self._broadcast_opponent_distribution()

        self.episode_count += len(all_experiences)
        return all_experiences, all_returns

    def update_policy(self, experiences: List[ExperienceBatch]) -> Dict:
        """Perform PPO update."""
        if not experiences:
            return {}

        all_nodes = torch.from_numpy(np.concatenate([e.node_feats for e in experiences])).to(self.device)
        all_globals = torch.from_numpy(np.concatenate([e.global_feats for e in experiences])).to(self.device)
        all_actions = torch.from_numpy(np.concatenate([e.actions for e in experiences])).to(self.device)
        all_old_logprobs = torch.from_numpy(np.concatenate([e.logprobs for e in experiences])).to(self.device)
        all_old_values = torch.from_numpy(np.concatenate([e.values for e in experiences])).to(self.device)
        all_masks = torch.from_numpy(np.concatenate([e.masks for e in experiences])).to(self.device)

        advantages = torch.from_numpy(np.concatenate([e.advantages for e in experiences])).to(self.device)
        returns = torch.from_numpy(np.concatenate([e.returns for e in experiences])).to(self.device)

        n_samples = all_nodes.shape[0]
        self.total_steps += n_samples

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
            indices = torch.randperm(n_samples, device=self.device)

            for start in range(0, n_samples, self.config.mini_batch_size):
                end = min(start + self.config.mini_batch_size, n_samples)
                idx = indices[start:end]

                with autocast('cuda', enabled=self.config.use_mixed_precision and self.device.type == 'cuda'):
                    logits, values = self.model(all_nodes[idx], all_globals[idx])

                    masked_logits = logits.float()
                    masked_logits[all_masks[idx] == 0] = -1e9

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

    def evaluate_vs_minimax_progressive(self) -> Tuple[int, str]:
        """Test AI against progressively harder minimax bots.

        Climb is unbounded: the loop only stops once the AI's win rate at
        some depth falls to/below 50%. The C++ orchestrator runs
        `games_per_depth=6` games concurrently and root-splits each
        game's minimax across `max_threads // games_per_depth` workers,
        so 6 games x 4 workers/game pegs a 24-core box during the bot's
        search (which dominates wall time at depth >= 4). As soon as a
        depth's W/D/L is known the printer emits the accumulating
        `Minimax: D1:... | D2:... | ...` line so progress is visible
        live instead of only at the end.
        """
        accumulated: list = []

        def _on_depth(d: int, r: Dict):
            accumulated.append(
                f"D{d}:{r['wins']}W/{r['draws']}D/{r['losses']}L"
            )
            # Overwrite the same line as each depth completes so progress is
            # visible live without spamming a new line per depth.
            print(f"\r  Minimax: {' | '.join(accumulated)}", end='', flush=True)

        cpu_count = os.cpu_count() or 4
        thread_cap = max(6, cpu_count)

        max_depth_beaten, results = evaluate_vs_minimax_cpp(
            self.model, self.device, self.num_actions,
            games_per_depth=6,
            max_threads=thread_cap,
            max_steps=150,
            use_mixed_precision=self.config.use_mixed_precision,
            unlimited=True,
            stone_distribution=self.curriculum.get_stone_distribution_for_phase(),
            progress_callback=_on_depth,
        )
        # Close the in-place-updated minimax line so subsequent prints land
        # on a fresh line.
        if accumulated:
            print(flush=True)
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

        curriculum_status = self.curriculum.get_status_string()
        config = self.curriculum.get_config()

        # Get per-opponent win rates (last 500 games each).
        # All phases are currently 'mixed'; the fallback is a defensive
        # placeholder for any future non-mixed phase.
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

        # Main progress line - matches original format
        print(
            f"[Phase {int(self.curriculum.current_phase)}] Ep {self.episode_count:,} | {curriculum_status} | "
            f"Ret: {avg_return:+.3f} | PL: {metrics.get('policy_loss', 0):+.4f} | VL: {metrics.get('value_loss', 0):.3f} | "
            f"LR: {metrics.get('lr', 0):.1e} | {eps_per_sec:.0f}/s",
            flush=True,
        )

        # Progressive minimax eval runs ONLY at phase graduations (and once
        # at startup) -- see `_on_phase_change` and `train()`. Here we just
        # read the cached depth so the CSV column stays populated (the
        # per-depth string was already printed live during the eval).
        max_depth_beaten = (
            self._last_minimax_eval[0] if self._last_minimax_eval is not None else 0
        )

        # Minimax evaluation results. The "(no-train)" / "*" notes reflect the
        # ACTUAL dampened flags used to drop games from PPO, so the log can
        # never disagree with what training does.
        active_max = opp_wr['active_mm_max_depth']
        ms = self.curriculum.mixed_state

        def _d_str(d: int) -> str:
            key = f'wr_vs_mm_d{d}'
            if active_max >= d:
                wr = opp_wr.get(key, 0.0)
                note = "*" if ms.minimax_depth_dominated.get(d, False) else ""
                return f" D{d}:{wr:.0%}{note}"
            return f" D{d}:locked"

        depth_str = "".join(_d_str(d) for d in range(1, 8))
        # Note: the per-depth "  Minimax: D1:… | D2:… | …" lines were already
        # printed live by `evaluate_vs_minimax_progressive`'s progress
        # callback, so we do not re-print the final accumulated line here.
        if config.opponent_type == 'mixed':
            wr_random = opp_wr['wr_vs_random']
            rnd_train_note = " (no-train)" if ms.random_dominated else ""
            self_paused = ms.total_episodes < ms.selfplay_train_cooldown_until
            self_note = f" (no-train, {ms.selfplay_train_cooldown_until - ms.total_episodes:,}ep left)" if self_paused else ""
            print(f"  WR(500):{depth_str} [MaxD:{active_max}] "
                  f"Rnd:{wr_random:.0%}{rnd_train_note} Self:{opp_wr['wr_vs_self']:.0%}{self_note}")
        else:
            # Defensive fallback for any future non-mixed phase.
            print(f"  WR({len(curr_stats.recent_results)}): Rnd:{win_rate:.0%}")

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
            fieldnames = [
                'episode', 'phase', 'starting_stones', 'steps', 'avg_return', 'ema_return',
                'win_rate', 'draw_rate', 'policy_loss', 'value_loss', 'entropy', 'kl_div',
                'clip_frac', 'grad_norm', 'lr', 'cycle_step', 'last_reset_event',
                'minimax_depth_beaten', 'clone_gen',
                'active_mm_max_depth', 'shaping_mult',
                'wr_vs_mm_d1', 'wr_vs_mm_d2', 'wr_vs_mm_d3', 'wr_vs_mm_d4',
                'wr_vs_mm_d5', 'wr_vs_mm_d6', 'wr_vs_mm_d7',
                'wr_vs_random', 'wr_vs_self',
                # Per-depth graduation diagnostics — slope angle (deg) over a 1M
                # episode horizon, and current samples in the per-depth window.
                'slope_angle_d1', 'slope_angle_d2', 'slope_angle_d3', 'slope_angle_d4',
                'slope_angle_d5', 'slope_angle_d6', 'slope_angle_d7',
                'samples_in_window_d1', 'samples_in_window_d2', 'samples_in_window_d3',
                'samples_in_window_d4', 'samples_in_window_d5', 'samples_in_window_d6',
                'samples_in_window_d7',
                # Training throughput
                'eps_per_sec',
            ]
            self.log_writer = csv.DictWriter(self.log_file, fieldnames=fieldnames, extrasaction='ignore')
            if write_header:
                self.log_writer.writeheader()

        is_mixed = (config.opponent_type == 'mixed')

        def _slope_angle(d: int) -> float:
            if not is_mixed:
                return 0.0
            angle = ms.get_slope_angle_for_depth(d)
            # CSV-friendly: clamp +inf (insufficient samples) to NaN-equivalent.
            return float('nan') if angle == float('inf') else angle

        def _samples(d: int) -> int:
            return ms.get_window_size_for_depth(d) if is_mixed else 0

        row = {
            'episode': self.episode_count,
            'phase': int(self.curriculum.current_phase),
            'starting_stones': self.curriculum.get_starting_stones_for_phase(),
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
            'cycle_step': metrics.get('cycle_step', self.lr_scheduler.cycle_step),
            'last_reset_event': metrics.get('last_reset_event', self.lr_scheduler.last_reset_event),
            'minimax_depth_beaten': max_depth_beaten,
            'clone_gen': ms.clone_generation,
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
            'eps_per_sec': eps_per_sec,
        }
        for d in range(1, 8):
            row[f'slope_angle_d{d}'] = _slope_angle(d)
            row[f'samples_in_window_d{d}'] = _samples(d)

        self.log_writer.writerow(row)
        self.log_file.flush()

    def _log_phase_graduation(self, old_phase: Phase, new_phase: Phase):
        """Emit a one-line graduation snapshot to a sidecar CSV plus stdout.

        Captures the per-depth WR/slope/samples state at graduation, alongside
        episodes/updates-in-phase, clone generations, and current LR. This is
        the diagnostic record for whether a phase graduated cleanly under the
        new criteria. Writing to a separate file (vs. the main per-update CSV)
        keeps the schemas clean.
        """
        snapshot = self.curriculum.last_graduation_snapshot
        if snapshot is None:
            return  # No per-depth data captured for this graduation.

        episodes_in_phase = snapshot.get('episodes_in_phase', 0)
        updates_in_phase = self.update_count - self.update_count_at_phase_start
        per_depth_wr = snapshot.get('per_depth_wr', {})
        per_depth_slope = snapshot.get('per_depth_slope_angle', {})
        per_depth_samples = snapshot.get('per_depth_samples', {})

        # Stdout: human-readable summary
        depths_str = " ".join(
            f"D{d}:{per_depth_wr.get(d, 0.0):.0%}({per_depth_slope.get(d, float('inf')):.2f}°,n={per_depth_samples.get(d, 0)})"
            for d in sorted(per_depth_wr.keys())
        )
        print(
            f"  [Graduation] Phase {int(old_phase)} → {int(new_phase)} | "
            f"eps_in_phase={episodes_in_phase:,} updates_in_phase={updates_in_phase:,} | "
            f"wr_top={snapshot.get('wr_vs_top_depth_at_graduation', 0.0):.0%} | "
            f"clone_gen={snapshot.get('clone_generations_in_phase', 0)} | "
            f"lr={self.lr_scheduler.get_lr():.2e} | {depths_str}"
        )

        # CSV sidecar
        path = os.path.join(self.config.log_dir, "phase_graduations.csv")
        write_header = not os.path.exists(path)
        fieldnames = [
            'episode', 'old_phase', 'new_phase',
            'episodes_in_phase', 'updates_in_phase',
            'wr_vs_top_depth_at_graduation', 'top_depth',
            'clone_generations_in_phase', 'graduation_reason',
            'lr_before_reset',
        ] + [f'wr_d{d}' for d in range(1, 8)] \
          + [f'slope_angle_d{d}' for d in range(1, 8)] \
          + [f'samples_d{d}' for d in range(1, 8)]

        with open(path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            if write_header:
                writer.writeheader()
            row = {
                'episode': self.episode_count,
                'old_phase': int(old_phase),
                'new_phase': int(new_phase),
                'episodes_in_phase': episodes_in_phase,
                'updates_in_phase': updates_in_phase,
                'wr_vs_top_depth_at_graduation': snapshot.get('wr_vs_top_depth_at_graduation', 0.0),
                'top_depth': snapshot.get('top_depth', 0),
                'clone_generations_in_phase': snapshot.get('clone_generations_in_phase', 0),
                'graduation_reason': snapshot.get('graduation_reason', ''),
                'lr_before_reset': self.lr_scheduler.get_lr(),
            }
            for d in range(1, 8):
                row[f'wr_d{d}'] = per_depth_wr.get(d, '')
                slope = per_depth_slope.get(d, float('inf'))
                row[f'slope_angle_d{d}'] = '' if slope == float('inf') else slope
                row[f'samples_d{d}'] = per_depth_samples.get(d, '')
            writer.writerow(row)

    def save_checkpoint(self, prefix="checkpoint"):
        """Save model checkpoint."""
        path = os.path.join(
            self.config.checkpoint_dir,
            f"{time.strftime('%Y%m%d_%H%M%S')}_{prefix}_ep{self.episode_count}.pt"
        )
        curriculum_state = self.curriculum.to_state_dict()
        torch.save({
            'episode': self.episode_count,
            'total_steps': self.total_steps,
            'update_count': self.update_count,
            'update_count_at_phase_start': self.update_count_at_phase_start,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'ema_return': self.ema_return,
            'curriculum_phase': int(self.curriculum.current_phase),
            'curriculum_state': curriculum_state,
            'lr_scheduler_state': self.lr_scheduler.get_state_dict(),
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
            except Exception as e:
                logger.warning("Could not restore GradScaler state from checkpoint: %r", e)

        self.episode_count = ckpt['episode']
        self.total_steps = ckpt.get('total_steps', 0)
        self.update_count = ckpt.get('update_count', 0)
        # Older checkpoints don't carry phase-start update count. Fall back to
        # current update_count so updates_in_phase resets to 0 on resume; it
        # will report correctly from the next phase graduation onward.
        self.update_count_at_phase_start = ckpt.get('update_count_at_phase_start', self.update_count)
        self.ema_return = ckpt.get('ema_return')

        # Load LR scheduler state. Falls back gracefully if the checkpoint
        # was produced by an older single-cosine scheduler.
        if 'lr_scheduler_state' in ckpt and ckpt['lr_scheduler_state']:
            try:
                self.lr_scheduler.load_state_dict(ckpt['lr_scheduler_state'])
                print(
                    f"  LR scheduler restored: update={self.lr_scheduler.global_update}, "
                    f"cycle_step={self.lr_scheduler.cycle_step}, "
                    f"in_warmup={self.lr_scheduler.in_warmup}, "
                    f"lr={self.lr_scheduler.get_lr():.1e}"
                )
            except Exception as e:
                print(f"  Warning: could not restore LR scheduler state ({e}); continuing from fresh state.")

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

        print("Training Phases (engine-configured starting stones):")
        for phase, phase_cfg in PHASE_CONFIGS.items():
            print(f"  Phase {int(phase)}: {phase_cfg.description[:60]}")
        print()

        eps_per_update = max(1, cfg.episodes_per_update)
        warmup_updates = max(1, cfg.lr_warmup_episodes // eps_per_update)
        cycle_updates = max(1, cfg.lr_cycle_t_max_episodes // eps_per_update)
        print(
            f"LR schedule: warmup 0 → {cfg.lr_peak:.1e} over "
            f"{cfg.lr_warmup_episodes:,} eps (~{warmup_updates} updates), then "
            f"cosine → {cfg.lr_min:.1e} cycles of {cfg.lr_cycle_t_max_episodes:,} eps "
            f"(~{cycle_updates} updates). "
            f"Phase grad ×{cfg.lr_phase_reset_factor}, clone bump ×{cfg.lr_clone_bump_factor}."
        )
        print()

        print(f"Mixed Training (Phase 1-11):")
        print(f"  Phase 1 mix: self + random only (no minimax) — 75% self / 25% random.")
        print(f"  Phase 2-11 mix: equal-share across self + unlocked minimax D1-D{MIXED_CONFIG['minimax_max_depth']} + random,")
        print(f"                  self-play weighted ×{MIXED_CONFIG['selfplay_weight']:.0f}. Dampened slots pinned at {MIXED_CONFIG['dampen_cap']:.0%}.")
        print(f"  Minimax depths unlock progressively (D1 → D{MIXED_CONFIG['minimax_max_depth']}) by win rate; "
              f"D6/D7 stay eval-only.")
        print(f"  Self-play: Clone update at {MIXED_CONFIG['selfplay_winrate_threshold']*100:.0f}% WR over the rolling 500-game self-play window (checked at log tick).")
        print(f"  Graduation (Phase 2-9): trend-based plateau detection < "
              f"{GRADUATION_CONFIG['trend_max_angle_degrees']:.1f}° over a 1M-episode horizon "
              f"for every unlocked depth.")
        print(f"  Phase 11 is infinite: alternates 2.5M full-game + 2.5M uniform-mix sub-phases until stopped.")
        print()

        config = self.curriculum.get_config()
        dist = self.curriculum.get_stone_distribution_for_phase()
        dist_str = ", ".join(f"{c}:{w:.0%}" for c, w in dist)
        print(f"Starting Phase {int(self.curriculum.current_phase)}: {config.description}")
        print(f"  Stone distribution per player: {dist_str}")
        print("=" * 70)

        self.start_time = time.time()
        self.start_episode_count = self.episode_count

        self.start_workers()

        # Seed the minimax-eval cache once at startup so the first log
        # cycle's `Minimax: D1:...` line reflects a real result. Without
        # this, log_progress would print `(no eval yet)` until the first
        # phase graduation -- which is many hours away in long phases.
        if self._last_minimax_eval is None:
            print("\n=== Startup minimax eval (seeds the per-log cache) ===",
                  flush=True)
            try:
                mb, mstr = self.evaluate_vs_minimax_progressive()
                self._last_minimax_eval = (mb, mstr)
            except Exception as e:
                logger.warning("Startup minimax eval failed: %r", e)

        try:
            while self.curriculum.current_phase != Phase.COMPLETED:
                self._heartbeat("collect")
                # Collect experiences (workers run, main serves inference)
                experiences, returns = self.collect_experiences(cfg.episodes_per_update)

                self._heartbeat("ppo_update")
                # PPO update — workers are NOT paused.
                # They keep playing (minimax, random, self-play turns) and
                # queue up inference requests. Requests pile up during PPO
                # and get served on the next collect_experiences call.
                metrics = self.update_policy(experiences)

                # Step the warm-restart cosine ONLY when the PPO update actually
                # ran (update_policy returns {} for an empty batch). This keeps
                # the LR schedule aligned with real gradient steps.
                if metrics:
                    self.lr_scheduler.step()
                # Decorate metrics with current scheduler state for CSV logging.
                metrics['lr'] = self.lr_scheduler.get_lr()
                metrics['cycle_step'] = self.lr_scheduler.cycle_step
                metrics['last_reset_event'] = self.lr_scheduler.last_reset_event

                if self.episode_count % cfg.log_interval < cfg.episodes_per_update:
                    # All slow-signal checks live here — one place, one cadence.
                    # Order matters:
                    #  1. sample per-depth WR so logging sees the freshest tick
                    #  2. log progress (uses pre-update state)
                    #  3. clone update (reuses the wr_vs_self we just logged)
                    #  4. depth unlock (may broadcast and save a checkpoint)
                    #  5. weight rebroadcast (after dominated-state refresh)
                    #  6. graduation check (terminal signal for the phase)
                    self._heartbeat("log_progress")
                    self.curriculum.sample_minimax_winrate()
                    self.log_progress(returns, metrics)
                    if self.curriculum.should_update_clone():
                        self.curriculum.do_clone_update()
                    if self.curriculum.check_and_unlock_minimax_depth():
                        new_max = self.curriculum.get_active_minimax_max_depth()
                        self._heartbeat("save_depth_unlock")
                        self.save_checkpoint(f"depth{new_max}_unlocked")
                    if self.curriculum.get_config().opponent_type == 'mixed':
                        # Self-play timed dampening: if wr_vs_self crossed
                        # the threshold, pin self-play sampling at
                        # `dampen_cap` and drop self-play from PPO for the
                        # configured window. Updated BEFORE the distribution
                        # broadcast so the new dampened state is reflected.
                        ms = self.curriculum.mixed_state
                        opp_wr = self.curriculum.get_opponent_win_rates()
                        wr_self = opp_wr['wr_vs_self']
                        if wr_self > cfg.selfplay_train_pause_threshold:
                            ms.selfplay_train_cooldown_until = (
                                ms.total_episodes + cfg.selfplay_train_pause_episodes
                            )
                            remaining = ms.selfplay_train_cooldown_until - ms.total_episodes
                            print(
                                f"  [Self-play dampen] wr_vs_self={wr_self:.1%} > "
                                f"{cfg.selfplay_train_pause_threshold:.0%}; "
                                f"pinning self-play sampling at 1% and "
                                f"dropping it from PPO for next {remaining:,} eps."
                            )
                        # Re-evaluate minimax/random dampener flags on the
                        # log cadence, consuming the same opp_wr the logger
                        # just emitted so "log says WR >= 90%" and "fire the
                        # 90% protocol" are the same event.
                        ms.update_dampened_state(opp_wr)
                        # If every other opponent (random + all unlocked-and-
                        # trainable minimax depths) is dampened, lift the
                        # self-play pause early. Otherwise every slot pins at
                        # `dampen_cap`, sampling collapses to uniform, and no
                        # opponent produces gradient signal.
                        if ms.is_selfplay_dampened():
                            top_d = min(
                                ms.active_minimax_max_depth,
                                MIXED_CONFIG['minimax_max_depth'],
                            )
                            mm_all_damp = top_d >= 1 and all(
                                ms.minimax_depth_dominated.get(d, False)
                                for d in range(1, top_d + 1)
                            )
                            if mm_all_damp and ms.random_dominated:
                                remaining = (
                                    ms.selfplay_train_cooldown_until
                                    - ms.total_episodes
                                )
                                ms.selfplay_train_cooldown_until = 0
                                print(
                                    f"  [Self-play un-dampen] all minimax "
                                    f"(D1-D{top_d}) and random dampened; "
                                    f"lifting self-play pause {remaining:,} "
                                    f"eps early to restore PPO signal."
                                )
                        self._broadcast_opponent_distribution()
                    # Phase 10: draw_penalty decays linearly over its first
                    # 4M episodes (see PHASE_10_DRAW_PENALTY_DECAY_EPISODES in
                    # curriculum.py). Push the freshly-computed value out to
                    # workers each log tick so terminal rewards track the
                    # schedule instead of staying pinned at the phase-start
                    # value.
                    if self.curriculum.current_phase == Phase.PHASE_10:
                        self._broadcast_reward_config()
                    # Phase 11: flip stone distribution on sub-phase change.
                    # Draw penalty is flat in phase 11 so no reward-config
                    # rebroadcast is required, but the stone distribution
                    # swaps between full-game and uniform-mix each cycle.
                    self._maybe_broadcast_phase11_subphase_change()
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

