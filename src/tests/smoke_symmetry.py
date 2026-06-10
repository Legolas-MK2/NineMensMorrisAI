"""Smoke test: one PPO update with symmetry augmentation on, then off.

Verifies:
  1. No shape mismatches in the worker -> trainer pipeline.
  2. No asserts trip on the augmented obs/mask/action path.
  3. The PPO loss is finite for both runs.

Run from src/:
    python -m tests.smoke_symmetry
"""

from __future__ import annotations

import os
import sys
import tempfile
import multiprocessing as mp

# Allow running as `python -m tests.smoke_symmetry` from src/.
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import numpy as np


def make_tiny_config(use_aug: bool, tmpdir: str, policy_head: str = 'pointer'):
    from config import Config
    cfg = Config()
    cfg.num_workers = 2
    cfg.envs_per_worker = 4
    cfg.episodes_per_update = 16
    cfg.ppo_epochs = 1
    cfg.mini_batch_size = 256
    cfg.minimax_threads_per_worker = 1
    cfg.reserved_display_cores = 0
    cfg.worker_nice = 0
    cfg.use_mixed_precision = False
    cfg.use_symmetry_aug = use_aug
    cfg.aug_granularity = 'game'
    cfg.policy_head = policy_head
    # Redirect persistent directories so the smoke run doesn't touch real state.
    cfg.model_dir = os.path.join(tmpdir, 'models')
    cfg.log_dir = os.path.join(tmpdir, 'logs')
    cfg.checkpoint_dir = os.path.join(tmpdir, 'checkpoints')
    cfg.curriculum_dir = os.path.join(tmpdir, 'curriculum')
    return cfg


def run_one_update(use_aug: bool, label: str, policy_head: str = 'pointer') -> dict:
    print(f"\n[{label}] use_symmetry_aug = {use_aug}, policy_head = {policy_head}")
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = make_tiny_config(use_aug, tmpdir, policy_head=policy_head)
        from trainer import PPOTrainer

        trainer = PPOTrainer(cfg)
        trainer.start_workers()
        try:
            experiences, returns = trainer.collect_experiences(cfg.episodes_per_update)
            print(f"  collected {len(experiences)} episodes, "
                  f"{sum(len(e.node_feats) for e in experiences)} transitions")

            # Validate batch shapes / dtypes.
            for i, e in enumerate(experiences):
                T = e.node_feats.shape[0]
                assert e.node_feats.shape == (T, 24, 3), (i, e.node_feats.shape)
                assert e.global_feats.shape[0] == T, (i, e.global_feats.shape)
                assert e.masks.shape == (T, trainer.num_actions), (i, e.masks.shape)
                assert e.actions.shape == (T,), (i, e.actions.shape)
                # Every recorded action must be legal in its (augmented) mask.
                legal = e.masks[np.arange(T), e.actions]
                assert legal.min() == 1.0, f"illegal action in experience {i}"
                # Each per-step row should be a valid one-hot per board point.
                assert np.allclose(e.node_feats.sum(axis=-1), 1.0), \
                    f"node_feats rows are not one-hot at experience {i}"

            metrics = trainer.update_policy(experiences)
            print(f"  metrics: {metrics}")

            # Loss finiteness checks.
            for k in ('policy_loss', 'value_loss', 'entropy', 'kl_div'):
                v = metrics[k]
                assert np.isfinite(v), f"{label}: {k} is not finite: {v}"

            return metrics
        finally:
            trainer.stop_workers()


def main():
    mp.set_start_method('spawn', force=True)

    print("=== Milestone B: pointer head ===")
    metrics_off_p = run_one_update(use_aug=False, label='AUG-OFF', policy_head='pointer')
    metrics_on_p  = run_one_update(use_aug=True,  label='AUG-ON ', policy_head='pointer')

    print("\n=== Milestone A: flat head ===")
    metrics_off_f = run_one_update(use_aug=False, label='AUG-OFF', policy_head='flat')
    metrics_on_f  = run_one_update(use_aug=True,  label='AUG-ON ', policy_head='flat')

    print("\n--- SMOKE TEST SUMMARY ---")
    print(f"{'metric':>12}  {'ptr off':>10}  {'ptr on':>10}  {'flat off':>10}  {'flat on':>10}")
    for k in ('policy_loss', 'value_loss', 'entropy', 'kl_div', 'clip_frac'):
        print(f"  {k:>10}  "
              f"{metrics_off_p[k]:+.5f}  {metrics_on_p[k]:+.5f}  "
              f"{metrics_off_f[k]:+.5f}  {metrics_on_f[k]:+.5f}")
    print("\nAll finite. Smoke test PASSED.")


if __name__ == "__main__":
    main()
