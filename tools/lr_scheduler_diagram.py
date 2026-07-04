"""
Visualize the WarmRestartLRScheduler from src/lr_scheduler.py.

Simulates a long training run and annotates the four behaviors:
  1. Linear warmup 0 -> lr_peak
  2. Cosine annealing from cycle_starting_lr -> lr_min
  3. Phase-graduation reset: phase_peak *= phase_reset_factor (compounds), fresh cycle
  4. Clone-replacement bump: cycle_starting_lr <- min(phase_peak, lr * clone_bump_factor),
     cycle progress preserved
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Use the real scheduler class — no reimplementation. (Script lives in
# tools/; the scheduler lives in <repo>/src.)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import torch
from lr_scheduler import WarmRestartLRScheduler


# Use small numbers for a readable plot; ratios match the real config.
LR_PEAK = 3e-4
LR_MIN = 1e-6
WARMUP = 200
CYCLE_T_MAX = 1600
PHASE_RESET = 0.8
CLONE_BUMP = 1.3
TOTAL_UPDATES = 7000

# Scripted events (update index): mix of phase graduations and clone bumps.
PHASE_EVENTS = [2200, 4600]
CLONE_EVENTS = [1600, 1900, 3100, 3400, 3700, 5400, 5700, 6000, 6300]


def simulate():
    dummy = torch.nn.Linear(1, 1)
    opt = torch.optim.SGD(dummy.parameters(), lr=LR_PEAK)
    sched = WarmRestartLRScheduler(
        optimizer=opt,
        lr_peak=LR_PEAK,
        lr_min=LR_MIN,
        warmup_updates=WARMUP,
        cycle_t_max_updates=CYCLE_T_MAX,
        phase_reset_factor=PHASE_RESET,
        clone_bump_factor=CLONE_BUMP,
    )

    xs, lrs, peaks = [], [], []
    phase_marks, clone_marks = [], []

    for step in range(TOTAL_UPDATES):
        if step in PHASE_EVENTS:
            sched.notify_phase_graduated()
            phase_marks.append((step, sched.get_lr(), sched.phase_peak))
        if step in CLONE_EVENTS:
            sched.notify_clone_replaced()
            clone_marks.append((step, sched.get_lr()))
        sched.step()
        xs.append(step)
        lrs.append(sched.get_lr())
        peaks.append(sched.phase_peak)

    return np.array(xs), np.array(lrs), np.array(peaks), phase_marks, clone_marks


def plot(xs, lrs, peaks, phase_marks, clone_marks):
    fig, (ax_main, ax_zoom) = plt.subplots(
        2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [2.2, 1]}
    )

    # ------- Main panel: full trajectory -------
    ax_main.plot(xs, lrs, color='#1f77b4', linewidth=1.8, label='Learning rate', zorder=3)
    ax_main.plot(xs, peaks, color='#d62728', linewidth=1.0, linestyle='--',
                 alpha=0.65, label='phase_peak (cosine ceiling)', zorder=2)
    ax_main.axhline(LR_PEAK, color='#888', linewidth=0.8, linestyle=':',
                    label=f'lr_peak = {LR_PEAK:.0e}')
    ax_main.axhline(LR_MIN, color='#888', linewidth=0.8, linestyle=':',
                    label=f'lr_min = {LR_MIN:.0e}')

    # Warmup shading
    ax_main.axvspan(0, WARMUP, color='#ffe7a3', alpha=0.55, zorder=1)
    ax_main.text(WARMUP / 2, LR_PEAK * 0.55, 'Warmup\n(linear 0 → lr_peak)',
                 ha='center', va='center', fontsize=9, color='#7a5300')

    # Phase graduation markers
    for i, (x, y, new_peak) in enumerate(phase_marks):
        ax_main.axvline(x, color='#d62728', linewidth=1.4, alpha=0.7, zorder=2)
        ax_main.annotate(
            f'Phase graduation #{i+1}\n'
            f'phase_peak ×{PHASE_RESET}\n'
            f'new peak={new_peak:.1e}\n'
            f'cycle_step → 0',
            xy=(x, new_peak), xytext=(x + 150, new_peak * 1.6),
            fontsize=8, color='#a01a1a',
            arrowprops=dict(arrowstyle='->', color='#a01a1a', lw=1.0),
            bbox=dict(boxstyle='round,pad=0.3', fc='#fde2e2', ec='#a01a1a', lw=0.6),
        )

    # Clone replacement markers
    for x, y in clone_marks:
        ax_main.plot(x, y, marker='^', color='#2ca02c', markersize=9,
                     markeredgecolor='black', markeredgewidth=0.5, zorder=4)

    # Custom legend entry for clone events
    ax_main.plot([], [], marker='^', linestyle='', color='#2ca02c',
                 markeredgecolor='black', markersize=9,
                 label=f'Clone replacement (×{CLONE_BUMP}, capped at phase_peak)')

    ax_main.set_yscale('log')
    ax_main.set_xlabel('PPO update step')
    ax_main.set_ylabel('Learning rate (log scale)')
    ax_main.set_title('WarmRestartLRScheduler — full trajectory\n'
                      'Warmup → cosine cycle → floor-hold, interrupted by phase / clone events',
                      fontsize=12)
    ax_main.legend(loc='upper right', fontsize=8, framealpha=0.95)
    ax_main.grid(True, which='both', alpha=0.25)

    # ------- Zoom panel: a phase graduation up close -------
    if phase_marks:
        pgx = phase_marks[0][0]
        lo, hi = max(0, pgx - 400), min(len(xs), pgx + 1200)
        zx, zy = xs[lo:hi], lrs[lo:hi]
        ax_zoom.plot(zx, zy, color='#1f77b4', linewidth=1.8)
        ax_zoom.axvline(pgx, color='#d62728', linewidth=1.4, alpha=0.7)
        ax_zoom.axhline(LR_MIN, color='#888', linewidth=0.8, linestyle=':')

        # Highlight a clone bump inside this window if present
        for x, y in clone_marks:
            if lo <= x < hi:
                ax_zoom.plot(x, y, marker='^', color='#2ca02c', markersize=10,
                             markeredgecolor='black', markeredgewidth=0.5)
                ax_zoom.annotate(
                    'Clone bump\nlr × 1.3\n(cycle_step preserved → \n'
                    'curve continues, not restarts)',
                    xy=(x, y), xytext=(x + 60, y * 2.5),
                    fontsize=8, color='#1a5e1a',
                    arrowprops=dict(arrowstyle='->', color='#1a5e1a', lw=1.0),
                    bbox=dict(boxstyle='round,pad=0.3', fc='#e2f5e2',
                              ec='#1a5e1a', lw=0.6),
                )

        ax_zoom.annotate(
            'Phase graduation:\n'
            'fresh cosine cycle\nfrom shrunken peak',
            xy=(pgx, lrs[pgx]), xytext=(pgx - 350, LR_PEAK * 0.4),
            fontsize=8, color='#a01a1a',
            arrowprops=dict(arrowstyle='->', color='#a01a1a', lw=1.0),
            bbox=dict(boxstyle='round,pad=0.3', fc='#fde2e2', ec='#a01a1a', lw=0.6),
        )

        ax_zoom.set_yscale('log')
        ax_zoom.set_xlabel('PPO update step')
        ax_zoom.set_ylabel('Learning rate (log)')
        ax_zoom.set_title('Zoom: event behavior around the first phase graduation',
                          fontsize=11)
        ax_zoom.grid(True, which='both', alpha=0.25)

    # ------- Side note: state-machine description -------
    note = (
        "State machine (driven by PPO update steps):\n\n"
        "  step():\n"
        "    if in_warmup:  lr = lr_peak · (global_update / warmup_updates)\n"
        "       when global_update ≥ warmup_updates → exit warmup,\n"
        "       cycle_step=0, cycle_starting_lr=phase_peak\n"
        "    else:          progress = min(1, cycle_step / cycle_t_max)\n"
        "                   lr = lr_min + ½·(cycle_starting_lr − lr_min)·(1 + cos(π·progress))\n"
        "                   ↳ at progress=1 the lr is pinned to lr_min (floor-hold)\n\n"
        "  notify_phase_graduated():\n"
        "    phase_peak       = max(lr_min, phase_peak · phase_reset_factor)   # 0.8, compounds\n"
        "    cycle_starting_lr = phase_peak\n"
        "    cycle_step       = 0   # fresh cosine from the shrunken peak\n\n"
        "  notify_clone_replaced():\n"
        "    if in_warmup: return\n"
        "    target = min(phase_peak, current_lr · clone_bump_factor)   # 1.3, capped\n"
        "    if target > cycle_starting_lr: cycle_starting_lr = target\n"
        "    # cycle_step is NOT reset — frequent clones can't pin LR at the ceiling"
    )
    fig.text(0.012, 0.012, note, fontsize=8, family='monospace',
             bbox=dict(boxstyle='round,pad=0.5', fc='#f5f5f5', ec='#999', lw=0.5))

    plt.tight_layout(rect=(0.0, 0.18, 1.0, 1.0))
    out = os.path.join(os.path.dirname(__file__), 'lr_scheduler_diagram.png')
    plt.savefig(out, dpi=140, bbox_inches='tight')
    print(f"Saved: {out}")


if __name__ == '__main__':
    xs, lrs, peaks, phase_marks, clone_marks = simulate()
    plot(xs, lrs, peaks, phase_marks, clone_marks)
