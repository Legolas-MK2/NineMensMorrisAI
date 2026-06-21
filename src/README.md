# Nine Men's Morris — Curriculum PPO Training

A reinforcement-learning system that trains an AI to play Nine Men's Morris
using an **11-phase curriculum** built on top of a relational token-based
ActorCritic and the [fastnmm](../fastnmm/) C++ game engine.

The training pipeline is asynchronous PPO: a pool of CPU worker processes
collects games (handling random / minimax / self-play opponents and reward
shaping), a single GPU thread serves batched policy inference, and the
trainer periodically runs PPO updates on the accumulated experience.

## Curriculum at a glance

Every phase samples opponents from a shared **mixed-opponent distribution**
(self-play, minimax depths D1..D5, random). Boards are seeded via fastnmm's
`starting_stones` engine option — no random moves are played to prepare
positions. The per-phase **stone distribution** is the main knob that walks
the model from short flying-only games up to the full 9-stone game.

| Phase | Per-player stones | Opponents (base) | Notes |
|-------|-------------------|------------------|-------|
| 1  | uniform over {3..9} | 75% self / 25% random | Warmup; minimax locked. Graduates on WR vs random ≥ 95%. |
| 2  | 3:78%, 4:18%, 5:5% | self + unlocked minimax + random | Trend-based graduation (per-depth plateau). |
| 3  | 3:22%, 4:55%, 5:18%, 6:5% | self + minimax + random | " |
| 4  | 3:5%, 4:18%, 5:55%, 6:18%, 7:5% | self + minimax + random | " |
| 5  | 4:5%, 5:18%, 6:55%, 7:18%, 8:5% | self + minimax + random | " |
| 6  | 5:5%, 6:18%, 7:55%, 8:18%, 9:5% | self + minimax + random | " |
| 7  | 6:5%, 7:18%, 8:55%, 9:22% | self + minimax + random | " |
| 8  | 7:5%, 8:18%, 9:78% | self + minimax + random | " |
| 9  | 9:100% (full game) | self + minimax + random | " |
| 10 | uniform over {3..9} | self + minimax + random | Duration anchored to the shaping-decay schedule. |
| 11 | alternates 9:100% (full) and uniform {3..9} every 2.5M eps | self + minimax + random | **Infinite** — Phase 9 + Phase 10 combined; never graduates, stop with Ctrl-C. |

Each player draws its stone count **independently** from the phase's
distribution at every env reset, so asymmetric pairs are common and adjacent
phases overlap by design — phase transitions are smooth instead of cliff
edges.

### Opponent sampling

Within "mixed" the sampling distribution is computed by
`compute_opponent_distribution(unlocked_depths, dampened_set)` in
[curriculum.py](curriculum.py):

- Every slot gets weight 1, except **self-play which gets weight 3**.
- Dampened slots are pinned at `dampen_cap` (1%); the rest of the mass is
  redistributed by weight.
- **Minimax depths unlock progressively**: D(n+1) unlocks once WR vs D(n)
  ≥ 50% over at least 100 games. The training cap is **D5** — D6/D7 are
  tracked only via the periodic progressive minimax eval.
- **Per-opponent hysteresis**: when WR vs a slot crosses 90% over 100+
  games, that slot collapses to `dampen_cap` and stops feeding the PPO
  gradient. It recovers when WR drops back below 85%.
- **Self-play has a timed dampener instead**: when `wr_vs_self` exceeds
  `selfplay_train_pause_threshold` (default 95%) at a log tick, self-play
  sampling pins at 1% and self-play experiences stop feeding PPO for
  `selfplay_train_pause_episodes` episodes (default 500k).
- **Self-play clone update**: when `wr_vs_self` over the rolling 500-game
  self-play window crosses 80% at a log tick, the clone is replaced with
  the current model and the self-play window resets.

### Reward shaping

Shaping uses **Potential-Based Reward Shaping (PBRS)**:

    r_shape = γ · Φ(s') − Φ(s) + step_penalty

This preserves the optimal policy for any shaping scale. The shaping scale
**decays linearly from 1.0 → 0.0 over the first 20M episodes of training
(global, not per-phase)**. Phase 10 deliberately keeps consuming the same
schedule rather than zeroing the multiplier at phase entry, so the value
function doesn't get shocked at the boundary. Phase 10's duration is
defined as "long enough for the last 5M episodes to be shaping-free".

Φ(s) is built from minimax-aligned board features (mills, potential mills,
piece counts, mobility, double mills) — see
`RewardCalculator.calculate_potential` in [utils.py](utils.py). Terminal
rewards (win/loss/draw) are not shaped; wins also include a speed bonus
proportional to how quickly the game ended.

### Graduation

- **Phase 1**: graduates when WR vs random ≥ 95% over the recent window.
- **Phases 2–9**: trend-based **per-depth plateau detection**. For every
  unlocked minimax depth, the slope of WR-vs-depth (over a 1M-episode
  horizon) must be below `trend_max_angle_degrees` (default **2°**). All
  unlocked depths must plateau simultaneously, and there is a `min_episodes`
  floor of 1.5M episodes per phase.
- **Phase 10**: ends after `PHASE_10_POST_SHAPING_EPISODES` (default 5M)
  shaping-free episodes — duration is the shaping overlap plus the
  shaping-free tail.
- **Phase 11**: **never graduates**. The phase runs forever, alternating
  between `PHASE_11_FULL_GAME_EPISODES` (default 2.5M) episodes of full-game
  9-stone play and `PHASE_11_MIX_EPISODES` (default 2.5M) episodes of
  uniform-{3..9} mix. Stop training with Ctrl-C when you've collected the
  amount of polish you want. Shaping is already 0 by the time training
  reaches phase 11; the draw penalty is held flat at the same end value
  Phase 10 decayed to (`PHASE_11_DRAW_PENALTY` = -0.2) so the cycling
  sub-phases don't re-trigger any decay schedules. On every sub-phase flip
  the trainer rebroadcasts the new stone distribution to workers; the
  log-line prefix shows the current sub-phase (`full` or `mix`) and the
  episode position within it.

Phase transitions automatically reset the per-phase WR windows, **carry
over** the active minimax depth ceiling so a graduated phase doesn't have
to re-unlock D1→D5, and trigger an LR scheduler reset (peak ×
`lr_phase_reset_factor`, fresh cosine cycle).

## Architecture

### Model

Relational token-based ActorCritic ([model.py](model.py)):

- 24 board points are tokens, embedded with per-position learned embeddings.
- A 25th **global token** carries phase / piece-count features.
- `n_layers` × (multi-head attention + feed-forward) over the 25 tokens.
- Attention adds **learned per-head structural biases between board tokens
  only** — one scalar for adjacency, one for mill cohabitation. The bias
  matrices are sigma-invariant under every board automorphism.
- Policy head is **pointer-style**: per-node placement/capture logits +
  inner-product `(from, to)` move logits, producing 600 action logits that
  match the engine's action layout.
- Value head is an MLP over the global token (optionally concatenated with
  the mean of board tokens).

### Symmetry augmentation

The 24-point board graph has an order-16 automorphism group (D4 of the 7×7
grid × inner/outer-ring swap). [symmetry.py](symmetry.py) precomputes the
16 permutations of board points and the corresponding action permutations.
During rollout each game (or each step, depending on `aug_granularity`)
gets a random sigma applied to `(obs, mask, action)`, giving 16× effective
position diversity at zero compute cost. The network's structural biases
are invariant to sigma, so the bias is correct in every augmented frame.

### Parallelism

- **N worker processes**, each managing `envs_per_worker` games. Workers
  run across all available CPU cores with a niceness offset.
- **One thread pool per worker** for asynchronous minimax (default 2
  threads per worker). The C++ minimax engine releases the GIL during
  search, so multiple bot threads run concurrently on different cores.
- **Single GPU inference server** in the trainer's main thread, batching
  observations from all workers each tick.

### LR schedule

Warmup + warm-restart cosine, driven by PPO updates rather than episodes
([lr_scheduler.py](lr_scheduler.py)):

1. Linear warmup `0 → lr_peak` over `lr_warmup_episodes`.
2. Cosine cycle anneals `lr_peak → lr_min` over `lr_cycle_t_max_episodes`.
3. On **phase graduation**: peak `*= lr_phase_reset_factor`, fresh cycle.
4. On **clone replacement**: raise the current cycle ceiling toward
   `phase_peak` (capped). Cycle progress is preserved so frequent clone
   churn cannot pin LR at the ceiling.

## Installation

```bash
# Install Python dependencies
pip install -r ../requirements.txt

# Install the local fastnmm game engine (C++ bindings)
pip install -e ../fastnmm
```

## Usage

```bash
# Standard training (use --help for the full option list)
python main.py train --workers 22 --envs 48

# Resume from the latest checkpoint
python main.py resume --workers 22 --envs 48

# Resume from a specific checkpoint
python main.py resume --checkpoint checkpoints/<file>.pt

# Skip to a specific phase (1-11) from a fresh model
python main.py train --workers 22 --envs 48 --phase 5

# Jump straight into the infinite final phase (stop with Ctrl-C)
python main.py train --workers 22 --envs 48 --phase 11

# Load model weights from a checkpoint but reset training state to ep 0 / Phase 1
python main.py train --workers 22 --envs 48 --use-last-checkpoint

# Play / watch a trained model
python main.py play

# Print the full per-phase curriculum description (including stone distributions)
python main.py info
```

## Files

| File | Description |
|------|-------------|
| `main.py` | CLI entry point (`train`, `resume`, `play`, `info`). |
| `trainer.py` | PPO training loop with curriculum integration. |
| `curriculum.py` | `CurriculumManager`, per-phase configs, opponent-distribution and graduation logic. |
| `worker.py` | Worker-process experience collection with curriculum-aware opponents and reward shaping. |
| `model.py` | Relational token-based ActorCritic. |
| `minimax.py` | Python wrappers around the fastnmm C++ minimax engine + progressive eval. |
| `utils.py` | Game helpers, PBRS reward calculator, GAE, experience batch dataclass. |
| `symmetry.py` | Board automorphism group + obs/mask/action permutations for augmentation. |
| `config.py` | Configuration dataclass. |
| `lr_scheduler.py` | Warmup + warm-restart cosine LR scheduler. |
| `logging_setup.py` | Process-wide logger configuration. |
| `system_monitor.py` | Heartbeat-watching sidecar (detects "main thread stuck in C++" stalls). |
| `model_loader.py` | Helper for loading model checkpoints in eval/play contexts. |
| `board_utils.py` | Board printing / pretty-printing helpers. |

## Monitoring

Training logs to `logs/<timestamp>_curriculum.csv` with one row per log
tick (default every `log_interval` = 25k episodes). Key columns:

- `episode`, `phase`, `starting_stones`, `steps`, `eps_per_sec`
- `avg_return`, `ema_return`, `win_rate`, `draw_rate`
- `policy_loss`, `value_loss`, `entropy`, `kl_div`, `clip_frac`, `grad_norm`
- `lr`, `cycle_step`, `last_reset_event`
- `minimax_depth_beaten` (cached from the progressive minimax eval)
- `clone_gen`, `active_mm_max_depth`, `shaping_mult`
- `wr_vs_mm_d1` .. `wr_vs_mm_d7`, `wr_vs_random`, `wr_vs_self`
- `slope_angle_d1` .. `slope_angle_d7`, `samples_in_window_d1` .. `_d7`
  (per-depth graduation diagnostics)

A second sidecar CSV `logs/phase_graduations.csv` records per-phase
graduation snapshots (episodes/updates in phase, WR vs every unlocked
depth, slope angles, samples, LR before reset).

The trainer also writes an atomic JSON heartbeat to
`logs/trainer_heartbeat.json` consumed by [system_monitor.py](system_monitor.py)
to detect main-thread stalls.

## Tips

1. **Phase 1 should be fast** — 75% self-play / 25% random; if it doesn't
   reach 95% WR vs random quickly, something is wrong with the env
   bootstrap.
2. **Watch the minimax-depth unlock chain.** `active_mm_max_depth` should
   creep up from D1 to D5 over the course of phases 2–9; if it stalls,
   the curriculum is gated on whatever depth is currently the ceiling.
3. **Per-depth dampening matters.** A `(no-train)` / `*` annotation on a
   depth in the WR(500) log line means that opponent's PPO contribution is
   pinned; if every minimax slot dampens at once, training collapses to
   self-play.
4. **Entropy.** Should decay from `entropy_coef_start` toward
   `entropy_coef_end` over `entropy_decay_episodes`. If entropy crashes
   early the policy is collapsing — raise the entropy floor.
5. **Resuming.** `python main.py resume` reloads model + optimizer + LR
   scheduler + curriculum state (per-depth WR windows, dominated flags,
   clone generation, etc.) and continues from the recorded episode count.
