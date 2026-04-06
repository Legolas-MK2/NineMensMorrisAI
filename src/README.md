# Nine Men's Morris - Curriculum PPO Training

A reinforcement learning system that trains an AI to play Nine Men's Morris using a **5-phase curriculum approach**.

## What's Different From Before?

The original code trained against mixed opponents (random + minimax + self-play) simultaneously, which confused the learning signal. This new version uses **phased curriculum learning**:

1. **Phase 1**: Master basics against RANDOM opponent (95% win rate to pass)
2. **Phase 2**: Beat weak MINIMAX D1-D2 (85% win rate to pass)
3. **Phase 3**: Reduced reward shaping vs MINIMAX D2-D3 (75% win rate to pass)
4. **Phase 4**: Sparse rewards only vs MINIMAX D3-D4 (70% win rate to pass)
5. **Phase 5**: Self-play refinement (1M episodes)

## Key Improvements

### Reward System
- **Speed-based win/loss rewards**: Fast wins get bonus (+2.0), fast losses are penalized more (-2.0)
- **Graduated reward shaping**: Full shaping in early phases, gradually removed
- **Mill detection rewards**: +0.3 for making mills, +0.5 for double mills

### Learning Rate Schedule
- Each phase has its own LR range
- Phase 1: 3e-4 → 1e-4 (high, fast learning)
- Phase 5: 5e-6 → 1e-6 (low, fine-tuning)

### Automatic Progression
- Training automatically moves to next phase when criteria are met
- Checkpoints saved at each phase transition
- Can resume from any checkpoint

## Installation

```bash
# Install dependencies
pip install torch pyspiel numpy

# Or with conda
conda install pytorch -c pytorch
pip install open_spiel
```

## Usage

### Start Training
```bash
# Basic training (uses GPU if available)
python main.py train --workers 16 --envs 32

# Quick test run
python main.py train --workers 8 --envs 8 --episodes 100000

# Start from a specific phase (e.g., skip Phase 1)
python main.py train --workers 16 --envs 32 --phase 2
```

### Resume Training
```bash
# Resume from latest checkpoint
python main.py resume --workers 16 --envs 32

# Resume from specific checkpoint
python main.py resume --checkpoint checkpoints/phase2_complete_ep500000.pt
```

### Play Against Model
```bash
python main.py play
```

### Show Curriculum Info
```bash
python main.py info
```

## Files

| File | Description |
|------|-------------|
| `main.py` | Entry point and CLI |
| `trainer.py` | PPO training loop with curriculum integration |
| `curriculum.py` | **NEW**: CurriculumManager handles phase transitions |
| `worker.py` | Experience collection with phase-aware rewards |
| `model.py` | Actor-Critic neural network |
| `minimax.py` | Minimax opponent with alpha-beta pruning |
| `utils.py` | Game utilities and reward calculation |
| `config.py` | Configuration settings |

## Curriculum Phase Details

### Phase 1: Learning the Game (vs Random)
- **Goal**: Learn legal moves, basic captures, form mills
- **Opponent**: 100% random moves
- **Rewards**: Full shaping (mills +0.3, captures +0.2)
- **Pass Criteria**: 95% win rate over 1000 games

### Phase 2: Learning Strategy (vs Minimax D1-D2)
- **Goal**: Beat weak strategic opponent
- **Opponent**: Minimax depth 1, promotes to depth 2 at 80% WR
- **Rewards**: Full shaping
- **Pass Criteria**: 85% win rate vs D2 over 500 games

### Phase 3: Reducing Reward Dependency (vs Minimax D2-D3)
- **Goal**: Learn without relying on shaping rewards
- **Opponent**: Minimax depth 2-3
- **Rewards**: 50% shaping (mills +0.15, etc.)
- **Pass Criteria**: 75% win rate vs D3 over 500 games

### Phase 4: Sparse Rewards Only (vs Minimax D3-D4)
- **Goal**: Win using game outcome signal only
- **Opponent**: Minimax depth 3-4
- **Rewards**: Win/loss/draw only (no shaping!)
- **Pass Criteria**: 70% win rate vs D4 over 500 games

### Phase 5: Self-Play Refinement
- **Goal**: Discover advanced strategies
- **Opponent**: 100% self-play
- **Rewards**: Sparse only
- **Duration**: 1 million episodes

## Monitoring Progress

The training logs to `logs/curriculum_*.csv` with columns:
- `episode`: Total episodes
- `phase`: Current phase (1-5)
- `win_rate`: Rolling win rate
- `minimax_depth`: Current minimax depth being faced
- `minimax_depth_beaten`: Highest depth AI can beat in evaluation

## Tips

1. **Phase 1 should be fast** - If stuck at <90% WR vs random after 500k episodes, something's wrong

2. **Phase transitions are automatic** - Watch for "🎓 GRADUATED" messages

3. **Save checkpoints frequently** - Training can resume from any phase

4. **Monitor entropy** - Should decrease over phases (exploration → exploitation)

5. **Expected training time** (16 workers, 32 envs each):
   - Phase 1: ~30 min
   - Phase 2: ~1-2 hours
   - Phase 3: ~2-3 hours
   - Phase 4: ~3-4 hours
   - Phase 5: ~2-3 hours
   - **Total**: ~8-12 hours to complete all phases

## Troubleshooting

### Win rate stuck at 50% in Phase 1
- Check that opponent_type is 'random', not 'self'
- Verify rewards are being calculated correctly

### Phase 2 not progressing
- Minimax D1 should be beatable with basic strategy
- Check if model is exploring (entropy > 0.5)

### Training too slow
- Increase `--workers` and `--envs`
- Ensure GPU is being used (check device output)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CurriculumManager                        │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │ Phase 1 │→ │ Phase 2 │→ │ Phase 3 │→ │ Phase 4 │→ ...   │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘        │
│       ↓            ↓            ↓            ↓              │
│   opponent     opponent     opponent     opponent           │
│    config       config       config       config            │
│   rewards      rewards      rewards      rewards            │
│      LR          LR           LR           LR               │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      PPOTrainer                             │
│  ┌──────────┐     ┌───────────────┐     ┌──────────────┐   │
│  │ Workers  │ ←→  │ Inference GPU │ ←→  │ PPO Update   │   │
│  │ (games)  │     │ (batch)       │     │ (gradients)  │   │
│  └──────────┘     └───────────────┘     └──────────────┘   │
└─────────────────────────────────────────────────────────────┘
```
