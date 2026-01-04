# Rewards Table for Each Phase

This table shows all the reward values for each phase of the Nine Men's Morris AI training curriculum.

| Reward Type | Description | Phase 1 | Phase 2 | Phase 3 | Phase 4 | Phase 5 |
|-------------|-------------|---------|---------|---------|---------|---------|
| **Terminal Rewards** |
| win_reward_base | Base reward for winning | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| win_reward_speed_bonus | Bonus for fast wins | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| loss_reward | Penalty for losing | -1.0 | -1.0 | -1.0 | -1.0 | -1.0 |
| draw_penalty | Penalty for draws | -0.8 | -0.8 | -0.8 | -0.8 | -0.8 |
| **Shaping Rewards** |
| shaping_multiplier | Multiplier for all shaping rewards | 1.0 | 0.7 | 0.3 | 0.1 | 0.0 |
| mill_reward | Reward for making a mill | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 |
| enemy_mill_penalty | Penalty for opponent making mill | -0.2 | -0.2 | -0.2 | 0.2 | 0.2 |
| block_mill_reward | Reward for blocking opponent's mill | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 |
| double_mill_reward | Reward for making 2 mills in one move | 0.3 | 0.3 | 0.3 | 0.3 | 0.3 |
| setup_capture_reward | Reward for moves that set up captures | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
| **Learning Rate** |
| lr_start | Starting learning rate | 3e-4 | 1e-4 | 5e-5 | 1e-5 | 5e-6 |
| lr_end | Ending learning rate | 1e-4 | 5e-5 | 1e-5 | 5e-6 | 1e-6 |

## Notes:
1. Shaping rewards are multiplied by the shaping_multiplier for each phase
2. In Phase 4 and Phase 5, all shaping rewards are disabled (multiplier = 0.0)
3. Learning rates decrease as training progresses through phases
4. Terminal rewards generally become less extreme as training progresses
5. Speed bonuses are reduced in later phases to focus on strategic play

## Phase Summaries:
- **Phase 1**: Focus on learning basics with reduced shaping rewards
- **Phase 2**: Learning strategy with moderate shaping
- **Phase 3**: Reducing dependency on shaping rewards
- **Phase 4**: Sparse rewards only, learning from game outcomes
- **Phase 5**: Self-play refinement with sparse rewards only


