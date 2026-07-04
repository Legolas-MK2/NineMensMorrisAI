"""Tests for the curriculum state machine in src/curriculum.py.

Covers opponent-distribution math, explicit game outcomes, depth unlock,
dampening hysteresis, graduation, schedules, and save/load round-trips —
the logic that drives multi-day training runs.
"""

from __future__ import annotations

import math
import os
import sys

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import pytest

from curriculum import (
    CurriculumManager, MixedTrainingState, Phase, PHASE_CONFIGS,
    MIXED_CONFIG, GRADUATION_CONFIG, RESULTS_WINDOW,
    PHASE_10_POST_SHAPING_EPISODES, PHASE_11_FULL_GAME_EPISODES,
    PHASE_11_CYCLE_EPISODES, SHAPING_DECAY_EPISODES,
    compute_opponent_distribution,
)


@pytest.fixture()
def manager(tmp_path):
    return CurriculumManager(save_dir=str(tmp_path))


# ---------------------------------------------------------------------------
# compute_opponent_distribution
# ---------------------------------------------------------------------------

def test_distribution_sums_to_one_and_matches_docstring_examples():
    d = compute_opponent_distribution(list(range(1, 8)), set())
    assert math.isclose(sum(d.values()), 1.0)
    assert math.isclose(d['self'], 3 / 11)
    assert math.isclose(d['random'], 1 / 11)
    assert math.isclose(d['minimax_d3'], 1 / 11)

    d = compute_opponent_distribution([], set())
    assert d == pytest.approx({'self': 0.75, 'random': 0.25})


def test_distribution_pins_dampened_slots():
    d = compute_opponent_distribution(list(range(1, 8)), {'random'})
    assert d['random'] == pytest.approx(0.01)
    assert math.isclose(sum(d.values()), 1.0)
    assert d['self'] == pytest.approx(0.99 * 3 / 10)


def test_distribution_all_dampened_degenerates_to_cap():
    slots = {'self', 'random', 'minimax_d1'}
    d = compute_opponent_distribution([1], slots)
    assert all(v == pytest.approx(0.01) for v in d.values())


# ---------------------------------------------------------------------------
# add_game_result (explicit outcomes)
# ---------------------------------------------------------------------------

def test_add_game_result_tracks_explicit_outcomes(manager):
    manager.add_game_result('win', opponent_type='random')
    manager.add_game_result('loss', opponent_type='minimax', minimax_depth=1)
    manager.add_game_result('draw', opponent_type='self')

    assert manager.total_episodes == 3
    assert manager.stats.wins == 1
    assert manager.stats.losses == 1
    assert manager.stats.draws == 1
    assert list(manager.mixed_state.results_vs_random) == ['win']
    assert list(manager.mixed_state.results_vs_minimax[1]) == ['loss']
    assert list(manager.mixed_state.results_vs_self) == ['draw']
    assert manager.mixed_state.games_vs_random == 1
    assert manager.mixed_state.games_vs_minimax == 1
    assert manager.mixed_state.games_vs_self == 1


def test_add_game_result_rejects_invalid_outcome(manager):
    with pytest.raises(ValueError):
        manager.add_game_result('victory')


def test_result_windows_are_bounded(manager):
    for _ in range(RESULTS_WINDOW + 50):
        manager.add_game_result('win', opponent_type='random')
    assert len(manager.mixed_state.results_vs_random) == RESULTS_WINDOW


# ---------------------------------------------------------------------------
# Depth unlock
# ---------------------------------------------------------------------------

def test_depth_unlock_progression(manager):
    manager.current_phase = Phase.PHASE_2
    ms = manager.mixed_state
    min_games = MIXED_CONFIG['minimax_depth_unlock_min_games']

    # Not enough games -> no unlock.
    for _ in range(min_games - 1):
        ms.record_game_result('minimax', 'win', 1)
    assert manager.check_and_unlock_minimax_depth() is False
    assert ms.active_minimax_max_depth == 1

    # Enough games at >= 50% WR -> unlock D2.
    ms.record_game_result('minimax', 'win', 1)
    assert manager.check_and_unlock_minimax_depth() is True
    assert ms.active_minimax_max_depth == 2


def test_depth_unlock_caps_at_training_max(manager):
    manager.current_phase = Phase.PHASE_2
    ms = manager.mixed_state
    ms.active_minimax_max_depth = MIXED_CONFIG['minimax_max_depth']
    for _ in range(200):
        ms.record_game_result('minimax', 'win', ms.active_minimax_max_depth)
    assert manager.check_and_unlock_minimax_depth() is False
    assert ms.active_minimax_max_depth == MIXED_CONFIG['minimax_max_depth']


def test_opponent_distribution_clamps_depths_beyond_training_cap(manager):
    manager.current_phase = Phase.PHASE_9
    manager.mixed_state.active_minimax_max_depth = 7  # e.g. old checkpoint
    dist = manager.get_opponent_distribution()
    assert 'minimax_d5' in dist
    assert 'minimax_d6' not in dist and 'minimax_d7' not in dist


# ---------------------------------------------------------------------------
# Dampening hysteresis
# ---------------------------------------------------------------------------

def test_dampening_sets_and_recovers_with_hysteresis(manager):
    ms = manager.mixed_state
    min_games = MIXED_CONFIG['minimax_depth_dominate_min_games']
    for _ in range(min_games):
        ms.record_game_result('minimax', 'win', 1)
        ms.record_game_result('random', 'win')

    # Above threshold -> dampened.
    ms.update_dampened_state({'wr_vs_mm_d1': 0.95, 'wr_vs_random': 0.95})
    assert ms.minimax_depth_dominated[1] is True
    assert ms.random_dominated is True
    assert {'minimax_d1', 'random'} <= ms.get_dampened_set()

    # In the hysteresis band (recover <= wr < threshold) -> stays dampened.
    ms.update_dampened_state({'wr_vs_mm_d1': 0.87, 'wr_vs_random': 0.87})
    assert ms.minimax_depth_dominated[1] is True

    # Below recover -> restored.
    ms.update_dampened_state({'wr_vs_mm_d1': 0.80, 'wr_vs_random': 0.80})
    assert ms.minimax_depth_dominated[1] is False
    assert ms.random_dominated is False


def test_selfplay_timed_dampening(manager):
    ms = manager.mixed_state
    ms.total_episodes = 100
    ms.selfplay_train_cooldown_until = 200
    assert ms.is_selfplay_dampened() is True
    assert 'self' in ms.get_dampened_set()
    ms.total_episodes = 200
    assert ms.is_selfplay_dampened() is False


# ---------------------------------------------------------------------------
# Graduation
# ---------------------------------------------------------------------------

def _fill_min_games(manager):
    cfg = manager.get_config()
    for _ in range(cfg.min_games_for_graduation):
        manager.stats.recent_results.append('win')


def test_phase1_graduates_on_wr_vs_random(manager):
    cfg = PHASE_CONFIGS[Phase.PHASE_1]
    _fill_min_games(manager)
    manager.stats.episodes_in_phase = cfg.min_episodes

    # Below threshold -> no graduation.
    for _ in range(100):
        manager.mixed_state.record_game_result('random', 'loss')
    assert manager.should_graduate() is False

    # Fill the window with wins so WR clears the threshold.
    for _ in range(RESULTS_WINDOW):
        manager.mixed_state.record_game_result('random', 'win')
    assert manager.should_graduate() is True
    assert manager.graduate() is True
    assert manager.current_phase == Phase.PHASE_2
    assert 'WR vs random' in manager.phase_history[-1]['graduation_reason']


def test_phase2_graduates_on_plateau(manager):
    manager.current_phase = Phase.PHASE_2
    manager.stats.phase = Phase.PHASE_2
    _fill_min_games(manager)
    manager.stats.episodes_in_phase = manager.graduation_min_episodes

    ms = manager.mixed_state
    # Flat WR window for the (only) unlocked depth -> plateau.
    for _ in range(GRADUATION_CONFIG['trend_window_samples']):
        ms.minimax_wr_history_by_depth[1].append(0.60)
    assert manager.should_graduate() is True

    # A clearly climbing window must block graduation.
    ms.minimax_wr_history_by_depth[1].clear()
    for i in range(GRADUATION_CONFIG['trend_window_samples']):
        ms.minimax_wr_history_by_depth[1].append(0.10 + 0.04 * i)
    assert manager.should_graduate() is False


def test_phase2_requires_min_episodes(manager):
    manager.current_phase = Phase.PHASE_2
    manager.stats.phase = Phase.PHASE_2
    _fill_min_games(manager)
    manager.stats.episodes_in_phase = manager.graduation_min_episodes - 1
    for _ in range(GRADUATION_CONFIG['trend_window_samples']):
        manager.mixed_state.minimax_wr_history_by_depth[1].append(0.60)
    assert manager.should_graduate() is False


def test_phase11_never_graduates(manager):
    manager.current_phase = Phase.PHASE_11
    manager.stats.phase = Phase.PHASE_11
    _fill_min_games(manager)
    manager.stats.episodes_in_phase = 10 * PHASE_11_CYCLE_EPISODES
    assert manager.should_graduate() is False


def test_graduation_carries_depth_unlock_and_sample_interval(tmp_path):
    manager = CurriculumManager(save_dir=str(tmp_path), wr_sample_interval=10_000)
    manager.current_phase = Phase.PHASE_2
    manager.stats.phase = Phase.PHASE_2
    manager.mixed_state.active_minimax_max_depth = 4
    manager.graduate()
    assert manager.current_phase == Phase.PHASE_3
    assert manager.mixed_state.active_minimax_max_depth == 4
    assert manager.mixed_state.wr_sample_interval == 10_000


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------

def test_shaping_multiplier_decays_linearly(manager):
    assert manager.get_shaping_multiplier() == pytest.approx(1.0)
    manager.total_episodes = SHAPING_DECAY_EPISODES // 2
    assert manager.get_shaping_multiplier() == pytest.approx(0.5)
    manager.total_episodes = SHAPING_DECAY_EPISODES * 2
    assert manager.get_shaping_multiplier() == 0.0


def test_phase10_draw_penalty_decays(manager):
    manager.current_phase = Phase.PHASE_10
    manager.stats.phase = Phase.PHASE_10
    start = PHASE_CONFIGS[Phase.PHASE_10].draw_penalty
    manager.stats.episodes_in_phase = 0
    assert manager.get_phase10_draw_penalty() == pytest.approx(start)
    manager.stats.episodes_in_phase = 10 ** 9
    from curriculum import PHASE_10_DRAW_PENALTY_END
    assert manager.get_phase10_draw_penalty() == pytest.approx(PHASE_10_DRAW_PENALTY_END)


def test_phase10_duration_anchored_to_shaping_tail(manager):
    manager.current_phase = Phase.PHASE_10
    manager.stats.phase = Phase.PHASE_10
    _fill_min_games(manager)
    # Shaping ended long ago -> phase 10 lasts exactly the post-shaping floor.
    manager.total_episodes = SHAPING_DECAY_EPISODES + PHASE_10_POST_SHAPING_EPISODES + 10
    manager.stats.episodes_in_phase = PHASE_10_POST_SHAPING_EPISODES - 1
    assert manager.should_graduate() is False
    manager.total_episodes += 1
    manager.stats.episodes_in_phase = PHASE_10_POST_SHAPING_EPISODES
    assert manager.should_graduate() is True


def test_phase11_subphase_cycles(manager):
    manager.current_phase = Phase.PHASE_11
    manager.stats.phase = Phase.PHASE_11
    manager.stats.episodes_in_phase = 0
    assert manager.get_phase11_subphase() == 'full'
    assert manager.get_starting_stones_for_phase() == 9
    assert manager.get_ai_disadvantage() is False

    manager.stats.episodes_in_phase = PHASE_11_FULL_GAME_EPISODES
    assert manager.get_phase11_subphase() == 'mix'
    assert manager.get_starting_stones_for_phase() == -1
    assert manager.get_ai_disadvantage() is True

    manager.stats.episodes_in_phase = PHASE_11_CYCLE_EPISODES
    assert manager.get_phase11_subphase() == 'full'


def test_slope_uses_configured_sample_interval():
    # Same WR history, half the sample interval -> double the per-episode slope.
    a = MixedTrainingState(wr_sample_interval=25_000)
    b = MixedTrainingState(wr_sample_interval=12_500)
    for i in range(20):
        a.minimax_wr_history_by_depth[1].append(0.01 * i)
        b.minimax_wr_history_by_depth[1].append(0.01 * i)
    sa = a.calculate_winrate_trend_for_depth(1)
    sb = b.calculate_winrate_trend_for_depth(1)
    assert sb == pytest.approx(2 * sa)


# ---------------------------------------------------------------------------
# Persistence round-trip
# ---------------------------------------------------------------------------

def test_state_dict_round_trip(tmp_path):
    m1 = CurriculumManager(save_dir=str(tmp_path / "a"), wr_sample_interval=10_000)
    m1.current_phase = Phase.PHASE_3
    m1.stats.phase = Phase.PHASE_3
    for outcome, opp, d in [('win', 'random', 0), ('loss', 'minimax', 2),
                            ('draw', 'self', 0), ('win', 'minimax', 1)]:
        m1.add_game_result(outcome, opponent_type=opp, minimax_depth=d)
    m1.mixed_state.active_minimax_max_depth = 3
    m1.mixed_state.minimax_depth_dominated[2] = True
    m1.mixed_state.random_dominated = True
    m1.mixed_state.minimax_wr_history_by_depth[1].extend([0.4, 0.5, 0.6])
    m1.mixed_state.selfplay_train_cooldown_until = 12345

    state = m1.to_state_dict()

    m2 = CurriculumManager(save_dir=str(tmp_path / "b"), wr_sample_interval=10_000)
    assert m2.load_state_dict(state) is True

    assert m2.current_phase == Phase.PHASE_3
    assert m2.total_episodes == m1.total_episodes
    ms1, ms2 = m1.mixed_state, m2.mixed_state
    assert ms2.active_minimax_max_depth == 3
    assert ms2.minimax_depth_dominated[2] is True
    assert ms2.random_dominated is True
    assert ms2.selfplay_train_cooldown_until == 12345
    assert ms2.wr_sample_interval == 10_000
    for d in range(1, 8):
        assert list(ms2.results_vs_minimax[d]) == list(ms1.results_vs_minimax[d])
        assert list(ms2.minimax_wr_history_by_depth[d]) == \
            list(ms1.minimax_wr_history_by_depth[d])
    assert list(ms2.results_vs_random) == list(ms1.results_vs_random)
    assert list(ms2.results_vs_self) == list(ms1.results_vs_self)


def test_save_and_load_state_file_round_trip(tmp_path):
    m1 = CurriculumManager(save_dir=str(tmp_path))
    m1.add_game_result('win', opponent_type='random')
    m1.save_state()

    m2 = CurriculumManager(save_dir=str(tmp_path))
    assert m2.load_state() is True
    assert m2.total_episodes == 1
    assert list(m2.mixed_state.results_vs_random) == ['win']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
