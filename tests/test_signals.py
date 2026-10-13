"""Core-logic tests on small synthetic players (no network needed). Run: pytest"""
import numpy as np
import pandas as pd

from src import signals, trend


def make_player(snaps, opp=None, passdown=None, pid="TestPl00", pos="WR", team="SD"):
    n = len(snaps)
    return pd.DataFrame({
        "season": 2025, "week": range(1, n + 1), "pfr_player_id": pid, "player": pid,
        "position": pos, "team": team, "gsis_id": pid, "espn_id": 1.0,
        "snap_pct": snaps,
        "opp_share": opp if opp is not None else [s * 0.2 for s in snaps],
        "passdown_share": passdown if passdown is not None else [s * 0.2 for s in snaps],
    })


SHIFT_NOISE = {"WR": {"snap_pct": 0.17, "opp_share": 0.036, "composite": 0.9}}
GAME_NOISE = {"WR": {"snap_pct": 0.24, "opp_share": 0.05, "passdown_share": 0.09}}


# --- sudden drops (likely in-game injuries) ---------------------------------

def test_sudden_drop_flags_collapse_of_a_starter():
    assert signals._sudden_drops(np.array([0.9, 0.9, 0.2])).tolist() == [False, False, True]


def test_sudden_drop_ignores_part_time_players():
    # usual snap share under 50% -> a low game is not treated as an injury
    assert not signals._sudden_drops(np.array([0.3, 0.3, 0.1])).any()


# --- shift score ------------------------------------------------------------

def test_shift_score_positive_for_a_riser_and_negative_for_a_faller():
    rise = signals.score(signals.build_windows(make_player([0.2, 0.2, 0.7, 0.8])), SHIFT_NOISE)
    fall = signals.score(signals.build_windows(make_player([0.8, 0.8, 0.4, 0.3])), SHIFT_NOISE)
    assert rise.shift_score.iloc[0] > 2
    assert fall.shift_score.iloc[0] < -2


def test_injury_blip_does_not_fake_a_falling_role():
    # Ja'Marr Chase, 2026: left week 4 hurt (20%), back to 89% in week 5. Not a role change.
    chase = signals.score(signals.build_windows(make_player([0.94, 0.88, 1.0, 0.20, 0.89])), SHIFT_NOISE)
    assert chase.injury_games_excluded.iloc[0] == 1
    assert chase.shift_score.iloc[0] > -1


def test_real_benching_is_still_flagged():
    # Two low games in a row, no bounce-back: that can be a real role loss, so it must count.
    benched = signals.score(signals.build_windows(make_player([0.9, 0.9, 0.9, 0.3, 0.3])), SHIFT_NOISE)
    assert benched.injury_games_excluded.iloc[0] == 0
    assert benched.shift_score.iloc[0] < -2


def test_shift_score_needs_enough_games():
    # 3 games = 1 baseline game, below MIN_BASELINE_GAMES -> no window
    assert signals.build_windows(make_player([0.5, 0.5, 0.9])).empty


# --- CUSUM slow bleed -------------------------------------------------------

def test_early_season_with_no_eligible_players_does_not_crash():
    usage = make_player([0.5, 0.6])
    assert signals.build_windows(usage).empty
    assert trend.build_trends(usage).empty
    assert trend.cusum_all(usage, GAME_NOISE).empty


def test_cusum_flat_player_stays_at_zero():
    path = trend.cusum_path(make_player([0.7] * 10), GAME_NOISE)
    assert (path.cusum_bleed == 0).all() and (path.cusum_climb == 0).all()


def test_cusum_slow_bleed_crosses_threshold():
    # loses at most 10 points of snap share in any one game, but keeps sliding
    snaps = [0.7, 0.7, 0.7, 0.6, 0.55, 0.5, 0.45, 0.4, 0.4, 0.35, 0.35, 0.35]
    path = trend.cusum_path(make_player(snaps), GAME_NOISE)
    assert path.cusum_bleed.iloc[-1] <= -3
    assert path.cusum_bleed.iloc[4] > -0.5     # the first small dips are absorbed by the allowance


def test_cusum_one_injury_game_cannot_trigger_a_flag_alone():
    # cap of 3 SD minus the 0.5 allowance -> one game moves the tally at most -2.5
    path = trend.cusum_path(make_player([0.9, 0.9, 0.9, 0.0]), GAME_NOISE)
    assert path.cusum_bleed.iloc[-1] == -(trend.CUSUM_CAP - trend.CUSUM_K)
    assert path.cusum_bleed.iloc[-1] > -3


def test_injury_check_survives_when_too_few_games_for_a_shift_score():
    # Saquon Barkley, 2026: early dip he bounced back from, then hurt again in his latest game.
    saquon = signals.build_windows(make_player([0.71, 0.16, 0.72, 0.07]))
    assert len(saquon) == 1 and bool(saquon.sudden_drop.iloc[0])
