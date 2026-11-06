import numpy as np
import pandas as pd
import pytest

from src.volatility import (
    MAX_WEEKLY_CHANGE,
    METRICS,
    calculate_volatility,
    eligibility_minimums,
    score_weeks,
    typical_weekly_change,
    weekly_changes,
)


def player_games(player_id, weeks, snaps, position="RB", team="AAA",
                 season=2025, opportunities=12):
    """One row per game for a test player."""
    return pd.DataFrame({
        "season": season,
        "week": weeks,
        "gsis_id": player_id,
        "player_name": f"Player {player_id}",
        "position": position,
        "team": team,
        "snap_pct": snaps,
        "opp_share": 0.20,
        "passdown_share": 0.20,
        "redzone_share": 0.20,
        "targets": opportunities / 2,
        "carries": opportunities / 2,
    })


def test_bye_week_keeps_comparison_but_missed_game_breaks_it():
    # The team plays weeks 1-3 and 5-6 (bye in week 4).
    full = player_games("a", [1, 2, 3, 5, 6], [0.5, 0.6, 0.5, 0.6, 0.5])
    missed = player_games("b", [1, 2, 5, 6], [0.5, 0.6, 0.5, 0.6])

    changes = weekly_changes(pd.concat([full, missed]))
    compared = changes.set_index(["gsis_id", "week"])["snap_change"].notna()

    assert compared["a"].to_dict() == {
        1: False, 2: True, 3: True, 5: True, 6: True,
    }
    # Week 5 isn't compared with week 2 because week 3 was missed.
    assert compared["b"].to_dict() == {
        1: False, 2: True, 5: False, 6: True,
    }


def test_weekly_change_is_capped():
    games = player_games("a", [1, 2], [0.90, 0.10])

    change = weekly_changes(games)["snap_change"].iloc[1]

    assert change == pytest.approx(MAX_WEEKLY_CHANGE)


def test_typical_change_scores_one():
    # Every game-to-game snap change is 10 points.
    snaps = [0.50, 0.60] * 5
    data = pd.concat([
        player_games("a", list(range(1, 11)), snaps),
        player_games("b", list(range(1, 11)), snaps[::-1]),
    ])

    weekly, seasons, typical = calculate_volatility(
        data, baseline_seasons=None
    )

    assert typical.loc["RB", "snap_pct"] == pytest.approx(0.10)
    scored = weekly["weekly_volatility"].dropna()
    assert len(scored) == 18
    assert np.allclose(scored, 1.0)
    assert np.allclose(seasons["volatility_score"], 1.0)


def test_missing_red_zone_keeps_the_week():
    changes = weekly_changes(
        player_games("a", [1, 2], [0.50, 0.60])
    )
    changes.loc[1, "redzone_change"] = np.nan
    typical = pd.DataFrame(
        {metric: [0.10] for metric in METRICS}, index=["RB"]
    )
    weights = {"RB": {
        "snap_pct": 0.5,
        "opp_share": 0.0,
        "passdown_share": 0.0,
        "redzone_share": 0.5,
    }}

    scored = score_weeks(changes, typical, weights)

    # Only snap share is left, so its weight is rescaled to 1.
    assert scored["weekly_volatility"].iloc[1] == pytest.approx(1.0)


def test_typical_change_needs_baseline_seasons():
    changes = weekly_changes(
        player_games("a", [1, 2, 3], [0.5, 0.6, 0.5])
    )

    with pytest.raises(ValueError):
        typical_weekly_change(changes, seasons=(2022,))


@pytest.mark.parametrize(
    ("weeks_so_far", "expected"),
    [(2, (3, 1)), (4, (3, 1)), (10, (6, 4)), (17, (10, 8)), (18, (10, 8))],
)
def test_eligibility_minimums_scale_with_the_season(weeks_so_far, expected):
    assert eligibility_minimums("RB", weeks_so_far) == expected


def test_consistency_eligibility():
    snaps = [0.60, 0.65] * 9
    data = pd.concat([
        player_games("starter", list(range(1, 11)), snaps[:10]),
        player_games("short", list(range(1, 10)), snaps[:9]),
        player_games("backup", list(range(1, 11)), snaps[:10],
                     opportunities=4),
        # Plays through Week 18, so the full-season minimum applies.
        player_games("full", list(range(1, 19)), snaps, team="BBB"),
    ])

    _, seasons, _ = calculate_volatility(data, baseline_seasons=None)
    eligible = seasons.set_index("gsis_id")["consistency_eligible"]

    assert eligible.to_dict() == {
        "backup": False,  # under 10 carries + targets per game
        "full": True,
        "short": False,  # 9 games, under the 10 required
        "starter": True,
    }


def test_early_season_uses_scaled_minimum():
    snaps = [0.60, 0.65, 0.60, 0.65]
    data = player_games("starter", [1, 2, 3, 4], snaps)

    _, seasons, _ = calculate_volatility(data, baseline_seasons=None)

    # Through Week 4: 3 games and 1 comparison are enough.
    assert bool(seasons["consistency_eligible"].iloc[0])
