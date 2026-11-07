import numpy as np
import pandas as pd
import pytest

from src.volatility import (
    MAX_WEEKLY_CHANGE,
    METRICS,
    MIN_GAMES_SHARE,
    MIN_RANKED_GAMES_SHARE,
    PART_COLUMNS,
    QUALITY_WEIGHTS,
    RELATIVE_FLOOR,
    calculate_volatility,
    consistency_quality,
    score_weeks,
    season_scores,
    typical_weekly_change,
    weekly_changes,
)


def player_games(player_id, weeks, snaps, position="RB", team="AAA",
                 season=2025, opportunities=12, opp_share=0.20):
    """One row per game for a test player."""
    return pd.DataFrame({
        "season": season,
        "week": weeks,
        "gsis_id": player_id,
        "player_name": f"Player {player_id}",
        "position": position,
        "team": team,
        "snap_pct": snaps,
        "opp_share": opp_share,
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


def test_opportunity_change_is_relative_to_the_players_share():
    games = player_games(
        "a", [1, 2, 3, 4], [0.5] * 4, opp_share=[0.20, 0.30, 0.0, 0.02]
    )

    change = weekly_changes(games)["opportunity_change"].tolist()

    # 20% -> 30%: 10 points out of a 25% average share.
    assert change[1] == pytest.approx(0.10 / 0.25)
    # 30% -> 0%: the largest possible relative change.
    assert change[2] == pytest.approx(2.0)
    # 0% -> 2%: divided by the floor, not by the 1% average.
    assert change[3] == pytest.approx(0.02 / RELATIVE_FLOOR)


def test_parts_add_up_to_the_weekly_score():
    data = pd.concat([
        player_games("a", [1, 2, 3], [0.5, 0.7, 0.6],
                     opp_share=[0.20, 0.10, 0.25]),
        player_games("b", [1, 2, 3], [0.6, 0.5, 0.6],
                     opp_share=[0.30, 0.20, 0.30], team="BBB"),
    ])
    typical = pd.DataFrame(
        {"snap_pct": [0.10], "opp_share": [0.40],
         "passdown_share": [np.nan], "redzone_share": [np.nan]},
        index=["RB"],
    )
    weights = {"RB": {
        "snap_pct": 0.7,
        "opp_share": 0.3,
        "passdown_share": 0.0,
        "redzone_share": 0.0,
    }}

    scored = score_weeks(weekly_changes(data), typical, weights)
    week_2 = scored.iloc[1]

    # Snap: 20 points / 10 typical * 0.7. Opportunity: 10 points out
    # of a 15% average share, / 40% typical * 0.3.
    assert week_2["snap_part"] == pytest.approx(0.7 * 0.20 / 0.10)
    assert week_2["opportunity_part"] == pytest.approx(
        0.3 * (0.10 / 0.15) / 0.40
    )
    parts = scored[list(PART_COLUMNS.values())].sum(axis=1, min_count=1)
    assert np.allclose(
        parts.dropna(), scored["weekly_volatility"].dropna()
    )


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


def test_games_share_counts_team_games_not_weeks():
    # The team plays weeks 1-3 and 5-6 (bye in week 4): 5 games.
    team = player_games("team", [1, 2, 3, 5, 6], [0.6, 0.7] * 2 + [0.6])
    played_3 = player_games("three", [1, 2, 3], [0.6, 0.7, 0.6])
    played_2 = player_games("two", [1, 2], [0.6, 0.7])

    weekly, _, _ = calculate_volatility(
        pd.concat([team, played_3, played_2]), baseline_seasons=None
    )
    window = season_scores(weekly, first_week=1, last_week=6)
    by_player = window.set_index("gsis_id")

    assert by_player["team_games"].tolist() == [5, 5, 5]
    assert by_player.loc["three", "games_share"] == pytest.approx(0.6)
    # 3 of 5 games clears 55%; 2 of 5 doesn't.
    assert MIN_GAMES_SHARE == pytest.approx(0.55)
    assert bool(by_player.loc["three", "consistency_eligible"])
    assert not bool(by_player.loc["two", "consistency_eligible"])
    # Both clear the 25% needed for the volatility rankings.
    assert MIN_RANKED_GAMES_SHARE == pytest.approx(0.25)
    assert by_player["volatility_ranked"].all()


def test_volatility_rankings_need_a_role_and_some_games():
    data = pd.concat([
        player_games("team", list(range(1, 17)), [0.6, 0.7] * 8),
        # 4 of 16 games is exactly 25%.
        player_games("four", [1, 2, 3, 4], [0.6, 0.7] * 2),
        player_games("three", [1, 2, 3], [0.6, 0.7, 0.6]),
        player_games("part_time", list(range(1, 17)), [0.10, 0.15] * 8),
    ])

    weekly, _, _ = calculate_volatility(data, baseline_seasons=None)
    ranked = season_scores(weekly).set_index("gsis_id")["volatility_ranked"]

    assert ranked.to_dict() == {
        "four": True,
        "part_time": False,  # under 20% of snaps
        "team": True,
        "three": False,  # 3 of 16 games
    }


def test_consistency_eligibility():
    snaps = [0.60, 0.65] * 9
    data = pd.concat([
        # The team plays all 18 weeks.
        player_games("full", list(range(1, 19)), snaps),
        # 10 of 18 games (56%) clears 55%; 9 of 18 (50%) doesn't.
        player_games("starter", list(range(1, 11)), snaps[:10]),
        player_games("short", list(range(1, 10)), snaps[:9]),
        player_games("backup", list(range(1, 11)), snaps[:10],
                     opportunities=4),
        player_games("low_snaps", list(range(1, 19)), [0.45] * 18),
    ])

    _, seasons, _ = calculate_volatility(data, baseline_seasons=None)
    eligible = seasons.set_index("gsis_id")["consistency_eligible"]

    assert eligible.to_dict() == {
        "backup": False,  # under 10 carries + targets per game
        "full": True,
        "low_snaps": False,  # under the 50% RB snap minimum
        "short": False,  # 9 of 18 games
        "starter": True,
    }


def test_short_range_uses_the_same_share():
    snaps = [0.60, 0.65, 0.60, 0.65]
    data = pd.concat([
        player_games("starter", [1, 2, 3, 4], snaps),
        player_games("missed_two", [1, 2], snaps[:2]),
    ])

    _, seasons, _ = calculate_volatility(data, baseline_seasons=None)
    eligible = seasons.set_index("gsis_id")["consistency_eligible"]

    # Through Week 4: 4 of 4 games is eligible, 2 of 4 (50%) isn't.
    assert eligible.to_dict() == {"missed_two": False, "starter": True}


def test_quality_score_adds_up_weighted_percentiles():
    players = pd.DataFrame({
        "volatility_score": [0.8, 1.0, 1.2, 1.4],
        "avg_snap_pct": [0.60, 0.90, 0.70, 0.80],
        "avg_opportunities_per_week": [12, 20, 16, 14],
    })

    scored = consistency_quality(players)

    # The steadiest player gets 100, like the largest role does.
    assert scored["steadiness_pctl"].tolist() == [100, 75, 50, 25]
    assert scored["snap_pctl"].tolist() == [25, 100, 50, 75]
    assert scored["opportunity_pctl"].tolist() == [25, 100, 75, 50]
    expected = (
        QUALITY_WEIGHTS["low_volatility"] * scored["steadiness_pctl"]
        + QUALITY_WEIGHTS["snap_pct"] * scored["snap_pctl"]
        + QUALITY_WEIGHTS["opportunities"] * scored["opportunity_pctl"]
    )
    assert np.allclose(scored["quality_score"], expected)
    # 0.50 * 75 + 0.25 * 100 + 0.25 * 100
    assert scored["quality_score"].iloc[1] == pytest.approx(87.5)


def test_previous_week_is_recorded():
    # Bye in week 4: week 5 is compared with week 3.
    changes = weekly_changes(
        player_games("a", [1, 2, 3, 5], [0.5, 0.6, 0.5, 0.6])
    )

    assert changes["previous_week"].tolist()[1:] == [1, 2, 3]
    assert np.isnan(changes["previous_week"].iloc[0])


def test_week_window_uses_changes_inside_it():
    # Snap changes: week 2 +10 pts, week 3 +30 pts, week 4 -30 pts.
    data = pd.concat([
        player_games("a", [1, 2, 3, 4], [0.40, 0.50, 0.80, 0.50]),
        player_games("b", [1, 2, 3, 4], [0.50, 0.60, 0.50, 0.60],
                     team="BBB"),
    ])
    weekly, full, _ = calculate_volatility(data, baseline_seasons=None)

    window = season_scores(weekly, first_week=2, last_week=3)
    player = window.set_index("gsis_id").loc["a"]

    # Only the Week 2 -> Week 3 change counts.
    assert player["weekly_comparisons"] == 1
    assert player["weeks"] == 2
    week_3 = weekly.set_index(["gsis_id", "week"]).loc[
        ("a", 3), "weekly_volatility"
    ]
    assert player["volatility_score"] == pytest.approx(week_3)

    # The whole season averages all three changes.
    assert full.set_index("gsis_id").loc["a", "weekly_comparisons"] == 3
