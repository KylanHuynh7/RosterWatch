"""Week-to-week usage volatility model for RosterWatch.

Each player-week is scored by how much the player's usage changed since
his previous game. Every change is divided by the typical weekly change
for that metric and position, so a score of 1.0 means a normal week and
2.0 means twice the usual swing. Because of that scaling, each metric's
share of the score matches its weight.

All weights and thresholds live at the top of this module so the
Streamlit app, tests and notebooks share one set of values. The reason
for each number is noted next to it.
"""

from math import ceil

import numpy as np
import pandas as pd

POSITIONS = ["RB", "WR", "TE"]

METRICS = [
    "snap_pct",
    "opp_share",
    "passdown_share",
    "redzone_share",
]

# A week needs this change to be scored. Any other metric that is
# missing that week (red-zone share is missing when the team ran no
# red-zone plays, about 2% of rows) is skipped and the remaining
# weights are rescaled, instead of dropping the week.
REQUIRED_METRICS = ["snap_pct"]

# Share of the weekly score each metric controls: snap share only.
# Tuned on 2022-24 and checked on 2025, a player's average volatility
# so far best warned of a role change in his next 3 games when built
# from snap share alone (AUC 0.63 / 0.57). Every blend tested did
# worse, and more weight on the other metrics lowered it further (the
# previous weights: 0.55 / 0.53). The other metrics' changes are still
# calculated and shown; give them weight here to blend them in.
POSITION_WEIGHTS = {
    position: {
        "snap_pct": 1.0,
        "opp_share": 0.0,
        "passdown_share": 0.0,
        "redzone_share": 0.0,
    }
    for position in POSITIONS
}

# Seasons used to measure each position's typical weekly change.
# 2025 is left out so it can be used to check the model.
BASELINE_SEASONS = (2022, 2023, 2024)

# Players averaging under 20% of snaps in a season are left out of the
# typical-change baseline and the volatility rankings. They are 19% of
# player-games but score 4% of PPR fantasy points.
MIN_ROLE_SNAP_PCT = 0.20

# Players need this many scored weeks to appear in the volatility
# rankings. With fewer, the average is mostly one game (in 2025 the
# top WR on the unfiltered list had played 2 games). Matches the
# 4-game window used elsewhere in the project.
MIN_RANKED_COMPARISONS = 4

# One game's change in any metric counts at most 40 percentage points,
# so a single injury exit can't dominate an average built on a few
# games. About 4% of snap-share changes are larger. Full-season results
# barely move with or without it (role-change AUC 0.630 either way).
MAX_WEEKLY_CHANGE = 0.40

# Minimum usage for the Most Consistent leaderboard. "min_weeks" is the
# full-season requirement; see eligibility_minimums for earlier weeks.
CONSISTENCY_THRESHOLDS = {
    "RB": {
        "min_weeks": 10,
        "min_snap_pct": 0.50,
        "min_opportunities_per_week": 10,
    },
    "WR": {
        "min_weeks": 10,
        "min_snap_pct": 0.65,
        "min_opportunities_per_week": 5,
    },
    "TE": {
        "min_weeks": 10,
        "min_snap_pct": 0.60,
        "min_opportunities_per_week": 4,
    },
}

# Before a season is far enough along for 10 games, require 60% of the
# weeks played so far, and never fewer than 3 games.
MIN_WEEKS_SHARE = 0.60
MIN_WEEKS_FLOOR = 3

# Valid weekly comparisons required = minimum weeks minus 2 (the first
# game has nothing to compare with, and one missed game is allowed).
COMPARISON_GAP = 2

PLAYER_SEASON = ["season", "gsis_id"]

CHANGE_COLUMNS = {
    "snap_pct": "snap_change",
    "opp_share": "opportunity_change",
    "passdown_share": "passdown_change",
    "redzone_share": "redzone_change",
}

WEEKLY_COLUMNS = [
    "season",
    "week",
    "gsis_id",
    "player_name",
    "position",
    "team",
    "weekly_volatility",
    *CHANGE_COLUMNS.values(),
    *METRICS,
    "targets",
    "carries",
    "opportunities",
]

SCORE_COLUMNS = [
    "season",
    "gsis_id",
    "player_name",
    "position",
    "team",
    "weeks",
    "weekly_comparisons",
    "volatility_score",
    "avg_snap_pct",
    "avg_opportunities_per_week",
    "consistency_eligible",
]


def eligibility_minimums(position, weeks_so_far):
    """Return the games and comparisons a player needs to be ranked.

    Args:
        position: "RB", "WR" or "TE".
        weeks_so_far: last NFL week included in the data for the season.

    Returns:
        (min_weeks, min_comparisons). Through Week 18 this is the
        full-season requirement (10 and 8).
    """
    full_season = CONSISTENCY_THRESHOLDS[position]["min_weeks"]
    scaled = ceil(MIN_WEEKS_SHARE * weeks_so_far)
    min_weeks = min(full_season, max(MIN_WEEKS_FLOOR, scaled))
    return min_weeks, max(1, min_weeks - COMPARISON_GAP)


def weekly_changes(data):
    """Measure each player's change in usage since his previous game.

    A change is measured only when the player didn't miss one of his
    team's games in between: a bye week doesn't break the comparison,
    a missed game does. Each change is capped at MAX_WEEKLY_CHANGE.

    Args:
        data: one row per player-game, with season, week, gsis_id,
            player_name, position, team, targets, carries and METRICS.

    Returns:
        One row per player-week with the usage columns, "opportunities"
        (targets + carries) and one change column per metric (see
        CHANGE_COLUMNS). Changes are NaN when there is no comparable
        previous game. Player name, position and team are the player's
        last values that season.
    """
    weekly = (
        data.sort_values(PLAYER_SEASON + ["week"])
        .groupby(PLAYER_SEASON + ["week"], as_index=False)
        .agg(
            player_name=("player_name", "last"),
            position=("position", "last"),
            game_team=("team", "last"),
            targets=("targets", "mean"),
            carries=("carries", "mean"),
            **{metric: (metric, "mean") for metric in METRICS},
        )
    )

    # The team's previous game, taken from every player who played it.
    schedule = (
        weekly[["season", "game_team", "week"]]
        .drop_duplicates()
        .sort_values(["season", "game_team", "week"])
    )
    schedule["team_previous_game"] = (
        schedule.groupby(["season", "game_team"])["week"].shift()
    )
    weekly = weekly.merge(
        schedule,
        on=["season", "game_team", "week"],
        how="left",
    ).sort_values(PLAYER_SEASON + ["week"], ignore_index=True)

    players = weekly.groupby(PLAYER_SEASON)
    previous_game = players["week"].shift()

    # ">=" rather than "==" so a player traded during his new team's
    # bye still gets compared with his last game for the old team.
    comparable = previous_game.notna() & (
        previous_game >= weekly["team_previous_game"]
    )

    changes = (
        players[METRICS].diff().abs().clip(upper=MAX_WEEKLY_CHANGE)
    )
    changes = changes.where(comparable)

    for metric, column in CHANGE_COLUMNS.items():
        weekly[column] = changes[metric]

    weekly["opportunities"] = (
        weekly["targets"].fillna(0) + weekly["carries"].fillna(0)
    )
    for column in ["player_name", "position"]:
        weekly[column] = players[column].transform("last")
    weekly["team"] = players["game_team"].transform("last")

    return weekly.drop(columns=["game_team", "team_previous_game"])


def typical_weekly_change(changes, seasons=BASELINE_SEASONS):
    """Average weekly change per metric for each position.

    Uses players averaging at least MIN_ROLE_SNAP_PCT of snaps that
    season, in the given seasons only.

    Args:
        changes: output of weekly_changes.
        seasons: seasons to measure; None uses every season in changes.

    Returns:
        DataFrame indexed by position with one column per metric, in
        share units (0.12 = 12 percentage points). A metric that never
        changed is NaN, so score_weeks skips it.

    Raises:
        ValueError: if a position has no typical change for one of
            the REQUIRED_METRICS.
    """
    role = changes.groupby(PLAYER_SEASON)["snap_pct"].transform("mean")
    sample = changes.loc[role >= MIN_ROLE_SNAP_PCT]
    if seasons is not None:
        sample = sample.loc[sample["season"].isin(seasons)]

    typical = (
        sample.groupby("position")[list(CHANGE_COLUMNS.values())]
        .mean()
        .rename(columns={v: k for k, v in CHANGE_COLUMNS.items()})
        .reindex(changes["position"].unique())
    )

    typical = typical.where(typical > 0)

    if typical[REQUIRED_METRICS].isna().any().any():
        raise ValueError(
            "Not enough weekly comparisons to measure a typical "
            f"change for every position in seasons {seasons}."
        )

    return typical


def score_weeks(changes, typical, weights=POSITION_WEIGHTS):
    """Add a "weekly_volatility" column to the output of weekly_changes.

    Each change is divided by the typical change for that metric and
    position, then combined with the position's weights. Weights are
    rescaled to sum to 1 over the metrics available that week.

    Args:
        changes: output of weekly_changes.
        typical: output of typical_weekly_change.
        weights: {position: {metric: weight}}.

    Returns:
        A copy of changes with "weekly_volatility" (1.0 = a typical
        week; NaN when a required change is missing).
    """
    scored = changes.copy()
    change_columns = [CHANGE_COLUMNS[m] for m in METRICS]
    positions = scored["position"]

    relative = (
        scored[change_columns].to_numpy()
        / typical.loc[positions, METRICS].to_numpy()
    )
    metric_weights = (
        pd.DataFrame(weights).T.loc[positions, METRICS].to_numpy()
    )
    available = ~np.isnan(relative)

    weighted_sum = np.where(available, relative * metric_weights, 0).sum(1)
    weight_used = np.where(available, metric_weights, 0).sum(1)

    required = scored[[CHANGE_COLUMNS[m] for m in REQUIRED_METRICS]]
    has_required = required.notna().all(axis=1).to_numpy()

    with np.errstate(invalid="ignore", divide="ignore"):
        score = weighted_sum / weight_used
    scored["weekly_volatility"] = np.where(has_required, score, np.nan)

    return scored


def season_scores(weekly):
    """Average weekly volatility and leaderboard eligibility per season.

    Args:
        weekly: output of score_weeks.

    Returns:
        One row per player-season with at least one scored week, with
        the columns in SCORE_COLUMNS. "consistency_eligible" applies
        CONSISTENCY_THRESHOLDS and eligibility_minimums.
    """
    valid = weekly.dropna(subset=["weekly_volatility"])
    if valid.empty:
        return pd.DataFrame(columns=SCORE_COLUMNS)

    seasons = weekly.groupby(PLAYER_SEASON).agg(
        player_name=("player_name", "last"),
        position=("position", "last"),
        team=("team", "last"),
        weeks=("week", "size"),
        avg_snap_pct=("snap_pct", "mean"),
        avg_opportunities_per_week=("opportunities", "mean"),
    )
    scored = valid.groupby(PLAYER_SEASON)["weekly_volatility"]
    seasons["weekly_comparisons"] = scored.size()
    seasons["volatility_score"] = scored.mean()
    seasons = seasons.dropna(subset=["volatility_score"]).reset_index()
    seasons["weekly_comparisons"] = (
        seasons["weekly_comparisons"].astype(int)
    )

    weeks_so_far = seasons["season"].map(
        weekly.groupby("season")["week"].max()
    )
    minimums = [
        eligibility_minimums(position, last_week)
        for position, last_week in zip(seasons["position"], weeks_so_far)
    ]
    min_weeks = np.array([m[0] for m in minimums])
    min_comparisons = np.array([m[1] for m in minimums])

    thresholds = pd.DataFrame(CONSISTENCY_THRESHOLDS).T.loc[
        seasons["position"]
    ]
    seasons["consistency_eligible"] = (
        (seasons["weeks"].to_numpy() >= min_weeks)
        & (seasons["weekly_comparisons"].to_numpy() >= min_comparisons)
        & (
            seasons["avg_snap_pct"].to_numpy()
            >= thresholds["min_snap_pct"].to_numpy()
        )
        & (
            seasons["avg_opportunities_per_week"].to_numpy()
            >= thresholds["min_opportunities_per_week"].to_numpy()
        )
    )

    return seasons[SCORE_COLUMNS]


def calculate_volatility(
    data,
    weights=POSITION_WEIGHTS,
    baseline_seasons=BASELINE_SEASONS,
):
    """Run the full volatility model.

    Args:
        data: one row per player-game (see weekly_changes).
        weights: {position: {metric: weight}}.
        baseline_seasons: seasons used for the typical weekly change;
            None uses every season in data.

    Returns:
        (weekly_scores, season_scores, typical_change):
        weekly_scores has WEEKLY_COLUMNS, one row per player-week;
        season_scores has SCORE_COLUMNS, one row per player-season;
        typical_change is the output of typical_weekly_change.
    """
    data = data.loc[data["position"].isin(list(weights))]
    changes = weekly_changes(data)
    typical = typical_weekly_change(changes, baseline_seasons)
    weekly = score_weeks(changes, typical, weights)

    return weekly[WEEKLY_COLUMNS], season_scores(weekly), typical
