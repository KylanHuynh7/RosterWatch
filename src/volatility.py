"""Week-to-week usage volatility model for RosterWatch.

Each player-week is scored by how much the player's usage changed since
his previous game:

    V = 0.70 * snap score + 0.30 * opportunity score
    score = this week's change / the position's typical weekly change

Snap share change is measured in percentage points. Opportunity share
change is measured relative to the player's own share (see
RELATIVE_METRICS). Dividing each change by the typical weekly change
for the position makes a score of 1.0 a normal week and 2.0 twice the
usual swing, and makes each metric's share of the score match its
weight: the two changes come in different sizes, so without it the
weights wouldn't mean 70/30.

All weights and thresholds live at the top of this module so the
Streamlit app, tests and notebooks share one set of values. The reason
for each number is noted next to it.
"""

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

# Share of the weekly score each metric controls (weights sum to 1).
# Snap share is the strongest single warning of a coming role change.
# Opportunity share, measured relative to the player's own share, adds
# information snap share misses: it warns better of lost touches and
# tracks how bumpy a player's fantasy points are (see the guide for
# the 2022-24 / 2025 checks). Passing-down share added nothing on top,
# and red-zone share is 0 in most games, so both are shown but not
# weighted.
POSITION_WEIGHTS = {
    position: {
        "snap_pct": 0.70,
        "opp_share": 0.30,
        "passdown_share": 0.0,
        "redzone_share": 0.0,
    }
    for position in POSITIONS
}

# Metrics whose change is measured relative to the player's own level:
# |change| / average of the two games. A raw change in opportunity
# share mostly measures workload size (a 25% target share swings more
# points than a 3% one), which made workhorses look volatile.
RELATIVE_METRICS = ["opp_share"]

# Smallest average share a relative change is divided by, so a player
# going from 0% to 1% doesn't count as a 200% swing.
RELATIVE_FLOOR = 0.05

# Seasons used to measure each position's typical weekly change.
# 2025 is left out so it can be used to check the model.
BASELINE_SEASONS = (2022, 2023, 2024)

# Players averaging under 20% of snaps in a season are left out of the
# typical-change baseline and the volatility rankings. They are 19% of
# player-games but score 4% of PPR fantasy points.
MIN_ROLE_SNAP_PCT = 0.20

# Share of his team's games in the selected weeks a player must have
# played to be ranked (a bye isn't a missed game). A share instead of a
# fixed count works the same for any week range, and a bye inside a
# short range no longer knocks players out.
#
# Most Volatile: 25%, about 5 of 17 games for a full season, close to
# the old 4 scored weeks. Keeps players whose role changed partway
# through the season.
MIN_RANKED_GAMES_SHARE = 0.25

# Most Consistent: 55%, the old 10 of 17 games for a full season, and
# 3 of 4 games for a 4-week range. 65% would also work for short ranges
# but needs 12 of 17 games, which in 2025 dropped Bucky Irving (the
# number 2 RB), George Kittle and Alvin Kamara.
MIN_GAMES_SHARE = 0.55

# One game's change in a percentage-point metric counts at most 40
# points, so a single injury exit can't dominate an average built on a
# few games. About 4% of snap-share changes are larger. Full-season
# results barely move with or without it. Relative changes are already
# limited to 200% (going to or from zero).
MAX_WEEKLY_CHANGE = 0.40

# Minimum usage for the Most Consistent leaderboard, averaged over the
# games played in the selected weeks (so neither depends on how many
# weeks are selected).
CONSISTENCY_THRESHOLDS = {
    "RB": {
        "min_snap_pct": 0.50,
        "min_opportunities_per_week": 10,
    },
    "WR": {
        "min_snap_pct": 0.65,
        "min_opportunities_per_week": 5,
    },
    "TE": {
        "min_snap_pct": 0.60,
        "min_opportunities_per_week": 4,
    },
}

# Most Consistent leaderboard score (0-100):
#     Q = 0.50 * steadiness + 0.25 * snaps + 0.25 * touches
# where each part is the player's percentile (0-100) among the eligible
# players at his position. Half rewards low volatility; the rest
# rewards a large role, so a steady backup doesn't outrank a steady
# starter.
QUALITY_WEIGHTS = {
    "low_volatility": 0.50,
    "snap_pct": 0.25,
    "opportunities": 0.25,
}

# Percentile column for each part of the quality score.
QUALITY_COLUMNS = {
    "low_volatility": "steadiness_pctl",
    "snap_pct": "snap_pctl",
    "opportunities": "opportunity_pctl",
}

PLAYER_SEASON = ["season", "gsis_id"]

CHANGE_COLUMNS = {
    "snap_pct": "snap_change",
    "opp_share": "opportunity_change",
    "passdown_share": "passdown_change",
    "redzone_share": "redzone_change",
}

# Each metric's part of the weekly score; the parts add up to it.
PART_COLUMNS = {
    "snap_pct": "snap_part",
    "opp_share": "opportunity_part",
    "passdown_share": "passdown_part",
    "redzone_share": "redzone_part",
}

WEEKLY_COLUMNS = [
    "season",
    "week",
    "previous_week",
    "gsis_id",
    "player_name",
    "position",
    "team",
    "game_team",
    "weekly_volatility",
    *CHANGE_COLUMNS.values(),
    *PART_COLUMNS.values(),
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
    "team_games",
    "games_share",
    "weekly_comparisons",
    "volatility_score",
    "avg_snap_pct",
    "avg_opportunities_per_week",
    "volatility_ranked",
    "consistency_eligible",
]


def weekly_changes(data):
    """Measure each player's change in usage since his previous game.

    A change is measured only when the player didn't miss one of his
    team's games in between: a bye week doesn't break the comparison,
    a missed game does. Changes are in share units (0.12 = 12
    percentage points) and capped at MAX_WEEKLY_CHANGE, except for
    RELATIVE_METRICS, where the change is divided by the average of the
    two games (0.40 = 40% of the player's share).

    Args:
        data: one row per player-game, with season, week, gsis_id,
            player_name, position, team, targets, carries and METRICS.

    Returns:
        One row per player-week with the usage columns, "opportunities"
        (targets + carries), "previous_week" (the game each change is
        measured from) and one change column per metric (see
        CHANGE_COLUMNS). Changes and previous_week are NaN when there is
        no comparable previous game. Player name, position and team are
        the player's last values that season; "game_team" is the team
        he played for that week.
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

    differences = players[METRICS].diff().abs()
    changes = differences.clip(upper=MAX_WEEKLY_CHANGE)
    for metric in RELATIVE_METRICS:
        average = (weekly[metric] + players[metric].shift()) / 2
        changes[metric] = (
            differences[metric] / average.clip(lower=RELATIVE_FLOOR)
        )
    changes = changes.where(comparable)

    for metric, column in CHANGE_COLUMNS.items():
        weekly[column] = changes[metric]
    weekly["previous_week"] = previous_game.where(comparable)

    weekly["opportunities"] = (
        weekly["targets"].fillna(0) + weekly["carries"].fillna(0)
    )
    for column in ["player_name", "position"]:
        weekly[column] = players[column].transform("last")
    weekly["team"] = players["game_team"].transform("last")

    return weekly.drop(columns=["team_previous_game"])


def typical_weekly_change(changes, seasons=BASELINE_SEASONS):
    """Average weekly change per metric for each position.

    Uses players averaging at least MIN_ROLE_SNAP_PCT of snaps that
    season, in the given seasons only.

    Args:
        changes: output of weekly_changes.
        seasons: seasons to measure; None uses every season in changes.

    Returns:
        DataFrame indexed by position with one column per metric, in
        the units of weekly_changes. A metric that never changed is
        NaN, so score_weeks skips it.

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
    """Score each week: a weighted average of the scaled changes.

    Each change is divided by the typical change for that metric and
    position, then combined with the position's weights. Weights are
    rescaled to sum to 1 over the metrics available that week.

    Args:
        changes: output of weekly_changes.
        typical: output of typical_weekly_change.
        weights: {position: {metric: weight}}.

    Returns:
        A copy of changes with "weekly_volatility" (1.0 = a typical
        week; NaN when a required change is missing) and each metric's
        part of it (PART_COLUMNS), which add up to the weekly score.
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
        parts = relative * metric_weights / weight_used[:, None]
    scored["weekly_volatility"] = np.where(has_required, score, np.nan)
    parts[~has_required] = np.nan
    for index, metric in enumerate(METRICS):
        scored[PART_COLUMNS[metric]] = parts[:, index]

    return scored


def season_scores(weekly, first_week=1, last_week=None):
    """Average weekly volatility and leaderboard eligibility per season.

    Args:
        weekly: output of score_weeks.
        first_week: first NFL week of the window.
        last_week: last NFL week of the window; None runs to the end of
            each season. Only games inside the window count, and only
            changes between two games inside it: Weeks 2-3 is the
            change from Week 2 to Week 3.

    Returns:
        One row per player-season with at least one scored week in the
        window, with the columns in SCORE_COLUMNS:
        "team_games" is the number of games his team (his last team that
        season) played in the window and "games_share" the share of
        them he played. "volatility_ranked" marks players in the
        volatility rankings (MIN_ROLE_SNAP_PCT and
        MIN_RANKED_GAMES_SHARE); "consistency_eligible" marks ranked
        players who also meet MIN_GAMES_SHARE and
        CONSISTENCY_THRESHOLDS.
    """
    season_end = weekly.groupby("season")["week"].max()
    if last_week is not None:
        season_end = season_end.clip(upper=last_week)

    weekly = weekly.loc[
        (weekly["week"] >= first_week)
        & (weekly["week"] <= weekly["season"].map(season_end))
    ]
    valid = weekly.loc[
        weekly["weekly_volatility"].notna()
        & (weekly["previous_week"] >= first_week)
    ]
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

    # Games each team played in the window, from every player who
    # played them, so a bye week doesn't count as a missed game.
    team_games = (
        weekly[["season", "game_team", "week"]]
        .drop_duplicates()
        .groupby(["season", "game_team"])
        .size()
    )
    seasons["team_games"] = pd.MultiIndex.from_frame(
        seasons[["season", "team"]]
    ).map(team_games.get).astype(float)
    seasons["games_share"] = seasons["weeks"] / seasons["team_games"]

    seasons["volatility_ranked"] = (
        (seasons["avg_snap_pct"] >= MIN_ROLE_SNAP_PCT)
        & (seasons["games_share"] >= MIN_RANKED_GAMES_SHARE)
    )

    thresholds = pd.DataFrame(CONSISTENCY_THRESHOLDS).T.loc[
        seasons["position"]
    ]
    seasons["consistency_eligible"] = (
        seasons["volatility_ranked"].to_numpy()
        & (seasons["games_share"].to_numpy() >= MIN_GAMES_SHARE)
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


def consistency_quality(players):
    """Quality score for the Most Consistent leaderboard.

    Args:
        players: rows of season_scores to compare with each other,
            usually the consistency-eligible players at one position.

    Returns:
        A copy with each player's percentile (0-100) among these
        players for low volatility, average snap share, and carries +
        targets per game (see QUALITY_COLUMNS), and "quality_score",
        their weighted sum using QUALITY_WEIGHTS. The steadiest player
        and the player with the most snaps or touches each get 100.
    """
    scored = players.copy()
    percentiles = {
        "low_volatility": scored["volatility_score"].rank(
            ascending=False, pct=True
        ),
        "snap_pct": scored["avg_snap_pct"].rank(pct=True),
        "opportunities": scored["avg_opportunities_per_week"].rank(
            pct=True
        ),
    }

    scored["quality_score"] = 0.0
    for part, column in QUALITY_COLUMNS.items():
        scored[column] = percentiles[part] * 100
        scored["quality_score"] += QUALITY_WEIGHTS[part] * scored[column]

    return scored


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
