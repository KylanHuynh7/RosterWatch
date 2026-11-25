"""Backtest Waiver Wire Alerts on 2022-2025 (the table in README section 11).

Every player is scored as of each of Weeks 5-15, using only games up to that
week. A flag "hits" if the player then averages starter-level fantasy points
over his next three games: at least the per-game average of the RB24, WR36 or
TE12 in 2022-2024 (nflverse standard scoring). The "waiver pool" columns keep
only players scoring under 75% of that level so far, a stand-in for players
who are probably available. The ESPN <=50% rostered filter is not applied:
ESPN gives one value per season, which already reflects later pickups and
would leak future information into past weeks.

Run from the repository root (takes about 30 seconds):

    python analysis/backtest_waiver.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rosterwatch.data import load_data  # noqa: E402
from rosterwatch.settings import WAIVER_FLAGS  # noqa: E402
from rosterwatch.waiver import calculate_waiver_model  # noqa: E402

DATA = ROOT / "notebooks" / "weekly_usage.parquet"
SEASONS = [2022, 2023, 2024, 2025]
WEEKS = range(5, 16)
STARTER_RANK = {"RB": 24, "WR": 36, "TE": 12}
POOL_SHARE = 0.75


def starter_levels(usage):
    """Per-game points of the RB24 / WR36 / TE12 (players with 8+ games), averaged over 2022-2024."""
    seasons = usage.loc[usage["season"] <= 2024]
    per_player = seasons.groupby(["season", "position", "gsis_id"]).agg(
        games=("week", "size"), points=("points", "mean")
    )
    per_player = per_player.loc[per_player["games"] >= 8]
    levels = {}
    for position, rank in STARTER_RANK.items():
        by_season = per_player.xs(position, level="position").groupby(level="season")
        levels[position] = (
            by_season["points"].apply(lambda s: s.nlargest(rank).iloc[-1]).mean()
        )
    return levels


def score_all_weeks(usage, levels):
    """Score every player as of each analysis week and attach what happened next."""
    rows = []
    for season in SEASONS:
        season_rows = usage.loc[usage["season"] == season]
        for week in WEEKS:
            flags = calculate_waiver_model(usage, season, week, ())
            so_far = (
                season_rows.loc[season_rows["week"] <= week]
                .groupby("gsis_id")["points"]
                .mean()
            )
            next_three = (
                season_rows.loc[season_rows["week"] > week]
                .sort_values("week")
                .groupby("gsis_id")
                .head(3)
                .groupby("gsis_id")["points"]
                .agg(["mean", "size"])
            )
            flags = flags.merge(
                so_far.rename("points_so_far"), on="gsis_id", how="left"
            )
            flags = flags.merge(next_three, on="gsis_id", how="left")
            flags = flags.loc[flags["size"] == 3].assign(season=season, week=week)
            rows.append(flags)
    result = pd.concat(rows, ignore_index=True)
    result["level"] = result["position"].map(levels)
    result["hit"] = result["mean"] >= result["level"]
    result["pool"] = result["points_so_far"] < POOL_SHARE * result["level"]
    return result


def summary_table(result, n_weeks):
    """Flags per week and hit rate by flag, for all players and for the waiver pool."""
    table = []
    for label, rows in [
        *[(f, result.loc[result["waiver_flag"] == f]) for f in WAIVER_FLAGS],
        ("Any flag (base rate)", result),
    ]:
        pool = rows.loc[rows["pool"]]
        table.append(
            {
                "Flag": label,
                "All: flags/week": round(len(rows) / n_weeks, 1),
                "All: hit rate": f"{rows['hit'].mean():.0%}",
                "Pool: flags/week": round(len(pool) / n_weeks, 1),
                "Pool: hit rate": f"{pool['hit'].mean():.0%}",
            }
        )
    return pd.DataFrame(table)


def main():
    usage = load_data(str(DATA))
    usage["points"] = usage["fantasy_points"].fillna(0)
    levels = starter_levels(usage)
    result = score_all_weeks(usage, levels)
    n_weeks = result[["season", "week"]].drop_duplicates().shape[0]

    print(
        "Starter level (points per game): "
        + ", ".join(f"{p} {v:.1f}" for p, v in levels.items())
    )
    print(
        f"Analysis weeks: {n_weeks}; player-weeks with three later games: {len(result):,}\n"
    )
    print(summary_table(result, n_weeks).to_string(index=False))

    must_pick = result.loc[result["waiver_flag"] == "Must Pick Up"]
    print(
        f"\nMust Pick Up flags on players already at starter level: {(~must_pick['pool']).mean():.0%}"
    )
    pool_pick = must_pick.loc[must_pick["pool"]]
    early, late = (
        pool_pick.loc[pool_pick["season"] <= 2024],
        pool_pick.loc[pool_pick["season"] == 2025],
    )
    print(
        f"Waiver-pool Must Pick Up hit rate: 2022-2024 {early['hit'].mean():.0%} (n={len(early)}), "
        f"2025 {late['hit'].mean():.0%} (n={len(late)})"
    )


if __name__ == "__main__":
    main()
