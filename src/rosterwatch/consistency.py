"""Player Consistency Model: weekly usage volatility and the consistency rankings."""

import numpy as np
import pandas as pd

from .settings import (
    CONSISTENCY_THRESHOLDS,
    MIN_COMPARISONS,
    POSITION_WEIGHTS,
    VOLATILITY_METRICS,
)


# ============================================================
# MODEL 1 — WEEKLY VOLATILITY
# ============================================================
def calculate_volatility(data):
    # Vectorized across all players at once (same results as a per-player loop,
    # about 50x faster). Row order: players in order of first appearance, each
    # player's rows sorted by week.
    if data.empty:
        return data.copy(), pd.DataFrame()
    group_keys = ["season", "gsis_id"]
    group_id = data.groupby(group_keys, sort=False).ngroup()
    keep = group_id.notna() & (group_id >= 0)
    weekly = (
        data.loc[keep]
        .assign(_player_group=group_id[keep].astype("int64"))
        .sort_values(["_player_group", "week"], kind="stable")
    )
    by_player = weekly.groupby("_player_group", sort=False)
    metrics = weekly[VOLATILITY_METRICS]
    changes = by_player[VOLATILITY_METRICS].diff().abs()
    valid = (
        by_player["week"].diff().eq(1)
        & metrics.notna().all(axis=1)
        & by_player[VOLATILITY_METRICS].shift().notna().all(axis=1)
    )

    # Each player's weights come from their position in their latest week.
    last_position = weekly.drop_duplicates("_player_group", keep="last").set_index(
        "_player_group"
    )["position"]
    position = weekly["_player_group"].map(last_position)
    weight_rows = np.array([POSITION_WEIGHTS[p] for p in position], dtype=float)
    weights = pd.DataFrame(
        weight_rows.reshape(len(weekly), len(VOLATILITY_METRICS)),
        index=weekly.index,
        columns=VOLATILITY_METRICS,
    )
    weekly["weekly_volatility"] = (
        changes.mul(weights).sum(axis=1, min_count=4).where(valid)
    )
    weekly = weekly.drop(columns="_player_group").reset_index(drop=True)
    summary = summarize_model1_window(weekly)
    return weekly, summary


# ============================================================
# MODEL 1 — CONSISTENCY SUMMARY
# ============================================================
def summarize_model1_window(weekly):
    if weekly.empty:
        return pd.DataFrame()
    summary = weekly.groupby(
        ["season", "gsis_id"],
        as_index=False,
    ).agg(
        player_name=("player_name", "last"),
        position=("position", "last"),
        team=("team", "last"),
        weeks=("week", "nunique"),
        weekly_comparisons=(
            "weekly_volatility",
            "count",
        ),
        volatility_score=(
            "weekly_volatility",
            "mean",
        ),
        avg_snap_pct=(
            "snap_pct",
            "mean",
        ),
        avg_opportunities_per_week=(
            "opportunities",
            "mean",
        ),
    )

    def eligible(row):
        min_weeks, min_snap, min_volume = CONSISTENCY_THRESHOLDS[row["position"]]
        return bool(
            row["weeks"] >= min_weeks
            and row["weekly_comparisons"] >= MIN_COMPARISONS
            and row["avg_snap_pct"] >= min_snap
            and row["avg_opportunities_per_week"] >= min_volume
        )

    summary["consistency_eligible"] = summary.apply(eligible, axis=1)
    summary["consistency_quality_score"] = np.nan
    eligible_summary = summary.loc[summary["consistency_eligible"]]
    for _, group in eligible_summary.groupby(["season", "position"]):
        volatility_rank = group["volatility_score"].rank(pct=True)
        snap_rank = group["avg_snap_pct"].rank(pct=True)
        opportunity_rank = group["avg_opportunities_per_week"].rank(pct=True)
        quality = 100 * (
            0.50 * (1 - volatility_rank) + 0.25 * snap_rank + 0.25 * opportunity_rank
        )
        summary.loc[
            group.index,
            "consistency_quality_score",
        ] = quality
    return summary
