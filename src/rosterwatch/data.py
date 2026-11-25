"""Load and validate the two input tables: weekly usage and ESPN % rostered."""

import numpy as np
import pandas as pd

from .settings import MAX_PERCENT_OWNED, POSITIONS


# ============================================================
# DATA LOADING
# ============================================================
def load_data(path):
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    required = [
        "season",
        "week",
        "gsis_id",
        "player_name",
        "position",
        "team",
        "offense_snaps",
        "snap_pct",
        "targets",
        "carries",
        "opp_share",
        "passdown_share",
        "redzone_share",
    ]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if "fantasy_points_ppr" not in df.columns:
        df["fantasy_points_ppr"] = np.nan
    if "opponent" not in df.columns:
        df["opponent"] = ""
    numeric_columns = [
        "season",
        "week",
        "offense_snaps",
        "snap_pct",
        "targets",
        "carries",
        "opp_share",
        "passdown_share",
        "redzone_share",
        "fantasy_points_ppr",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )
    df["position"] = df["position"].astype(str).str.upper().str.strip()
    df["gsis_id"] = df["gsis_id"].astype("string")
    df = df.loc[
        df["season"].between(2022, 2025)
        & df["week"].between(1, 18)
        & df["position"].isin(POSITIONS)
        & df["gsis_id"].notna()
    ].copy()
    df["season"] = df["season"].astype(int)
    df["week"] = df["week"].astype(int)
    df = df.sort_values(["season", "gsis_id", "week"]).drop_duplicates(
        ["season", "gsis_id", "week"],
        keep="last",
    )
    df["opportunities"] = df["targets"] + df["carries"]
    return df.reset_index(drop=True)


def load_ownership(path):
    ownership = pd.read_csv(path, dtype={"gsis_id": "string"})
    required = {"ownership_season", "gsis_id", "percentOwned"}
    missing = required - set(ownership.columns)
    if missing:
        raise ValueError(f"ESPN ownership CSV missing columns: {sorted(missing)}")
    ownership = ownership.copy()
    ownership["ownership_season"] = pd.to_numeric(
        ownership["ownership_season"], errors="coerce"
    )
    ownership["percentOwned"] = pd.to_numeric(
        ownership["percentOwned"], errors="coerce"
    )
    ownership["gsis_id"] = ownership["gsis_id"].str.strip()
    ownership = ownership.loc[
        ownership["gsis_id"].notna()
        & ownership["gsis_id"].ne("")
        & ownership["ownership_season"].between(2022, 2025)
        & ownership["percentOwned"].between(0, MAX_PERCENT_OWNED)
    ]

    # One ownership value per player/season. When duplicate rows exist,
    # retain the most recently collected record, if timestamps are supplied.
    if "collected_at_utc" in ownership.columns:
        ownership = ownership.sort_values("collected_at_utc", kind="stable")
    ownership = ownership.drop_duplicates(["ownership_season", "gsis_id"], keep="last")
    return ownership[["ownership_season", "gsis_id", "percentOwned"]]
