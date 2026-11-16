
from pathlib import Path
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RosterWatch 🏈",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parents[2]

DATA_PATHS = [
    ROOT / "data/processed/weekly_usage.parquet",
    ROOT / "RosterWatch-matt/notebooks/weekly_usage.parquet",
    ROOT / "data/processed/weekly_usage.csv",
    ROOT / "RosterWatch-matt/notebooks/weekly_usage.csv",
]

POSITIONS = ["RB", "WR", "TE"]
WEEKS = list(range(1, 19))
PAGE_SIZE = 15

VOLATILITY_METRICS = [
    "snap_pct",
    "opp_share",
    "passdown_share",
    "redzone_share",
]

POSITION_WEIGHTS = {
    "RB": [0.30, 0.45, 0.15, 0.10],
    "WR": [0.35, 0.35, 0.20, 0.10],
    "TE": [0.40, 0.30, 0.20, 0.10],
}

CONSISTENCY_THRESHOLDS = {
    "RB": (10, 0.50, 10),
    "WR": (10, 0.65, 5),
    "TE": (10, 0.60, 4),
}

MIN_COMPARISONS = 8

MODEL2_WEIGHTS = {
    "fantasy": 0.40,
    "opportunity": 0.20,
    "snap": 0.15,
    "stability": 0.15,
    "confirmation": 0.10,
}

# Heuristic benchmarks, not statistically fitted values.
MODEL2_SETTINGS = {
    "RB": {
        "ppr": 12.0,
        "snap": 0.50,
        "opp": 0.18,
        "volume": 10,
    },
    "WR": {
        "ppr": 12.0,
        "snap": 0.65,
        "opp": 0.17,
        "volume": 6,
    },
    "TE": {
        "ppr": 8.0,
        "snap": 0.60,
        "opp": 0.13,
        "volume": 4,
    },
}

WAIVER_FLAGS = [
    "Must Pick Up",
    "Slow Riser",
    "Slow Drop",
    "Must Drop",
    "Stable / No Clear Signal",
    "Insufficient Data",
]

FLAG_ORDER = {
    "Must Pick Up": 0,
    "Slow Riser": 1,
    "Slow Drop": 2,
    "Must Drop": 3,
    "Stable / No Clear Signal": 4,
    "Insufficient Data": 5,
}

COLORS = {
    "blue": "#38BDF8",
    "green": "#4ADE80",
    "orange": "#F59E0B",
    "purple": "#A78BFA",
    "red": "#F87171",
    "panel": "#13243B",
    "grid": "#29415D",
    "text": "#E2E8F0",
}


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
<style>
.stApp {
    background:
        radial-gradient(
            circle at top right,
            #17375b 0%,
            #08111f 50%
        );
    color: #e2e8f0;
}

.block-container {
    max-width: 1450px;
    padding-top: 1.6rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: #101e32;
}

[data-testid="stMetric"] {
    background: #152841;
    border: 1px solid #2b4967;
    border-radius: 14px;
    padding: 15px;
}

[data-testid="stMetricValue"] {
    color: #4ade80;
}

.hero {
    border: 1px solid #2b4967;
    border-radius: 18px;
    padding: 26px 30px;
    background: linear-gradient(
        115deg, #10243b, #15395b
    );
    margin-bottom: 24px;
}

.hero-title {
    font-size: clamp(2rem, 4vw, 3rem);
    font-weight: 800;
    color: #f8fafc;
}

.hero-subtitle {
    color: #a7bfd7;
    margin: 8px 0 14px;
}

.pills {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 18px;
}

.pill {
    border: 1px solid #315779;
    background: #173655;
    border-radius: 999px;
    padding: 6px 12px;
    color: #c8e9ff;
    font-size: .8rem;
}

.stButton > button {
    border-radius: 9px;
    border: 1px solid #315779;
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data(show_spinner="Loading NFL data...")
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
        raise ValueError(
            f"Missing required columns: {missing}"
        )

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

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df["position"] = (
        df["position"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    df["gsis_id"] = df["gsis_id"].astype("string")

    df = df.loc[
        df["season"].between(2022, 2025)
        & df["week"].between(1, 18)
        & df["position"].isin(POSITIONS)
        & df["gsis_id"].notna()
    ].copy()

    df["season"] = df["season"].astype(int)
    df["week"] = df["week"].astype(int)

    df = (
        df.sort_values(
            ["season", "gsis_id", "week"]
        )
        .drop_duplicates(
            ["season", "gsis_id", "week"],
            keep="last",
        )
    )

    df["opportunities"] = (
        df["targets"] + df["carries"]
    )

    return df.reset_index(drop=True)


# ============================================================
# SHARED HELPERS
# ============================================================

def safe_mean(series):

    values = pd.to_numeric(
        series, errors="coerce"
    ).dropna()

    if values.empty:
        return np.nan

    return float(values.mean())


def clip01(value):

    if pd.isna(value):
        return 0.0

    return float(np.clip(value, 0, 1))


def normalized_change(
    recent,
    previous,
    column,
    scale,
):

    current = safe_mean(recent[column])
    baseline = safe_mean(previous[column])

    if pd.isna(current) or pd.isna(baseline):
        return 0.0

    return float(
        np.clip(
            (current - baseline) / scale,
            -1,
            1,
        )
    )


# ============================================================
# MODEL 1 — WEEKLY VOLATILITY
# ============================================================

@st.cache_data(
    show_spinner="Calculating weekly volatility..."
)
def calculate_volatility(data):

    if data.empty:
        return data.copy(), pd.DataFrame()

    frames = []

    for _, player in data.groupby(
        ["season", "gsis_id"],
        sort=False,
    ):

        player = player.sort_values("week").copy()
        position = player["position"].iloc[-1]

        changes = (
            player[VOLATILITY_METRICS]
            .diff()
            .abs()
        )

        valid = (
            player["week"].diff().eq(1)
            & player[VOLATILITY_METRICS]
            .notna().all(axis=1)
            & player[VOLATILITY_METRICS]
            .shift().notna().all(axis=1)
        )

        player["weekly_volatility"] = (
            changes.mul(
                POSITION_WEIGHTS[position],
                axis=1,
            )
            .sum(axis=1, min_count=4)
            .where(valid)
        )

        frames.append(player)

    weekly = pd.concat(
        frames,
        ignore_index=True,
    )

    summary = (
        weekly.groupby(
            ["season", "gsis_id"],
            as_index=False,
        )
        .agg(
            player_name=("player_name", "last"),
            position=("position", "last"),
            team=("team", "last"),
            weeks=("week", "nunique"),
            weekly_comparisons=(
                "weekly_volatility", "count"
            ),
            volatility_score=(
                "weekly_volatility", "mean"
            ),
            avg_snap_pct=("snap_pct", "mean"),
            avg_opportunities_per_week=(
                "opportunities", "mean"
            ),
        )
    )

    def eligible(row):

        min_weeks, min_snap, min_volume = (
            CONSISTENCY_THRESHOLDS[row["position"]]
        )

        return bool(
            row["weeks"] >= min_weeks
            and row["weekly_comparisons"]
            >= MIN_COMPARISONS
            and row["avg_snap_pct"] >= min_snap
            and row["avg_opportunities_per_week"]
            >= min_volume
        )

    summary["consistency_eligible"] = (
        summary.apply(eligible, axis=1)
    )

    summary["consistency_quality_score"] = np.nan

    eligible_summary = summary.loc[
        summary["consistency_eligible"]
    ]

    for _, group in eligible_summary.groupby(
        ["season", "position"]
    ):

        volatility_rank = (
            group["volatility_score"].rank(pct=True)
        )

        snap_rank = (
            group["avg_snap_pct"].rank(pct=True)
        )

        opportunity_rank = (
            group["avg_opportunities_per_week"]
            .rank(pct=True)
        )

        quality = 100 * (
            0.50 * (1 - volatility_rank)
            + 0.25 * snap_rank
            + 0.25 * opportunity_rank
        )

        summary.loc[
            group.index,
            "consistency_quality_score",
        ] = quality

    return weekly, summary


# ============================================================
# MODEL 2 — FANTASY PERFORMANCE
# ============================================================

def calculate_fantasy_score(
    player,
    recent,
    previous,
    settings,
):

    baseline = settings["ppr"]

    season_ppr = safe_mean(
        player["fantasy_points_ppr"]
    )

    recent_ppr = safe_mean(
        recent["fantasy_points_ppr"]
    )

    recent_median = float(
        recent["fantasy_points_ppr"].median()
    )

    ppr_trend = normalized_change(
        recent,
        previous,
        "fantasy_points_ppr",
        baseline * 0.50,
    )

    # Recent mean and median prevent one
    # exceptional game from dominating.
    scoring_level = (
        0.65 * clip01(recent_ppr / baseline)
        + 0.35 * clip01(recent_median / baseline)
    )

    trend_component = (
        0.5 + 0.5 * ppr_trend
    )

    fantasy_score = clip01(
        0.75 * scoring_level
        + 0.25 * trend_component
    )

    return {
        "score": fantasy_score,
        "season_ppr": season_ppr,
        "recent_ppr": recent_ppr,
        "recent_median": recent_median,
        "trend": ppr_trend,
    }


# ============================================================
# MODEL 2 — SNAP AND OPPORTUNITY
# ============================================================

def calculate_usage_scores(
    recent,
    previous,
    settings,
):

    snap_trend = normalized_change(
        recent,
        previous,
        "snap_pct",
        0.10,
    )

    opp_trend = normalized_change(
        recent,
        previous,
        "opp_share",
        0.06,
    )

    recent_snap = safe_mean(
        recent["snap_pct"]
    )

    recent_opp = safe_mean(
        recent["opp_share"]
    )

    snap_level = clip01(
        recent_snap / settings["snap"]
    )

    opp_level = clip01(
        recent_opp / settings["opp"]
    )

    snap_score = clip01(
        0.80 * snap_level
        + 0.20 * (0.5 + 0.5 * snap_trend)
    )

    opportunity_score = clip01(
        0.80 * opp_level
        + 0.20 * (0.5 + 0.5 * opp_trend)
    )

    return {
        "snap_score": snap_score,
        "opportunity_score": opportunity_score,
        "snap_trend": snap_trend,
        "opp_trend": opp_trend,
        "recent_snap": recent_snap,
        "recent_opp": recent_opp,
    }


# ============================================================
# MODEL 2 — VOLATILITY AND STABILITY
# ============================================================

def calculate_stability(
    player,
    recent,
    season_ppr,
):

    # Fantasy production variability.
    ppr_std = float(
        player["fantasy_points_ppr"].std(ddof=0)
    )

    ppr_cv = (
        ppr_std / max(abs(season_ppr), 2.0)
    )

    fantasy_stability = (
        1 / (1 + ppr_cv)
    )

    # Model 1 usage volatility.
    # Recent volatility receives greater weight.
    season_volatility = safe_mean(
        player["weekly_volatility"]
    )

    recent_volatility = safe_mean(
        recent["weekly_volatility"]
    )

    if pd.notna(recent_volatility):
        if pd.notna(season_volatility):
            combined_volatility = (
                0.65 * recent_volatility
                + 0.35 * season_volatility
            )
        else:
            combined_volatility = recent_volatility

    elif pd.notna(season_volatility):
        combined_volatility = season_volatility

    else:
        combined_volatility = np.nan

    if pd.notna(combined_volatility):
        usage_stability = (
            1 / (1 + combined_volatility / 0.12)
        )
    else:
        usage_stability = 0.50

    stability_score = clip01(
        0.60 * fantasy_stability
        + 0.40 * usage_stability
    )

    return {
        "score": stability_score,
        "fantasy_stability": fantasy_stability,
        "usage_stability": usage_stability,
        "volatility": combined_volatility,
    }


# ============================================================
# MODEL 2 — TREND CONFIRMATION
# ============================================================

def calculate_trend_confirmation(
    recent,
    previous,
    ppr_trend,
    opp_trend,
    snap_trend,
):

    # Directional agreement across metrics.
    weighted_trend = (
        0.50 * ppr_trend
        + 0.30 * opp_trend
        + 0.20 * snap_trend
    )

    alignment_score = clip01(
        0.5 + 0.5 * weighted_trend
    )

    baseline_ppr = safe_mean(
        previous["fantasy_points_ppr"]
    )

    baseline_opp = safe_mean(
        previous["opp_share"]
    )

    baseline_snap = safe_mean(
        previous["snap_pct"]
    )

    # Each recent game contributes evidence.
    ppr_up = (
        recent["fantasy_points_ppr"]
        > baseline_ppr
    )

    opp_up = (
        recent["opp_share"]
        > baseline_opp
    )

    snap_up = (
        recent["snap_pct"]
        > baseline_snap
    )

    confirmed_up = (
        ppr_up & (opp_up | snap_up)
    )

    ppr_down = (
        recent["fantasy_points_ppr"]
        < baseline_ppr
    )

    opp_down = (
        recent["opp_share"]
        < baseline_opp
    )

    snap_down = (
        recent["snap_pct"]
        < baseline_snap
    )

    confirmed_down = (
        ppr_down & (opp_down | snap_down)
    )

    upward_confirmation = float(
        confirmed_up.mean()
    )

    downward_confirmation = float(
        confirmed_down.mean()
    )

    if weighted_trend >= 0:
        persistence = upward_confirmation
    else:
        persistence = downward_confirmation

    # Confirmation rewards positive momentum.
    # Declining trends should not receive a
    # positive bonus merely for being persistent.
    confirmation_score = clip01(
        0.65 * alignment_score
        + 0.35 * upward_confirmation
    )

    return {
        "score": confirmation_score,
        "weighted_trend": weighted_trend,
        "upward_confirmation": upward_confirmation,
        "downward_confirmation": downward_confirmation,
        "persistence": persistence,
    }


# ============================================================
# MODEL 2 — SCENARIO CLASSIFICATION
# ============================================================

def classify_waiver_scenario(
    score,
    recent_ppr,
    recent_median,
    latest_ppr,
    ppr_trend,
    opp_trend,
    snap_trend,
    upward_confirmation,
    downward_confirmation,
    recent_volatility,
    games_analyzed,
    settings,
):

    baseline = settings["ppr"]

    usage_rising = (
        opp_trend >= 0.10
        or snap_trend >= 0.10
    )

    usage_falling = (
        opp_trend <= -0.10
        or snap_trend <= -0.10
    )

    fantasy_rising = (
        ppr_trend >= 0.20
    )

    fantasy_falling = (
        ppr_trend <= -0.20
    )

    # Detect an exceptional latest game.
    scoring_spike = (
        latest_ppr
        > max(
            1.8 * recent_median,
            1.5 * baseline,
        )
        and ppr_trend > 0
    )

    high_volatility = (
        pd.notna(recent_volatility)
        and recent_volatility >= 0.15
    )

    # --------------------------------------------
    # Scenario 1: Possible one-week spike
    # --------------------------------------------

    if (
        scoring_spike
        and upward_confirmation < 0.67
    ):

        return (
            "Stable / No Clear Signal",
            "Possible one-week fantasy spike",
        )

    # --------------------------------------------
    # Scenario 2: Confirmed breakout
    # --------------------------------------------

    if (
        fantasy_rising
        and usage_rising
        and upward_confirmation >= 0.67
        and score >= 72
        and recent_ppr >= baseline * 0.85
        and games_analyzed >= 5
    ):

        return (
            "Must Pick Up",
            "Confirmed fantasy breakout",
        )

    # --------------------------------------------
    # Scenario 3: Rising fantasy production
    # --------------------------------------------

    if (
        fantasy_rising
        and usage_rising
        and upward_confirmation >= 0.50
        and score >= 52
    ):

        return (
            "Slow Riser",
            "Rising fantasy points and offensive usage",
        )

    # --------------------------------------------
    # Scenario 4: Opportunities increasing
    # --------------------------------------------

    if (
        usage_rising
        and ppr_trend > -0.30
        and recent_ppr < baseline
    ):

        return (
            "Slow Riser",
            "Increasing opportunities; fantasy upside",
        )

    # --------------------------------------------
    # Scenario 5: Major decline
    # --------------------------------------------

    if (
        fantasy_falling
        and usage_falling
        and downward_confirmation >= 0.67
        and score <= 43
        and recent_ppr < baseline * 0.60
    ):

        return (
            "Must Drop",
            "Sustained fantasy and usage decline",
        )

    # --------------------------------------------
    # Scenario 6: Gradual decline
    # --------------------------------------------

    if (
        fantasy_falling
        and usage_falling
        and downward_confirmation >= 0.50
    ):

        return (
            "Slow Drop",
            "Declining fantasy production and usage",
        )

    # --------------------------------------------
    # Scenario 7: Unsustainable scoring
    # --------------------------------------------

    if (
        fantasy_rising
        and opp_trend <= -0.10
    ):

        return (
            "Stable / No Clear Signal",
            "Fantasy scoring rising despite fewer opportunities",
        )

    # --------------------------------------------
    # Scenario 8: Scoring lagging behind usage
    # --------------------------------------------

    if (
        fantasy_falling
        and opp_trend >= 0.10
    ):

        return (
            "Stable / No Clear Signal",
            "Opportunities increasing but scoring declining",
        )

    # --------------------------------------------
    # Scenario 9: High volatility
    # --------------------------------------------

    if high_volatility:

        return (
            "Stable / No Clear Signal",
            "High weekly usage volatility",
        )

    # --------------------------------------------
    # Scenario 10: Reliable producer
    # --------------------------------------------

    if (
        recent_ppr >= baseline
        and score >= 65
        and abs(ppr_trend) < 0.20
        and not usage_falling
    ):

        return (
            "Stable / No Clear Signal",
            "Reliable fantasy producer",
        )

    return (
        "Stable / No Clear Signal",
        "Stable or mixed fantasy indicators",
    )


# ============================================================
# MODEL 2 — COMPLETE WAIVER MODEL
# ============================================================

@st.cache_data(
    show_spinner="Analyzing fantasy and usage trends..."
)
def calculate_waiver_model(
    weekly_data,
    selected_season,
    as_of_week,
):

    data = weekly_data.loc[
        (weekly_data["season"] == selected_season)
        & (weekly_data["week"] <= as_of_week)
    ].copy()

    # Calculate volatility from the full available
    # history through the selected week.
    data, _ = calculate_volatility(data)

    columns = [
        "gsis_id",
        "player_name",
        "position",
        "team",
        "waiver_flag",
        "waiver_score",
        "scenario",
        "fantasy_score",
        "opportunity_score",
        "snap_score",
        "stability_score",
        "confirmation_score",
        "season_ppr",
        "recent_ppr",
        "ppr_trend",
        "snap_trend",
        "opportunity_trend",
        "volatility",
        "upward_confirmation",
        "downward_confirmation",
        "games_analyzed",
    ]

    results = []

    for player_id, player in data.groupby(
        "gsis_id"
    ):

        player = (
            player.sort_values("week").copy()
        )

        latest = player.iloc[-1]

        # Do not recommend players absent
        # for more than one NFL week.
        if (
            as_of_week - int(latest["week"])
            > 1
        ):
            continue

        position = latest["position"]

        if position not in MODEL2_SETTINGS:
            continue

        settings = MODEL2_SETTINGS[position]

        result = {
            column: np.nan
            for column in columns
        }

        result.update({
            "gsis_id": player_id,
            "player_name": latest["player_name"],
            "position": position,
            "team": latest["team"],
            "waiver_flag": "Insufficient Data",
            "scenario": "Insufficient history",
            "games_analyzed": len(player),
        })

        required = [
            "snap_pct",
            "opp_share",
            "opportunities",
            "fantasy_points_ppr",
        ]

        valid_player = player.dropna(
            subset=required
        ).copy()

        # Require enough actual fantasy data.
        if (
            len(valid_player) < 4
            or len(valid_player) / len(player) < 0.75
        ):
            results.append(result)
            continue

        recent_count = min(
            4,
            max(2, len(valid_player) // 3),
        )

        recent = valid_player.tail(
            recent_count
        )

        previous = valid_player.iloc[
            :-recent_count
        ]

        if len(previous) < 2:
            results.append(result)
            continue

        # ----------------------------------------
        # 1. Fantasy PPR performance — 40%
        # ----------------------------------------

        fantasy = calculate_fantasy_score(
            valid_player,
            recent,
            previous,
            settings,
        )

        # ----------------------------------------
        # 2. Opportunity and snap usage — 35%
        # ----------------------------------------

        usage_scores = calculate_usage_scores(
            recent,
            previous,
            settings,
        )

        # ----------------------------------------
        # 3. Volatility and stability — 15%
        # ----------------------------------------

        stability = calculate_stability(
            valid_player,
            recent,
            fantasy["season_ppr"],
        )

        # ----------------------------------------
        # 4. Trend confirmation — 10%
        # ----------------------------------------

        confirmation = (
            calculate_trend_confirmation(
                recent,
                previous,
                fantasy["trend"],
                usage_scores["opp_trend"],
                usage_scores["snap_trend"],
            )
        )

        # ----------------------------------------
        # 5. Weighted waiver score
        # ----------------------------------------

        raw_score = 100 * (
            MODEL2_WEIGHTS["fantasy"]
            * fantasy["score"]

            + MODEL2_WEIGHTS["opportunity"]
            * usage_scores["opportunity_score"]

            + MODEL2_WEIGHTS["snap"]
            * usage_scores["snap_score"]

            + MODEL2_WEIGHTS["stability"]
            * stability["score"]

            + MODEL2_WEIGHTS["confirmation"]
            * confirmation["score"]
        )

        # Evidence adjustment for sparse histories.
        coverage = min(
            1.0,
            len(valid_player)
            / max(as_of_week - 1, 1),
        )

        evidence_factor = (
            0.80 + 0.20 * coverage
        )

        waiver_score = float(
            np.clip(
                raw_score * evidence_factor,
                0,
                100,
            )
        )

        # ----------------------------------------
        # 6. Scenario detection
        # ----------------------------------------

        latest_ppr = float(
            valid_player[
                "fantasy_points_ppr"
            ].iloc[-1]
        )

        flag, scenario = classify_waiver_scenario(
            score=waiver_score,
            recent_ppr=fantasy["recent_ppr"],
            recent_median=fantasy["recent_median"],
            latest_ppr=latest_ppr,
            ppr_trend=fantasy["trend"],
            opp_trend=usage_scores["opp_trend"],
            snap_trend=usage_scores["snap_trend"],
            upward_confirmation=confirmation[
                "upward_confirmation"
            ],
            downward_confirmation=confirmation[
                "downward_confirmation"
            ],
            recent_volatility=stability[
                "volatility"
            ],
            games_analyzed=len(valid_player),
            settings=settings,
        )

        result.update({
            "waiver_flag": flag,
            "waiver_score": round(
                waiver_score, 1
            ),
            "scenario": scenario,
            "fantasy_score": round(
                fantasy["score"], 3
            ),
            "opportunity_score": round(
                usage_scores["opportunity_score"], 3
            ),
            "snap_score": round(
                usage_scores["snap_score"], 3
            ),
            "stability_score": round(
                stability["score"], 3
            ),
            "confirmation_score": round(
                confirmation["score"], 3
            ),
            "season_ppr": round(
                fantasy["season_ppr"], 2
            ),
            "recent_ppr": round(
                fantasy["recent_ppr"], 2
            ),
            "ppr_trend": round(
                fantasy["trend"], 3
            ),
            "snap_trend": round(
                usage_scores["snap_trend"], 3
            ),
            "opportunity_trend": round(
                usage_scores["opp_trend"], 3
            ),
            "volatility": (
                round(stability["volatility"], 3)
                if pd.notna(stability["volatility"])
                else np.nan
            ),
            "upward_confirmation": round(
                confirmation["upward_confirmation"], 3
            ),
            "downward_confirmation": round(
                confirmation["downward_confirmation"], 3
            ),
            "games_analyzed": len(valid_player),
        })

        results.append(result)

    return pd.DataFrame(
        results,
        columns=columns,
    )


# ============================================================
# PAGINATED TABLE
# ============================================================

def paginated_table(
    data,
    table_key,
    filter_signature,
    column_config=None,
):

    if data.empty:
        st.info("No players match these filters.")
        return

    total_pages = max(
        1,
        math.ceil(len(data) / PAGE_SIZE),
    )

    page_key = f"{table_key}_page"
    signature_key = f"{table_key}_signature"

    if (
        st.session_state.get(signature_key)
        != filter_signature
    ):
        st.session_state[signature_key] = (
            filter_signature
        )
        st.session_state[page_key] = 1

    current_page = min(
        max(
            st.session_state.get(page_key, 1),
            1,
        ),
        total_pages,
    )

    st.session_state[page_key] = current_page

    start = (current_page - 1) * PAGE_SIZE
    end = start + PAGE_SIZE

    st.dataframe(
        data.iloc[start:end].copy(),
        hide_index=True,
        use_container_width=True,
        height=570,
        column_config=column_config or {},
    )

    previous_col, page_col, next_col = st.columns(
        [1, 1.4, 1]
    )

    with previous_col:
        if st.button(
            "← Previous",
            key=f"{table_key}_previous",
            disabled=current_page == 1,
            use_container_width=True,
        ):
            st.session_state[page_key] -= 1
            st.rerun()

    with page_col:
        st.markdown(
            (
                "<p style='text-align:center;"
                "padding-top:8px;color:#a7bfd7'>"
                f"Page {current_page} of {total_pages}"
                "</p>"
            ),
            unsafe_allow_html=True,
        )

    with next_col:
        if st.button(
            "Next →",
            key=f"{table_key}_next",
            disabled=current_page == total_pages,
            use_container_width=True,
        ):
            st.session_state[page_key] += 1
            st.rerun()

    st.caption(
        f"Showing {start + 1}–"
        f"{min(end, len(data))} "
        f"of {len(data)} players"
    )


# ============================================================
# CHART HELPERS
# ============================================================

def weekly_chart(
    data,
    metric,
    title,
    color,
    percentage=False,
    bar=False,
):

    series = (
        data.groupby("week")[metric]
        .mean()
        .reindex(WEEKS)
    )

    fig = go.Figure()

    if bar:
        fig.add_trace(
            go.Bar(
                x=WEEKS,
                y=series.values,
                marker_color=color,
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=WEEKS,
                y=series.values,
                mode="lines+markers",
                connectgaps=False,
                line=dict(
                    color=color,
                    width=3,
                ),
                marker=dict(size=7),
            )
        )

    fig.update_layout(
        title=title,
        paper_bgcolor=COLORS["panel"],
        plot_bgcolor=COLORS["panel"],
        font=dict(color=COLORS["text"]),
        height=350,
        margin=dict(
            l=45,
            r=15,
            t=60,
            b=40,
        ),
        showlegend=False,
        xaxis=dict(
            title="NFL Week",
            tickmode="array",
            tickvals=WEEKS,
            range=[0.5, 18.5],
            gridcolor=COLORS["grid"],
        ),
        yaxis=dict(
            gridcolor=COLORS["grid"],
            rangemode="tozero",
        ),
    )

    if percentage:
        fig.update_yaxes(
            tickformat=".0%",
            range=[0, 1],
        )

    return fig


def component_chart(result):

    names = [
        "Fantasy PPR",
        "Opportunity Share",
        "Snap Share",
        "Stability",
        "Trend Confirmation",
    ]

    values = [
        result["fantasy_score"] * 100,
        result["opportunity_score"] * 100,
        result["snap_score"] * 100,
        result["stability_score"] * 100,
        result["confirmation_score"] * 100,
    ]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker_color=[
                COLORS["blue"],
                COLORS["green"],
                COLORS["purple"],
                COLORS["orange"],
                COLORS["red"],
            ],
            text=[
                f"{value:.1f}"
                for value in values
            ],
            textposition="auto",
        )
    )

    fig.update_layout(
        title="Model 2 Component Scores",
        paper_bgcolor=COLORS["panel"],
        plot_bgcolor=COLORS["panel"],
        font=dict(color=COLORS["text"]),
        height=350,
        margin=dict(
            l=150,
            r=20,
            t=60,
            b=40,
        ),
        xaxis=dict(
            title="Component Score",
            range=[0, 105],
            gridcolor=COLORS["grid"],
        ),
        yaxis=dict(
            autorange="reversed",
        ),
    )

    return fig


# ============================================================
# LOAD DATA
# ============================================================

source = next(
    (path for path in DATA_PATHS if path.is_file()),
    None,
)

if source is None:
    st.error(
        "Could not find weekly_usage.parquet "
        "or weekly_usage.csv."
    )
    st.stop()

try:
    usage = load_data(str(source))

    if usage.empty:
        st.error("No NFL records found.")
        st.stop()

except Exception as error:
    st.error(f"Data error: {error}")
    st.stop()


# ============================================================
# HEADER
# ============================================================

st.html("""
<div class="hero">
    <div class="hero-title">
        RosterWatch 🏈
    </div>

    <div class="hero-subtitle">
        NFL Fantasy Analytics &amp; Waiver Opportunity |
        2022–2025
    </div>

    <div class="hero-description">
        Analyze fantasy production, offensive usage,
        player consistency, and weekly waiver trends.
    </div>

    <div class="pills">
        <span class="pill">Model 1 · Volatility</span>
        <span class="pill">Model 2 · Fantasy Waiver Score</span>
        <span class="pill">PPR-Based Analysis</span>
        <span class="pill">RB / WR / TE</span>
        <span class="pill">Weeks 1–18</span>
    </div>
</div>
""")


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Control Center")

season = st.sidebar.selectbox(
    "NFL Season",
    sorted(
        usage["season"].unique().tolist(),
        reverse=True,
    ),
)

position = st.sidebar.selectbox(
    "Position",
    ["All Positions"] + POSITIONS,
)


st.sidebar.divider()

st.sidebar.markdown("**Model 1**")
st.sidebar.caption("Weekly Usage Volatility")

st.sidebar.markdown("**Model 2**")
st.sidebar.caption(
    "Fantasy PPR & Waiver Opportunity"
)


# ============================================================
# MODEL 1 DATA
# ============================================================
st.divider()
st.header("Model 1 — Weekly Usage Volatility")

model1_weeks = st.slider(
    "Select Model 1 Week Range",
    min_value=1,
    max_value=18,
    value=(1, 18),
    step=1,
    key="model1_week_range",
)

st.caption(
    f"Analyzing NFL Weeks {model1_weeks[0]}–"
    f"{model1_weeks[1]} for the {season} season."
)
# Calculate on full-season data first, so the
# first selected week can retain its previous
# week's volatility comparison.
full_model1_input = usage.loc[
    usage["season"] == season
].copy()

full_model1_weekly, _ = calculate_volatility(
    full_model1_input
)

model1_input = full_model1_weekly.loc[
    full_model1_weekly["week"].between(
        model1_weeks[0],
        model1_weeks[1],
    )
].copy()

# Summary uses only the selected window,
# while weekly volatility retains the
# correct preceding-week comparison.
def summarize_model1_window(weekly):

    if weekly.empty:
        return pd.DataFrame()

    summary = (
        weekly.groupby(
            ["season", "gsis_id"],
            as_index=False,
        )
        .agg(
            player_name=("player_name", "last"),
            position=("position", "last"),
            team=("team", "last"),
            weeks=("week", "nunique"),
            weekly_comparisons=(
                "weekly_volatility", "count"
            ),
            volatility_score=(
                "weekly_volatility", "mean"
            ),
            avg_snap_pct=("snap_pct", "mean"),
            avg_opportunities_per_week=(
                "opportunities", "mean"
            ),
        )
    )

    def eligible(row):
        min_weeks, min_snap, min_volume = (
            CONSISTENCY_THRESHOLDS[row["position"]]
        )

        return bool(
            row["weeks"] >= min_weeks
            and row["weekly_comparisons"]
            >= MIN_COMPARISONS
            and row["avg_snap_pct"] >= min_snap
            and row["avg_opportunities_per_week"]
            >= min_volume
        )

    summary["consistency_eligible"] = (
        summary.apply(eligible, axis=1)
    )

    summary["consistency_quality_score"] = np.nan

    for _, group in summary.loc[
        summary["consistency_eligible"]
    ].groupby(["season", "position"]):

        quality = 100 * (
            0.50 * (
                1 - group["volatility_score"]
                .rank(pct=True)
            )
            + 0.25 * group["avg_snap_pct"]
            .rank(pct=True)
            + 0.25 * group[
                "avg_opportunities_per_week"
            ].rank(pct=True)
        )

        summary.loc[
            group.index,
            "consistency_quality_score",
        ] = quality

    return summary


model1_summary = summarize_model1_window(
    model1_input
)

filtered_usage = model1_input.copy()
filtered_summary = model1_summary.copy()

if position != "All Positions":

    filtered_usage = filtered_usage.loc[
        filtered_usage["position"] == position
    ]

    filtered_summary = filtered_summary.loc[
        filtered_summary["position"] == position
    ]


# ============================================================
# ABOUT ROSTERWATCH
# ============================================================

st.header("About RosterWatch")

st.write(
    "RosterWatch analyzes NFL offensive usage "
    "and fantasy production to evaluate player "
    "consistency and identify potential waiver "
    "wire opportunities."
)

st.write(
    "**Model 1** measures week-to-week changes "
    "in offensive involvement."
)

st.write(
    "**Model 2** prioritizes PPR fantasy points "
    "while evaluating snap share, opportunity "
    "share, volatility, and trend confirmation."
)


# ============================================================
# ANALYTICS MODEL EXPLANATION
# ============================================================

with st.expander(
    "Our Analytics Models — Equations and Thresholds",
    expanded=True,
):

    st.subheader("Model 1 — Weekly Volatility")

    st.latex(
        r"V_t="
        r"w_S|S_t-S_{t-1}|"
        r"+w_O|O_t-O_{t-1}|"
        r"+w_P|P_t-P_{t-1}|"
        r"+w_R|R_t-R_{t-1}|"
    )

    st.write(
        "S = snap share; O = opportunity share; "
        "P = passing-down share; R = red-zone share."
    )

    weights_table = pd.DataFrame(
        POSITION_WEIGHTS,
        index=VOLATILITY_METRICS,
    ).T

    weights_table.columns = [
        "Snap %",
        "Opportunity Share",
        "Passing-Down Share",
        "Red-Zone Share",
    ]

    st.table(weights_table)

    st.latex(
        r"\bar{V}=\frac{1}{m}\sum_{t\in T}V_t"
    )

    st.subheader(
        "Model 2 — Fantasy Waiver Opportunity"
    )

    st.latex(
        r"W_t=100("
        r"0.40F_t+0.20O_t+0.15S_t"
        r"+0.15V_t+0.10T_t)"
    )

    st.write(
        "**F:** Fantasy PPR performance, "
        "including recent production and trends."
    )

    st.write(
        "**O:** Opportunity share strength."
    )

    st.write(
        "**S:** Offensive snap share strength."
    )

    st.write(
        "**V:** Volatility-adjusted stability "
        "of fantasy production and offensive usage."
    )

    st.write(
        "**T:** Trend confirmation based on "
        "fantasy production and usage changes."
    )

    st.markdown("#### Component Weights")

    st.table(
        pd.DataFrame({
            "Component": [
                "Fantasy PPR Performance",
                "Opportunity Share",
                "Snap Share",
                "Volatility & Stability",
                "Trend Confirmation",
            ],
            "Weight": [
                "40%",
                "20%",
                "15%",
                "15%",
                "10%",
            ],
        })
    )

    st.markdown("#### Position Benchmarks")

    benchmark_table = pd.DataFrame({
        pos: {
            "PPR Points/Game": cfg["ppr"],
            "Snap Share": cfg["snap"],
            "Opportunity Share": cfg["opp"],
            "Opportunities/Game": cfg["volume"],
        }
        for pos, cfg in MODEL2_SETTINGS.items()
    }).T

    st.table(benchmark_table)

    st.info(
        "Model 2 uses heuristic benchmarks and "
        "scenario rules. Waiver scores are not "
        "calibrated probabilities or guarantees. "
        "Actual waiver availability is not verified."
    )


# ============================================================
# SEASON OVERVIEW
# ============================================================

st.divider()
st.header("Season Overview")

st.caption(
    f"{season} Season | "
    f"Model 1 Weeks {model1_weeks[0]}–"
    f"{model1_weeks[1]}"
)

a, b, c, d = st.columns(4)

a.metric(
    "Player-Week Records",
    f"{len(filtered_usage):,}",
)

b.metric(
    "Unique Players",
    f"{filtered_usage['gsis_id'].nunique():,}",
)

c.metric(
    "Offensive Snaps",
    f"{filtered_usage['offense_snaps'].sum():,.0f}",
)

d.metric(
    "Total Targets",
    f"{filtered_usage['targets'].sum():,.0f}",
)


# ============================================================
# MODEL 1 — PLAYER EXPLORER
# ============================================================

st.divider()
st.header("Player Explorer")

players = (
    filtered_usage[
        ["gsis_id", "player_name", "position"]
    ]
    .drop_duplicates("gsis_id")
    .sort_values("player_name")
)

if not players.empty:

    labels = {
        row.gsis_id: (
            f"{row.player_name} · {row.position}"
        )
        for row in players.itertuples(index=False)
    }

    player_id = st.selectbox(
        "Select NFL Player",
        players["gsis_id"].tolist(),
        format_func=lambda pid: labels.get(
            pid, str(pid)
        ),
    )

    player_data = (
        filtered_usage.loc[
            filtered_usage["gsis_id"] == player_id
        ]
        .sort_values("week")
    )

    player_summary = filtered_summary.loc[
        filtered_summary["gsis_id"] == player_id
    ]

    a, b, c, d = st.columns(4)

    a.metric(
        "Average Snap Share",
        f"{player_data['snap_pct'].mean():.1%}",
    )

    b.metric(
        "Targets",
        f"{player_data['targets'].sum():,.0f}",
    )

    c.metric(
        "Carries",
        f"{player_data['carries'].sum():,.0f}",
    )

    volatility = (
        player_summary["volatility_score"].iloc[0]
        if not player_summary.empty
        else np.nan
    )

    d.metric(
        "Avg Weekly Volatility",
        f"{volatility:.3f}"
        if pd.notna(volatility)
        else "N/A",
    )

    left, right = st.columns(2)

    with left:

        st.plotly_chart(
            weekly_chart(
                player_data,
                "snap_pct",
                "Offensive Snap Percentage",
                COLORS["blue"],
                percentage=True,
            ),
            use_container_width=True,
        )

        st.plotly_chart(
            weekly_chart(
                player_data,
                "targets",
                "Weekly Targets",
                COLORS["orange"],
            ),
            use_container_width=True,
        )

    with right:

        st.plotly_chart(
            weekly_chart(
                player_data,
                "opp_share",
                "Opportunity Share",
                COLORS["green"],
                percentage=True,
            ),
            use_container_width=True,
        )

        st.plotly_chart(
            weekly_chart(
                player_data,
                "carries",
                "Weekly Carries",
                COLORS["purple"],
            ),
            use_container_width=True,
        )

    left, right = st.columns(2)

    with left:

        st.plotly_chart(
            weekly_chart(
                player_data,
                "fantasy_points_ppr",
                "Sleeper PPR Points",
                COLORS["green"],
                bar=True,
            ),
            use_container_width=True,
        )

    with right:

        st.plotly_chart(
            weekly_chart(
                player_data,
                "weekly_volatility",
                "Weekly Usage Volatility",
                COLORS["blue"],
                bar=True,
            ),
            use_container_width=True,
        )

    with st.expander("View Weekly Player Data"):

        st.dataframe(
            player_data[
                [
                    "week",
                    "team",
                    "opponent",
                    "snap_pct",
                    "opp_share",
                    "targets",
                    "carries",
                    "fantasy_points_ppr",
                    "weekly_volatility",
                ]
            ],
            hide_index=True,
            use_container_width=True,
        )


# ============================================================
# MODEL 1 — RANKINGS
# ============================================================

st.divider()
st.header("Model 1 — Player Volatility Rankings")

ranking_position = st.selectbox(
    "Ranking Position",
    POSITIONS,
    index=(
        POSITIONS.index(position)
        if position in POSITIONS
        else 0
    ),
)

ranked = filtered_summary.loc[
    (filtered_summary["position"] == ranking_position)
    & filtered_summary["volatility_score"].notna()
].copy()

eligible = ranked.loc[
    ranked["consistency_eligible"]
].copy()

a, b, c, d = st.columns(4)

a.metric("Players Scored", len(ranked))

b.metric(
    "Average Volatility",
    f"{ranked['volatility_score'].mean():.3f}"
    if not ranked.empty
    else "N/A",
)

c.metric(
    "Highest Volatility",
    f"{ranked['volatility_score'].max():.3f}"
    if not ranked.empty
    else "N/A",
)

d.metric(
    "Consistency Eligible",
    len(eligible),
)

left, right = st.columns(2)

with left:

    st.subheader("📈 Most Volatile Players")

    volatile_display = (
        ranked.sort_values(
            ["volatility_score", "player_name"],
            ascending=[False, True],
        )[
            [
                "player_name",
                "team",
                "weeks",
                "weekly_comparisons",
                "volatility_score",
            ]
        ]
    )

    paginated_table(
        volatile_display,
        "volatile",
        (
            season,
            position,
            ranking_position,
            model1_weeks,
        ),
    )

with right:

    st.subheader("📉 Most Consistent Players")

    consistent_display = (
        eligible.sort_values(
            [
                "consistency_quality_score",
                "volatility_score",
            ],
            ascending=[False, True],
        )[
            [
                "player_name",
                "team",
                "weeks",
                "avg_snap_pct",
                "avg_opportunities_per_week",
                "volatility_score",
                "consistency_quality_score",
            ]
        ]
    )

    paginated_table(
        consistent_display,
        "consistent",
        (
            season,
            position,
            ranking_position,
            model1_weeks,
        ),
    )

    min_weeks, min_snap, min_volume = (
        CONSISTENCY_THRESHOLDS[ranking_position]
    )

    st.caption(
        f"Eligibility: {min_weeks}+ weeks, "
        f"{MIN_COMPARISONS}+ comparisons, "
        f"{min_snap:.0%}+ average snaps, "
        f"{min_volume}+ opportunities/game."
    )


# ============================================================
# MODEL 2 — WAIVER DASHBOARD
# ============================================================

st.divider()
st.header("Model 2 — Fantasy Waiver Opportunity")

st.write(
    "Model 2 prioritizes fantasy PPR production "
    "while analyzing offensive snap share, "
    "opportunity share, weekly volatility, "
    "and recent performance trends."
)

selected_week = st.selectbox(
    "Analyze Through NFL Week",
    list(range(4, 19)),
    index=14,
    key="waiver_week",
)

st.caption(
    f"Analyzing player history from Week 1 "
    f"through Week {selected_week} of {season}."
)

waiver_results = calculate_waiver_model(
    usage,
    season,
    selected_week,
)

if position != "All Positions":

    waiver_results = waiver_results.loc[
        waiver_results["position"] == position
    ].copy()


if not waiver_results.empty:

    # --------------------------------------------------------
    # FLAG COUNTS
    # --------------------------------------------------------

    a, b, c, d = st.columns(4)

    a.metric(
        "🟢 Must Pick Up",
        int(
            (
                waiver_results["waiver_flag"]
                == "Must Pick Up"
            ).sum()
        ),
    )

    b.metric(
        "🔵 Slow Riser",
        int(
            (
                waiver_results["waiver_flag"]
                == "Slow Riser"
            ).sum()
        ),
    )

    c.metric(
        "🟠 Slow Drop",
        int(
            (
                waiver_results["waiver_flag"]
                == "Slow Drop"
            ).sum()
        ),
    )

    d.metric(
        "🔴 Must Drop",
        int(
            (
                waiver_results["waiver_flag"]
                == "Must Drop"
            ).sum()
        ),
    )

    # --------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------

    left, right = st.columns(2)

    with left:

        flag_filter = st.selectbox(
            "Waiver Flag",
            ["All Flags"] + WAIVER_FLAGS,
            key="waiver_flag_filter",
        )

    with right:

        scenario_options = sorted(
            waiver_results["scenario"]
            .dropna()
            .unique()
            .tolist()
        )

        scenario_filter = st.selectbox(
            "Fantasy Scenario",
            ["All Scenarios"] + scenario_options,
            key="waiver_scenario_filter",
        )

    if flag_filter != "All Flags":

        waiver_results = waiver_results.loc[
            waiver_results["waiver_flag"]
            == flag_filter
        ].copy()

    if scenario_filter != "All Scenarios":

        waiver_results = waiver_results.loc[
            waiver_results["scenario"]
            == scenario_filter
        ].copy()

    # --------------------------------------------------------
    # SORTING
    # --------------------------------------------------------

    waiver_results["sort_order"] = (
        waiver_results["waiver_flag"]
        .map(FLAG_ORDER)
    )

    waiver_results = waiver_results.sort_values(
        [
            "sort_order",
            "waiver_score",
            "player_name",
        ],
        ascending=[True, False, True],
    )

    # --------------------------------------------------------
    # WAIVER RANKINGS
    # --------------------------------------------------------

    st.subheader("Weekly Waiver Rankings")

    waiver_display = waiver_results[
    [
        "player_name",
        "position",
        "team",
        "waiver_flag",
        "waiver_score",
        "recent_ppr",
        "season_ppr",
        "ppr_trend",
        "volatility",
    ]
].copy()

    paginated_table(
        data=waiver_display,
        table_key="waiver",
        filter_signature=(
            season,
            position,
            selected_week,
            flag_filter,
            scenario_filter,
        ),
        column_config={
            "player_name": "Player",
            "position": "Position",
            "team": "Team",
            "waiver_flag": "Waiver Flag",
            "waiver_score": (
                st.column_config.NumberColumn(
                    "Waiver Score",
                    format="%.1f",
                )
            ),
            "recent_ppr": (
                st.column_config.NumberColumn(
                    "Recent PPR Avg",
                    format="%.2f",
                )
            ),
            "season_ppr": (
                st.column_config.NumberColumn(
                    "Season PPR Avg",
                    format="%.2f",
                )
            ),
            "ppr_trend": (
                st.column_config.NumberColumn(
                    "PPR Trend",
                    format="%.3f",
                    help=(
                        "Normalized recent scoring "
                        "change, from -1 to +1."
                    ),
                )
            ),
            "volatility": (
                st.column_config.NumberColumn(
                    "Usage Volatility",
                    format="%.3f",
                )
            ),
        },
    )

    # --------------------------------------------------------
    # PLAYER WAIVER EXPLORER
    # --------------------------------------------------------

    st.divider()
    st.subheader("Player Waiver Trend Explorer")

    choices = (
        waiver_results[
            ["gsis_id", "player_name", "position"]
        ]
        .drop_duplicates("gsis_id")
        .sort_values("player_name")
    )

    if not choices.empty:

        labels = {
            row.gsis_id: (
                f"{row.player_name} · {row.position}"
            )
            for row in choices.itertuples(index=False)
        }

        selected_player_id = st.selectbox(
            "Inspect Player",
            choices["gsis_id"].tolist(),
            format_func=lambda pid: labels.get(
                pid, str(pid)
            ),
            key="waiver_player",
        )

        selected_result = waiver_results.loc[
            waiver_results["gsis_id"]
            == selected_player_id
        ].iloc[0]

        history = usage.loc[
            (usage["season"] == season)
            & (
                usage["gsis_id"]
                == selected_player_id
            )
            & (usage["week"] <= selected_week)
        ].sort_values("week")

        st.markdown(
            f"### {selected_result['player_name']}"
        )

        st.write(
            f"**Waiver Flag:** "
            f"{selected_result['waiver_flag']}"
        )

        st.write(
            f"**Fantasy Scenario:** "
            f"{selected_result['scenario']}"
        )

        score = selected_result["waiver_score"]

        a, b, c, d = st.columns(4)

        a.metric(
            "Waiver Score",
            f"{score:.1f}/100"
            if pd.notna(score)
            else "N/A",
        )

        b.metric(
            "Recent PPR Average",
            f"{selected_result['recent_ppr']:.2f}"
            if pd.notna(selected_result["recent_ppr"])
            else "N/A",
        )

        c.metric(
            "Season PPR Average",
            f"{selected_result['season_ppr']:.2f}"
            if pd.notna(selected_result["season_ppr"])
            else "N/A",
        )

        d.metric(
            "Usage Volatility",
            f"{selected_result['volatility']:.3f}"
            if pd.notna(selected_result["volatility"])
            else "N/A",
        )

        # ----------------------------------------------------
        # FANTASY PPR CHART
        # ----------------------------------------------------

        st.plotly_chart(
            weekly_chart(
                history,
                "fantasy_points_ppr",
                "Sleeper PPR Fantasy Points by Week",
                COLORS["purple"],
                bar=True,
            ),
            use_container_width=True,
        )

        # ----------------------------------------------------
        # COMPONENT BREAKDOWN
        # ----------------------------------------------------

        if pd.notna(score):

            st.plotly_chart(
                component_chart(selected_result),
                use_container_width=True,
            )

            st.caption(
                "Component scores are normalized "
                "to 0–100. The final waiver score "
                "also includes an evidence-coverage "
                "adjustment."
            )

        # ----------------------------------------------------
        # PLAYER TREND DETAILS
        # ----------------------------------------------------

        st.markdown("#### Player Trend Analysis")

        if pd.notna(score):

            a, b, c = st.columns(3)

            a.metric(
                "PPR Trend",
                f"{selected_result['ppr_trend']:+.3f}",
            )

            b.metric(
                "Snap Share Trend",
                f"{selected_result['snap_trend']:+.3f}",
            )

            c.metric(
                "Opportunity Share Trend",
                f"{selected_result['opportunity_trend']:+.3f}",
            )

            st.caption(
                "Trend values represent normalized "
                "recent changes, not raw percentages."
            )

        # ----------------------------------------------------
        # WEEKLY HISTORY
        # ----------------------------------------------------

        with st.expander(
            "View Full Player History"
        ):

            st.dataframe(
                history[
                    [
                        "week",
                        "team",
                        "opponent",
                        "snap_pct",
                        "opp_share",
                        "targets",
                        "carries",
                        "fantasy_points_ppr",
                    ]
                ],
                hide_index=True,
                use_container_width=True,
            )

else:

    st.info(
        "No players available for Model 2."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown("### RosterWatch 🏈")

st.caption(
    "Model 1: Weekly Usage Volatility | "
    "Model 2: Fantasy PPR Waiver Opportunity"
)

st.caption(
    "2022–2025 | RB · WR · TE | Weeks 1–18"
)

st.caption(
    "Built with Python, Pandas, Plotly, and Streamlit."
)
