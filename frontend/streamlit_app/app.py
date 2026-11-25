

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



# ESPN ownership CSV is kept separately from historical weekly usage data.
OWNERSHIP_PATHS = [
    ROOT / "data/processed/espn_ownership.csv",
    ROOT / "espn_ownership.csv",
]
MAX_PERCENT_OWNED = 50.0

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
]




FLAG_ORDER = {

    flag: index

    for index, flag in enumerate(WAIVER_FLAGS)

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



    missing = sorted(

        set(required) - set(df.columns)

    )



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



    for column in numeric_columns:

        df[column] = pd.to_numeric(

            df[column],

            errors="coerce",

        )



    df["position"] = (

        df["position"]

        .astype(str)

        .str.upper()

        .str.strip()

    )



    df["gsis_id"] = (

        df["gsis_id"].astype("string")

    )



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

@st.cache_data(show_spinner="Loading ESPN ownership data...")
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
    ownership = ownership.drop_duplicates(
        ["ownership_season", "gsis_id"], keep="last"
    )
    return ownership[["ownership_season", "gsis_id", "percentOwned"]]


# SHARED HELPERS

# ============================================================



def safe_mean(series):

    # Fast path for plain float/int columns (called thousands of times per page).
    # sum / count matches pandas' mean for NaN-free float64 data exactly.
    if isinstance(series.dtype, np.dtype) and series.dtype.kind in "fi":

        values = series.to_numpy(dtype="float64")

        values = values[~np.isnan(values)]

        if values.size == 0:

            return np.nan

        return float(values.sum() / values.size)

    values = pd.to_numeric(

        series,

        errors="coerce",

    ).dropna()



    if values.empty:

        return np.nan



    return float(values.mean())





def clip01(value):



    if pd.isna(value):

        return 0.0



    return float(

        np.clip(value, 0, 1)

    )





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

    )



    def eligible(row):



        min_weeks, min_snap, min_volume = (

            CONSISTENCY_THRESHOLDS[

                row["position"]

            ]

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

            group["volatility_score"].rank(

                pct=True

            )

        )



        snap_rank = (

            group["avg_snap_pct"].rank(

                pct=True

            )

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



    return summary





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



    scoring_level = (

        0.65 * clip01(

            recent_ppr / baseline

        )

        + 0.35 * clip01(

            recent_median / baseline

        )

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

        + 0.20 * (

            0.5 + 0.5 * snap_trend

        )

    )



    opportunity_score = clip01(

        0.80 * opp_level

        + 0.20 * (

            0.5 + 0.5 * opp_trend

        )

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



    ppr_std = float(

        player["fantasy_points_ppr"].std(

            ddof=0

        )

    )



    ppr_cv = (

        ppr_std

        / max(abs(season_ppr), 2.0)

    )



    fantasy_stability = (

        1 / (1 + ppr_cv)

    )



    season_volatility = safe_mean(

        player["weekly_volatility"]

    )



    recent_volatility = safe_mean(

        recent["weekly_volatility"]

    )



    if (

        pd.notna(recent_volatility)

        and pd.notna(season_volatility)

    ):



        combined_volatility = (

            0.65 * recent_volatility

            + 0.35 * season_volatility

        )



    elif pd.notna(recent_volatility):



        combined_volatility = (

            recent_volatility

        )



    else:



        combined_volatility = (

            season_volatility

        )



    if pd.notna(combined_volatility):



        usage_stability = (

            1

            / (

                1

                + combined_volatility / 0.12

            )

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

        ppr_up

        & (opp_up | snap_up)

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

        ppr_down

        & (opp_down | snap_down)

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

# MODEL 2 — WAIVER CLASSIFICATION

# ============================================================



def conservative_waiver_flag(score, ppr_trend=0.0, opp_trend=0.0,
                             snap_trend=0.0, recent_ppr=np.nan,
                             baseline=np.nan):
    """Choose an existing directional flag when evidence is mixed or sparse.

    Neutral/mixed cases use Slow Riser or Slow Drop, not a strong action.
    This is a heuristic classification, not a verified trend.
    """
    signals = [(ppr_trend, 0.50), (opp_trend, 0.30), (snap_trend, 0.20)]
    available = [(float(x), weight) for x, weight in signals if pd.notna(x)]
    direction = (sum(x * weight for x, weight in available) /
                 sum(weight for _, weight in available)) if available else 0.0
    if direction > 0.05:
        return "Slow Riser"
    if direction < -0.05:
        return "Slow Drop"
    if pd.notna(recent_ppr) and pd.notna(baseline) and baseline > 0:
        if recent_ppr >= baseline:
            return "Slow Riser"
        return "Slow Drop"
    return "Slow Riser" if pd.notna(score) and score >= 50 else "Slow Drop"


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



    if (

        scoring_spike

        and upward_confirmation < 0.67

    ):



        return (

            conservative_waiver_flag(score, ppr_trend, opp_trend, snap_trend, recent_ppr, baseline),

            "Possible one-week fantasy spike",

        )



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



    if (

        usage_rising

        and ppr_trend > -0.30

        and recent_ppr < baseline

    ):



        return (

            "Slow Riser",

            "Increasing opportunities; fantasy upside",

        )



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



    if (

        fantasy_falling

        and usage_falling

        and downward_confirmation >= 0.50

    ):



        return (

            "Slow Drop",

            "Declining fantasy production and usage",

        )



    if (

        fantasy_rising

        and opp_trend <= -0.10

    ):



        return (

            conservative_waiver_flag(score, ppr_trend, opp_trend, snap_trend, recent_ppr, baseline),

            "Fantasy scoring rising despite fewer opportunities",

        )



    if (

        fantasy_falling

        and opp_trend >= 0.10

    ):



        return (

            conservative_waiver_flag(score, ppr_trend, opp_trend, snap_trend, recent_ppr, baseline),

            "Opportunities increasing but scoring declining",

        )



    if high_volatility:



        return (

            conservative_waiver_flag(score, ppr_trend, opp_trend, snap_trend, recent_ppr, baseline),

            "High weekly usage volatility",

        )



    if (

        recent_ppr >= baseline

        and score >= 65

        and abs(ppr_trend) < 0.20

        and not usage_falling

    ):



        return (

            conservative_waiver_flag(score, ppr_trend, opp_trend, snap_trend, recent_ppr, baseline),

            "Reliable fantasy producer",

        )



    return (

        conservative_waiver_flag(score, ppr_trend, opp_trend, snap_trend, recent_ppr, baseline),

        "Stable or mixed fantasy indicators",

    )





# ============================================================

# MODEL 2 — COMPLETE WAIVER MODEL

# ============================================================



# ============================================================
# MODEL 2 — ADAPTIVE SCORING FOR LIMITED / PARTIAL HISTORY
# ============================================================

def calculate_adaptive_waiver(player, settings, as_of_week):
    """Score limited history using observed metrics only.

    Missing metrics are excluded and remaining weights renormalized.
    One game has no trend evidence, so it receives a neutral flag.
    Short histories never qualify for high-confidence Must Pick Up/Drop.
    """
    player = player.sort_values("week")
    metrics = {
        "fantasy_points_ppr": ("fantasy_score", MODEL2_WEIGHTS["fantasy"], settings["ppr"]),
        "opp_share": ("opportunity_score", MODEL2_WEIGHTS["opportunity"], settings["opp"]),
        "snap_pct": ("snap_score", MODEL2_WEIGHTS["snap"], settings["snap"]),
    }
    n = len(player)
    recent_count = min(2, max(1, n // 2)) if n > 1 else 1
    recent = player.tail(recent_count)
    previous = player.iloc[:-recent_count]
    result = {
        "fantasy_score": np.nan,
        "opportunity_score": np.nan,
        "snap_score": np.nan,
        "stability_score": np.nan,
        "confirmation_score": np.nan,
        "season_ppr": safe_mean(player["fantasy_points_ppr"]),
        "recent_ppr": safe_mean(recent["fantasy_points_ppr"]),
        "ppr_trend": 0.0,
        "snap_trend": 0.0,
        "opportunity_trend": 0.0,
        "volatility": safe_mean(player["weekly_volatility"]),
        "upward_confirmation": 0.0,
        "downward_confirmation": 0.0,
        "games_analyzed": n,
    }
    weighted = []
    trend_values = {}
    for column, (output, weight, benchmark) in metrics.items():
        current = safe_mean(recent[column])
        if not np.isfinite(current):
            continue
        level = clip01(current / benchmark)
        baseline = safe_mean(previous[column]) if not previous.empty else np.nan
        scale = {"fantasy_points_ppr": settings["ppr"] * 0.5,
                 "opp_share": 0.06, "snap_pct": 0.10}[column]
        trend = (float(np.clip((current - baseline) / scale, -1, 1))
                 if np.isfinite(baseline) else 0.0)
        trend_values[column] = trend
        component = clip01(0.80 * level + 0.20 * (0.5 + 0.5 * trend))
        result[output] = round(component, 3)
        weighted.append((weight, component))
    result["ppr_trend"] = trend_values.get("fantasy_points_ppr", 0.0)
    result["snap_trend"] = trend_values.get("snap_pct", 0.0)
    result["opportunity_trend"] = trend_values.get("opp_share", 0.0)

    # Stability is neutral when only one observation exists.
    stability_values = []
    for column, floor in [("fantasy_points_ppr", 2.0),
                          ("opp_share", 0.03), ("snap_pct", 0.05)]:
        values = pd.to_numeric(player[column], errors="coerce").dropna()
        if len(values) >= 2:
            cv = float(values.std(ddof=0)) / max(abs(float(values.mean())), floor)
            stability_values.append(1.0 / (1.0 + cv))
    if stability_values:
        stability = float(np.mean(stability_values))
        result["stability_score"] = round(stability, 3)
        weighted.append((MODEL2_WEIGHTS["stability"], stability))
    else:
        result["stability_score"] = 0.5

    # Confirmation is counted only where both a recent value and an
    # earlier baseline exist; otherwise it remains neutral and unweighted.
    available = [c for c in ("fantasy_points_ppr", "opp_share", "snap_pct")
                 if pd.notna(safe_mean(previous[c])) and
                 pd.notna(safe_mean(recent[c]))]
    if available:
        directions = [trend_values.get(c, 0.0) for c in available]
        direction = float(np.mean(directions))
        confirmed = []
        for _, game in recent.iterrows():
            checks = []
            for column in available:
                value = game[column]
                if pd.isna(value):
                    continue
                baseline = safe_mean(previous[column])
                checks.append(value > baseline if direction >= 0 else value < baseline)
            if checks:
                confirmed.append(all(checks))
        confirmation = float(np.mean(confirmed)) if confirmed else 0.5
        result["confirmation_score"] = round(confirmation, 3)
        if confirmed:
            weighted.append((MODEL2_WEIGHTS["confirmation"], confirmation))
            if direction >= 0:
                result["upward_confirmation"] = confirmation
            else:
                result["downward_confirmation"] = confirmation
    else:
        result["confirmation_score"] = 0.5

    total_weight = sum(weight for weight, _ in weighted)
    raw_score = (100 * sum(w * value for w, value in weighted) / total_weight
                 if total_weight else 0.0)
    # Sparse histories and missing fields receive a transparent discount.
    observed = sum(int(player[c].notna().sum()) for c in metrics)
    completeness = observed / max(3 * n, 1)
    evidence = (0.75 + 0.25 * min(n / 5, 1)) * (0.80 + 0.20 * completeness)
    score = float(np.clip(raw_score * evidence, 0, 100))

    # Never claim a strong breakout/decline without enough observations.
    usage_directions = [trend_values[c] for c in ("opp_share", "snap_pct")
                        if c in trend_values and not previous.empty]
    usage_direction = float(np.mean(usage_directions)) if usage_directions else 0.0
    if n >= 2 and usage_direction >= 0.10:
        flag = "Slow Riser"
    elif n >= 2 and usage_direction <= -0.10:
        flag = "Slow Drop"
    else:
        flag = conservative_waiver_flag(
            score, result["ppr_trend"], result["opportunity_trend"],
            result["snap_trend"], safe_mean(recent["fantasy_points_ppr"]),
            settings["ppr"],
        )
    available_names = ", ".join(c for c in metrics if player[c].notna().any())
    scenario = (f"Limited-history estimate ({n} recorded game(s)); "
                f"available metrics: {available_names or 'none'}. "
                "Confidence discounted; no strong waiver flag assigned.")
    result.update(waiver_flag=flag, waiver_score=round(score, 1), scenario=scenario)
    return result


@st.cache_data(

    show_spinner="Analyzing fantasy and usage trends..."

)

def calculate_waiver_model(

    weekly_data,

    selected_season,

    as_of_week,

    consistent_player_ids,

):



    data = weekly_data.loc[

        (weekly_data["season"] == selected_season)

        & (weekly_data["week"] <= as_of_week)

    ].copy()



    data, _ = calculate_volatility(data)



    # Model 2 classification is independent of Model 1 consistency.

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



    def add_result(result):
        # Every scored player receives exactly one of the four waiver flags.
        # Model 1 consistency does not override Model 2 classification.
        if result.get("waiver_flag") not in WAIVER_FLAGS:
            result["waiver_flag"] = conservative_waiver_flag(
                result.get("waiver_score"),
                result.get("ppr_trend", 0.0),
                result.get("opportunity_trend", 0.0),
                result.get("snap_trend", 0.0),
                result.get("recent_ppr", np.nan),
                MODEL2_SETTINGS[result["position"]]["ppr"],
            )
        results.append(result)



    for player_id, player in data.groupby(

        "gsis_id"

    ):



        player = (

            player.sort_values("week").copy()

        )



        latest = player.iloc[-1]



        # Include every ownership-eligible player with recorded data through
        # the selected week, even if their last appearance was earlier.

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

            "waiver_flag": "Slow Drop",

            "scenario": "Limited history; using available data",

            "games_analyzed": len(player),

        })



        required = ["snap_pct", "opp_share", "opportunities", "fantasy_points_ppr"]
        valid_player = player.dropna(subset=required).copy()

        # Use adaptive scoring instead of returning Insufficient Data.
        # This also handles players with missing fantasy-point observations.
        if len(valid_player) < 4 or len(valid_player) / len(player) < 0.75:
            result.update(calculate_adaptive_waiver(player, settings, as_of_week))
            add_result(result)
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

            result.update(calculate_adaptive_waiver(player, settings, as_of_week))
            add_result(result)
            continue



        fantasy = calculate_fantasy_score(

            valid_player,

            recent,

            previous,

            settings,

        )



        usage_scores = calculate_usage_scores(

            recent,

            previous,

            settings,

        )



        stability = calculate_stability(

            valid_player,

            recent,

            fantasy["season_ppr"],

        )



        confirmation = (

            calculate_trend_confirmation(

                recent,

                previous,

                fantasy["trend"],

                usage_scores["opp_trend"],

                usage_scores["snap_trend"],

            )

        )



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

                round(

                    stability["volatility"], 3

                )

                if pd.notna(

                    stability["volatility"]

                )

                else np.nan

            ),

            "upward_confirmation": round(

                confirmation[

                    "upward_confirmation"

                ],

                3,

            ),

            "downward_confirmation": round(

                confirmation[

                    "downward_confirmation"

                ],

                3,

            ),

            "games_analyzed": len(valid_player),

        })



        add_result(result)



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

        st.info(

            "No players match these filters."

        )

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

            st.session_state.get(

                page_key, 1

            ),

            1,

        ),

        total_pages,

    )



    st.session_state[page_key] = (

        current_page

    )



    start = (

        current_page - 1

    ) * PAGE_SIZE



    end = start + PAGE_SIZE



    st.dataframe(

        data.iloc[start:end].copy(),

        hide_index=True,

        use_container_width=True,

        height=570,

        column_config=column_config or {},

    )



    previous_col, page_col, next_col = (

        st.columns([1, 1.4, 1])

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

                f"Page {current_page} of "

                f"{total_pages}"

                "</p>"

            ),

            unsafe_allow_html=True,

        )



    with next_col:



        if st.button(

            "Next →",

            key=f"{table_key}_next",

            disabled=(

                current_page == total_pages

            ),

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

        font=dict(

            color=COLORS["text"]

        ),

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





# ============================================================

# LOAD DATA

# ============================================================



source = next(

    (

        path

        for path in DATA_PATHS

        if path.is_file()

    ),

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



        st.error(

            "No NFL records found."

        )



        st.stop()



except Exception as error:



    st.error(

        f"Data error: {error}"

    )



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

        NFL Fantasy Analytics & Waiver Opportunity |

        2022–2025

    </div>



    <div>

        Analyze fantasy production, offensive usage,

        player consistency, and weekly waiver trends.

    </div>



    <div class="pills">

        <span class="pill">Player Consistency Model</span>

        <span class="pill">Waiver Wire Alerts</span>

        <span class="pill">Four Waiver Flags</span>

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



st.sidebar.markdown("**Player Consistency Model**")

st.sidebar.caption(

    "Weekly Usage Volatility"

)



st.sidebar.markdown("**Waiver Wire Alerts**")

st.sidebar.caption(

    "Fantasy PPR & Waiver Opportunity"

)





# ============================================================

# TABS

# ============================================================



about_tab, consistency_tab, waiver_tab = st.tabs(

    [

        "About RosterWatch",

        "Player Consistency Model",

        "Waiver Wire Alerts",

    ]

)



with about_tab:



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

        "**Player Consistency Model** measures week-to-week changes "

    "in offensive involvement."

    )



    st.write(

        "**Waiver Wire Alerts** prioritize PPR fantasy points "

    "while evaluating snap share, opportunity "

    "share, volatility, and trend confirmation."

    )



    st.write("**Waiver Wire Alerts:** Every eligible player is assigned one of four waiver flags; Player Consistency Model rankings do not override the classification.")





    # ============================================================

    # ANALYTICS MODEL EXPLANATION

    # ============================================================



    with st.expander(

        "Our Analytics Models — Equations and Thresholds",

        expanded=True,

    ):



        st.subheader(

            "Player Consistency Model — Weekly Volatility"

        )



        st.latex(

            r"V_t="

        r"w_S|S_t-S_{t-1}|"

        r"+w_O|O_t-O_{t-1}|"

        r"+w_P|P_t-P_{t-1}|"

        r"+w_R|R_t-R_{t-1}|"

        )



        st.write(

            "S = snap share; "

        "O = opportunity share; "

        "P = passing-down share; "

        "R = red-zone share."

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

            r"\bar{V}="

        r"\frac{1}{m}"

        r"\sum_{t\in T}V_t"

        )

        st.write(
            "T = the weeks in the selected range with a valid comparison "
        "(the player also played the week before, and all four shares exist in both games); "
        "m = the number of those weeks."
        )

        st.latex(
            r"Q=100\left(0.50\,C+0.25\,S_{\text{rank}}+0.25\,O_{\text{rank}}\right),"
        r"\qquad C=1-\operatorname{pct}(\bar{V})"
        )

        st.write(
            "**Q** is the consistency quality score used to sort the Most Consistent table. "
        "C, S_rank and O_rank are percentile ranks (0–1) among consistency-eligible players "
        "at the same position and season: lower average volatility, higher average snap share "
        "and more carries + targets per game rank higher."
        )



        st.subheader(

            "Waiver Wire Alerts — Fantasy Waiver Opportunity"

        )



        st.latex(

            r"W=100("

        r"0.40F"

        r"+0.20O"

        r"+0.15S"

        r"+0.15V"

        r"+0.10T)"

        r"\times E"

        )



        st.write(

            "**F:** Fantasy PPR performance."

        )



        st.write(

            "**O:** Opportunity share strength."

        )



        st.write(

            "**S:** Offensive snap share strength."

        )



        st.write(

            "**V:** Fantasy and usage stability."

        )



        st.write(

            "**T:** Trend confirmation."

        )

        st.write(
            "**E:** Evidence factor, 0.80 + 0.20 × min(1, valid games ÷ (week − 1)). "
        "It lowers the score for players with few valid games; players with a "
        "limited history use a stricter version."
        )



        st.markdown(

            "#### Component Weights"

        )



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



        st.markdown(

            "#### Position Benchmarks"

        )



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



        st.info("Waiver Wire Alerts use heuristic benchmarks and four directional waiver flags. Ownership is based on the ESPN snapshot, not verified historical weekly availability.")








with consistency_tab:



    # ============================================================

    # MODEL 1 DATA

    # ============================================================






    st.header(

        "Player Consistency Model — Weekly Usage Volatility"

    )



    model1_weeks = st.slider(

        "Select Player Consistency Model Week Range",

        min_value=1,

        max_value=18,

        value=(1, 18),

        step=1,

        key="model1_week_range",

    )



    st.caption(

        f"Analyzing NFL Weeks "

    f"{model1_weeks[0]}–"

    f"{model1_weeks[1]} "

    f"for the {season} season."

    )



    full_model1_input = usage.loc[

        usage["season"] == season

    ].copy()



    full_model1_weekly, _ = (

        calculate_volatility(

            full_model1_input

        )

    )



    model1_input = full_model1_weekly.loc[

        full_model1_weekly["week"].between(

            model1_weeks[0],

            model1_weeks[1],

        )

    ].copy()



    model1_summary = summarize_model1_window(

        model1_input

    )



    filtered_usage = model1_input.copy()

    filtered_summary = model1_summary.copy()



    if position != "All Positions":



        filtered_usage = filtered_usage.loc[

            filtered_usage["position"]

            == position

        ]



        filtered_summary = filtered_summary.loc[

            filtered_summary["position"]

            == position

        ]





    # ============================================================

    # MUST START PLAYER IDS

    # ============================================================



    # Exactly the players listed in the

    # Model 1 consistency rankings.



    consistent_player_ids = tuple(

        sorted(

            model1_summary.loc[

                model1_summary[

                    "consistency_eligible"

                ],

                "gsis_id",

            ].astype(str)

        )

    )





    # ============================================================

    # SEASON OVERVIEW

    # ============================================================



    st.divider()



    st.header("Season Overview")



    st.caption(

        f"{season} Season | "

    f"Player Consistency Model Weeks "

    f"{model1_weeks[0]}–"

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

            [

                "gsis_id",

                "player_name",

                "position",

            ]

        ]

        .drop_duplicates("gsis_id")

        .sort_values("player_name")

    )



    if not players.empty:



        labels = {

            row.gsis_id: (

                f"{row.player_name} · "

            f"{row.position}"

            )

            for row in players.itertuples(

                index=False

            )

        }



        player_id = st.selectbox(

            "Select NFL Player",

            players["gsis_id"].tolist(),

            format_func=lambda pid: labels.get(

                pid,

                str(pid),

            ),

        )



        player_data = (

            filtered_usage.loc[

                filtered_usage["gsis_id"]

                == player_id

            ]

            .sort_values("week")

        )



        player_summary = (

            filtered_summary.loc[

                filtered_summary["gsis_id"]

                == player_id

            ]

        )



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

            player_summary[

                "volatility_score"

            ].iloc[0]

            if not player_summary.empty

            else np.nan

        )



        d.metric(

            "Avg Weekly Volatility",

            (

                f"{volatility:.3f}"

                if pd.notna(volatility)

                else "N/A"

            ),

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



        with st.expander(

            "View Weekly Player Data"

        ):



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



    st.header(

        "Player Consistency Model — Player Volatility Rankings"

    )



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

        (

            filtered_summary["position"]

            == ranking_position

        )

        & filtered_summary[

            "volatility_score"

        ].notna()

    ].copy()



    eligible = ranked.loc[

        ranked["consistency_eligible"]

    ].copy()



    a, b, c, d = st.columns(4)



    a.metric(

        "Players Scored",

        len(ranked),

    )



    b.metric(

        "Average Volatility",

        (

            f"{ranked['volatility_score'].mean():.3f}"

            if not ranked.empty

            else "N/A"

        ),

    )



    c.metric(

        "Highest Volatility",

        (

            f"{ranked['volatility_score'].max():.3f}"

            if not ranked.empty

            else "N/A"

        ),

    )



    d.metric(

        "Consistency Eligible",

        len(eligible),

    )



    left, right = st.columns(2)



    with left:



        st.subheader(

            "📈 Most Volatile Players"

        )



        volatile_display = (

            ranked.sort_values(

                [

                    "volatility_score",

                    "player_name",

                ],

                ascending=[

                    False,

                    True,

                ],

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



        st.subheader(

            "📉 Most Consistent Players"

        )



        consistent_display = (

            eligible.sort_values(

                [

                    "consistency_quality_score",

                    "volatility_score",

                ],

                ascending=[

                    False,

                    True,

                ],

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

            CONSISTENCY_THRESHOLDS[

                ranking_position

            ]

        )



        st.caption(

            f"Eligibility: {min_weeks}+ weeks, "

        f"{MIN_COMPARISONS}+ comparisons, "

        f"{min_snap:.0%}+ average snaps, "

        f"{min_volume}+ opportunities/game."

        )








with waiver_tab:



    # ============================================================

    # MODEL 2 — WAIVER DASHBOARD

    # ============================================================






    st.header(

        "Waiver Wire Alerts — Fantasy Waiver Opportunity"

    )



    st.write(

        "Waiver Wire Alerts prioritize fantasy PPR production "

    "while analyzing offensive snap share, "

    "opportunity share, weekly volatility, "

    "and recent performance trends."

    )



    st.write("**Four flags:** Must Pick Up, Slow Riser, Slow Drop, and Must Drop. All ownership-eligible players returned by Waiver Wire Alerts receive one flag.")

    with st.expander("How players qualify for each waiver flag"):
        st.markdown(
            f"""
**Who is scored:** players matched to the ESPN snapshot and rostered in at most {MAX_PERCENT_OWNED:g}% of leagues.

**Trends** compare a player's last 2–4 valid games with his earlier games, scaled from −1 to +1.
*Usage rising* means the snap or opportunity share trend is at least +0.10 (falling: −0.10 or lower).
*Fantasy rising* means the PPR trend is at least +0.20 (falling: −0.20 or lower).

| Flag | Qualifies when |
|---|---|
| 🟢 Must Pick Up | Fantasy and usage rising, at least 67% of recent games confirm the rise, waiver score ≥ 72, recent PPR ≥ 85% of the position benchmark, 5+ valid games |
| 🔵 Slow Riser | Fantasy and usage rising with at least 50% of recent games confirming and score ≥ 52, **or** usage rising with the PPR trend above −0.30 while recent PPR is still below the benchmark, **or** mixed signals that lean upward |
| 🟠 Slow Drop | Fantasy and usage falling with at least 50% of recent games confirming the drop, **or** mixed signals that lean downward |
| 🔴 Must Drop | Fantasy and usage falling, at least 67% of recent games confirm the drop, waiver score ≤ 43, recent PPR under 60% of the benchmark |

**Full history** means at least 4 valid games, covering at least 75% of the games he played, with 2+ games before the recent window.
Players with less are scored on the data they have and never receive Must Pick Up or Must Drop.
Position benchmarks (PPR points, snap share, opportunity share) are listed in the model explanation on the About RosterWatch tab.
"""
        )



    selected_week = st.selectbox(

        "Analyze Through NFL Week",

        list(range(4, 19)),

        index=14,

        key="waiver_week",

    )



    st.caption(

        f"Analyzing player history from Week 1 "

    f"through Week {selected_week} "

    f"of {season}."

    )



    # Model 2 uses the season-specific ESPN ownership snapshot.
    # Only verified GSIS-ID matches with percentOwned <= 50 are eligible.
    ownership_source = next((p for p in OWNERSHIP_PATHS if p.is_file()), None)
    if ownership_source is None:
        st.error("Missing espn_ownership.csv. Place it in data/processed/ or the repository root.")
        st.stop()
    try:
        espn_ownership = load_ownership(str(ownership_source))
    except Exception as error:
        st.error(f"Unable to load ESPN ownership: {error}")
        st.stop()

    season_ownership = espn_ownership.loc[
        espn_ownership["ownership_season"].eq(season),
        ["gsis_id", "percentOwned"],
    ].copy()
    season_ownership["gsis_id"] = season_ownership["gsis_id"].astype("string").str.strip()
    eligible_ids = set(season_ownership["gsis_id"].dropna())

    # The original dashboard prioritizes Parquet. Some locally stored Parquet
    # versions may have different player IDs or season coverage from the supplied
    # ESPN ownership CSV. Select the existing weekly-usage source with the most
    # ownership-matched, recently active players for this season/week.
    model2_usage = usage
    model2_source = source
    best_matches = -1
    source_diagnostics = []
    for candidate in DATA_PATHS:
        if not candidate.is_file():
            continue
        try:
            candidate_data = usage if candidate == source else load_data(str(candidate))
            scoped = candidate_data.loc[
                candidate_data["season"].eq(season)
                & candidate_data["week"].le(selected_week)
            ]
            if scoped.empty:
                matched_active = 0
            else:
                last_rows = scoped.sort_values("week").groupby("gsis_id").tail(1)
                active_ids = set(last_rows.loc[
                    last_rows["week"].ge(selected_week - 1), "gsis_id"
                ].astype("string").str.strip())
                matched_active = len(active_ids & eligible_ids)
            source_diagnostics.append(f"{candidate.name}: {matched_active} matches")
            if matched_active > best_matches:
                best_matches = matched_active
                model2_usage = candidate_data
                model2_source = candidate
        except Exception as exc:
            source_diagnostics.append(f"{candidate.name}: load failed ({exc})")

    # Restrict input BEFORE scoring, so Model 2 never includes >50%-owned players.
    model2_usage = model2_usage.copy()
    model2_usage["gsis_id"] = model2_usage["gsis_id"].astype("string").str.strip()
    model2_usage = model2_usage.loc[
        model2_usage["gsis_id"].isin(eligible_ids)
    ].copy()

    all_waiver_results = calculate_waiver_model(
        model2_usage, season, selected_week, consistent_player_ids
    )
    all_waiver_results["gsis_id"] = all_waiver_results["gsis_id"].astype("string").str.strip()
    all_waiver_results = all_waiver_results.merge(
        season_ownership, on="gsis_id", how="inner", validate="many_to_one"
    )
    all_waiver_results = all_waiver_results.loc[
        all_waiver_results["percentOwned"].notna()
        & all_waiver_results["percentOwned"].between(0, MAX_PERCENT_OWNED)
    ].reset_index(drop=True)

    st.caption(
        f"Waiver Wire Alerts ownership limit: ≤ {MAX_PERCENT_OWNED:g}% | "
    f"{len(all_waiver_results)} eligible players | "
    f"Weekly source: {model2_source.relative_to(ROOT)}. "
    "Ownership is a collected snapshot, not historical weekly ownership."
    )
    if all_waiver_results.empty:
        st.warning(
            "No eligible Waiver Wire Alerts players for this selection. "
        f"Season {season}, through Week {selected_week}; "
        f"{len(season_ownership)} ownership-eligible IDs. "
        "Available source matches: " + "; ".join(source_diagnostics)
        )


    if position != "All Positions":



        all_waiver_results = (

            all_waiver_results.loc[

                all_waiver_results["position"]

                == position

            ].copy()

        )





    if not all_waiver_results.empty:



        # --------------------------------------------------------

        # FLAG COUNTS

        # --------------------------------------------------------



        a, b, c, d = st.columns(4)
        for metric_col, flag, emoji in zip(
            (a, b, c, d), WAIVER_FLAGS, ("🟢", "🔵", "🟠", "🔴")
        ):
            metric_col.metric(
                f"{emoji} {flag}",
                int(all_waiver_results["waiver_flag"].eq(flag).sum()),
            )

        # --------------------------------------------------------

        # FILTERS — ALL FLAGS FIXED

        # --------------------------------------------------------



        flag_filter = st.selectbox(

            "Waiver Flag",

            ["All Flags"] + WAIVER_FLAGS,

            key="waiver_flag_filter",

        )



        if flag_filter == "All Flags":



            waiver_results = (

                all_waiver_results.copy()

            )



        else:



            waiver_results = (

                all_waiver_results.loc[

                    all_waiver_results[

                        "waiver_flag"

                    ]

                    == flag_filter

                ].copy()

            )



        waiver_results = (

            waiver_results.reset_index(

                drop=True

            )

        )



        # --------------------------------------------------------

        # SORTING

        # --------------------------------------------------------



        waiver_results["sort_order"] = (

            waiver_results["waiver_flag"]

            .map(FLAG_ORDER)

        )



        waiver_results = (

            waiver_results.sort_values(

                [

                    "sort_order",

                    "waiver_score",

                    "player_name",

                ],

                ascending=[

                    True,

                    False,

                    True,

                ],

                na_position="last",

            )

            .reset_index(drop=True)

        )



        st.caption(

            f"Showing {len(waiver_results):,} "

        f"of {len(all_waiver_results):,} "

        f"players."

        )



        # --------------------------------------------------------

        # WAIVER RANKINGS

        # --------------------------------------------------------



        st.subheader(

            "Weekly Waiver Rankings"

        )



        waiver_display = waiver_results[

            [

                "player_name",

                "position",

                "team",

                "waiver_flag",

                "waiver_score",

                "percentOwned",

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

                model1_weeks,

            ),

            column_config={

                "player_name": "Player",

                "position": "Position",

                "team": "Team",

                "percentOwned": st.column_config.NumberColumn(
                    "ESPN Owned (%)", format="%.1f%%"
                ),

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



        st.subheader(

            "Player Waiver Trend Explorer"

        )



        choices = (

            waiver_results[

                [

                    "gsis_id",

                    "player_name",

                    "position",

                ]

            ]

            .drop_duplicates("gsis_id")

            .sort_values("player_name")

        )



        if not choices.empty:



            labels = {

                row.gsis_id: (

                    f"{row.player_name} · "

                f"{row.position}"

                )

                for row in choices.itertuples(

                    index=False

                )

            }



            selected_player_id = st.selectbox(

                "Inspect Player",

                choices["gsis_id"].tolist(),

                format_func=lambda pid: labels.get(

                    pid,

                    str(pid),

                ),

                key="waiver_player",

            )



            selected_result = (

                waiver_results.loc[

                    waiver_results["gsis_id"]

                    == selected_player_id

                ].iloc[0]

            )



            history = usage.loc[

                (usage["season"] == season)

                & (

                    usage["gsis_id"]

                    == selected_player_id

                )

                & (

                    usage["week"]

                    <= selected_week

                )

            ].sort_values("week")



            st.markdown(

                f"### {selected_result['player_name']}"

            )



            st.write(

                f"**Waiver Flag:** "

            f"{selected_result['waiver_flag']}"

            )



            score = (

                selected_result["waiver_score"]

            )



            a, b, c, d = st.columns(4)



            a.metric(

                "Waiver Score",

                (

                    f"{score:.1f}/100"

                    if pd.notna(score)

                    else "N/A"

                ),

            )



            b.metric(

                "Recent PPR Average",

                (

                    f"{selected_result['recent_ppr']:.2f}"

                    if pd.notna(

                        selected_result["recent_ppr"]

                    )

                    else "N/A"

                ),

            )



            c.metric(

                "Season PPR Average",

                (

                    f"{selected_result['season_ppr']:.2f}"

                    if pd.notna(

                        selected_result["season_ppr"]

                    )

                    else "N/A"

                ),

            )



            d.metric(

                "Usage Volatility",

                (

                    f"{selected_result['volatility']:.3f}"

                    if pd.notna(

                        selected_result["volatility"]

                    )

                    else "N/A"

                ),

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

            if pd.notna(score):

                # PLAYER TREND ANALYSIS

                # ------------------------------------------------



                st.markdown(

                    "#### Player Trend Analysis"

                )



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



            else:



                st.info(

                    "Insufficient data to calculate "

                "a complete waiver score."

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

            "No players available for Waiver Wire Alerts "

        "at this season, position, and week."

        )





# ============================================================

# FOOTER

# ============================================================



st.divider()



st.markdown(

    "### RosterWatch 🏈"

)



st.caption(

    "Player Consistency Model: Weekly Usage Volatility | "

    "Waiver Wire Alerts: Fantasy PPR Waiver Opportunity"

)





st.caption(

    "2022–2025 | RB · WR · TE | Weeks 1–18"

)



st.caption(

    "Built with Python, Pandas, Plotly, "

    "and Streamlit."

)
