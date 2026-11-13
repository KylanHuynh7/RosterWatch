
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

WAIVER_FLAGS = [
    "Must Pick Up",
    "Slow Riser",
    "Slow Drop",
    "Must Drop",
    "Stable / No Clear Signal",
    "Insufficient Data",
]

# Position-specific usage benchmarks.
# These are heuristic starting values, not fitted parameters.

WAIVER_SETTINGS = {
    "RB": {
        "role_snap": 0.50,
        "role_opp": 0.18,
        "role_volume": 10.0,
        "trend_snap": 0.08,
        "trend_opp": 0.045,
        "trend_volume": 2.5,
        "weights": {
            "snap_pct": 0.25,
            "opp_share": 0.40,
            "opportunities": 0.35,
        },
    },
    "WR": {
        "role_snap": 0.65,
        "role_opp": 0.17,
        "role_volume": 6.0,
        "trend_snap": 0.09,
        "trend_opp": 0.04,
        "trend_volume": 2.0,
        "weights": {
            "snap_pct": 0.30,
            "opp_share": 0.40,
            "opportunities": 0.30,
        },
    },
    "TE": {
        "role_snap": 0.60,
        "role_opp": 0.13,
        "role_volume": 4.0,
        "trend_snap": 0.08,
        "trend_opp": 0.035,
        "trend_volume": 1.5,
        "weights": {
            "snap_pct": 0.30,
            "opp_share": 0.40,
            "opportunities": 0.30,
        },
    },
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
    background:
        linear-gradient(115deg, #10243b, #15395b);
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
# MODEL 1 — WEEKLY VOLATILITY
# ============================================================

@st.cache_data(show_spinner="Calculating volatility...")
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
            .notna()
            .all(axis=1)
            & player[VOLATILITY_METRICS]
            .shift()
            .notna()
            .all(axis=1)
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
# MODEL 2 — HELPERS
# ============================================================

def safe_mean(series):

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

    return float(np.clip(value, 0, 1))


def role_strength(frame, settings):

    if frame.empty:
        return 0.0

    snap = safe_mean(frame["snap_pct"])
    opp = safe_mean(frame["opp_share"])
    volume = safe_mean(frame["opportunities"])

    snap_score = clip01(
        snap / settings["role_snap"]
    )

    opp_score = clip01(
        opp / settings["role_opp"]
    )

    volume_score = clip01(
        volume / settings["role_volume"]
    )

    return float(
        0.30 * snap_score
        + 0.40 * opp_score
        + 0.30 * volume_score
    )


def usage_consistency(frame, settings):

    """
    Evaluate consistency across all observed
    weeks, including the size of the player's role.
    """

    if len(frame) < 2:
        return 0.0

    variation_scores = []

    metrics = [
        ("snap_pct", 0.05, 0.30),
        ("opp_share", 0.03, 0.40),
        ("opportunities", 1.0, 0.30),
    ]

    for metric, floor, weight in metrics:

        values = frame[metric].dropna()

        if len(values) < 2:
            return 0.0

        cv = (
            values.std(ddof=0)
            / max(float(values.mean()), floor)
        )

        stability = 1 / (1 + cv)

        variation_scores.append(
            weight * stability
        )

    statistical_consistency = sum(
        variation_scores
    )

    meaningful_weeks = (
        (
            (frame["snap_pct"]
             >= settings["role_snap"] * 0.50)
            | (
                frame["opportunities"]
                >= settings["role_volume"] * 0.50
            )
        ).mean()
    )

    return clip01(
        statistical_consistency
        * meaningful_weeks
    )


def full_season_trend(frame, settings):

    """
    Analyze every available player week.

    Combines:
    1. Recent usage vs all earlier usage
    2. Season-long linear trend
    3. Position-specific normalization
    """

    frame = frame.sort_values("week").copy()

    n = len(frame)

    if n < 4:
        return 0.0, 0.0, 0.0

    recent_count = min(
        4,
        max(2, n // 3),
    )

    recent = frame.tail(recent_count)
    previous = frame.iloc[:-recent_count]

    if len(previous) < 2:
        return 0.0, 0.0, 0.0

    thresholds = {
        "snap_pct": settings["trend_snap"],
        "opp_share": settings["trend_opp"],
        "opportunities": settings["trend_volume"],
    }

    weights = settings["weights"]

    recent_trend = 0.0
    season_trend = 0.0

    for metric, threshold in thresholds.items():

        baseline = safe_mean(previous[metric])
        current = safe_mean(recent[metric])

        change = current - baseline

        normalized_change = float(
            np.clip(
                change / threshold,
                -1,
                1,
            )
        )

        recent_trend += (
            weights[metric] * normalized_change
        )

        # Full-season linear trend.
        # Every observed week contributes.
        valid = frame[
            ["week", metric]
        ].dropna()

        if len(valid) >= 3:

            slope = np.polyfit(
                valid["week"].to_numpy(dtype=float),
                valid[metric].to_numpy(dtype=float),
                1,
            )[0]

            # Normalize approximately four
            # weeks of change.
            slope_signal = float(
                np.clip(
                    (slope * 4) / threshold,
                    -1,
                    1,
                )
            )

        else:
            slope_signal = 0.0

        season_trend += (
            weights[metric] * slope_signal
        )

    # Recent trends matter more, but the entire
    # historical trajectory remains influential.
    combined = (
        0.65 * recent_trend
        + 0.35 * season_trend
    )

    return (
        float(np.clip(combined, -1, 1)),
        float(np.clip(recent_trend, -1, 1)),
        float(np.clip(season_trend, -1, 1)),
    )


def trend_persistence(frame):

    """
    Evaluate persistence using the complete
    season history through the selected week.

    Compares every observed game to the
    previous observed game, while avoiding
    comparisons across long gaps.
    """

    frame = frame.sort_values("week").copy()

    if len(frame) < 3:
        return 0.0, 0.0

    snap_diff = frame["snap_pct"].diff()
    opp_diff = frame["opp_share"].diff()
    volume_diff = frame["opportunities"].diff()

    valid = (
        frame["week"].diff().between(1, 2)
    )

    up = (
        (snap_diff > 0)
        & (opp_diff > 0)
        & (volume_diff > 0)
        & valid
    )

    down = (
        (snap_diff < 0)
        & (opp_diff < 0)
        & (volume_diff < 0)
        & valid
    )

    count = int(valid.sum())

    if count == 0:
        return 0.0, 0.0

    return (
        float(up.sum() / count),
        float(down.sum() / count),
    )


def recent_trend_confirmation(
    frame,
    trend,
):

    """
    Determine whether recent games confirm
    the direction found across the season.
    """

    frame = frame.sort_values("week")

    n = len(frame)

    recent_count = min(
        4,
        max(2, n // 3),
    )

    recent = frame.tail(recent_count)
    previous = frame.iloc[:-recent_count]

    if previous.empty:
        return 0.0

    baseline_snap = safe_mean(
        previous["snap_pct"]
    )

    baseline_opp = safe_mean(
        previous["opp_share"]
    )

    baseline_volume = safe_mean(
        previous["opportunities"]
    )

    if trend >= 0:

        confirmed = (
            (recent["snap_pct"] > baseline_snap)
            & (recent["opp_share"] > baseline_opp)
            & (
                recent["opportunities"]
                >= baseline_volume
            )
        )

    else:

        confirmed = (
            (recent["snap_pct"] < baseline_snap)
            & (recent["opp_share"] < baseline_opp)
            & (
                recent["opportunities"]
                <= baseline_volume
            )
        )

    return float(confirmed.mean())


# ============================================================
# MODEL 2 — FULL HISTORY WAIVER MODEL
# ============================================================

@st.cache_data(
    show_spinner="Analyzing all previous NFL weeks..."
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

    columns = [
        "gsis_id",
        "player_name",
        "position",
        "team",
        "waiver_flag",
        "waiver_score",
        "season_role",
        "recent_role",
        "consistency",
        "trend",
        "persistence",
        "games_analyzed",
    ]

    results = []

    for player_id, player in data.groupby("gsis_id"):

        player = (
            player.sort_values("week")
            .copy()
        )

        latest = player.iloc[-1]

        # Avoid making a current recommendation
        # based on a player absent for several weeks.
        if (
            as_of_week - int(latest["week"])
            > 1
        ):
            continue

        position = latest["position"]
        settings = WAIVER_SETTINGS[position]

        result = {
            "gsis_id": player_id,
            "player_name": latest["player_name"],
            "position": position,
            "team": latest["team"],
            "waiver_flag": "Insufficient Data",
            "waiver_score": np.nan,
            "season_role": np.nan,
            "recent_role": np.nan,
            "consistency": np.nan,
            "trend": np.nan,
            "persistence": np.nan,
            "games_analyzed": len(player),
        }

        required = [
            "snap_pct",
            "opp_share",
            "opportunities",
        ]

        # Do not silently drop invalid historical
        # rows and mistake missing usage for a
        # legitimate improvement or decline.
        valid_player = player.dropna(
            subset=required
        ).copy()

        if (
            len(valid_player) < 4
            or len(valid_player) / len(player) < 0.75
        ):
            results.append(result)
            continue

        # --------------------------------------------
        # 1. SEASON-LONG ROLE — 20%
        # --------------------------------------------

        season_role = role_strength(
            valid_player,
            settings,
        )

        # --------------------------------------------
        # 2. RECENT ROLE — 25%
        # --------------------------------------------

        recent_count = min(
            4,
            max(2, len(valid_player) // 3),
        )

        recent = valid_player.tail(
            recent_count
        )

        recent_role = role_strength(
            recent,
            settings,
        )

        # --------------------------------------------
        # 3. FULL-SEASON CONSISTENCY — 15%
        # --------------------------------------------

        consistency = usage_consistency(
            valid_player,
            settings,
        )

        # --------------------------------------------
        # 4. FULL-SEASON USAGE TREND — 25%
        # --------------------------------------------

        (
            trend,
            recent_trend,
            season_trend,
        ) = full_season_trend(
            valid_player,
            settings,
        )

        # --------------------------------------------
        # 5. TREND PERSISTENCE — 15%
        # --------------------------------------------

        (
            upward_rate,
            downward_rate,
        ) = trend_persistence(
            valid_player
        )

        historical_persistence = (
            upward_rate
            if trend >= 0
            else downward_rate
        )

        recent_confirmation = (
            recent_trend_confirmation(
                valid_player,
                trend,
            )
        )

        persistence = (
            0.35 * historical_persistence
            + 0.65 * recent_confirmation
        )

        # --------------------------------------------
        # 6. DATA COVERAGE
        # --------------------------------------------

        # Treat absent player-week records
        # cautiously. A missing row can represent
        # a bye, injury, or unavailable data.
        #
        # Penalize sparse evidence without
        # interpreting every missing week as
        # zero offensive usage.

        expected_weeks = as_of_week

        coverage = min(
            1.0,
            len(valid_player)
            / max(expected_weeks - 1, 1),
        )

        evidence_factor = (
            0.80 + 0.20 * coverage
        )

        # --------------------------------------------
        # 7. WAIVER SCORE
        # --------------------------------------------

        # Season role: 20%
        # Recent role: 25%
        # Trend: 25%
        # Consistency: 15%
        # Persistence: 15%
        #
        # A neutral trend contributes 50/100
        # to the trend component.
        #
        # Score represents a heuristic assessment
        # of offensive opportunity, not a
        # probability of fantasy success.

        trend_score = (
            50 + 50 * trend
        )

        # Persistence should not excessively
        # penalize players with stable roles.
        # Neutral trends receive a neutral
        # persistence component.
        persistence_score = (
            50
            + 50 * persistence
            if trend >= 0.15
            else (
                50 - 50 * persistence
                if trend <= -0.15
                else 50
            )
        )

        raw_score = (
            0.20 * season_role * 100
            + 0.25 * recent_role * 100
            + 0.25 * trend_score
            + 0.15 * consistency * 100
            + 0.15 * persistence_score
        )

        waiver_score = float(
            np.clip(
                raw_score * evidence_factor,
                0,
                100,
            )
        )

        # --------------------------------------------
        # 8. WAIVER FLAGS
        # --------------------------------------------

        # Strong classifications require:
        # - Meaningful current involvement
        # - A clear historical trend
        # - Recent confirmation
        #
        # This avoids labeling a one-week
        # spike as a Must Pick Up.

        if (
            waiver_score >= 75
            and recent_role >= 0.70
            and trend >= 0.35
            and recent_confirmation >= 0.67
            and len(valid_player) >= 5
        ):

            flag = "Must Pick Up"

        elif (
            trend <= -0.35
            and recent_role <= 0.40
            and recent_confirmation >= 0.67
            and waiver_score <= 45
        ):

            flag = "Must Drop"

        elif (
            waiver_score >= 58
            and recent_role >= 0.45
            and trend >= 0.15
            and recent_confirmation >= 0.50
        ):

            flag = "Slow Riser"

        elif (
            trend <= -0.15
            and recent_confirmation >= 0.50
            and (
                waiver_score <= 55
                or recent_role <= 0.45
            )
        ):

            flag = "Slow Drop"

        else:

            flag = "Stable / No Clear Signal"

        result.update({
            "waiver_flag": flag,
            "waiver_score": round(
                waiver_score, 1
            ),
            "season_role": round(
                season_role, 3
            ),
            "recent_role": round(
                recent_role, 3
            ),
            "consistency": round(
                consistency, 3
            ),
            "trend": round(
                trend, 3
            ),
            "persistence": round(
                persistence, 3
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
        NFL Weekly Volatility &amp; Waiver Opportunity |
        2022–2025
    </div>

    <div class="hero-description">
        Measure player role stability, analyze
        season-long offensive usage, and identify
        potential waiver opportunities.
    </div>

    <div class="pills">
        <span class="pill">Model 1 · Volatility</span>
        <span class="pill">Model 2 · Waiver Score</span>
        <span class="pill">Full-Season Analysis</span>
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

model1_weeks = st.sidebar.slider(
    "Model 1 — Week Range",
    min_value=1,
    max_value=18,
    value=(1, 18),
    step=1,
)

st.sidebar.divider()

st.sidebar.markdown("**Model 1**")
st.sidebar.caption("Weekly Usage Volatility")

st.sidebar.markdown("**Model 2**")
st.sidebar.caption(
    "Full-Season Waiver Opportunity"
)


# ============================================================
# MODEL 1 DATA
# ============================================================

model1_input = usage.loc[
    (usage["season"] == season)
    & usage["week"].between(
        model1_weeks[0],
        model1_weeks[1],
    )
].copy()

model1_weekly, model1_summary = (
    calculate_volatility(model1_input)
)

filtered_usage = model1_weekly.copy()
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
    "RosterWatch analyzes weekly NFL offensive "
    "usage to evaluate player role stability "
    "and identify potential waiver opportunities."
)

st.write(
    "**Model 1** measures week-to-week changes "
    "in offensive involvement."
)

st.write(
    "**Model 2** analyzes all available historical "
    "weeks through the selected NFL week, combining "
    "season-long usage, current role, consistency, "
    "trend direction, and persistence."
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
        "Model 2 — Full-Season Waiver Opportunity"
    )

    st.write(
        "Model 2 analyzes every available player "
        "week from Week 1 through the selected week."
    )

    st.markdown("#### Waiver Score")

    st.latex(
        r"W=100\left("
        r"0.20L+0.25R+0.25T"
        r"+0.15C+0.15P"
        r"\right)"
    )

    st.write(
        "**L:** Season-long offensive role strength "
        "(0–1)."
    )

    st.write(
        "**R:** Recent offensive role strength "
        "(0–1)."
    )

    st.write(
        "**T:** Trend score normalized to 0–1, "
        "with 0.5 representing neutral usage."
    )

    st.write(
        "**C:** Season-long usage consistency "
        "(0–1)."
    )

    st.write(
        "**P:** Persistence component normalized "
        "to 0–1."
    )

    st.caption(
        "The implementation also applies a small "
        "data-coverage adjustment to the raw score."
    )

    st.markdown("#### Component Weights")

    st.table(
        pd.DataFrame({
            "Component": [
                "Season-long role",
                "Recent role",
                "Usage trend",
                "Consistency",
                "Persistence",
            ],
            "Weight": [
                "20%",
                "25%",
                "25%",
                "15%",
                "15%",
            ],
        })
    )

    st.markdown("#### Position Benchmarks")

    benchmark_table = pd.DataFrame({
        pos: {
            "Snap Share": cfg["role_snap"],
            "Opportunity Share": cfg["role_opp"],
            "Opportunities/Game": cfg["role_volume"],
        }
        for pos, cfg in WAIVER_SETTINGS.items()
    }).T

    st.table(benchmark_table)

    st.info(
        "The score measures historical offensive "
        "opportunity using heuristic thresholds. "
        "It is not a calibrated probability, and "
        "actual waiver availability is not verified."
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
            pid,
            str(pid),
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
st.header("Model 2 — Waiver Opportunity")

st.write(
    "Evaluate a player's entire season history "
    "through the selected week. The model combines "
    "long-term involvement with recent role changes "
    "to identify potential waiver opportunities."
)

selected_week = st.selectbox(
    "Analyze Through NFL Week",
    list(range(4, 19)),
    index=14,
    key="waiver_week",
)

st.caption(
    f"Analyzing all available player history "
    f"from Week 1 through Week {selected_week}."
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
    # FLAG FILTER
    # --------------------------------------------------------

    flag_filter = st.selectbox(
        "Waiver Flag",
        ["All Flags"] + WAIVER_FLAGS,
        key="waiver_flag_filter",
    )

    if flag_filter != "All Flags":

        waiver_results = waiver_results.loc[
            waiver_results["waiver_flag"]
            == flag_filter
        ].copy()

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    flag_order = {
        "Must Pick Up": 0,
        "Slow Riser": 1,
        "Slow Drop": 2,
        "Must Drop": 3,
        "Stable / No Clear Signal": 4,
        "Insufficient Data": 5,
    }

    waiver_results["sort_order"] = (
        waiver_results["waiver_flag"]
        .map(flag_order)
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
    # FIVE-COLUMN WAIVER TABLE
    # --------------------------------------------------------

    st.subheader("Weekly Waiver Rankings")

    waiver_display = waiver_results[
        [
            "player_name",
            "position",
            "team",
            "waiver_flag",
            "waiver_score",
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
                    help=(
                        "0–100 offensive opportunity "
                        "score based on all available "
                        "weeks through the selected week."
                    ),
                )
            ),
        },
    )

    # --------------------------------------------------------
    # PLAYER WAIVER TREND EXPLORER
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
                pid,
                str(pid),
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

        score = selected_result["waiver_score"]

        st.metric(
            "Waiver Score",
            (
                f"{score:.1f}/100"
                if pd.notna(score)
                else "N/A"
            ),
        )

        # Model 2 retains only the PPR graph.
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

        st.caption(
            "Fantasy points are shown for context "
            "and do not directly affect the "
            "waiver score or flag."
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
    "Model 2: Full-Season Waiver Opportunity"
)

st.caption(
    "2022–2025 | RB · WR · TE | Weeks 1–18"
)


st.caption(
    "Built with Python, Pandas, Plotly, and Streamlit."
)