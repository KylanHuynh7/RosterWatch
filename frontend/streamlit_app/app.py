
from pathlib import Path
import textwrap

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RosterWatch 🏈 | NFL Volatility Analytics",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"

POSITIONS = ["RB", "WR", "TE"]
WEEKS = list(range(1, 19))
TOP_N = 15

METRICS = [
    "snap_pct",
    "opp_share",
    "passdown_share",
    "redzone_share",
]

METRIC_LABELS = {
    "snap_pct": "Snap Percentage",
    "opp_share": "Opportunity Share",
    "passdown_share": "Passing-Down Share",
    "redzone_share": "Red-Zone Share",
}

METRIC_COLORS = {
    "snap_pct": "#38BDF8",
    "opp_share": "#4ADE80",
    "passdown_share": "#F59E0B",
    "redzone_share": "#A78BFA",
}

# ORIGINAL MODEL 1 WEIGHTS — UNCHANGED

POSITION_WEIGHTS = {
    "RB": {
        "snap_pct": 0.30,
        "opp_share": 0.45,
        "passdown_share": 0.15,
        "redzone_share": 0.10,
    },
    "WR": {
        "snap_pct": 0.35,
        "opp_share": 0.35,
        "passdown_share": 0.20,
        "redzone_share": 0.10,
    },
    "TE": {
        "snap_pct": 0.40,
        "opp_share": 0.30,
        "passdown_share": 0.20,
        "redzone_share": 0.10,
    },
}

# ORIGINAL CONSISTENCY THRESHOLDS — UNCHANGED

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

MIN_WEEKLY_COMPARISONS = 8

COLORS = {
    "blue": "#38BDF8",
    "green": "#4ADE80",
    "orange": "#F59E0B",
    "purple": "#A78BFA",
    "panel": "#13243B",
    "text": "#E2E8F0",
    "grid": "#2A405A",
    "muted": "#152841",
    "border": "#2B4967",
    "muted_text": "#A7BFD7",
}


# ============================================================
# 2. WEBSITE STYLING
# ============================================================

st.markdown(
    """
<style>

.stApp {
    background: radial-gradient(
        circle at top right,
        #17375b 0%,
        #08111f 48%
    );
    color: #e2e8f0;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: #f8fafc !important;
}

[data-testid="stSidebar"] {
    background: #101e32;
}

[data-testid="stMetric"] {
    background: #152841;
    border: 1px solid #2b4967;
    border-radius: 14px;
    padding: 18px;
    transition: transform .18s ease, border-color .18s ease;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    border-color: #38bdf8;
}

[data-testid="stMetricValue"] {
    color: #4ade80;
}

.hero {
    border: 1px solid #2b4967;
    border-radius: 18px;
    padding: 28px 32px;
    background: linear-gradient(115deg, #10243b, #15395b);
    margin-bottom: 24px;
}

.hero-title {
    font-size: clamp(2rem, 4vw, 3.2rem);
    font-weight: 800;
    letter-spacing: -.04em;
    color: #f8fafc;
}

.hero-subtitle {
    color: #a7bfd7;
    margin: 8px 0 16px;
}

.hero-description {
    line-height: 1.7;
    max-width: 850px;
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
    padding: 7px 12px;
    color: #c8e9ff;
    font-size: .8rem;
}

.pill:hover {
    background: #23537c;
}

@media (prefers-reduced-motion: reduce) {
    * {
        transition: none !important;
    }
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# 3. LOAD MATT'S NFL DATA
# ============================================================

DATA_SOURCES = [
    DATA_DIR / "weekly_usage.parquet",
    ROOT / "RosterWatch-matt" / "notebooks" / "weekly_usage.parquet",
    DATA_DIR / "weekly_usage.csv",
    ROOT / "RosterWatch-matt" / "notebooks" / "weekly_usage.csv",
]


@st.cache_data(show_spinner="Loading Matt's NFL data...")
def load_data(path):

    if str(path).endswith(".parquet"):
        usage = pd.read_parquet(path)
    else:
        usage = pd.read_csv(path)

    required = {
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
    }

    missing = required - set(usage.columns)

    if missing:
        raise ValueError(
            "Matt's dataset is missing columns: "
            + ", ".join(sorted(missing))
        )

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
    ]

    if "fantasy_points_ppr" in usage.columns:
        numeric_columns.append("fantasy_points_ppr")
    else:
        usage["fantasy_points_ppr"] = np.nan

    for column in numeric_columns:
        usage[column] = pd.to_numeric(
            usage[column],
            errors="coerce",
        )

    usage["position"] = (
        usage["position"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    usage["gsis_id"] = usage["gsis_id"].astype("string")

    usage = usage.loc[
        usage["season"].between(2022, 2025)
        & usage["week"].between(1, 18)
        & usage["position"].isin(POSITIONS)
    ].copy()

    usage = usage.dropna(
        subset=["gsis_id", "player_name"]
    )

    usage["season"] = usage["season"].astype(int)
    usage["week"] = usage["week"].astype(int)

    usage = usage.drop_duplicates()

    usage["opportunities"] = (
        usage["targets"].fillna(0)
        + usage["carries"].fillna(0)
    )

    return usage


source = next(
    (path for path in DATA_SOURCES if path.exists()),
    None,
)

if source is None:

    st.error("Matt's NFL dataset could not be found.")

    st.code(
        "mkdir -p data/processed\n"
        "cp RosterWatch-matt/notebooks/weekly_usage.parquet "
        "data/processed/weekly_usage.parquet",
        language="bash",
    )

    st.stop()

try:
    usage = load_data(str(source))

except Exception as error:
    st.error(f"Could not load RosterWatch data: {error}")
    st.stop()

if usage.empty:
    st.warning("No NFL usage records were found.")
    st.stop()


# ============================================================
# 4. ORIGINAL WEEK-TO-WEEK VOLATILITY MODEL
# ============================================================
#
# MODEL 1 IS UNCHANGED.
#
# V_t =
#   wS * abs(S_t - S_t-1)
# + wO * abs(O_t - O_t-1)
# + wP * abs(P_t - P_t-1)
# + wR * abs(R_t - R_t-1)
#
# Only consecutive NFL weeks are compared.
# Missing values are not treated as zero volatility.
#
# ============================================================

@st.cache_data(show_spinner="Calculating weekly volatility...")
def calculate_volatility(data):

    weekly_results = []
    season_results = []

    for (season, player_id), player in data.groupby(
        ["season", "gsis_id"]
    ):

        player = player.sort_values("week")

        position = player["position"].iloc[-1]

        if position not in POSITION_WEIGHTS:
            continue

        weights = POSITION_WEIGHTS[position]
        thresholds = CONSISTENCY_THRESHOLDS[position]

        weekly = (
            player.groupby("week")
            .agg({
                "snap_pct": "mean",
                "opp_share": "mean",
                "passdown_share": "mean",
                "redzone_share": "mean",
                "targets": "mean",
                "carries": "mean",
                "fantasy_points_ppr": "mean",
            })
            .sort_index()
        )

        weekly["opportunities"] = (
            weekly["targets"].fillna(0)
            + weekly["carries"].fillna(0)
        )

        # NFL usage metrics only.
        # Sleeper fantasy points are NOT included.

        changes = weekly[METRICS].diff().abs()

        # Compare consecutive NFL weeks only.

        consecutive = (
            weekly.index.to_series().diff() == 1
        )

        changes.loc[~consecutive, :] = np.nan

        weekly["weekly_volatility"] = (
            changes.mul(pd.Series(weights))
            .sum(
                axis=1,
                min_count=len(METRICS),
            )
        )

        weekly["snap_change"] = changes["snap_pct"]
        weekly["opportunity_change"] = changes["opp_share"]
        weekly["passdown_change"] = changes["passdown_share"]
        weekly["redzone_change"] = changes["redzone_share"]

        latest = player.iloc[-1]

        for week, row in weekly.iterrows():

            weekly_results.append({
                "season": season,
                "week": int(week),
                "gsis_id": player_id,
                "player_name": latest["player_name"],
                "position": position,
                "team": latest["team"],
                "weekly_volatility": row["weekly_volatility"],
                "snap_change": row["snap_change"],
                "opportunity_change": row["opportunity_change"],
                "passdown_change": row["passdown_change"],
                "redzone_change": row["redzone_change"],
                "snap_pct": row["snap_pct"],
                "opp_share": row["opp_share"],
                "passdown_share": row["passdown_share"],
                "redzone_share": row["redzone_share"],
                "targets": row["targets"],
                "carries": row["carries"],
                "opportunities": row["opportunities"],
                "fantasy_points_ppr": row["fantasy_points_ppr"],
            })

        valid_scores = weekly[
            "weekly_volatility"
        ].dropna()

        if valid_scores.empty:
            continue

        recorded_weeks = len(weekly)
        weekly_comparisons = len(valid_scores)

        avg_snap_pct = weekly["snap_pct"].mean()
        avg_opportunities = weekly["opportunities"].mean()

        consistency_eligible = (
            recorded_weeks >= thresholds["min_weeks"]
            and weekly_comparisons >= MIN_WEEKLY_COMPARISONS
            and pd.notna(avg_snap_pct)
            and avg_snap_pct >= thresholds["min_snap_pct"]
            and pd.notna(avg_opportunities)
            and avg_opportunities
            >= thresholds["min_opportunities_per_week"]
        )

        season_results.append({
            "season": season,
            "gsis_id": player_id,
            "player_name": latest["player_name"],
            "position": position,
            "team": latest["team"],
            "weeks": recorded_weeks,
            "weekly_comparisons": weekly_comparisons,
            "volatility_score": valid_scores.mean(),
            "avg_snap_pct": avg_snap_pct,
            "avg_opportunities_per_week": avg_opportunities,
            "consistency_eligible": bool(
                consistency_eligible
            ),
            "total_ppr": weekly["fantasy_points_ppr"].sum(
                min_count=1
            ),
            "avg_ppr": weekly["fantasy_points_ppr"].mean(),
        })

    weekly_scores = pd.DataFrame(weekly_results)
    season_scores = pd.DataFrame(season_results)

    return weekly_scores, season_scores


weekly_volatility, volatility = calculate_volatility(usage)


# ============================================================
# 5. ORIGINAL CONSISTENCY QUALITY SCORE
# ============================================================

def calculate_consistency_quality(data):

    if data.empty:
        return data.copy()

    scored = data.copy()

    volatility_rank = scored[
        "volatility_score"
    ].rank(
        ascending=True,
        pct=True,
    )

    snap_rank = scored[
        "avg_snap_pct"
    ].rank(
        ascending=True,
        pct=True,
    )

    opportunity_rank = scored[
        "avg_opportunities_per_week"
    ].rank(
        ascending=True,
        pct=True,
    )

    scored["consistency_quality_score"] = (
        0.50 * (1 - volatility_rank)
        + 0.25 * snap_rank
        + 0.25 * opportunity_rank
    ) * 100

    return scored


# ============================================================
# 6. NEW UI HELPER FUNCTIONS
# ============================================================

def table_height(data):
    return min(max((len(data) + 1) * 35 + 3, 100), 650)


def format_number(value, decimals=1):

    if pd.isna(value):
        return "—"

    return f"{value:,.{decimals}f}"


def format_percent(value):

    if pd.isna(value):
        return "—"

    return f"{value:.1%}"


def csv_data(data):
    return data.to_csv(index=False).encode("utf-8")


def style_figure(fig, height=365):

    fig.update_layout(
        paper_bgcolor=COLORS["panel"],
        plot_bgcolor=COLORS["panel"],
        font=dict(
            color=COLORS["text"],
            size=13,
        ),
        height=height,
        margin=dict(
            l=55,
            r=20,
            t=65,
            b=55,
        ),
        hoverlabel=dict(
            bgcolor=COLORS["muted"],
            bordercolor=COLORS["border"],
            font=dict(color=COLORS["text"]),
        ),
        xaxis=dict(
            title="NFL Week",
            tickmode="array",
            tickvals=WEEKS,
            range=[0.5, 18.5],
            gridcolor=COLORS["grid"],
            zeroline=False,
        ),
        yaxis=dict(
            gridcolor=COLORS["grid"],
            zeroline=False,
            rangemode="tozero",
        ),
    )

    return fig


def shade_selected_weeks(fig, first_week, last_week):

    if first_week == 1 and last_week == 18:
        return

    fig.add_vrect(
        x0=first_week - 0.5,
        x1=last_week + 0.5,
        fillcolor="rgba(56,189,248,0.10)",
        line_width=0,
        layer="below",
    )


def mark_highlighted_week(fig, week):

    fig.add_vrect(
        x0=week - 0.45,
        x1=week + 0.45,
        fillcolor="rgba(255,255,255,0.07)",
        line=dict(
            color=COLORS["muted_text"],
            width=1,
            dash="dot",
        ),
        layer="below",
    )


def label_week(fig, week, value, label, top):

    if week <= 2:
        anchor = "left"
    elif week >= 17:
        anchor = "right"
    else:
        anchor = "center"

    if pd.isna(value):

        fig.add_annotation(
            x=week,
            y=top,
            text=label,
            showarrow=False,
            yanchor="top",
            xanchor=anchor,
            font=dict(
                size=12,
                color=COLORS["text"],
            ),
            bgcolor=COLORS["muted"],
            bordercolor=COLORS["muted_text"],
            borderpad=4,
        )

        return

    fig.add_annotation(
        x=week,
        y=value,
        text=label,
        showarrow=True,
        arrowhead=0,
        arrowwidth=1,
        arrowcolor=COLORS["muted_text"],
        ax=0,
        ay=-38 if value < top * 0.75 else 38,
        xanchor=anchor,
        font=dict(
            size=12,
            color=COLORS["text"],
        ),
        bgcolor=COLORS["muted"],
        bordercolor=COLORS["muted_text"],
        borderpad=4,
    )


# ============================================================
# 7. HERO HEADER
# ============================================================

hero_html = """
<div class="hero">

    <div class="hero-title">
        RosterWatch 🏈
    </div>

    <div class="hero-subtitle">
        NFL Weekly Volatility Analytics | 2022–2025
    </div>

    <div class="hero-description">
        Track how NFL players' offensive roles change
        from week to week. Explore playing time,
        offensive opportunities, passing-down involvement,
        and red-zone usage with position-aware analytics.
    </div>

    <div class="pills">
        <span class="pill">Weekly Volatility Model</span>
        <span class="pill">Position-Aware Scoring</span>
        <span class="pill">Matt's NFL Data</span>
        <span class="pill">Sleeper PPR Tracking</span>
        <span class="pill">2022–2025 Seasons</span>
        <span class="pill">RB / WR / TE</span>
        <span class="pill">Weeks 1–18</span>
    </div>

</div>
"""

hero_html = textwrap.dedent(hero_html)

hero_html = "\n".join(
    line.strip()
    for line in hero_html.splitlines()
    if line.strip()
)

st.markdown(
    hero_html,
    unsafe_allow_html=True,
)


# ============================================================
# 8. ABOUT ROSTERWATCH
# ============================================================

st.header("About RosterWatch")

st.write(
    "RosterWatch is an interactive NFL analytics dashboard "
    "that tracks how players' offensive involvement changes "
    "from one week to the next. Rather than calculating "
    "volatility using season-long standard deviation, "
    "the model measures changes between consecutive weeks."
)

st.write(
    "The model uses position-specific weights for running "
    "backs, wide receivers, and tight ends. Its consistency "
    "rankings also account for meaningful playing time "
    "and offensive opportunities."
)

st.write(
    "NFL usage data and Sleeper PPR fantasy points are "
    "loaded from Matt's dataset. Fantasy points are "
    "displayed separately and do not affect the model."
)


# ============================================================
# 9. ORIGINAL MODEL EXPLANATION
# ============================================================

st.header("Our Analytics Model")

with st.container(border=True):

    st.subheader("Weekly Volatility Model")

    st.write(
        "The Weekly Volatility Model measures how much a "
        "player's offensive involvement changes compared "
        "with the previous NFL week."
    )

    st.markdown("#### Weekly Volatility Equation")

    st.latex(
        r"V_t = w_S|S_t-S_{t-1}|"
        r"+w_O|O_t-O_{t-1}|"
        r"+w_P|P_t-P_{t-1}|"
        r"+w_R|R_t-R_{t-1}|"
    )

    st.markdown(
        """
**Variables:**

- **Vₜ:** Volatility score for the current week
- **S:** Offensive snap percentage
- **O:** Opportunity share
- **P:** Passing-down share
- **R:** Red-zone share
- **w:** Position-specific weight
"""
    )

    st.markdown("#### Position-Specific Weights")

    selected_formula_position = st.selectbox(
        "View Position Formula",
        POSITIONS,
        key="formula_position",
    )

    weights = POSITION_WEIGHTS[selected_formula_position]

    st.latex(
        rf"V_{{{selected_formula_position},t}}="
        rf"{weights['snap_pct']:.2f}|\Delta S_t|"
        rf"+{weights['opp_share']:.2f}|\Delta O_t|"
        rf"+{weights['passdown_share']:.2f}|\Delta P_t|"
        rf"+{weights['redzone_share']:.2f}|\Delta R_t|"
    )

    st.divider()

    st.markdown("#### Average Weekly Volatility")

    st.write(
        "To compare players across the season, "
        "RosterWatch averages their valid weekly "
        "volatility scores."
    )

    st.latex(
        r"\bar{V}=\frac{1}{m}"
        r"\sum_{t \in T} V_t"
    )

    st.caption(
        "T contains weeks with valid consecutive-week "
        "comparisons, and m is the number of those comparisons."
    )

    left, right = st.columns(2)

    with left:

        st.markdown("**Higher Weekly Volatility**")

        st.write(
            "The player's offensive role changed "
            "more significantly from the previous week."
        )

    with right:

        st.markdown("**Lower Weekly Volatility**")

        st.write(
            "The player's offensive involvement "
            "remained more similar to the previous week."
        )


with st.expander(
    "How are Weekly Volatility and Consistency calculated?"
):

    st.markdown("#### Model Weights")

    st.table(
        pd.DataFrame({
            "Metric": [
                "Snap Percentage",
                "Opportunity Share",
                "Passing-Down Share",
                "Red-Zone Share",
            ],
            "RB": ["30%", "45%", "15%", "10%"],
            "WR": ["35%", "35%", "20%", "10%"],
            "TE": ["40%", "30%", "20%", "10%"],
        })
    )

    st.markdown("#### Consistency Eligibility")

    st.table(
        pd.DataFrame({
            "Requirement": [
                "Minimum Recorded Weeks",
                "Minimum Valid Weekly Comparisons",
                "Minimum Average Snap %",
                "Minimum Carries + Targets / Week",
            ],
            "RB": ["10", "8", "50%", "10"],
            "WR": ["10", "8", "65%", "5"],
            "TE": ["10", "8", "60%", "4"],
        })
    )

    st.markdown("#### Consistency Quality Score")

    st.latex(
        r"Q=100(0.50C+0.25S+0.25O)"
    )

    st.write(
        "C rewards lower average weekly volatility, "
        "S rewards higher snap participation, and "
        "O rewards more offensive opportunities. "
        "Each component is ranked relative to eligible "
        "players at the same position and season."
    )

    st.info(
        "Week 1 has no previous week for comparison. "
        "Missing weeks prevent comparisons with the "
        "next recorded week. Missing data is never "
        "treated as zero volatility."
    )

    st.caption(
        "Model weights, thresholds, and quality-score "
        "coefficients are experimental."
    )

    st.warning(
        "Sleeper PPR points are NOT used in either "
        "the volatility or consistency model."
    )


# ============================================================
# 10. SIDEBAR — NEW WEEK DROPDOWNS
# ============================================================

st.sidebar.title("Control Center")

season_options = sorted(
    usage["season"].unique().tolist(),
    reverse=True,
)

season = st.sidebar.selectbox(
    "NFL Season",
    season_options,
)

season_last_week = int(
    usage.loc[
        usage["season"] == season,
        "week",
    ].max()
)

season_weeks = [
    week for week in WEEKS
    if week <= season_last_week
]

st.sidebar.markdown("### Select NFL Weeks")

from_column, to_column = st.sidebar.columns(2)

with from_column:

    first_week = st.selectbox(
        "From Week",
        season_weeks,
        index=0,
        key="first_week",
    )

to_options = [
    week for week in season_weeks
    if week >= first_week
]

with to_column:

    last_week = st.selectbox(
        "To Week",
        to_options,
        index=len(to_options) - 1,
        key="last_week",
    )

full_season = (
    first_week == 1
    and last_week == season_last_week
)

range_label = (
    "Full Season"
    if full_season
    else f"Weeks {first_week}–{last_week}"
)

st.sidebar.caption(
    "Select a week range to update player totals, "
    "average volatility, and rankings."
)

st.sidebar.caption(
    "Volatility uses only consecutive-week comparisons "
    "fully contained in the selected range."
)

position = st.sidebar.selectbox(
    "Position",
    ["All Positions"] + POSITIONS,
)

st.sidebar.divider()

st.sidebar.markdown("**Weekly Volatility Model**")

st.sidebar.caption(
    "Original position-weighted absolute changes "
    "in offensive usage."
)

st.sidebar.markdown("**Ranking Method**")

st.sidebar.caption(
    "Average weekly volatility and original "
    "consistency eligibility thresholds."
)

st.sidebar.markdown("**Data Coverage**")

st.sidebar.caption(
    "2022–2025 · RB / WR / TE · Weeks 1–18"
)


# ============================================================
# 11. FILTER DATA AND RECALCULATE WINDOW SUMMARIES
# ============================================================
#
# The original model has already calculated every
# weekly volatility score.
#
# For a selected week range, we use those original
# weekly values and summarize only comparisons
# whose two weeks are inside the selected range.
#
# This does not change the volatility equation.
#
# ============================================================

@st.cache_data(show_spinner=False)
def calculate_window_scores(
    data,
    weekly_scores,
    first,
    last,
):

    window_usage = data.loc[
        data["week"].between(first, last)
    ].copy()

    window_weekly = weekly_scores.loc[
        weekly_scores["week"].between(first, last)
    ].copy()

    # A comparison ending at the first selected week
    # would involve a week outside the selected range.

    window_weekly.loc[
        window_weekly["week"] <= first,
        "weekly_volatility",
    ] = np.nan

    results = []

    for (season_value, player_id), player in (
        window_usage.groupby(["season", "gsis_id"])
    ):

        player = player.sort_values("week")

        position_value = player["position"].iloc[-1]

        if position_value not in CONSISTENCY_THRESHOLDS:
            continue

        thresholds = CONSISTENCY_THRESHOLDS[position_value]

        player_weekly = window_weekly.loc[
            (window_weekly["season"] == season_value)
            & (window_weekly["gsis_id"] == player_id)
        ]

        valid_scores = (
            player_weekly["weekly_volatility"].dropna()
        )

        if valid_scores.empty:
            continue

        weekly_aggregated = (
            player.groupby("week")
            .agg({
                "snap_pct": "mean",
                "targets": "mean",
                "carries": "mean",
                "fantasy_points_ppr": "mean",
            })
        )

        weekly_aggregated["opportunities"] = (
            weekly_aggregated["targets"].fillna(0)
            + weekly_aggregated["carries"].fillna(0)
        )

        recorded_weeks = len(weekly_aggregated)
        comparisons = len(valid_scores)

        avg_snap = weekly_aggregated["snap_pct"].mean()

        avg_opportunities = (
            weekly_aggregated["opportunities"].mean()
        )

        eligible = (
            recorded_weeks >= thresholds["min_weeks"]
            and comparisons >= MIN_WEEKLY_COMPARISONS
            and pd.notna(avg_snap)
            and avg_snap >= thresholds["min_snap_pct"]
            and pd.notna(avg_opportunities)
            and avg_opportunities
            >= thresholds["min_opportunities_per_week"]
        )

        latest = player.iloc[-1]

        results.append({
            "season": season_value,
            "gsis_id": player_id,
            "player_name": latest["player_name"],
            "position": position_value,
            "team": latest["team"],
            "weeks": recorded_weeks,
            "weekly_comparisons": comparisons,
            "volatility_score": valid_scores.mean(),
            "avg_snap_pct": avg_snap,
            "avg_opportunities_per_week": avg_opportunities,
            "consistency_eligible": bool(eligible),
            "total_ppr": weekly_aggregated[
                "fantasy_points_ppr"
            ].sum(min_count=1),
            "avg_ppr": weekly_aggregated[
                "fantasy_points_ppr"
            ].mean(),
        })

    if not results:
        return pd.DataFrame(columns=[
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
            "total_ppr",
            "avg_ppr",
        ])

    return pd.DataFrame(results)


# Preserve the original full-season results exactly.

if full_season:
    window_volatility = volatility.copy()
else:
    window_volatility = calculate_window_scores(
        usage,
        weekly_volatility,
        first_week,
        last_week,
    )

season_usage = usage.loc[
    usage["season"] == season
].copy()

filtered_usage = season_usage.loc[
    season_usage["week"].between(
        first_week,
        last_week,
    )
].copy()

season_volatility = window_volatility.loc[
    window_volatility["season"] == season
].copy()

filtered_volatility = season_volatility.copy()

filtered_weekly = weekly_volatility.loc[
    weekly_volatility["season"] == season
].copy()

if position != "All Positions":

    filtered_usage = filtered_usage.loc[
        filtered_usage["position"] == position
    ]

    filtered_volatility = filtered_volatility.loc[
        filtered_volatility["position"] == position
    ]

    filtered_weekly = filtered_weekly.loc[
        filtered_weekly["position"] == position
    ]


# ============================================================
# 12. SEASON OVERVIEW
# ============================================================

st.divider()

st.header("📊 Season Overview")

st.caption(
    f"{season} NFL Season · {range_label} · {position}"
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
# 13. PLAYER EXPLORER
# ============================================================

st.divider()

st.header("🔎 Player Explorer")

st.write(
    "Select a player to explore weekly offensive usage, "
    "Sleeper fantasy points, and week-to-week volatility."
)

players = (
    filtered_usage[
        ["gsis_id", "player_name", "position"]
    ]
    .dropna(subset=["gsis_id", "player_name"])
    .drop_duplicates(subset=["gsis_id"])
    .sort_values(["player_name", "position"])
)

if players.empty:

    st.info("No players match these filters.")

else:

    labels = {
        row.gsis_id: f"{row.player_name} · {row.position}"
        for row in players.itertuples(index=False)
    }

    player_options = players["gsis_id"].tolist()

    previous_selection = st.session_state.get(
        "selected_player_id"
    )

    player_id = st.selectbox(
        "Select an NFL Player",
        player_options,
        index=(
            player_options.index(previous_selection)
            if previous_selection in player_options
            else 0
        ),
        format_func=lambda pid: labels.get(
            pid,
            str(pid),
        ),
    )

    st.session_state["selected_player_id"] = player_id

    # Selected range for totals.

    player_data = (
        filtered_usage.loc[
            filtered_usage["gsis_id"] == player_id
        ]
        .sort_values("week")
        .copy()
    )

    # Full season for charts.

    player_season = (
        season_usage.loc[
            season_usage["gsis_id"] == player_id
        ]
        .sort_values("week")
        .copy()
    )

    player_weekly = (
        filtered_weekly.loc[
            filtered_weekly["gsis_id"] == player_id
        ]
        .sort_values("week")
        .copy()
    )

    player_summary = filtered_volatility.loc[
        filtered_volatility["gsis_id"] == player_id
    ]

    player_name = player_data["player_name"].iloc[0]
    player_position = player_data["position"].iloc[-1]

    st.subheader(
        f"{player_name} — {season} Weekly Usage"
    )

    st.caption(
        f"Totals and averages: {range_label}. "
        "Charts display the full season."
    )

    # ========================================================
    # PLAYER SUMMARY METRICS
    # ========================================================

    m1, m2, m3, m4 = st.columns(4)

    mean_snap = player_data["snap_pct"].mean()

    m1.metric(
        "Average Snap Percentage",
        format_percent(mean_snap),
    )

    m2.metric(
        f"{range_label} Targets",
        f"{player_data['targets'].sum():,.0f}",
    )

    m3.metric(
        f"{range_label} Carries",
        f"{player_data['carries'].sum():,.0f}",
    )

    if not player_summary.empty:

        summary = player_summary.iloc[0]

        player_score = summary["volatility_score"]

        m4.metric(
            "Avg Weekly Volatility",
            f"{player_score:.3f}",
        )

        eligible = bool(
            summary["consistency_eligible"]
        )

        if eligible:

            st.success(
                "This player meets the original "
                "consistency ranking requirements."
            )

        else:

            st.caption(
                "This player does not meet all "
                "consistency leaderboard thresholds."
            )

    else:

        m4.metric(
            "Avg Weekly Volatility",
            "N/A",
        )

    st.caption(
        f"Weekly volatility uses {player_position}-specific "
        "weights and consecutive NFL weeks."
    )

    # ========================================================
    # HIGHLIGHTED WEEK AND CHART STYLE
    # ========================================================

    valid_weeks = player_weekly.loc[
        player_weekly["week"].between(
            first_week,
            last_week,
        )
        & player_weekly["weekly_volatility"].notna(),
        "week",
    ]

    default_week = (
        int(valid_weeks.max())
        if not valid_weeks.empty
        else last_week
    )

    controls1, controls2 = st.columns([2, 1])

    with controls1:

        highlighted_week = st.slider(
            "Highlight NFL Week",
            min_value=1,
            max_value=18,
            value=default_week,
            key=f"highlight_{season}_{player_id}",
        )

    with controls2:

        chart_style = st.selectbox(
            "Chart Style",
            [
                "Line + markers",
                "Bars",
                "Line only",
            ],
        )

    # ========================================================
    # NEW: HIGHLIGHTED WEEK AT A GLANCE
    # ========================================================

    st.markdown("### Highlighted Week at a Glance")

    usage_by_week = (
        player_season.groupby("week")[
            [
                "snap_pct",
                "opp_share",
                "targets",
                "carries",
                "fantasy_points_ppr",
            ]
        ]
        .mean()
        .sort_index()
    )

    weekly_by_week = (
        player_weekly
        .set_index("week")
        .sort_index()
    )

    if highlighted_week not in usage_by_week.index:

        st.info(
            f"{player_name} has no recorded usage "
            f"for Week {highlighted_week}."
        )

    else:

        now = usage_by_week.loc[highlighted_week]

        previous_week = highlighted_week - 1

        previous_exists = (
            previous_week in usage_by_week.index
        )

        before = (
            usage_by_week.loc[previous_week]
            if previous_exists
            else None
        )

        st.caption(
            f"Week {highlighted_week} compared with "
            f"Week {previous_week}."
            if previous_exists
            else "No consecutive previous-week record "
            "is available."
        )

        def show_glance_metric(
            column,
            label,
            metric,
            percentage=False,
        ):

            value = now[metric]

            if percentage:
                displayed = format_percent(value)
            else:
                displayed = format_number(value, 0)

            delta = None

            if before is not None:

                previous_value = before[metric]

                if (
                    pd.notna(value)
                    and pd.notna(previous_value)
                ):

                    difference = value - previous_value

                    if percentage:
                        delta = (
                            f"{difference * 100:+.1f} pp"
                        )
                    else:
                        delta = f"{difference:+.0f}"

            column.metric(
                label,
                displayed,
                delta=delta,
                delta_color="off",
            )

        g1, g2, g3, g4, g5 = st.columns(5)

        show_glance_metric(
            g1,
            "Snap %",
            "snap_pct",
            True,
        )

        show_glance_metric(
            g2,
            "Opportunity Share",
            "opp_share",
            True,
        )

        show_glance_metric(
            g3,
            "Targets",
            "targets",
        )

        show_glance_metric(
            g4,
            "Carries",
            "carries",
        )

        week_score = (
            weekly_by_week.loc[
                highlighted_week,
                "weekly_volatility",
            ]
            if highlighted_week in weekly_by_week.index
            else np.nan
        )

        g5.metric(
            "Weekly Volatility",
            f"{week_score:.3f}"
            if pd.notna(week_score)
            else "—",
        )

    # ========================================================
    # CHART BUILDER WITH NEW ANNOTATIONS
    # ========================================================

    def make_chart(
        data,
        metric,
        title,
        ylabel,
        color,
        percentage=False,
    ):

        series = (
            data.groupby("week")[metric]
            .mean()
            .reindex(WEEKS)
        )

        fig = go.Figure()

        hover = (
            "Week %{x}<br>"
            + ylabel
            + (
                ": %{y:.1%}"
                if percentage
                else ": %{y:.1f}"
            )
            + "<extra></extra>"
        )

        if chart_style == "Bars":

            fig.add_trace(
                go.Bar(
                    x=WEEKS,
                    y=series.values,
                    marker_color=color,
                    hovertemplate=hover,
                )
            )

        else:

            fig.add_trace(
                go.Scatter(
                    x=WEEKS,
                    y=series.values,
                    mode=(
                        "lines+markers"
                        if chart_style == "Line + markers"
                        else "lines"
                    ),
                    connectgaps=False,
                    line=dict(
                        color=color,
                        width=3,
                    ),
                    marker=dict(size=7),
                    hovertemplate=hover,
                )
            )

        shade_selected_weeks(
            fig,
            first_week,
            last_week,
        )

        mark_highlighted_week(
            fig,
            highlighted_week,
        )

        selected_value = series.loc[highlighted_week]

        if pd.notna(selected_value):

            shown = (
                f"{selected_value:.0%}"
                if percentage
                else f"{selected_value:.1f}"
            )

        else:
            shown = "No data"

        maximum = series.max(skipna=True)

        top = (
            1.0
            if percentage
            else max(
                float(maximum) * 1.15,
                1.0,
            )
            if pd.notna(maximum)
            else 1.0
        )

        label_week(
            fig,
            highlighted_week,
            selected_value,
            f"Wk {highlighted_week} · {shown}",
            top,
        )

        style_figure(fig)

        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=17),
                x=0.03,
            ),
            showlegend=False,
            yaxis_title=ylabel,
        )

        if percentage:

            fig.update_yaxes(
                tickformat=".0%",
                range=[0, 1.05],
            )

        return fig

    # ========================================================
    # PLAYER USAGE CHARTS
    # ========================================================

    chart1, chart2 = st.columns(2)

    with chart1:

        st.plotly_chart(
            make_chart(
                player_season,
                "snap_pct",
                "Offensive Snap Percentage",
                "Share of Offensive Snaps",
                COLORS["blue"],
                True,
            ),
            use_container_width=True,
        )

    with chart2:

        st.plotly_chart(
            make_chart(
                player_season,
                "opp_share",
                "Opportunity Share",
                "Share of Team Opportunities",
                COLORS["green"],
                True,
            ),
            use_container_width=True,
        )

    chart3, chart4 = st.columns(2)

    with chart3:

        st.plotly_chart(
            make_chart(
                player_season,
                "targets",
                "Weekly Targets",
                "Targets",
                COLORS["orange"],
            ),
            use_container_width=True,
        )

    with chart4:

        st.plotly_chart(
            make_chart(
                player_season,
                "carries",
                "Weekly Carries",
                "Carries",
                COLORS["purple"],
            ),
            use_container_width=True,
        )

    # ========================================================
    # SLEEPER FANTASY POINTS — PRESERVED
    # ========================================================

    st.subheader("Sleeper Fantasy Points")

    st.write(
        "Weekly PPR fantasy points from Matt's dataset. "
        "These values are displayed separately and do "
        "not influence the volatility model."
    )

    st.plotly_chart(
        make_chart(
            player_season,
            "fantasy_points_ppr",
            "Weekly Sleeper PPR Fantasy Points",
            "PPR Fantasy Points",
            COLORS["green"],
        ),
        use_container_width=True,
    )

    # ========================================================
    # WEEKLY VOLATILITY CHART — ORIGINAL MODEL
    # ========================================================

    st.subheader("Weekly Volatility")

    st.write(
        "This chart shows how much the player's "
        "NFL offensive usage changed compared "
        "with the previous consecutive NFL week."
    )

    weekly_series = (
        player_weekly
        .set_index("week")["weekly_volatility"]
        .reindex(WEEKS)
    )

    volatility_fig = go.Figure()

    volatility_fig.add_trace(
        go.Bar(
            x=WEEKS,
            y=weekly_series.values,
            marker_color=COLORS["blue"],
            hovertemplate=(
                "Week %{x}<br>"
                "Weekly Volatility: %{y:.3f}"
                "<extra></extra>"
            ),
        )
    )

    shade_selected_weeks(
        volatility_fig,
        first_week,
        last_week,
    )

    mark_highlighted_week(
        volatility_fig,
        highlighted_week,
    )

    selected_volatility = weekly_series.loc[
        highlighted_week
    ]

    max_volatility = weekly_series.max(skipna=True)

    volatility_top = (
        max(float(max_volatility) * 1.15, 0.1)
        if pd.notna(max_volatility)
        else 0.1
    )

    label_week(
        volatility_fig,
        highlighted_week,
        selected_volatility,
        (
            f"Wk {highlighted_week} · "
            f"{selected_volatility:.3f}"
            if pd.notna(selected_volatility)
            else f"Wk {highlighted_week} · No score"
        ),
        volatility_top,
    )

    style_figure(
        volatility_fig,
        height=400,
    )

    volatility_fig.update_layout(
        title=dict(
            text="Week-to-Week Usage Volatility",
            x=0.03,
        ),
        yaxis_title="Weekly Volatility Score",
        showlegend=False,
    )

    st.plotly_chart(
        volatility_fig,
        use_container_width=True,
    )

    # ========================================================
    # NEW: WEEKLY VOLATILITY BREAKDOWN
    # ========================================================

    st.markdown("#### Highlighted Week Breakdown")

    selected_week_data = player_weekly.loc[
        player_weekly["week"] == highlighted_week
    ]

    if selected_week_data.empty:

        st.info(
            "No player record is available for "
            f"Week {highlighted_week}."
        )

    else:

        selected_row = selected_week_data.iloc[0]

        selected_score = selected_row[
            "weekly_volatility"
        ]

        if pd.isna(selected_score):

            st.info(
                "No valid consecutive-week comparison "
                "is available for this week."
            )

        else:

            player_weights = POSITION_WEIGHTS[
                player_position
            ]

            components = [
                (
                    "Snap Percentage",
                    "snap_change",
                    "snap_pct",
                ),
                (
                    "Opportunity Share",
                    "opportunity_change",
                    "opp_share",
                ),
                (
                    "Passing-Down Share",
                    "passdown_change",
                    "passdown_share",
                ),
                (
                    "Red-Zone Share",
                    "redzone_change",
                    "redzone_share",
                ),
            ]

            breakdown = []

            for label, change_column, weight_key in components:

                change = selected_row[change_column]
                weight = player_weights[weight_key]

                contribution = change * weight

                breakdown.append({
                    "Metric": label,
                    "Absolute Change": change,
                    "Position Weight": weight,
                    "Weighted Contribution": contribution,
                })

            breakdown_df = pd.DataFrame(breakdown)

            st.dataframe(
                breakdown_df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Absolute Change": st.column_config.NumberColumn(
                        "Absolute Change",
                        format="%.3f",
                    ),
                    "Position Weight": st.column_config.NumberColumn(
                        "Weight",
                        format="%.0f%%",
                    ),
                    "Weighted Contribution": st.column_config.NumberColumn(
                        "Contribution",
                        format="%.4f",
                    ),
                },
            )

            st.latex(
                rf"V_{{{highlighted_week}}}="
                rf"{selected_row['snap_change']:.3f}"
                rf"\times{player_weights['snap_pct']:.2f}"
                rf"+{selected_row['opportunity_change']:.3f}"
                rf"\times{player_weights['opp_share']:.2f}"
                rf"+{selected_row['passdown_change']:.3f}"
                rf"\times{player_weights['passdown_share']:.2f}"
                rf"+{selected_row['redzone_change']:.3f}"
                rf"\times{player_weights['redzone_share']:.2f}"
            )

            st.success(
                f"Week {highlighted_week} "
                f"Volatility Score: {selected_score:.4f}"
            )

    # ========================================================
    # WEEKLY PLAYER DATA
    # ========================================================

    with st.expander("View Weekly Player Data"):

        st.dataframe(
            player_weekly[
                [
                    "week",
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

    with st.expander("View Weekly Volatility Components"):

        st.dataframe(
            player_weekly[
                [
                    "week",
                    "snap_change",
                    "opportunity_change",
                    "passdown_change",
                    "redzone_change",
                    "weekly_volatility",
                ]
            ],
            hide_index=True,
            use_container_width=True,
        )

    # ========================================================
    # NEW: PLAYER DATA DOWNLOAD
    # ========================================================

    st.download_button(
        "Download Player Weekly Data",
        data=csv_data(player_weekly),
        file_name=(
            f"rosterwatch_{season}_{player_id}_weekly.csv"
        ),
        mime="text/csv",
    )


# ============================================================
# 14. PLAYER VOLATILITY RANKINGS
# ============================================================

st.divider()

st.header("📊 Player Volatility Rankings")

st.write(
    "Rankings use average week-to-week NFL usage volatility. "
    "The Most Consistent leaderboard also rewards "
    "meaningful playing time and offensive opportunities."
)

st.caption(
    f"{season} NFL Season · {range_label}"
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

ranked_all = (
    season_volatility.loc[
        season_volatility["position"] == ranking_position
    ]
    .sort_values(
        "volatility_score",
        ascending=False,
    )
    .copy()
)

if ranked_all.empty:

    st.info(
        "No volatility scores are available "
        "for this position and selected week range."
    )

else:

    thresholds = CONSISTENCY_THRESHOLDS[
        ranking_position
    ]

    # ========================================================
    # NEW: INDEPENDENT LEADERBOARD FILTERS
    # ========================================================

    st.markdown("### Leaderboard Filters")

    filter_left, filter_right = st.columns(2)

    with filter_left:

        volatile_min_snap = st.slider(
            "Most Volatile — Minimum Average Snap %",
            min_value=0,
            max_value=100,
            value=0,
            step=5,
            key="volatile_min_snap",
        )

    with filter_right:

        consistent_min_snap = st.slider(
            "Most Consistent — Minimum Average Snap %",
            min_value=0,
            max_value=100,
            value=int(
                thresholds["min_snap_pct"] * 100
            ),
            step=5,
            key=f"consistent_min_snap_{ranking_position}",
        )

    # Filter display only.
    # Model 1's underlying scores are unchanged.

    ranked = ranked_all.loc[
        ranked_all["avg_snap_pct"]
        >= volatile_min_snap / 100
    ].copy()

    eligible_players = ranked_all.loc[
        ranked_all["consistency_eligible"]
    ].copy()

    # Quality percentiles are calculated across the
    # full original eligible pool before display filters.

    eligible_players = calculate_consistency_quality(
        eligible_players
    )

    eligible_players = eligible_players.loc[
        eligible_players["avg_snap_pct"]
        >= consistent_min_snap / 100
    ].copy()

    average_volatility = ranked_all[
        "volatility_score"
    ].mean()

    r1, r2, r3, r4 = st.columns(4)

    r1.metric(
        "Players Scored",
        f"{len(ranked_all):,}",
    )

    r2.metric(
        "Avg Weekly Volatility",
        f"{average_volatility:.3f}",
    )

    r3.metric(
        "Highest Avg Volatility",
        f"{ranked_all['volatility_score'].max():.3f}",
    )

    r4.metric(
        "Consistency Eligible",
        f"{ranked_all['consistency_eligible'].sum():,}",
    )

    ranking_left, ranking_right = st.columns(2)

    # ========================================================
    # MOST VOLATILE PLAYERS
    # ========================================================

    with ranking_left:

        st.subheader("📈 Most Volatile Players")

        st.caption(
            f"{ranking_position}s with the largest "
            "average week-to-week usage changes."
        )

        most_volatile = ranked.head(TOP_N).copy()

        most_volatile["volatility_score"] = (
            most_volatile["volatility_score"].round(3)
        )

        volatile_display = most_volatile[
            [
                "player_name",
                "team",
                "weeks",
                "weekly_comparisons",
                "volatility_score",
            ]
        ]

        st.dataframe(
            volatile_display,
            hide_index=True,
            use_container_width=True,
            height=table_height(volatile_display),
            column_config={
                "player_name": "Player",
                "team": "Team",
                "weeks": "Weeks",
                "weekly_comparisons": "Comparisons",
                "volatility_score": "Avg Volatility",
            },
        )

        st.download_button(
            "Download Most Volatile Rankings",
            data=csv_data(ranked),
            file_name=(
                f"rosterwatch_{season}_"
                f"{ranking_position}_volatile.csv"
            ),
            mime="text/csv",
        )

    # ========================================================
    # MOST CONSISTENT PLAYERS
    # ========================================================

    with ranking_right:

        st.subheader("📉 Most Consistent Players")

        st.caption(
            f"{ranking_position}s with stable weekly usage "
            "and substantial offensive involvement."
        )

        if eligible_players.empty:

            st.info(
                "No players meet the minimum playing-time "
                "and opportunity thresholds."
            )

        else:

            most_consistent = (
                eligible_players
                .sort_values(
                    [
                        "consistency_quality_score",
                        "volatility_score",
                    ],
                    ascending=[False, True],
                )
                .head(TOP_N)
                .copy()
            )

            most_consistent["avg_snap_pct"] = (
                most_consistent["avg_snap_pct"] * 100
            ).round(1)

            most_consistent["volatility_score"] = (
                most_consistent["volatility_score"].round(3)
            )

            most_consistent["consistency_quality_score"] = (
                most_consistent[
                    "consistency_quality_score"
                ].round(1)
            )

            most_consistent["avg_opportunities_per_week"] = (
                most_consistent[
                    "avg_opportunities_per_week"
                ].round(1)
            )

            consistent_display = most_consistent[
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

            st.dataframe(
                consistent_display,
                hide_index=True,
                use_container_width=True,
                height=table_height(consistent_display),
                column_config={
                    "player_name": "Player",
                    "team": "Team",
                    "weeks": "Weeks",
                    "avg_snap_pct": "Avg Snap %",
                    "avg_opportunities_per_week": "Opp / Week",
                    "volatility_score": "Avg Volatility",
                    "consistency_quality_score": "Quality Score",
                },
            )

            st.download_button(
                "Download Most Consistent Rankings",
                data=csv_data(eligible_players),
                file_name=(
                    f"rosterwatch_{season}_"
                    f"{ranking_position}_consistent.csv"
                ),
                mime="text/csv",
            )

        st.caption(
            f"Original minimum requirements: "
            f"{thresholds['min_weeks']} recorded weeks, "
            f"{MIN_WEEKLY_COMPARISONS} valid comparisons, "
            f"{thresholds['min_snap_pct']:.0%} average snaps, "
            f"and {thresholds['min_opportunities_per_week']} "
            "opportunities per week."
        )

    # ========================================================
    # DETAILED RANKINGS
    # ========================================================

    with st.expander("View Detailed Player Volatility Data"):

        detailed = ranked_all[
            [
                "player_name",
                "position",
                "team",
                "weeks",
                "weekly_comparisons",
                "avg_snap_pct",
                "avg_opportunities_per_week",
                "volatility_score",
                "consistency_eligible",
            ]
        ].copy()

        st.dataframe(
            detailed,
            hide_index=True,
            use_container_width=True,
        )

        st.download_button(
            "Download All Detailed Rankings",
            data=csv_data(detailed),
            file_name=(
                f"rosterwatch_{season}_"
                f"{ranking_position}_all_rankings.csv"
            ),
            mime="text/csv",
        )


# ============================================================
# 15. FOOTER
# ============================================================

st.divider()

st.markdown("### RosterWatch 🏈")

st.caption(
    "NFL Weekly Volatility Analytics | "
    "2022–2025 | RB · WR · TE"
)

st.caption(
    "NFL usage and Sleeper PPR data from Matt's dataset."
)

st.caption(
    "Built with Python, Pandas, Plotly, and Streamlit."
)

