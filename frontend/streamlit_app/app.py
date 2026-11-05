
from pathlib import Path

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

METRICS = [
    "snap_pct",
    "opp_share",
    "passdown_share",
    "redzone_share",
]

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

# These are experimental minimum requirements for
# the Most Consistent Players leaderboard.
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

# Require enough valid week-to-week comparisons
# before including a player in consistency rankings.
MIN_WEEKLY_COMPARISONS = 8

COLORS = {
    "blue": "#38BDF8",
    "green": "#4ADE80",
    "orange": "#F59E0B",
    "purple": "#A78BFA",
    "panel": "#13243B",
    "text": "#E2E8F0",
}

WEEKLY_COLUMNS = [
    "season",
    "week",
    "gsis_id",
    "player_name",
    "position",
    "team",
    "weekly_volatility",
    "snap_change",
    "opportunity_change",
    "passdown_change",
    "redzone_change",
    "snap_pct",
    "opp_share",
    "passdown_share",
    "redzone_share",
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
# 3. LOAD AND VALIDATE DATA
# ============================================================

@st.cache_data(show_spinner="Loading RosterWatch data...")
def load_data():

    usage = pd.read_csv(
        DATA_DIR / "weekly_usage.csv"
    )

    required = {
        "season",
        "week",
        "gsis_id",
        "player_name",
        "position",
        "team",
        "snap_pct",
        "opp_share",
        "passdown_share",
        "redzone_share",
        "targets",
        "carries",
        "offense_snaps",
    }

    missing = required - set(usage.columns)

    if missing:
        raise ValueError(
            "weekly_usage.csv is missing columns: "
            + ", ".join(sorted(missing))
        )

    numeric_columns = [
        "season",
        "week",
        "snap_pct",
        "opp_share",
        "passdown_share",
        "redzone_share",
        "targets",
        "carries",
        "offense_snaps",
    ]

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

    usage["gsis_id"] = (
        usage["gsis_id"].astype("string")
    )

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

    # Remove exact duplicate rows.
    usage = usage.drop_duplicates()

    return usage


try:
    usage = load_data()

except (OSError, ValueError, pd.errors.ParserError) as error:

    st.error(
        f"Could not load RosterWatch data: {error}"
    )

    st.info(
        "Make sure weekly_usage.csv exists "
        "inside data/processed/."
    )

    st.stop()


if usage.empty:
    st.warning("No NFL usage records were found.")
    st.stop()


# ============================================================
# 4. WEEK-TO-WEEK VOLATILITY MODEL
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

        # Create one observation per player-week.
        # Use mean to avoid double-counting repeated
        # player-week rows.
        weekly = (
            player.groupby("week")
            .agg({
                "snap_pct": "mean",
                "opp_share": "mean",
                "passdown_share": "mean",
                "redzone_share": "mean",
                "targets": "mean",
                "carries": "mean",
            })
            .sort_index()
        )

        weekly["opportunities"] = (
            weekly["targets"].fillna(0)
            + weekly["carries"].fillna(0)
        )

        # Calculate absolute changes from the
        # previous recorded week.
        changes = weekly[METRICS].diff().abs()

        # Only score consecutive calendar weeks.
        # Example: Week 7 cannot be compared with
        # Week 5 if Week 6 is missing.
        consecutive = (
            weekly.index.to_series().diff() == 1
        )

        changes.loc[~consecutive, :] = np.nan

        # Position-specific weighted weekly volatility.
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

        # Store the score for every recorded week.
        for week, row in weekly.iterrows():

            weekly_results.append({
                "season": season,
                "week": int(week),
                "gsis_id": player_id,
                "player_name": latest["player_name"],
                "position": position,
                "team": latest["team"],
                "weekly_volatility": row[
                    "weekly_volatility"
                ],
                "snap_change": row["snap_change"],
                "opportunity_change": row[
                    "opportunity_change"
                ],
                "passdown_change": row[
                    "passdown_change"
                ],
                "redzone_change": row[
                    "redzone_change"
                ],
                "snap_pct": row["snap_pct"],
                "opp_share": row["opp_share"],
                "passdown_share": row["passdown_share"],
                "redzone_share": row["redzone_share"],
                "targets": row["targets"],
                "carries": row["carries"],
                "opportunities": row["opportunities"],
            })

        valid_scores = (
            weekly["weekly_volatility"].dropna()
        )

        # A player needs at least one valid comparison
        # to receive an average weekly volatility score.
        if valid_scores.empty:
            continue

        recorded_weeks = len(weekly)
        weekly_comparisons = len(valid_scores)

        avg_snap_pct = weekly["snap_pct"].mean()

        avg_opportunities = (
            weekly["opportunities"].mean()
        )

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
        })

    weekly_scores = pd.DataFrame(
        weekly_results,
        columns=WEEKLY_COLUMNS,
    )

    season_scores = pd.DataFrame(
        season_results,
        columns=SCORE_COLUMNS,
    )

    return weekly_scores, season_scores


weekly_volatility, volatility = calculate_volatility(usage)


# ============================================================
# 5. CONSISTENCY QUALITY SCORE
# ============================================================

def calculate_consistency_quality(data):

    if data.empty:
        return data.copy()

    scored = data.copy()

    # Higher consistency = lower volatility.
    # Higher playing time and opportunities = better.

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
# 6. HERO HEADER
# ============================================================

hero_html = """
<div class="hero">
<div class="hero-title">RosterWatch 🏈</div>
<div class="hero-subtitle">
NFL Weekly Volatility Analytics | 2022–2025
</div>
<div class="hero-description">
Track how NFL players' offensive roles change from week to week.
Explore playing time, offensive opportunities, passing-down
involvement, and red-zone usage with position-aware analytics.
</div>
<div class="pills">
<span class="pill">Weekly Volatility Model</span>
<span class="pill">Position-Aware Scoring</span>
<span class="pill">Meaningful Usage Thresholds</span>
<span class="pill">2022–2025 Seasons</span>
<span class="pill">RB / WR / TE</span>
<span class="pill">Weeks 1–18</span>
</div>
</div>
"""

st.markdown(
    hero_html.strip(),
    unsafe_allow_html=True,
)


# ============================================================
# 7. ABOUT ROSTERWATCH
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


# ============================================================
# 8. MODEL EXPLANATION
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

    st.markdown("**Variables:**")

    st.markdown(
        """
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

    if selected_formula_position == "RB":

        st.latex(
            r"V_{RB,t}=0.30|\Delta S_t|"
            r"+0.45|\Delta O_t|"
            r"+0.15|\Delta P_t|"
            r"+0.10|\Delta R_t|"
        )

    elif selected_formula_position == "WR":

        st.latex(
            r"V_{WR,t}=0.35|\Delta S_t|"
            r"+0.35|\Delta O_t|"
            r"+0.20|\Delta P_t|"
            r"+0.10|\Delta R_t|"
        )

    else:

        st.latex(
            r"V_{TE,t}=0.40|\Delta S_t|"
            r"+0.30|\Delta O_t|"
            r"+0.20|\Delta P_t|"
            r"+0.10|\Delta R_t|"
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


with st.expander("How are Weekly Volatility and Consistency calculated?"):

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
        "players at the same position."
    )

    st.info(
        "Week 1 has no previous week for comparison. "
        "A missing week also prevents a comparison "
        "with the next recorded week. Missing data "
        "is never treated as zero volatility."
    )

    st.caption(
        "Weights, thresholds, and quality-score "
        "coefficients are experimental."
    )


# ============================================================
# 9. SIDEBAR
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

position = st.sidebar.selectbox(
    "Position",
    ["All Positions"] + POSITIONS,
)

st.sidebar.divider()

st.sidebar.markdown("**Active Analytics Model**")
st.sidebar.caption("Weekly Volatility Model")

st.sidebar.markdown("**Ranking Method**")
st.sidebar.caption(
    "Average weekly volatility + meaningful usage thresholds"
)

st.sidebar.markdown("**Data Coverage**")
st.sidebar.caption(
    "2022–2025 · RB / WR / TE · Weeks 1–18"
)


# ============================================================
# 10. FILTER DATA
# ============================================================

filtered_usage = usage.loc[
    usage["season"] == season
].copy()

filtered_volatility = volatility.loc[
    volatility["season"] == season
].copy()

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
# 11. SEASON OVERVIEW
# ============================================================

st.divider()

st.header("📊 Season Overview")

st.caption(f"{season} NFL Season · {position}")

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
# 12. PLAYER EXPLORER
# ============================================================

st.divider()

st.header("🔎 Player Explorer")

st.write(
    "Select a player to explore weekly offensive usage "
    "and their week-to-week volatility."
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

    player_id = st.selectbox(
        "Select an NFL Player",
        players["gsis_id"].tolist(),
        format_func=lambda pid: labels.get(pid, str(pid)),
    )

    player_data = (
        filtered_usage.loc[
            filtered_usage["gsis_id"] == player_id
        ]
        .sort_values("week")
        .copy()
    )

    player_name = str(
        player_data["player_name"].iloc[0]
    )

    player_position = str(
        player_data["position"].iloc[-1]
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

    st.subheader(
        f"{player_name} — {season} Weekly Usage"
    )

    m1, m2, m3, m4 = st.columns(4)

    mean_snap = player_data["snap_pct"].mean()

    m1.metric(
        "Average Snap Percentage",
        f"{mean_snap:.1%}"
        if pd.notna(mean_snap)
        else "—",
    )

    m2.metric(
        "Season Targets",
        f"{player_data['targets'].sum():,.0f}",
    )

    m3.metric(
        "Season Carries",
        f"{player_data['carries'].sum():,.0f}",
    )

    if not player_summary.empty:

        player_score = player_summary[
            "volatility_score"
        ].iloc[0]

        m4.metric(
            "Avg Weekly Volatility",
            f"{player_score:.3f}",
        )

        is_eligible = bool(
            player_summary[
                "consistency_eligible"
            ].iloc[0]
        )

        if is_eligible:

            st.success(
                "This player meets the playing-time, "
                "opportunity, and weekly comparison "
                "requirements for consistency rankings."
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
        "weights and compares consecutive NFL weeks."
    )

    controls1, controls2 = st.columns([2, 1])

    with controls1:

        highlighted_week = st.slider(
            "Highlight NFL Week",
            min_value=1,
            max_value=18,
            value=1,
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

    # --------------------------------------------------------
    # CHART BUILDER
    # --------------------------------------------------------

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

        fig.add_vrect(
            x0=highlighted_week - 0.45,
            x1=highlighted_week + 0.45,
            fillcolor="rgba(255,255,255,0.07)",
            line_width=0,
            layer="below",
        )

        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=17),
                x=0.03,
            ),
            paper_bgcolor=COLORS["panel"],
            plot_bgcolor=COLORS["panel"],
            font=dict(color=COLORS["text"]),
            height=365,
            margin=dict(
                l=55,
                r=20,
                t=65,
                b=55,
            ),
            showlegend=False,
            xaxis=dict(
                title="NFL Week",
                tickmode="array",
                tickvals=WEEKS,
                range=[0.5, 18.5],
                gridcolor="#2A405A",
                zeroline=False,
            ),
            yaxis=dict(
                title=ylabel,
                gridcolor="#2A405A",
                zeroline=False,
                rangemode="tozero",
            ),
        )

        if percentage:

            fig.update_yaxes(
                tickformat=".0%",
                range=[0, 1],
            )

        return fig

    # --------------------------------------------------------
    # FOUR PLAYER USAGE CHARTS
    # --------------------------------------------------------

    chart1, chart2 = st.columns(2)

    with chart1:

        st.plotly_chart(
            make_chart(
                player_data,
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
                player_data,
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
                player_data,
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
                player_data,
                "carries",
                "Weekly Carries",
                "Carries",
                COLORS["purple"],
            ),
            use_container_width=True,
        )

    # --------------------------------------------------------
    # WEEKLY VOLATILITY CHART
    # --------------------------------------------------------

    st.subheader("Weekly Volatility")

    st.write(
        "This chart shows how much the player's offensive "
        "usage changed compared with the previous week."
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

    volatility_fig.add_vrect(
        x0=highlighted_week - 0.45,
        x1=highlighted_week + 0.45,
        fillcolor="rgba(255,255,255,0.07)",
        line_width=0,
        layer="below",
    )

    volatility_fig.update_layout(
        title="Week-to-Week Usage Volatility",
        paper_bgcolor=COLORS["panel"],
        plot_bgcolor=COLORS["panel"],
        font=dict(color=COLORS["text"]),
        height=400,
        xaxis=dict(
            title="NFL Week",
            tickmode="array",
            tickvals=WEEKS,
            range=[0.5, 18.5],
            gridcolor="#2A405A",
        ),
        yaxis=dict(
            title="Weekly Volatility Score",
            rangemode="tozero",
            gridcolor="#2A405A",
        ),
        margin=dict(
            l=55,
            r=20,
            t=65,
            b=55,
        ),
    )

    st.plotly_chart(
        volatility_fig,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # SELECTED WEEK DETAILS
    # --------------------------------------------------------

    selected_week_data = player_weekly.loc[
        player_weekly["week"] == highlighted_week
    ]

    if not selected_week_data.empty:

        selected_row = selected_week_data.iloc[0]

        selected_score = selected_row[
            "weekly_volatility"
        ]

        if pd.notna(selected_score):

            st.info(
                f"Week {highlighted_week} volatility: "
                f"{selected_score:.3f}"
            )

        else:

            st.caption(
                f"Week {highlighted_week} has no valid "
                "consecutive-week comparison."
            )

    else:

        st.caption(
            f"No player record for Week {highlighted_week}."
        )

    with st.expander("View Weekly Player Data"):

        st.dataframe(
            player_weekly[
                [
                    "week",
                    "snap_pct",
                    "opp_share",
                    "targets",
                    "carries",
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


# ============================================================
# 13. PLAYER VOLATILITY RANKINGS
# ============================================================

st.divider()

st.header("📊 Player Volatility Rankings")

st.write(
    "Rankings use average week-to-week volatility. "
    "The Most Consistent leaderboard also rewards "
    "meaningful playing time and offensive opportunities."
)

ranking_position = st.selectbox(
    "Ranking Position",
    POSITIONS,
    index=(
        POSITIONS.index(position)
        if position in POSITIONS
        else 0
    ),
    help=(
        "Compare players within the same position "
        "because RB, WR, and TE use different weights."
    ),
)

ranked = (
    filtered_volatility.loc[
        filtered_volatility["position"] == ranking_position
    ]
    .sort_values(
        "volatility_score",
        ascending=False,
    )
    .copy()
)

if ranked.empty:

    st.info(
        "No volatility scores are available "
        "for this position and season."
    )

else:

    thresholds = CONSISTENCY_THRESHOLDS[
        ranking_position
    ]

    eligible_players = ranked.loc[
        (ranked["weeks"] >= thresholds["min_weeks"])
        & (
            ranked["weekly_comparisons"]
            >= MIN_WEEKLY_COMPARISONS
        )
        & (
            ranked["avg_snap_pct"]
            >= thresholds["min_snap_pct"]
        )
        & (
            ranked["avg_opportunities_per_week"]
            >= thresholds["min_opportunities_per_week"]
        )
    ].copy()

    eligible_players = calculate_consistency_quality(
        eligible_players
    )

    average_volatility = ranked[
        "volatility_score"
    ].mean()

    r1, r2, r3, r4 = st.columns(4)

    r1.metric(
        "Players Scored",
        f"{len(ranked):,}",
    )

    r2.metric(
        "Avg Weekly Volatility",
        f"{average_volatility:.3f}",
    )

    r3.metric(
        "Highest Avg Volatility",
        f"{ranked['volatility_score'].max():.3f}",
    )

    r4.metric(
        "Consistency Eligible",
        f"{len(eligible_players):,}",
    )

    ranking_left, ranking_right = st.columns(2)

    # --------------------------------------------------------
    # MOST VOLATILE PLAYERS
    # --------------------------------------------------------

    with ranking_left:

        st.subheader("📈 Most Volatile Players")

        st.caption(
            f"{ranking_position}s with the largest "
            "average week-to-week usage changes."
        )

        most_volatile = ranked.head(15).copy()

        most_volatile["volatility_score"] = (
            most_volatile["volatility_score"].round(3)
        )

        st.dataframe(
            most_volatile[
                [
                    "player_name",
                    "team",
                    "weeks",
                    "weekly_comparisons",
                    "volatility_score",
                ]
            ],
            hide_index=True,
            use_container_width=True,
            column_config={
                "player_name": "Player",
                "team": "Team",
                "weeks": "Weeks",
                "weekly_comparisons": "Comparisons",
                "volatility_score": "Avg Volatility",
            },
        )

    # --------------------------------------------------------
    # MOST CONSISTENT PLAYERS
    # --------------------------------------------------------

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
                .head(15)
                .copy()
            )

            most_consistent["volatility_score"] = (
                most_consistent[
                    "volatility_score"
                ].round(3)
            )

            most_consistent["avg_snap_pct"] = (
                most_consistent["avg_snap_pct"] * 100
            ).round(1)

            most_consistent[
                "avg_opportunities_per_week"
            ] = (
                most_consistent[
                    "avg_opportunities_per_week"
                ].round(1)
            )

            most_consistent[
                "consistency_quality_score"
            ] = (
                most_consistent[
                    "consistency_quality_score"
                ].round(1)
            )

            st.dataframe(
                most_consistent[
                    [
                        "player_name",
                        "team",
                        "weeks",
                        "avg_snap_pct",
                        "avg_opportunities_per_week",
                        "volatility_score",
                        "consistency_quality_score",
                    ]
                ],
                hide_index=True,
                use_container_width=True,
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

        st.caption(
            f"Minimum requirements: "
            f"{thresholds['min_weeks']} recorded weeks, "
            f"{MIN_WEEKLY_COMPARISONS} valid weekly comparisons, "
            f"{thresholds['min_snap_pct']:.0%} average snaps, "
            f"and {thresholds['min_opportunities_per_week']} "
            f"opportunities per week."
        )

    # --------------------------------------------------------
    # DETAILED RANKINGS
    # --------------------------------------------------------

    with st.expander("View Detailed Player Volatility Data"):

        detailed = ranked[
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

        numeric_columns = [
            "avg_snap_pct",
            "avg_opportunities_per_week",
            "volatility_score",
        ]

        detailed[numeric_columns] = (
            detailed[numeric_columns].round(3)
        )

        st.dataframe(
            detailed,
            hide_index=True,
            use_container_width=True,
        )


# ============================================================
# 14. FOOTER
# ============================================================

st.divider()

st.markdown("### RosterWatch 🏈")

st.caption(
    "NFL Weekly Volatility Analytics | "
    "2022–2025 | RB · WR · TE"
)

st.caption(
    "Built with Python, Pandas, Plotly, and Streamlit."
)


