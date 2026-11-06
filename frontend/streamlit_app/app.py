
import sys
from pathlib import Path

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

# Make the project's src/ package importable when Streamlit runs
# this file directly.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Weights, thresholds and the model itself live in src/volatility.py.
from src.volatility import (  # noqa: E402
    BASELINE_SEASONS,
    COMPARISON_GAP,
    CONSISTENCY_THRESHOLDS,
    MAX_WEEKLY_CHANGE,
    METRICS,
    MIN_RANKED_COMPARISONS,
    MIN_ROLE_SNAP_PCT,
    MIN_WEEKS_FLOOR,
    MIN_WEEKS_SHARE,
    POSITION_WEIGHTS,
    POSITIONS,
    calculate_volatility as run_volatility_model,
    eligibility_minimums,
)

WEEKS = list(range(1, 19))

METRIC_LABELS = {
    "snap_pct": "Snap Percentage",
    "opp_share": "Opportunity Share",
    "passdown_share": "Passing-Down Share",
    "redzone_share": "Red-Zone Share",
}

METRIC_SYMBOLS = {
    "snap_pct": "S",
    "opp_share": "O",
    "passdown_share": "P",
    "redzone_share": "R",
}

COLORS = {
    "blue": "#38BDF8",
    "green": "#4ADE80",
    "orange": "#F59E0B",
    "purple": "#A78BFA",
    "panel": "#13243B",
    "text": "#E2E8F0",
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
    """Run the volatility model in src/volatility.py, cached."""
    return run_volatility_model(data)


weekly_volatility, volatility, typical_change = calculate_volatility(usage)


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
    "from one game to the next. Rather than calculating "
    "volatility using season-long standard deviation, "
    "the model compares each game with the player's "
    "previous game."
)

st.write(
    "Each change is scaled by the typical weekly change for "
    "the player's position, so a score of 1.0 means a normal "
    "week for a running back, wide receiver, or tight end. "
    "Its consistency rankings also account for meaningful "
    "playing time and offensive opportunities."
)


# ============================================================
# 8. MODEL EXPLANATION
# ============================================================

def weighted_metrics(position):
    """Metrics with a nonzero weight for a position."""
    return [
        metric
        for metric in METRICS
        if POSITION_WEIGHTS[position][metric] > 0
    ]


def position_formula(position):
    """LaTeX formula for one position's weekly volatility score."""
    terms = []

    for metric in weighted_metrics(position):
        weight = POSITION_WEIGHTS[position][metric]
        typical = typical_change.loc[position, metric] * 100
        symbol = METRIC_SYMBOLS[metric]
        terms.append(
            rf"{weight:.2f}\cdot"
            rf"\frac{{|\Delta {symbol}_t|}}{{{typical:.1f}}}"
        )

    return rf"V_{{{position},t}}=" + "+".join(terms)


def metric_table(values):
    """Table with one row per metric and one column per position."""
    return pd.DataFrame({
        "Metric": [METRIC_LABELS[metric] for metric in METRICS],
        **{
            position: [values(position, metric) for metric in METRICS]
            for position in POSITIONS
        },
    })


st.header("Our Analytics Model")

with st.container(border=True):

    st.subheader("Weekly Volatility Model")

    st.write(
        "The Weekly Volatility Model measures how much a "
        "player's offensive involvement changes compared "
        "with the player's previous game."
    )

    st.markdown("#### Weekly Volatility Equation")

    st.latex(
        r"V_t = \sum_{m} w_m"
        r"\frac{|m_t - m_{t-1}|}{\bar{\Delta}_m}"
    )

    st.markdown("**Variables:**")

    st.markdown(
        """
- **Vₜ:** Volatility score for the current week (1.0 = a typical week)
- **m:** A usage metric: snap percentage (S), opportunity share (O),
  passing-down share (P), or red-zone share (R)
- **mₜ − mₜ₋₁:** Change since the player's previous game, in percentage points
- **Δ̄ₘ:** Typical weekly change in that metric for the position
- **wₘ:** Weight of that metric
"""
    )

    st.markdown("#### Position Formulas")

    selected_formula_position = st.selectbox(
        "View Position Formula",
        POSITIONS,
        key="formula_position",
    )

    st.latex(position_formula(selected_formula_position))

    st.caption(
        "Changes and typical changes are in percentage points. "
        "Metrics in the score: "
        + ", ".join(
            METRIC_LABELS[metric]
            for metric in weighted_metrics(selected_formula_position)
        )
        + ". Metrics with zero weight are still shown in each "
        "player's weekly components."
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
        "T contains the weeks compared with the player's "
        "previous game, and m is the number of those comparisons."
    )

    left, right = st.columns(2)

    with left:

        st.markdown("**Above 1.0**")

        st.write(
            "The player's offensive role changed more "
            "than in a typical week for the position."
        )

    with right:

        st.markdown("**Below 1.0**")

        st.write(
            "The player's offensive role stayed steadier "
            "than in a typical week for the position."
        )


with st.expander("How are Weekly Volatility and Consistency calculated?"):

    st.markdown("#### Model Weights")

    st.table(
        metric_table(
            lambda position, metric:
            f"{POSITION_WEIGHTS[position][metric]:.0%}"
        )
    )

    st.markdown("#### Typical Weekly Change")

    st.table(
        metric_table(
            lambda position, metric:
            f"{typical_change.loc[position, metric] * 100:.1f} pts"
        )
    )

    st.caption(
        "Average change from one game to the next in "
        f"{BASELINE_SEASONS[0]}–{BASELINE_SEASONS[-1]}, for players "
        f"averaging at least {MIN_ROLE_SNAP_PCT:.0%} of snaps. "
        "Each game's change counts at most "
        f"{MAX_WEEKLY_CHANGE * 100:.0f} percentage points."
    )

    st.markdown("#### Volatility Rankings")

    st.table(
        pd.DataFrame({
            "Requirement": [
                "Minimum Average Snap %",
                "Minimum Scored Weeks",
            ],
            "All Positions": [
                f"{MIN_ROLE_SNAP_PCT:.0%}",
                str(MIN_RANKED_COMPARISONS),
            ],
        })
    )

    st.markdown("#### Consistency Eligibility")

    full_season = {
        position: eligibility_minimums(position, WEEKS[-1])
        for position in POSITIONS
    }

    st.table(
        pd.DataFrame({
            "Requirement": [
                "Minimum Recorded Weeks",
                "Minimum Valid Weekly Comparisons",
                "Minimum Average Snap %",
                "Minimum Carries + Targets / Week",
            ],
            **{
                position: [
                    str(full_season[position][0]),
                    str(full_season[position][1]),
                    f"{CONSISTENCY_THRESHOLDS[position]['min_snap_pct']:.0%}",
                    str(
                        CONSISTENCY_THRESHOLDS[position][
                            "min_opportunities_per_week"
                        ]
                    ),
                ]
                for position in POSITIONS
            },
        })
    )

    st.caption(
        "Through Week 18. Earlier in a season, the minimum is "
        f"{MIN_WEEKS_SHARE:.0%} of the weeks played so far "
        f"(at least {MIN_WEEKS_FLOOR}), with {COMPARISON_GAP} "
        "fewer comparisons."
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
        "A player's first game has nothing to compare with. "
        "A bye week doesn't break the comparison, but a missed "
        "game does: the game after an absence isn't compared "
        "with the one before it. Missing data is never treated "
        "as zero volatility."
    )

    st.caption(
        "Weights, thresholds, and quality-score coefficients "
        "are experimental. They are set in src/volatility.py."
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
            f"{player_score:.2f}",
            help="1.0 = a typical week for the position.",
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
        "Each game is compared with the player's previous game "
        "(a bye week doesn't break the comparison). "
        f"1.0 = a typical week for a {player_position}."
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
        "usage changed compared with the previous game. "
        "The dotted line is a typical week (1.0)."
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
                "Weekly Volatility: %{y:.2f}"
                "<extra></extra>"
            ),
        )
    )

    volatility_fig.add_hline(
        y=1.0,
        line_dash="dot",
        line_color=COLORS["text"],
        opacity=0.6,
        annotation_text="Typical week",
        annotation_position="top left",
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
                f"{selected_score:.2f}"
            )

        else:

            st.caption(
                f"Week {highlighted_week} has no comparison: it is "
                "the player's first game, or the player missed "
                "the game before it."
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
        "because each position has its own typical "
        "weekly change."
    ),
)

# Leave out part-time players and averages built on a few games.
ranked = (
    filtered_volatility.loc[
        (filtered_volatility["position"] == ranking_position)
        & (filtered_volatility["avg_snap_pct"] >= MIN_ROLE_SNAP_PCT)
        & (
            filtered_volatility["weekly_comparisons"]
            >= MIN_RANKED_COMPARISONS
        )
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

    min_weeks, min_comparisons = eligibility_minimums(
        ranking_position,
        usage.loc[usage["season"] == season, "week"].max(),
    )

    # Eligibility is set in src/volatility.py (season_scores).
    eligible_players = ranked.loc[
        ranked["consistency_eligible"]
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
        help=(
            f"Players averaging {MIN_ROLE_SNAP_PCT:.0%}+ of snaps "
            f"with {MIN_RANKED_COMPARISONS}+ scored weeks."
        ),
    )

    r2.metric(
        "Avg Weekly Volatility",
        f"{average_volatility:.2f}",
        help="1.0 = a typical week for the position.",
    )

    r3.metric(
        "Highest Avg Volatility",
        f"{ranked['volatility_score'].max():.2f}",
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
            f"{ranking_position}s with the largest average "
            "week-to-week usage changes (averaging "
            f"{MIN_ROLE_SNAP_PCT:.0%}+ of snaps, "
            f"{MIN_RANKED_COMPARISONS}+ scored weeks)."
        )

        most_volatile = ranked.head(15).copy()

        most_volatile["volatility_score"] = (
            most_volatile["volatility_score"].round(2)
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
                ].round(2)
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
            f"{min_weeks} recorded weeks, "
            f"{min_comparisons} valid weekly comparisons, "
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


