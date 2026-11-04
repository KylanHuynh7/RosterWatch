
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

POSITIONS = ["RB", "WR", "TE"]
WEEKS = list(range(1, 19))

# Position-specific weights for the Volatility Model.
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

# Stricter eligibility requirements for consistency rankings.
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

COLORS = {
    "blue": "#38BDF8",
    "green": "#4ADE80",
    "orange": "#F59E0B",
    "purple": "#A78BFA",
    "panel": "#13243B",
    "text": "#E2E8F0",
}

SCORE_COLUMNS = [
    "season",
    "gsis_id",
    "player_name",
    "position",
    "team",
    "weeks",
    "volatility_score",
    "snap_volatility",
    "opportunity_volatility",
    "passdown_volatility",
    "redzone_volatility",
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
# 3. LOAD DATA
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

    usage["season"] = usage["season"].astype(int)
    usage["week"] = usage["week"].astype(int)

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
# 4. VOLATILITY MODEL
# ============================================================

@st.cache_data(show_spinner="Calculating volatility scores...")
def calculate_volatility(data):

    results = []

    for (season, player_id), player in data.groupby(
        ["season", "gsis_id"]
    ):

        player = player.sort_values("week")
        position = player["position"].iloc[-1]

        if position not in POSITION_WEIGHTS:
            continue

        weights = POSITION_WEIGHTS[position]
        thresholds = CONSISTENCY_THRESHOLDS[position]

        # One observation per player-week.
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
        )

        metric_volatility = {}

        for metric in weights:

            values = weekly[metric].dropna()

            if len(values) < 2:
                break

            metric_volatility[metric] = (
                values.std(ddof=0)
            )

        if len(metric_volatility) != len(weights):
            continue

        score = sum(
            weights[metric] * metric_volatility[metric]
            for metric in weights
        )

        recorded_weeks = len(weekly)
        avg_snap_pct = weekly["snap_pct"].mean()

        weekly_opportunities = (
            weekly["targets"].fillna(0)
            + weekly["carries"].fillna(0)
        )

        avg_opportunities = (
            weekly_opportunities.mean()
        )

        consistency_eligible = (
            recorded_weeks >= thresholds["min_weeks"]
            and pd.notna(avg_snap_pct)
            and avg_snap_pct >= thresholds["min_snap_pct"]
            and pd.notna(avg_opportunities)
            and avg_opportunities
            >= thresholds["min_opportunities_per_week"]
        )

        latest = player.iloc[-1]

        results.append({
            "season": season,
            "gsis_id": player_id,
            "player_name": latest["player_name"],
            "position": position,
            "team": latest["team"],
            "weeks": recorded_weeks,
            "volatility_score": score,
            "snap_volatility": metric_volatility[
                "snap_pct"
            ],
            "opportunity_volatility": metric_volatility[
                "opp_share"
            ],
            "passdown_volatility": metric_volatility[
                "passdown_share"
            ],
            "redzone_volatility": metric_volatility[
                "redzone_share"
            ],
            "avg_snap_pct": avg_snap_pct,
            "avg_opportunities_per_week": avg_opportunities,
            "consistency_eligible": bool(
                consistency_eligible
            ),
        })

    return pd.DataFrame(
        results,
        columns=SCORE_COLUMNS,
    )


volatility = calculate_volatility(usage)


# ============================================================
# 5. CONSISTENCY QUALITY SCORE
# ============================================================

def calculate_consistency_quality(data):

    if data.empty:
        return data.copy()

    scored = data.copy()

    # Low volatility is better.
    # High playing time and opportunities are better.

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

    # 50% consistency
    # 25% snap participation
    # 25% offensive opportunities
    #
    # Convert rankings to a 0–100 quality score.

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
NFL Player Volatility Analytics | 2022–2025
</div>
<div class="hero-description">
Explore how consistently NFL players are involved in their
team's offense. Compare playing time, offensive opportunities,
passing-down involvement, and red-zone usage across the season.
</div>
<div class="pills">
<span class="pill">Volatility Model</span>
<span class="pill">Position-Aware Scoring</span>
<span class="pill">Meaningful Usage Thresholds</span>
<span class="pill">Consistency Quality Rankings</span>
<span class="pill">2022–2025 Seasons</span>
<span class="pill">RB / WR / TE</span>
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
    "that helps users understand how consistently players "
    "are involved in their team's offense."
)

st.write(
    "The Volatility Model adjusts its weights for RBs, "
    "WRs, and TEs. To identify consistently involved "
    "players rather than rarely used backups, the "
    "Most Consistent Players leaderboard requires "
    "minimum playing time and opportunities, then "
    "combines stability and offensive involvement "
    "into a Consistency Quality Score."
)


# ============================================================
# 8. MODEL EXPLANATION
# ============================================================

st.header("Our Analytics Model")

with st.container(border=True):

    st.subheader("Volatility Model")

    st.write(
        "The Volatility Model measures how much a player's "
        "offensive involvement varies throughout a season. "
        "It calculates the standard deviation of four "
        "usage metrics and combines them using weights "
        "specific to the player's position."
    )

    st.markdown("#### General Volatility Equation")

    st.latex(
        r"V = w_S\sigma_S + w_O\sigma_O "
        r"+ w_P\sigma_P + w_R\sigma_R"
    )

    st.markdown("#### Position-Specific Equations")

    selected_formula_position = st.selectbox(
        "View Position Formula",
        POSITIONS,
        key="formula_position",
    )

    if selected_formula_position == "RB":

        st.latex(
            r"V_{RB} = 0.30\sigma_S + 0.45\sigma_O "
            r"+ 0.15\sigma_P + 0.10\sigma_R"
        )

    elif selected_formula_position == "WR":

        st.latex(
            r"V_{WR} = 0.35\sigma_S + 0.35\sigma_O "
            r"+ 0.20\sigma_P + 0.10\sigma_R"
        )

    else:

        st.latex(
            r"V_{TE} = 0.40\sigma_S + 0.30\sigma_O "
            r"+ 0.20\sigma_P + 0.10\sigma_R"
        )

    st.divider()

    left, right = st.columns(2)

    with left:

        st.markdown("**Higher Volatility Score**")

        st.write(
            "Greater variation in offensive involvement."
        )

    with right:

        st.markdown("**Lower Volatility Score**")

        st.write(
            "More stable offensive involvement."
        )

    st.info(
        "Low volatility alone does not guarantee a "
        "high consistency ranking. Players must also "
        "receive meaningful playing time and opportunities."
    )


with st.expander("How are Volatility and Consistency calculated?"):

    st.markdown("#### Volatility Weights")

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

    st.markdown("#### Standard Deviation")

    st.latex(
        r"\sigma_X = "
        r"\sqrt{\frac{1}{n}"
        r"\sum_{i=1}^{n}(X_i-\bar{X})^2}"
    )

    st.markdown("#### Consistency Eligibility")

    st.table(
        pd.DataFrame({
            "Requirement": [
                "Minimum Recorded Weeks",
                "Minimum Average Snap %",
                "Minimum Carries + Targets / Week",
            ],
            "RB": ["10", "50%", "10"],
            "WR": ["10", "65%", "5"],
            "TE": ["10", "60%", "4"],
        })
    )

    st.markdown("#### Consistency Quality Score")

    st.latex(
        r"Q = 100\left("
        r"0.50C + 0.25S + 0.25O"
        r"\right)"
    )

    st.write(
        "**C** represents the player's relative consistency "
        "ranking, **S** represents their snap participation "
        "ranking, and **O** represents their opportunity "
        "ranking among eligible players at the same position."
    )

    st.caption(
        "Higher quality scores indicate a stronger "
        "combination of low volatility, meaningful "
        "playing time, and offensive opportunities. "
        "Weights and thresholds are experimental."
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
st.sidebar.caption("Position-Aware Volatility Model")

st.sidebar.markdown("**Consistency Rankings**")
st.sidebar.caption(
    "Meaningful usage thresholds + Quality Score"
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

if position != "All Positions":

    filtered_usage = filtered_usage.loc[
        filtered_usage["position"] == position
    ]

    filtered_volatility = filtered_volatility.loc[
        filtered_volatility["position"] == position
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
    "and their position-adjusted Volatility Score."
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

    st.subheader(
        f"{player_name} — {season} Weekly Usage"
    )

    player_volatility = filtered_volatility.loc[
        filtered_volatility["gsis_id"] == player_id
    ]

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

    if not player_volatility.empty:

        player_score = player_volatility[
            "volatility_score"
        ].iloc[0]

        m4.metric(
            "Volatility Score",
            f"{player_score:.3f}",
        )

        is_eligible = bool(
            player_volatility[
                "consistency_eligible"
            ].iloc[0]
        )

        if is_eligible:

            st.success(
                "This player meets the playing-time "
                "and opportunity thresholds for "
                "consistency rankings."
            )

        else:

            st.caption(
                "This player does not meet the "
                "minimum consistency eligibility "
                "thresholds."
            )

    else:

        m4.metric(
            "Volatility Score",
            "N/A",
        )

    st.caption(
        f"Volatility uses the {player_position}-specific "
        "model weights."
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
    # FOUR PLAYER CHARTS
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
    # HIGHLIGHTED WEEK SUMMARY
    # --------------------------------------------------------

    week_values = (
        player_data.groupby("week")[
            [
                "snap_pct",
                "opp_share",
                "targets",
                "carries",
            ]
        ]
        .mean()
    )

    if highlighted_week in week_values.index:

        current = week_values.loc[
            highlighted_week
        ]

        def percent_text(value):
            return (
                f"{value:.1%}"
                if pd.notna(value)
                else "—"
            )

        def number_text(value):
            return (
                f"{value:.1f}"
                if pd.notna(value)
                else "—"
            )

        st.caption(
            f"Week {highlighted_week}: "
            f"Snap share "
            f"{percent_text(current['snap_pct'])} · "
            f"Opportunity share "
            f"{percent_text(current['opp_share'])} · "
            f"Targets "
            f"{number_text(current['targets'])} · "
            f"Carries "
            f"{number_text(current['carries'])}"
        )

    else:

        st.caption(
            f"Week {highlighted_week}: "
            "No recorded player data."
        )

    with st.expander("View Weekly Player Data"):

        st.dataframe(
            player_data[
                [
                    "week",
                    "team",
                    "snap_pct",
                    "opp_share",
                    "targets",
                    "carries",
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
    "Compare players based on offensive usage volatility. "
    "The Most Consistent leaderboard prioritizes "
    "players with both stable usage and meaningful "
    "offensive involvement."
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
        "because RB, WR, and TE scores use "
        "different metric weights."
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

    # Reapply thresholds directly to ensure that
    # low-usage players cannot enter the leaderboard.
    eligible_players = ranked.loc[
        (ranked["weeks"] >= thresholds["min_weeks"])
        & (
            ranked["avg_snap_pct"]
            >= thresholds["min_snap_pct"]
        )
        & (
            ranked["avg_opportunities_per_week"]
            >= thresholds["min_opportunities_per_week"]
        )
    ].copy()

    # Calculate the quality score only among eligible
    # players of the selected position and season.
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
        "Average Volatility",
        f"{average_volatility:.3f}",
    )

    r3.metric(
        "Highest Volatility",
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
            "offensive usage fluctuations."
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
                    "volatility_score",
                ]
            ],
            hide_index=True,
            use_container_width=True,
            column_config={
                "player_name": "Player",
                "team": "Team",
                "weeks": "Weeks",
                "volatility_score": "Volatility",
            },
        )

    # --------------------------------------------------------
    # MOST CONSISTENT PLAYERS
    # --------------------------------------------------------

    with ranking_right:

        st.subheader("📉 Most Consistent Players")

        st.caption(
            f"{ranking_position}s with stable offensive "
            "usage, significant playing time, and "
            "meaningful offensive opportunities."
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
                    "volatility_score": "Volatility",
                    "consistency_quality_score": "Quality Score",
                },
            )

        st.caption(
            f"Minimum requirements: "
            f"{thresholds['min_weeks']} recorded weeks, "
            f"{thresholds['min_snap_pct']:.0%} average snaps, "
            f"and {thresholds['min_opportunities_per_week']} "
            f"opportunities per week."
        )

    # --------------------------------------------------------
    # DETAILED COMPONENTS
    # --------------------------------------------------------

    with st.expander(
        "View Detailed Volatility Components"
    ):

        detailed = ranked[
            [
                "player_name",
                "position",
                "team",
                "weeks",
                "avg_snap_pct",
                "avg_opportunities_per_week",
                "snap_volatility",
                "opportunity_volatility",
                "passdown_volatility",
                "redzone_volatility",
                "volatility_score",
                "consistency_eligible",
            ]
        ].copy()

        numeric_columns = [
            "avg_snap_pct",
            "avg_opportunities_per_week",
            "snap_volatility",
            "opportunity_volatility",
            "passdown_volatility",
            "redzone_volatility",
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
    "NFL Volatility Analytics | 2022–2025 | RB · WR · TE"
)

st.caption(
    "Built with Python, Pandas, Plotly, and Streamlit."
)

st.caption(
    "Volatility measures historical usage variability. "
    "Consistency Quality Score additionally considers "
    "playing time and offensive opportunities."
)
