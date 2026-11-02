
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RosterWatch 🏈 | NFL Usage Analytics",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"

POSITIONS = ["RB", "WR", "TE"]
WEEKS = list(range(1, 19))

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

[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 14px !important;
}

.hero {
    border: 1px solid #2b4967;
    border-radius: 18px;
    padding: 28px 32px;
    background: linear-gradient(
        115deg,
        #10243b,
        #15395b
    );
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
    transition: background .18s ease;
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

    scores = pd.read_csv(
        DATA_DIR / "player_scores.csv"
    )

    required_usage = {
        "season",
        "week",
        "gsis_id",
        "player_name",
        "position",
        "team",
        "snap_pct",
        "opp_share",
        "targets",
        "carries",
        "offense_snaps",
    }

    required_scores = {
        "season",
        "week",
        "player_name",
        "position",
        "team",
        "waiver_score",
        "role_score",
    }

    for name, frame, required in [
        ("weekly_usage.csv", usage, required_usage),
        ("player_scores.csv", scores, required_scores),
    ]:

        missing = required - set(frame.columns)

        if missing:
            raise ValueError(
                f"{name} is missing columns: "
                + ", ".join(sorted(missing))
            )

        for col in ("season", "week"):
            frame[col] = pd.to_numeric(
                frame[col],
                errors="coerce",
            )

        frame["position"] = (
            frame["position"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

    for col in [
        "snap_pct",
        "opp_share",
        "targets",
        "carries",
        "offense_snaps",
    ]:
        usage[col] = pd.to_numeric(
            usage[col],
            errors="coerce",
        )

    for col in ["waiver_score", "role_score"]:
        scores[col] = pd.to_numeric(
            scores[col],
            errors="coerce",
        )

    usage = usage.loc[
        usage["season"].between(2022, 2025)
        & usage["week"].between(1, 18)
        & usage["position"].isin(POSITIONS)
    ].copy()

    scores = scores.loc[
        scores["season"].between(2022, 2025)
        & scores["week"].between(1, 18)
        & scores["position"].isin(POSITIONS)
    ].copy()

    usage["week"] = usage["week"].astype(int)
    scores["week"] = scores["week"].astype(int)

    usage["gsis_id"] = usage["gsis_id"].astype("string")

    return usage, scores


try:
    usage, scores = load_data()

except (OSError, ValueError, pd.errors.ParserError) as error:

    st.error(
        f"Could not load data: {error}"
    )

    st.info(
        "Check data/processed/weekly_usage.csv "
        "and data/processed/player_scores.csv."
    )

    st.stop()


if usage.empty:
    st.warning(
        "No player usage records were found for 2022–2025."
    )
    st.stop()


# ============================================================
# 4. WEBSITE HEADER — HTML FIXED
# ============================================================

hero_html = """
<div class="hero">
<div class="hero-title">RosterWatch 🏈</div>
<div class="hero-subtitle">
NFL Player Usage Intelligence | 2022–2025
</div>
<div class="hero-description">
Understand how NFL offensive roles evolve throughout a season.
Explore playing time, opportunity share, and weekly trends for
running backs, wide receivers, and tight ends.
</div>
<div class="pills">
<span class="pill">2022–2025 seasons</span>
<span class="pill">RB / WR / TE</span>
<span class="pill">Weeks 1–18</span>
<span class="pill">Usage analytics</span>
</div>
</div>
"""

st.markdown(
    hero_html.strip(),
    unsafe_allow_html=True,
)


# ============================================================
# 5. ABOUT ROSTERWATCH
# ============================================================

st.header("About RosterWatch")

st.write(
    "RosterWatch is an interactive NFL analytics platform "
    "that highlights changing offensive roles and emerging "
    "opportunities. It focuses on how often players take "
    "the field and what share of their team's offensive "
    "workload they receive, rather than only traditional "
    "box-score totals."
)


# ============================================================
# 6. ANALYTICS MODELS
# ============================================================

st.header("Our Analytics Models")

st.write(
    "Two experimental models compare recent player usage "
    "with an earlier baseline."
)

left, right = st.columns(2)

with left:

    with st.container(border=True):

        st.subheader("Waiver Opportunity Model")

        st.write(
            "Highlights players whose offensive involvement "
            "has increased, considering snaps, opportunities, "
            "passing-down usage, and red-zone usage."
        )

        st.markdown(
            "**Higher score:** Stronger measured usage growth."
        )

        st.caption(
            "Not a fantasy-point prediction or a check "
            "of waiver availability."
        )


with right:

    with st.container(border=True):

        st.subheader("Role Instability Model")

        st.write(
            "Highlights players whose offensive involvement "
            "has declined compared with their earlier games."
        )

        st.markdown(
            "**Higher score:** Greater measured usage decline."
        )

        st.caption(
            "Historical indicator, not a forecast "
            "of future performance."
        )


with st.expander("How the models work"):

    st.write(
        "The models compare the two most recent games "
        "with an earlier three-game baseline."
    )

    st.markdown(
        """
| Metric | Weight |
|---|---:|
| Offensive snap percentage | 35% |
| Opportunity share | 30% |
| Passing-down opportunity share | 20% |
| Red-zone opportunity share | 15% |
"""
    )

    st.caption(
        "Early weeks may have no scores because the "
        "models need a historical baseline. "
        "These are experimental indicators."
    )


# ============================================================
# 7. SIDEBAR CONTROLS
# ============================================================

st.sidebar.title("Control Center")

season_options = sorted(
    usage["season"]
    .dropna()
    .astype(int)
    .unique()
    .tolist(),
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

st.sidebar.markdown("**Data Coverage**")

st.sidebar.caption(
    "2022–2025 · RB / WR / TE · "
    "Player charts: Weeks 1–18"
)


# ============================================================
# 8. FILTER DATA
# ============================================================

filtered_usage = usage.loc[
    usage["season"] == season
].copy()

filtered_scores = scores.loc[
    scores["season"] == season
].copy()

if position != "All Positions":

    filtered_usage = filtered_usage.loc[
        filtered_usage["position"] == position
    ]

    filtered_scores = filtered_scores.loc[
        filtered_scores["position"] == position
    ]


# ============================================================
# 9. SEASON OVERVIEW
# ============================================================

st.divider()

st.header("📊 Season Overview")

st.caption(
    f"{season} NFL season · {position}"
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

with st.expander("What do these metrics mean?"):

    st.write(
        "Player-week records count weekly player entries. "
        "Unique players count distinct player IDs. "
        "Offensive snaps and targets are totals across "
        "the current selection."
    )


# ============================================================
# 10. PLAYER EXPLORER
# ============================================================

st.divider()

st.header("🔎 Player Explorer")

st.write(
    "Choose a player to see weekly usage. "
    "All four charts display Weeks 1–18, "
    "even when some weeks have no recorded data."
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

    st.info(
        "No players match this season and position."
    )

else:

    labels = {
        row.gsis_id: f"{row.player_name} · {row.position}"
        for row in players.itertuples(index=False)
    }

    player_id = st.selectbox(
        "Select an NFL Player",
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
        .copy()
    )

    player_name = str(
        player_data["player_name"].iloc[0]
    )

    st.subheader(
        f"{player_name} — {season} Weekly Usage"
    )

    # --------------------------------------------------------
    # PLAYER SUMMARY METRICS
    # --------------------------------------------------------

    m1, m2, m3 = st.columns(3)

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

    # --------------------------------------------------------
    # INTERACTIVE CHART CONTROLS
    # --------------------------------------------------------

    controls1, controls2 = st.columns([2, 1])

    with controls1:

        highlighted_week = st.slider(
            "Highlight NFL week",
            min_value=1,
            max_value=18,
            value=1,
        )

    with controls2:

        chart_style = st.selectbox(
            "Chart style",
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
                    marker=dict(
                        size=7,
                    ),
                    hovertemplate=hover,
                )
            )

        # Highlight selected week
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

            font=dict(
                color=COLORS["text"],
            ),

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

    # --------------------------------------------------------
    # WEEK DATA CHECKER
    # --------------------------------------------------------

    with st.expander(
        "📚 Understanding the graphs and weekly records"
    ):

        st.write(
            "Snap percentage is the share of team "
            "offensive snaps played. Opportunity share "
            "is the share of team carries and targets. "
            "Targets and carries are weekly counts."
        )

        st.write(
            "Missing records are not zero values."
        )

        present = sorted(
            player_data["week"]
            .unique()
            .tolist()
        )

        missing = [
            week
            for week in WEEKS
            if week not in present
        ]

        st.write(
            "Weeks with records:",
            ", ".join(map(str, present)) or "None",
        )

        st.write(
            "Weeks without records:",
            ", ".join(map(str, missing)) or "None",
        )

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
# 11. PLAYER RANKINGS
# ============================================================

st.divider()

st.header("🚀 Player Role & Opportunity Rankings")

st.write(
    "Compare historical usage-change scores "
    "for a selected week."
)

score_weeks = sorted(
    filtered_scores["week"]
    .dropna()
    .astype(int)
    .unique()
    .tolist(),
    reverse=True,
)

if not score_weeks:

    st.info(
        "No scored player records match "
        "the current filters."
    )

else:

    selected_week = st.selectbox(
        "Scoring Week",
        score_weeks,
        format_func=lambda week: f"Week {week}",
    )

    week_scores = filtered_scores.loc[
        filtered_scores["week"] == selected_week
    ].copy()

    left, right = st.columns(2)

    rankings = [
        (
            left,
            "📈 Waiver Opportunity",
            "waiver_score",
            "Largest measured increases in recent usage.",
        ),
        (
            right,
            "📉 Role Instability",
            "role_score",
            "Largest measured declines in recent usage.",
        ),
    ]

    for (
        column,
        title,
        score_column,
        explanation,
    ) in rankings:

        with column:

            st.subheader(title)

            st.caption(explanation)

            leaders = (
                week_scores
                .dropna(subset=[score_column])
                .sort_values(
                    score_column,
                    ascending=False,
                )
                .head(15)
                .copy()
            )

            leaders[score_column] = (
                leaders[score_column].round(2)
            )

            st.dataframe(
                leaders[
                    [
                        "player_name",
                        "position",
                        "team",
                        score_column,
                    ]
                ],
                hide_index=True,
                use_container_width=True,
                column_config={
                    "player_name": "Player",
                    "position": "Position",
                    "team": "Team",
                    score_column: "Score",
                },
            )

    st.caption(
        "Scores describe historical usage changes, "
        "not fantasy-point predictions "
        "or waiver availability."
    )


# ============================================================
# 12. FOOTER
# ============================================================

st.divider()

st.markdown("### RosterWatch 🏈")

st.caption(
    "NFL Usage Analytics | 2022–2025 | RB · WR · TE"
)

st.caption(
    "Built with Python, Pandas, Plotly, and Streamlit."
)
