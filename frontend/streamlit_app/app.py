import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]

# The models live in src/rosterwatch/; put src/ on the import path so
# `streamlit run frontend/streamlit_app/app.py` finds them from any folder.
sys.path.insert(0, str(ROOT / "src"))

from rosterwatch.consistency import (  # noqa: E402
    calculate_volatility,
    summarize_model1_window,
)
from rosterwatch.data import load_data, load_ownership  # noqa: E402
from rosterwatch.settings import (  # noqa: E402
    CONSISTENCY_THRESHOLDS,
    MAX_PERCENT_OWNED,
    MIN_COMPARISONS,
    MODEL2_SETTINGS,
    POSITIONS,
    POSITION_WEIGHTS,
    VOLATILITY_METRICS,
    WAIVER_FLAGS,
)
from rosterwatch.waiver import calculate_waiver_model  # noqa: E402

# ============================================================
# CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="RosterWatch 🏈",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Streamlit reruns this script on every click, so the slow steps are cached:
# the same inputs return the stored result instead of recomputing.
load_data = st.cache_data(show_spinner="Loading NFL data...")(load_data)
load_ownership = st.cache_data(show_spinner="Loading ESPN ownership data...")(
    load_ownership
)
calculate_volatility = st.cache_data(show_spinner="Calculating weekly volatility...")(
    calculate_volatility
)
calculate_waiver_model = st.cache_data(
    show_spinner="Analyzing fantasy and usage trends..."
)(calculate_waiver_model)
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
WEEKS = list(range(1, 19))
PAGE_SIZE = 15
FLAG_ORDER = {flag: index for index, flag in enumerate(WAIVER_FLAGS)}
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
# PAGINATED TABLE
# ============================================================
def paginated_table(
    data,
    table_key,
    filter_signature,
    column_config=None,
):
    """Show a table 15 rows at a time with Previous / Next buttons.

    The current page is kept in ``st.session_state`` under ``table_key`` and goes
    back to page 1 whenever ``filter_signature`` changes (for example a new season
    or position).

    Parameters
    ----------
    data : pandas.DataFrame
        Rows to show.
    table_key : str
        Unique name for this table's page state and buttons.
    filter_signature : hashable
        The filter values the table depends on.
    column_config : dict, optional
        Streamlit column formats.
    """
    if data.empty:
        st.info("No players match these filters.")
        return
    total_pages = max(
        1,
        math.ceil(len(data) / PAGE_SIZE),
    )
    page_key = f"{table_key}_page"
    signature_key = f"{table_key}_signature"
    if st.session_state.get(signature_key) != filter_signature:
        st.session_state[signature_key] = filter_signature
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
    previous_col, page_col, next_col = st.columns([1, 1.4, 1])
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
            disabled=(current_page == total_pages),
            use_container_width=True,
        ):
            st.session_state[page_key] += 1
            st.rerun()
    st.caption(f"Showing {start + 1}–{min(end, len(data))} of {len(data)} players")


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
    """Build an 18-week chart of one metric for the Player Explorer.

    Weeks the player did not play are left as gaps, so every chart lines up on
    Weeks 1-18.

    Parameters
    ----------
    data : pandas.DataFrame
        One player's weekly rows.
    metric : str
        Column to plot.
    title : str
        Chart title.
    color : str
        Line or bar color (hex).
    percentage : bool, default False
        Show the y-axis as 0-100%.
    bar : bool, default False
        Draw bars instead of a line.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    series = data.groupby("week")[metric].mean().reindex(WEEKS)
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
    st.error("Could not find weekly_usage.parquet or weekly_usage.csv.")
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
st.sidebar.caption("Weekly Usage Volatility")
st.sidebar.markdown("**Waiver Wire Alerts**")
st.sidebar.caption("Fantasy PPR & Waiver Opportunity")

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
    st.write(
        "**Waiver Wire Alerts:** Every eligible player is assigned one of four waiver flags; Player Consistency Model rankings do not override the classification."
    )

    # ============================================================
    # ANALYTICS MODEL EXPLANATION
    # ============================================================
    with st.expander(
        "Our Analytics Models — Equations and Thresholds",
        expanded=True,
    ):
        st.subheader("Player Consistency Model — Weekly Volatility")
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
        st.subheader("Waiver Wire Alerts — Fantasy Waiver Opportunity")
        st.latex(
            r"W=100("
            r"0.40F"
            r"+0.20O"
            r"+0.15S"
            r"+0.15V"
            r"+0.10T)"
            r"\times E"
        )
        st.write("**F:** Fantasy PPR performance.")
        st.write("**O:** Opportunity share strength.")
        st.write("**S:** Offensive snap share strength.")
        st.write("**V:** Fantasy and usage stability.")
        st.write("**T:** Trend confirmation.")
        st.write(
            "**E:** Evidence factor, 0.80 + 0.20 × min(1, valid games ÷ (week − 1)). "
            "It lowers the score for players with few valid games; players with a "
            "limited history use a stricter version."
        )
        st.markdown("#### Component Weights")
        st.table(
            pd.DataFrame(
                {
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
                }
            )
        )
        st.markdown("#### Position Benchmarks")
        benchmark_table = pd.DataFrame(
            {
                pos: {
                    "PPR Points/Game": cfg["ppr"],
                    "Snap Share": cfg["snap"],
                    "Opportunity Share": cfg["opp"],
                    "Opportunities/Game": cfg["volume"],
                }
                for pos, cfg in MODEL2_SETTINGS.items()
            }
        ).T
        st.table(benchmark_table)
        st.info(
            "Waiver Wire Alerts use heuristic benchmarks and four directional waiver flags. Ownership is based on the ESPN snapshot, not verified historical weekly availability."
        )
with consistency_tab:
    # ============================================================
    # MODEL 1 DATA
    # ============================================================
    st.header("Player Consistency Model — Weekly Usage Volatility")
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
    full_model1_input = usage.loc[usage["season"] == season].copy()
    full_model1_weekly, _ = calculate_volatility(full_model1_input)
    model1_input = full_model1_weekly.loc[
        full_model1_weekly["week"].between(
            model1_weeks[0],
            model1_weeks[1],
        )
    ].copy()
    model1_summary = summarize_model1_window(model1_input)
    filtered_usage = model1_input.copy()
    filtered_summary = model1_summary.copy()
    if position != "All Positions":
        filtered_usage = filtered_usage.loc[filtered_usage["position"] == position]
        filtered_summary = filtered_summary.loc[
            filtered_summary["position"] == position
        ]

    # ============================================================
    # MUST START PLAYER IDS
    # ============================================================
    # Exactly the players listed in the
    # Model 1 consistency rankings.
    consistent_player_ids = tuple(
        sorted(
            model1_summary.loc[
                model1_summary["consistency_eligible"],
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
            row.gsis_id: (f"{row.player_name} · {row.position}")
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
        player_data = filtered_usage.loc[
            filtered_usage["gsis_id"] == player_id
        ].sort_values("week")
        player_summary = filtered_summary.loc[filtered_summary["gsis_id"] == player_id]
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
            (f"{volatility:.3f}" if pd.notna(volatility) else "N/A"),
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
    st.header("Player Consistency Model — Player Volatility Rankings")
    ranking_position = st.selectbox(
        "Ranking Position",
        POSITIONS,
        index=(POSITIONS.index(position) if position in POSITIONS else 0),
    )
    ranked = filtered_summary.loc[
        (filtered_summary["position"] == ranking_position)
        & filtered_summary["volatility_score"].notna()
    ].copy()
    eligible = ranked.loc[ranked["consistency_eligible"]].copy()
    a, b, c, d = st.columns(4)
    a.metric(
        "Players Scored",
        len(ranked),
    )
    b.metric(
        "Average Volatility",
        (f"{ranked['volatility_score'].mean():.3f}" if not ranked.empty else "N/A"),
    )
    c.metric(
        "Highest Volatility",
        (f"{ranked['volatility_score'].max():.3f}" if not ranked.empty else "N/A"),
    )
    d.metric(
        "Consistency Eligible",
        len(eligible),
    )
    left, right = st.columns(2)
    with left:
        st.subheader("📈 Most Volatile Players")
        volatile_display = ranked.sort_values(
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
        consistent_display = eligible.sort_values(
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
        min_weeks, min_snap, min_volume = CONSISTENCY_THRESHOLDS[ranking_position]
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
    st.header("Waiver Wire Alerts — Fantasy Waiver Opportunity")
    st.write(
        "Waiver Wire Alerts prioritize fantasy PPR production "
        "while analyzing offensive snap share, "
        "opportunity share, weekly volatility, "
        "and recent performance trends."
    )
    st.write(
        "**Four flags:** Must Pick Up, Slow Riser, Slow Drop, and Must Drop. All ownership-eligible players returned by Waiver Wire Alerts receive one flag."
    )
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
        st.error(
            "Missing espn_ownership.csv. Place it in data/processed/ or the repository root."
        )
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
    season_ownership["gsis_id"] = (
        season_ownership["gsis_id"].astype("string").str.strip()
    )
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
                active_ids = set(
                    last_rows.loc[last_rows["week"].ge(selected_week - 1), "gsis_id"]
                    .astype("string")
                    .str.strip()
                )
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
    model2_usage = model2_usage.loc[model2_usage["gsis_id"].isin(eligible_ids)].copy()
    all_waiver_results = calculate_waiver_model(
        model2_usage, season, selected_week, consistent_player_ids
    )
    all_waiver_results["gsis_id"] = (
        all_waiver_results["gsis_id"].astype("string").str.strip()
    )
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
        all_waiver_results = all_waiver_results.loc[
            all_waiver_results["position"] == position
        ].copy()
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
            waiver_results = all_waiver_results.copy()
        else:
            waiver_results = all_waiver_results.loc[
                all_waiver_results["waiver_flag"] == flag_filter
            ].copy()
        waiver_results = waiver_results.reset_index(drop=True)

        # --------------------------------------------------------
        # SORTING
        # --------------------------------------------------------
        waiver_results["sort_order"] = waiver_results["waiver_flag"].map(FLAG_ORDER)
        waiver_results = waiver_results.sort_values(
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
        ).reset_index(drop=True)
        st.caption(
            f"Showing {len(waiver_results):,} of {len(all_waiver_results):,} players."
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
        st.subheader("Player Waiver Trend Explorer")
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
                row.gsis_id: (f"{row.player_name} · {row.position}")
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
                waiver_results["gsis_id"] == selected_player_id
            ].iloc[0]
            history = usage.loc[
                (usage["season"] == season)
                & (usage["gsis_id"] == selected_player_id)
                & (usage["week"] <= selected_week)
            ].sort_values("week")
            st.markdown(f"### {selected_result['player_name']}")
            st.write(f"**Waiver Flag:** {selected_result['waiver_flag']}")
            score = selected_result["waiver_score"]
            a, b, c, d = st.columns(4)
            a.metric(
                "Waiver Score",
                (f"{score:.1f}/100" if pd.notna(score) else "N/A"),
            )
            b.metric(
                "Recent PPR Average",
                (
                    f"{selected_result['recent_ppr']:.2f}"
                    if pd.notna(selected_result["recent_ppr"])
                    else "N/A"
                ),
            )
            c.metric(
                "Season PPR Average",
                (
                    f"{selected_result['season_ppr']:.2f}"
                    if pd.notna(selected_result["season_ppr"])
                    else "N/A"
                ),
            )
            d.metric(
                "Usage Volatility",
                (
                    f"{selected_result['volatility']:.3f}"
                    if pd.notna(selected_result["volatility"])
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
                st.markdown("#### Player Trend Analysis")
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
                st.info("Insufficient data to calculate a complete waiver score.")

            # ----------------------------------------------------
            # WEEKLY HISTORY
            # ----------------------------------------------------
            with st.expander("View Full Player History"):
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
st.markdown("### RosterWatch 🏈")
st.caption(
    "Player Consistency Model: Weekly Usage Volatility | "
    "Waiver Wire Alerts: Fantasy PPR Waiver Opportunity"
)
st.caption("2022–2025 | RB · WR · TE | Weeks 1–18")
st.caption("Built with Python, Pandas, Plotly, and Streamlit.")
