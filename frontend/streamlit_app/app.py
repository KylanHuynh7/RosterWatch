
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
    CHANGE_COLUMNS,
    CONSISTENCY_THRESHOLDS,
    MAX_WEEKLY_CHANGE,
    METRICS,
    MIN_GAMES_SHARE,
    MIN_RANKED_GAMES_SHARE,
    MIN_ROLE_SNAP_PCT,
    PART_COLUMNS,
    POSITION_WEIGHTS,
    POSITIONS,
    QUALITY_COLUMNS,
    QUALITY_WEIGHTS,
    RELATIVE_FLOOR,
    RELATIVE_METRICS,
    calculate_volatility as run_volatility_model,
    consistency_quality,
    season_scores,
)

WEEKS = list(range(1, 19))

# Rows shown in each leaderboard.
TOP_N = 15

METRIC_LABELS = {
    "snap_pct": "Snap Percentage",
    "opp_share": "Opportunity Share",
    "passdown_share": "Passing-Down Share",
    "redzone_share": "Red-Zone Share",
}

# Short names for the parts of the weekly score.
METRIC_SHORT = {
    "snap_pct": "Snap",
    "opp_share": "Opportunity",
    "passdown_share": "Passing-down",
    "redzone_share": "Red-zone",
}

METRIC_SYMBOLS = {
    "snap_pct": "S",
    "opp_share": "O",
    "passdown_share": "P",
    "redzone_share": "R",
}

QUALITY_LABELS = {
    "low_volatility": "Steadiness",
    "snap_pct": "Snaps",
    "opportunities": "Touches",
}

# Blue-gradient palette; the CSS below uses the same colors.
THEME = {
    "card": "#13243B",
    "muted": "#152841",
    "border": "#2B4967",
    "grid": "#2A405A",
    "text": "#E2E8F0",
    "muted_text": "#A7BFD7",
    "accent": "#38BDF8",
}

# Each metric keeps its color on every chart.
METRIC_COLORS = {
    "snap_pct": "#38BDF8",
    "opp_share": "#4ADE80",
    "passdown_share": "#F59E0B",
    "redzone_share": "#A78BFA",
}

COUNT_COLORS = {
    "targets": "#F59E0B",
    "carries": "#A78BFA",
}

# Shares are stored as 0-1 and shown as percentages. Volatility is shown
# as a % of a normal week (1.0 = 100%), and weekly changes with a sign.
PERCENT_FORMAT = "%.1f%%"
VOLATILITY_FORMAT = "%.0f%%"
CHANGE_FORMAT = "%+.1f%%"


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
# 5. HELPERS
# ============================================================

def is_relative(metric):
    """True when a metric's change is relative to the player's share."""
    return metric in RELATIVE_METRICS


def format_change(metric, change, signed=False):
    """A weekly change as a percentage.

    Snap-type changes are in percentage points (77% -> 91% is 14.0%).
    Relative changes are a % of the player's own share.
    """
    sign = "+" if signed else ""
    if is_relative(metric):
        return f"{change:{sign}.0%}"
    return f"{change * 100:{sign}.1f}%"


def weighted_metrics(position):
    """Metrics with a nonzero weight for a position."""
    return [
        metric
        for metric in METRICS
        if POSITION_WEIGHTS[position][metric] > 0
    ]


def same_weights_everywhere():
    """True when every position uses the same weights."""
    return len({str(POSITION_WEIGHTS[p]) for p in POSITIONS}) == 1


def weight_summary():
    """e.g. "snap percentage (70%) and opportunity share (30%)"."""
    if not same_weights_everywhere():
        return "usage metrics, with weights set by position"
    weights = POSITION_WEIGHTS[POSITIONS[0]]
    parts = [
        f"{METRIC_LABELS[metric].lower()} ({weights[metric]:.0%})"
        for metric in weighted_metrics(POSITIONS[0])
    ]
    if len(parts) <= 2:
        return " and ".join(parts)
    return ", ".join(parts[:-1]) + ", and " + parts[-1]


def score_formula(position):
    """e.g. "0.70 × Snap score + 0.30 × Opportunity score"."""
    return " + ".join(
        f"{POSITION_WEIGHTS[position][metric]:.2f} × "
        f"{METRIC_SHORT[metric]} score"
        for metric in weighted_metrics(position)
    )


def score_latex(position):
    """The weekly score as weights times scores, in LaTeX."""
    terms = [
        rf"{POSITION_WEIGHTS[position][metric]:.2f}\times"
        rf"\text{{{METRIC_SHORT[metric]} score}}"
        for metric in weighted_metrics(position)
    ]
    return r"V_t=" + "+".join(terms)


def position_latex(position):
    """One position's weekly score with its weights and normal changes."""
    terms = []
    for metric in weighted_metrics(position):
        symbol = METRIC_SYMBOLS[metric]
        change = rf"|{symbol}_t-{symbol}_{{t-1}}|"
        if is_relative(metric):
            change = rf"{change}\,/\,\bar{{{symbol}}}_t"
        normal = format_change(
            metric, typical_change.loc[position, metric]
        ).replace("%", r"\%")
        terms.append(
            rf"{POSITION_WEIGHTS[position][metric]:.2f}\cdot"
            rf"\frac{{{change}}}{{{normal}}}"
        )
    return rf"V_{{{position},t}}=" + "+".join(terms)


def metric_table(values):
    """Table with one row per metric and one column per position."""
    return pd.DataFrame({
        "Metric": [METRIC_LABELS[metric] for metric in METRICS],
        "Change Measured As": [
            "% of player's share" if is_relative(metric)
            else "Percentage points"
            for metric in METRICS
        ],
        **{
            position: [values(position, metric) for metric in METRICS]
            for position in POSITIONS
        },
    })


def normal_week_table():
    """Each weighted metric's weight and normal change, by position."""
    rows = {}
    for metric in METRICS:
        if not any(POSITION_WEIGHTS[p][metric] > 0 for p in POSITIONS):
            continue
        rows[f"{METRIC_LABELS[metric]}: weight"] = {
            position: f"{POSITION_WEIGHTS[position][metric]:.0%}"
            for position in POSITIONS
        }
        rows[f"{METRIC_LABELS[metric]}: normal weekly change"] = {
            position: format_change(
                metric, typical_change.loc[position, metric]
            )
            for position in POSITIONS
        }
    return pd.DataFrame(rows).T


def explain_change(metric, before, now, change, typical, position):
    """e.g. "77% → 91% (+14.0%) ÷ 12.4% (a normal RB week)"."""
    text = f"{before:.0%} → {now:.0%} ({(now - before) * 100:+.1f}%)"
    if is_relative(metric):
        average = max((before + now) / 2, RELATIVE_FLOOR)
        text += (
            f", a {change:.0%} change for his {average:.0%} "
            "average share"
        )
    elif abs(now - before) > MAX_WEEKLY_CHANGE + 1e-9:
        text += f", counted as {MAX_WEEKLY_CHANGE:.0%} (the cap)"
    return (
        text
        + f" ÷ {format_change(metric, typical)} (a normal {position} week)"
    )


# Made-up week used to show the arithmetic.
EXAMPLE_POSITION = "RB"
EXAMPLE_WEEK = {
    "snap_pct": (0.60, 0.75),
    "opp_share": (0.20, 0.30),
    "passdown_share": (0.30, 0.40),
    "redzone_share": (0.20, 0.30),
}


def example_change(metric, before, now):
    """A change in the units of weekly_changes in src/volatility.py."""
    if is_relative(metric):
        return abs(now - before) / max((before + now) / 2, RELATIVE_FLOOR)
    return min(abs(now - before), MAX_WEEKLY_CHANGE)


def ranked_players(scores, position):
    """Players in the volatility rankings, most volatile first.

    The rules (snap minimum and share of team games played) are set in
    src/volatility.py (season_scores).
    """
    return scores.loc[
        (scores["position"] == position)
        & scores["volatility_ranked"]
    ].sort_values("volatility_score", ascending=False)


def quality_pool(ranked):
    """Consistency-eligible players with their quality score, best first.

    Eligibility is set in src/volatility.py (season_scores). Percentiles
    are taken among all of these players, so a player's quality score
    doesn't change when the leaderboard's snap slider hides others.
    """
    pool = consistency_quality(ranked.loc[ranked["consistency_eligible"]])
    return pool.sort_values(
        ["quality_score", "volatility_score"],
        ascending=[False, True],
    )


def quality_breakdown(row):
    """e.g. "0.50 × 93.1 + 0.25 × 100.0 + 0.25 × 100.0 = 96.6"."""
    terms = " + ".join(
        f"{QUALITY_WEIGHTS[part]:.2f} × {row[column]:.1f}"
        for part, column in QUALITY_COLUMNS.items()
    )
    return f"{terms} = {row['quality_score']:.1f}"


def table_height(table):
    """Pixel height that shows every row of a leaderboard table."""
    return (len(table) + 1) * 35 + 3


def style_figure(fig, height):
    """Shared look for every chart."""
    fig.update_layout(
        paper_bgcolor=THEME["card"],
        plot_bgcolor=THEME["card"],
        font=dict(color=THEME["text"], size=13),
        title_font=dict(size=16),
        height=height,
        margin=dict(l=55, r=20, t=60, b=50),
        hoverlabel=dict(
            bgcolor=THEME["muted"],
            bordercolor=THEME["border"],
            font=dict(color=THEME["text"]),
        ),
        xaxis=dict(
            title="NFL Week",
            tickmode="array",
            tickvals=WEEKS,
            range=[0.5, 18.5],
            gridcolor=THEME["grid"],
            linecolor=THEME["border"],
            zeroline=False,
        ),
        yaxis=dict(
            gridcolor=THEME["grid"],
            linecolor=THEME["border"],
            zeroline=False,
            rangemode="tozero",
        ),
    )
    return fig


def chart_top(series):
    """Highest value on a chart, at least 1, for placing labels."""
    top = series.max(skipna=True)
    return 1.0 if pd.isna(top) else max(float(top), 1.0)


def label_week(fig, week, value, text, top, marker=True):
    """Label the highlighted week on a chart.

    Args:
        fig: the chart.
        week: the highlighted NFL week.
        value: the y value to point at, or NaN when the week has none
            (the label then sits at the top of the chart).
        text: the label, e.g. "Wk 3 · 91%".
        top: the highest y value on the chart, to keep the label inside.
        marker: also draw a dot on the point.
    """
    # Keep labels near the edges from running off the chart.
    if week <= 2:
        xanchor = "left"
    elif week >= 17:
        xanchor = "right"
    else:
        xanchor = "center"

    box = dict(
        font=dict(size=12, color=THEME["text"]),
        bgcolor=THEME["muted"],
        bordercolor=THEME["muted_text"],
        borderpad=4,
        xanchor=xanchor,
    )

    if pd.isna(value):
        fig.add_annotation(
            x=week,
            y=top,
            text=text,
            showarrow=False,
            yanchor="top",
            **box,
        )
        return

    if marker:
        fig.add_trace(
            go.Scatter(
                x=[week],
                y=[value],
                mode="markers",
                marker=dict(
                    size=12,
                    color=THEME["text"],
                    line=dict(width=2, color=THEME["card"]),
                ),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.add_annotation(
        x=week,
        y=value,
        text=text,
        showarrow=True,
        arrowhead=0,
        arrowwidth=1,
        arrowcolor=THEME["muted_text"],
        ax=0,
        # Above the point, or below it when the point is near the top.
        ay=-36 if value < 0.75 * top else 36,
        **box,
    )


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
    "RosterWatch is an interactive NFL analytics dashboard that tracks "
    "how players' offensive involvement changes from one game to the "
    "next. Rather than calculating volatility using season-long "
    "standard deviation, the model compares each game with the "
    "player's previous game."
)

st.write(
    "Each week's score is a weighted average of the change in "
    f"{weight_summary()}, each compared with a normal week's change "
    "for the player's position. 100% means a normal week for a "
    "running back, wide receiver, or tight end, and 200% means twice "
    "the usual change. The consistency rankings also account for "
    "meaningful playing time and offensive opportunities."
)

with st.expander(
    "How the Weekly Volatility Model works: the equation and its weights",
):

    st.markdown("#### Weekly Volatility Equation")

    st.write(
        "Each week's score is a weighted average. For each metric, take "
        "this week minus the player's previous game, divide it by a "
        "normal week's change for his position, and multiply by the "
        "metric's weight. Then add the metrics together."
    )

    st.latex(
        r"V_t = \sum_{m} w_m"
        r"\frac{\Delta m_t}{\bar{\Delta}_m}"
    )

    st.markdown("**Variables:**")

    scored_metrics_any = [
        m for m in METRICS
        if any(POSITION_WEIGHTS[p][m] > 0 for p in POSITIONS)
    ]
    unscored_metrics = [m for m in METRICS if m not in scored_metrics_any]
    metric_names = " or ".join(
        f"{METRIC_LABELS[m].lower()} ({METRIC_SYMBOLS[m]})"
        for m in scored_metrics_any
    )
    weights_text = (
        ", ".join(
            f"{METRIC_LABELS[m].lower()} {POSITION_WEIGHTS[POSITIONS[0]][m]:.0%}"
            for m in weighted_metrics(POSITIONS[0])
        )
        if same_weights_everywhere()
        else "set by position (see the table below)"
    )
    example_snap = EXAMPLE_WEEK["snap_pct"]
    example_opp = EXAMPLE_WEEK["opp_share"]

    st.markdown(
        f"""
- **Vₜ:** the volatility score for week t, as a % of a normal week for
  the position: 100% = a normal week, 200% = twice the usual change.
- **m:** a usage metric in the score: {metric_names}.
- **wₘ:** the metric's weight: {weights_text}. The weights add up
  to 100%.
- **Δmₜ:** the change since the player's previous game.
    - **Snap percentage:** |Sₜ − Sₜ₋₁|, in percentage points.
      {example_snap[0]:.0%} → {example_snap[1]:.0%} is a
      {(example_snap[1] - example_snap[0]) * 100:.0f}% change. At most
      {MAX_WEEKLY_CHANGE:.0%} counts per game.
    - **Opportunity share** (his carries + targets ÷ the team's):
      |Oₜ − Oₜ₋₁| divided by the average of the two games (Ōₜ), so the
      change is relative to his own workload.
      {example_opp[0]:.0%} → {example_opp[1]:.0%} is
      {(example_opp[1] - example_opp[0]) * 100:.0f} points on a
      {sum(example_opp) / 2:.0%} average: a
      {example_change('opp_share', *example_opp):.0%} change. A backup
      going 5% → 10% counts as much as a starter going 20% → 40%.
- **Δ̄ₘ:** a normal week's change in that metric for the position: the
  average game-to-game change in
  {BASELINE_SEASONS[0]}–{BASELINE_SEASONS[-1]} among players averaging
  {MIN_ROLE_SNAP_PCT:.0%}+ of snaps.
"""
    )

    # Why the zero-weight metrics are left out (see POSITION_WEIGHTS in
    # src/volatility.py).
    unscored_reasons = {
        "passdown_share": "added nothing on top of snap percentage and "
        "opportunity share when tested",
        "redzone_share": "is 0 in most games",
    }
    if unscored_metrics:
        st.caption(
            " and ".join(METRIC_LABELS[m] for m in unscored_metrics)
            + " are not in the score (0% weight) and are shown in each "
            "player's weekly data only. "
            + "; ".join(
                f"{METRIC_LABELS[m]} {unscored_reasons[m]}"
                for m in unscored_metrics
                if m in unscored_reasons
            )
            + "."
        )

    st.markdown("#### Why divide by a normal week?")

    st.write(
        "Snap percentage and opportunity share change by different "
        f"amounts: in a normal {EXAMPLE_POSITION} week, snap percentage "
        "moves "
        + format_change(
            "snap_pct", typical_change.loc[EXAMPLE_POSITION, "snap_pct"]
        )
        + " and opportunity share moves "
        + format_change(
            "opp_share", typical_change.loc[EXAMPLE_POSITION, "opp_share"]
        )
        + " of the player's share. Dividing each change by its normal "
        "size puts both on the same scale (100% = normal), so the "
        "weights really decide how much each metric counts."
    )

    st.markdown("#### Position Formulas")

    st.write(
        "The same equation with each position's weights and "
        "normal-week changes filled in:"
    )

    for formula_position in POSITIONS:
        st.latex(position_latex(formula_position))

    st.write(
        "In words, each metric gets a score: its change ÷ a normal "
        "week's change, so 100% = normal. Then"
    )

    if same_weights_everywhere():
        st.latex(score_latex(POSITIONS[0]))

    st.table(normal_week_table())

    example_terms = []
    example_steps = []
    for metric in weighted_metrics(EXAMPLE_POSITION):
        before, now = EXAMPLE_WEEK[metric]
        change = example_change(metric, before, now)
        typical = typical_change.loc[EXAMPLE_POSITION, metric]
        score = change / typical
        example_terms.append(
            f"{POSITION_WEIGHTS[EXAMPLE_POSITION][metric]:.2f} × "
            f"{score:.0%}"
        )
        example_steps.append(
            f"- **{METRIC_SHORT[metric]} score {score:.0%}:** "
            f"{METRIC_LABELS[metric].lower()} "
            + explain_change(
                metric, before, now, change, typical, EXAMPLE_POSITION
            )
        )
    example_total = sum(
        POSITION_WEIGHTS[EXAMPLE_POSITION][metric]
        * example_change(metric, *EXAMPLE_WEEK[metric])
        / typical_change.loc[EXAMPLE_POSITION, metric]
        for metric in weighted_metrics(EXAMPLE_POSITION)
    )

    st.markdown(
        f"**Example ({EXAMPLE_POSITION}):** weekly volatility = "
        + " + ".join(example_terms)
        + f" = **{example_total:.0%}** of a normal week.\n\n"
        + "\n".join(example_steps)
    )

    st.markdown("#### Average Weekly Volatility")

    st.write(
        "To compare players over the selected weeks, RosterWatch "
        "averages their weekly scores."
    )

    st.latex(
        r"\bar{V}=\frac{1}{m}"
        r"\sum_{t \in T} V_t"
    )

    st.caption(
        "T contains the weeks compared with the player's previous "
        "game, and m is the number of those comparisons."
    )

    above, below = st.columns(2)

    with above:

        st.markdown("**Above 100%**")

        st.write(
            "The player's offensive role changed more than in a "
            "normal week for the position."
        )

    with below:

        st.markdown("**Below 100%**")

        st.write(
            "The player's offensive role stayed steadier than in a "
            "normal week for the position."
        )


with st.expander(
    "How players are ranked: thresholds and the quality score",
):

    st.markdown("#### All Metrics")

    st.table(
        metric_table(
            lambda position, metric:
            f"{POSITION_WEIGHTS[position][metric]:.0%} weight, "
            + format_change(metric, typical_change.loc[position, metric])
            + " normal change"
        )
    )

    st.caption(
        "Normal change: the average change from one game to the next "
        f"in {BASELINE_SEASONS[0]}–{BASELINE_SEASONS[-1]}, for players "
        f"averaging at least {MIN_ROLE_SNAP_PCT:.0%} of snaps. A "
        f"percentage-point change counts at most {MAX_WEEKLY_CHANGE:.0%} "
        "per game. A change relative to the player's share is divided "
        "by his average share across the two games (at least "
        f"{RELATIVE_FLOOR:.0%}), so it tops out at 200% (going to or "
        "from zero)."
    )

    st.markdown("#### Who Gets Ranked")

    st.table(
        pd.DataFrame({
            "Requirement": [
                "Played (share of his team's games)",
                "Average snap %",
                "Carries + targets per game",
            ],
            "Most Volatile": [
                f"{MIN_RANKED_GAMES_SHARE:.0%}+",
                f"{MIN_ROLE_SNAP_PCT:.0%}+",
                "—",
            ],
            **{
                f"Most Consistent {position}": [
                    f"{MIN_GAMES_SHARE:.0%}+",
                    f"{CONSISTENCY_THRESHOLDS[position]['min_snap_pct']:.0%}+",
                    f"{CONSISTENCY_THRESHOLDS[position]['min_opportunities_per_week']}+",
                ]
                for position in POSITIONS
            },
        })
    )

    st.caption(
        "Every requirement is a share or a per-game average over the "
        "selected weeks, so the same rules work for any week range. "
        "Games played count his team's games only, so a bye week "
        f"doesn't count against him: {MIN_GAMES_SHARE:.0%} is 10 of 17 "
        "games for a full season and 3 of 4 for Weeks 1–4. Each list's "
        "snap % slider can raise its minimum."
    )

    st.markdown("#### Consistency Quality Score")

    st.latex(
        rf"Q={QUALITY_WEIGHTS['low_volatility']:.2f}\,C"
        rf"+{QUALITY_WEIGHTS['snap_pct']:.2f}\,S"
        rf"+{QUALITY_WEIGHTS['opportunities']:.2f}\,O"
    )

    st.write(
        "C, S and O are percentiles from 0 to 100 among the players who "
        "meet the consistency requirements at the same position, for "
        "the selected season and weeks. C ranks low volatility (the "
        "steadiest player gets 100), S ranks average snap percentage "
        "and O ranks carries + targets per game. The Most Consistent "
        "table shows all three as Steadiness, Snaps and Touches. A "
        "score of 100 would mean the steadiest player also has the "
        "largest role."
    )

    st.info(
        "A player's first game has nothing to compare with. "
        "A bye week doesn't break the comparison, but a missed "
        "game does: the game after an absence isn't compared "
        "with the one before it. Missing data is never treated "
        "as zero volatility."
    )

    st.caption(
        "Weights, thresholds, and quality-score weights are set in "
        "src/volatility.py; every number above is read from there."
    )


# ============================================================
# 8. SIDEBAR
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
    usage.loc[usage["season"] == season, "week"].max()
)
season_weeks = [week for week in WEEKS if week <= season_last_week]

from_column, to_column = st.sidebar.columns(2)

first_week = from_column.selectbox(
    "From Week",
    season_weeks,
    index=0,
)

to_options = [week for week in season_weeks if week >= first_week]

last_week = to_column.selectbox(
    "To Week",
    to_options,
    index=len(to_options) - 1,
)

weeks_covered = last_week - first_week + 1
full_season = first_week == 1 and last_week == season_last_week
range_label = (
    "Season"
    if full_season
    else f"Weeks {first_week}–{last_week}"
)

st.sidebar.caption(
    "Volatility uses changes between games inside this range: "
    "Weeks 2–3 is the change from Week 2 to Week 3."
)

position = st.sidebar.selectbox(
    "Position",
    ["All Positions"] + POSITIONS,
)

st.sidebar.divider()

st.sidebar.markdown("**Weekly Volatility Model**")
st.sidebar.caption(
    f"Weighted change in {weight_summary()} since the player's "
    "previous game, compared with a normal week for his position: "
    "100% = normal."
)

st.sidebar.markdown("**Ranking Method**")
st.sidebar.caption(
    "Average weekly volatility + meaningful usage thresholds"
)

st.sidebar.markdown("**Data Coverage**")
st.sidebar.caption(
    "2022–2025 · RB / WR / TE · Weeks 1–18"
)


# ============================================================
# 9. FILTER DATA
# ============================================================

@st.cache_data(show_spinner=False)
def window_scores(first, last):
    """Season scores using only games from Week first to Week last."""
    return season_scores(weekly_volatility, first, last)


# Full season for one position: used for the weekly charts, which
# show the whole season with the selected weeks shaded.
season_usage = usage.loc[
    usage["season"] == season
].copy()

# All positions, for the rankings (which pick their own position).
season_volatility = window_scores(first_week, last_week)
season_volatility = season_volatility.loc[
    season_volatility["season"] == season
].copy()

filtered_volatility = season_volatility.copy()

filtered_weekly = weekly_volatility.loc[
    weekly_volatility["season"] == season
].copy()

if position != "All Positions":

    season_usage = season_usage.loc[
        season_usage["position"] == position
    ]

    filtered_volatility = filtered_volatility.loc[
        filtered_volatility["position"] == position
    ]

    filtered_weekly = filtered_weekly.loc[
        filtered_weekly["position"] == position
    ]

# Selected weeks only: used for totals, averages and the player list.
filtered_usage = season_usage.loc[
    season_usage["week"].between(first_week, last_week)
].copy()


def shade_selected_weeks(fig):
    """Shade the From-To weeks on a weekly chart."""
    if full_season:
        return
    fig.add_vrect(
        x0=first_week - 0.5,
        x1=last_week + 0.5,
        fillcolor="rgba(56,189,248,0.10)",
        line_width=0,
        layer="below",
    )


def mark_highlighted_week(fig):
    """Outline the week picked with the Highlight NFL Week slider."""
    fig.add_vrect(
        x0=highlighted_week - 0.45,
        x1=highlighted_week + 0.45,
        fillcolor="rgba(255,255,255,0.07)",
        line=dict(color=THEME["muted_text"], width=1, dash="dot"),
        layer="below",
    )


# ============================================================
# 10. SEASON OVERVIEW
# ============================================================

st.divider()

st.header("📊 Season Overview")

st.caption(f"{season} NFL Season · {range_label} · {position}")

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
# 11. PLAYER EXPLORER
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

    player_options = players["gsis_id"].tolist()

    # Keep the selected player when the season, weeks or position
    # change, as long as he played in the new selection.
    selected_player = st.session_state.get("selected_player")

    player_id = st.selectbox(
        "Select an NFL Player",
        player_options,
        index=(
            player_options.index(selected_player)
            if selected_player in player_options
            else 0
        ),
        format_func=lambda pid: labels.get(pid, str(pid)),
    )

    st.session_state["selected_player"] = player_id

    # Selected weeks, for the player's averages and totals.
    player_data = (
        filtered_usage.loc[
            filtered_usage["gsis_id"] == player_id
        ]
        .sort_values("week")
        .copy()
    )

    # Whole season, for the weekly charts.
    player_season = (
        season_usage.loc[
            season_usage["gsis_id"] == player_id
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

    st.caption(
        f"Averages, totals and volatility: {range_label.lower()}"
        if full_season
        else f"Averages, totals and volatility: {range_label} "
        "(shaded on the charts)"
    )

    m1, m2, m3, m4 = st.columns(4)

    mean_snap = player_data["snap_pct"].mean()

    m1.metric(
        "Average Snap Percentage",
        f"{mean_snap:.1%}"
        if pd.notna(mean_snap)
        else "—",
        help="Average share of his team's offensive snaps in the games "
        "he played.",
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

        m4.metric(
            "Avg Weekly Volatility",
            f"{summary['volatility_score']:.1%}",
            help=(
                "How much his snap percentage and opportunity share "
                "changed from game to game, as a % of a normal "
                f"{player_position} week (100% = normal). Average of "
                f"{int(summary['weekly_comparisons'])} game-to-game "
                f"changes over {range_label.lower()}."
            ),
        )

        # Same pool as the Most Consistent leaderboard.
        pool = quality_pool(
            ranked_players(season_volatility, player_position)
        )
        in_pool = (pool["gsis_id"] == player_id).to_numpy()

        if in_pool.any():

            row = pool.loc[in_pool].iloc[0]
            rank = int(in_pool.argmax()) + 1

            st.success(
                f"**Consistency quality score: {row['quality_score']:.1f}**"
                f" (#{rank} of {len(pool)} eligible {player_position}s)"
            )

            with st.expander("How his quality score is calculated"):
                st.markdown(
                    f"{quality_breakdown(row)}: "
                    + ", ".join(
                        f"{QUALITY_LABELS[part].lower()} "
                        f"{row[column]:.1f}"
                        for part, column in QUALITY_COLUMNS.items()
                    )
                    + f". Each part is a percentile among the {len(pool)} "
                    f"eligible {player_position}s (100 = best): steadiness "
                    "ranks low volatility, snaps ranks average snap %, and "
                    "touches ranks carries + targets per game."
                )

        else:

            thresholds = CONSISTENCY_THRESHOLDS[player_position]
            missing = []
            if summary["games_share"] < MIN_GAMES_SHARE:
                missing.append(
                    f"to play {MIN_GAMES_SHARE:.0%}+ of his team's games "
                    f"(played {int(summary['weeks'])} of "
                    f"{int(summary['team_games'])}, "
                    f"{summary['games_share']:.0%})"
                )
            if summary["avg_snap_pct"] < thresholds["min_snap_pct"]:
                missing.append(
                    f"{thresholds['min_snap_pct']:.0%}+ average snaps "
                    f"(has {summary['avg_snap_pct']:.0%})"
                )
            if (
                summary["avg_opportunities_per_week"]
                < thresholds["min_opportunities_per_week"]
            ):
                missing.append(
                    f"{thresholds['min_opportunities_per_week']}+ "
                    "carries + targets per game (has "
                    f"{summary['avg_opportunities_per_week']:.1f})"
                )

            st.caption(
                "Not in the Most Consistent rankings for these weeks: "
                "needs " + "; ".join(missing or ["more games"]) + "."
            )

    else:

        m4.metric(
            "Avg Weekly Volatility",
            "N/A",
            help=(
                "No two games in a row inside the selected weeks "
                "to compare."
            ),
        )

    st.caption(
        "Each game is compared with the player's previous game "
        "(a bye week doesn't break the comparison). "
        f"100% = a normal week for a {player_position}."
    )

    # --------------------------------------------------------
    # HIGHLIGHTED WEEK
    # --------------------------------------------------------

    scored_weeks = player_weekly.loc[
        player_weekly["week"].between(first_week, last_week)
        & player_weekly["weekly_volatility"].notna(),
        "week",
    ]
    default_week = (
        int(scored_weeks.max()) if not scored_weeks.empty else last_week
    )

    controls1, controls2 = st.columns([2, 1])

    with controls1:

        highlighted_week = st.slider(
            "Highlight NFL Week",
            min_value=1,
            max_value=max(season_last_week, 2),
            value=default_week,
            help="Labels this week on every chart, shows its numbers "
            "below, and explains its volatility score under the "
            "Weekly Volatility chart.",
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

    usage_by_week = player_season.groupby("week")[
        ["snap_pct", "opp_share", "targets", "carries"]
    ].mean()
    weekly_by_week = player_weekly.set_index("week")
    player_team = str(player_season["team"].iloc[-1])
    team_weeks = set(
        usage.loc[
            (usage["season"] == season) & (usage["team"] == player_team),
            "week",
        ]
    )

    def week_status(week):
        """Why a week has no number: "bye", "did not play" or "no score"."""
        if week not in usage_by_week.index:
            return "bye" if week not in team_weeks else "did not play"
        if week not in weekly_by_week.index or pd.isna(
            weekly_by_week.loc[week, "weekly_volatility"]
        ):
            earlier = usage_by_week.index[usage_by_week.index < week]
            return "first game back" if len(earlier) else "first game"
        return "played"

    highlighted_status = week_status(highlighted_week)

    with st.container(border=True):

        if highlighted_week not in usage_by_week.index:

            st.markdown(
                f"**Week {highlighted_week}:** "
                + (
                    f"bye week for {player_team}."
                    if highlighted_status == "bye"
                    else f"{player_name} did not play."
                )
            )

        else:

            now = usage_by_week.loc[highlighted_week]
            previous = (
                weekly_by_week.loc[highlighted_week, "previous_week"]
                if highlighted_week in weekly_by_week.index
                else float("nan")
            )
            before = (
                usage_by_week.loc[int(previous)]
                if pd.notna(previous)
                else None
            )

            st.markdown(
                f"**Week {highlighted_week} at a glance**"
                + (
                    f" · changes vs Week {int(previous)}, his previous game"
                    if before is not None
                    else " · no change shown: "
                    + (
                        "he missed the game before"
                        if highlighted_status == "first game back"
                        else "his first game of the season"
                    )
                )
            )

            def glance_metric(column, label, value, change):
                """One week's value with its change since last game."""
                if change is None:
                    column.metric(label, value)
                    return
                flat = change.strip("+-0.%") == ""
                column.metric(
                    label,
                    value,
                    delta=change,
                    delta_color="off" if flat else "normal",
                    delta_arrow="off" if flat else "auto",
                )

            g1, g2, g3, g4, g5 = st.columns(5)

            glance_metric(
                g1,
                "Snap %",
                f"{now['snap_pct']:.0%}",
                f"{(now['snap_pct'] - before['snap_pct']) * 100:+.1f}%"
                if before is not None
                else None,
            )
            glance_metric(
                g2,
                "Opportunity Share",
                f"{now['opp_share']:.0%}",
                f"{(now['opp_share'] - before['opp_share']) * 100:+.1f}%"
                if before is not None
                else None,
            )
            glance_metric(
                g3,
                "Targets",
                f"{now['targets']:.0f}",
                f"{now['targets'] - before['targets']:+.0f}"
                if before is not None
                else None,
            )
            glance_metric(
                g4,
                "Carries",
                f"{now['carries']:.0f}",
                f"{now['carries'] - before['carries']:+.0f}"
                if before is not None
                else None,
            )

            week_score = (
                weekly_by_week.loc[highlighted_week, "weekly_volatility"]
                if highlighted_week in weekly_by_week.index
                else float("nan")
            )
            g5.metric(
                "Weekly Volatility",
                f"{week_score:.0%}" if pd.notna(week_score) else "—",
                help=f"% of a normal {player_position} week's change. "
                "The math is under the Weekly Volatility chart.",
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
                else ": %{y:.0f}"
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
                        width=2.5,
                    ),
                    marker=dict(size=7),
                    hovertemplate=hover,
                )
            )

        shade_selected_weeks(fig)
        mark_highlighted_week(fig)

        value = series.loc[highlighted_week]
        top = 1.0 if percentage else chart_top(series)
        if pd.notna(value):
            shown = f"{value:.0%}" if percentage else f"{value:.0f}"
        else:
            shown = highlighted_status
        label_week(
            fig,
            highlighted_week,
            value,
            f"Wk {highlighted_week} · {shown}",
            top,
            marker=chart_style != "Bars",
        )

        style_figure(fig, 340)

        fig.update_layout(
            title=dict(text=title, x=0.02),
            showlegend=False,
            yaxis_title=ylabel,
        )

        if percentage:

            fig.update_yaxes(
                tickformat=".0%",
                range=[0, 1.05],
            )

        return fig

    # --------------------------------------------------------
    # FOUR PLAYER USAGE CHARTS
    # --------------------------------------------------------

    chart1, chart2 = st.columns(2)

    with chart1:

        st.plotly_chart(
            make_chart(
                player_season,
                "snap_pct",
                "Offensive Snap Percentage",
                "Share of Offensive Snaps",
                METRIC_COLORS["snap_pct"],
                True,
            ),
            width="stretch",
        )

    with chart2:

        st.plotly_chart(
            make_chart(
                player_season,
                "opp_share",
                "Opportunity Share",
                "Share of Team Carries + Targets",
                METRIC_COLORS["opp_share"],
                True,
            ),
            width="stretch",
        )

    chart3, chart4 = st.columns(2)

    with chart3:

        st.plotly_chart(
            make_chart(
                player_season,
                "targets",
                "Weekly Targets",
                "Targets",
                COUNT_COLORS["targets"],
            ),
            width="stretch",
        )

    with chart4:

        st.plotly_chart(
            make_chart(
                player_season,
                "carries",
                "Weekly Carries",
                "Carries",
                COUNT_COLORS["carries"],
            ),
            width="stretch",
        )

    # --------------------------------------------------------
    # WEEKLY VOLATILITY CHART
    # --------------------------------------------------------

    st.subheader("Weekly Volatility")

    st.write(
        "How much the player's usage changed since his previous game, "
        f"as a % of a normal {player_position} week. Each bar is split "
        "into the part snap share and opportunity share add; the dotted "
        "line is a normal week (100%)."
    )

    by_week = player_weekly.set_index("week").reindex(WEEKS)
    scored_metrics = weighted_metrics(player_position)

    def metric_score(frame, metric):
        """Change ÷ the position's normal change (1.0 = 100%)."""
        return (
            frame[CHANGE_COLUMNS[metric]]
            / typical_change.loc[player_position, metric]
        )

    volatility_fig = go.Figure()

    for metric in scored_metrics:
        volatility_fig.add_trace(
            go.Bar(
                x=WEEKS,
                y=by_week[PART_COLUMNS[metric]].values,
                name=(
                    f"{METRIC_SHORT[metric]} score × "
                    f"{POSITION_WEIGHTS[player_position][metric]:.2f}"
                ),
                marker_color=METRIC_COLORS[metric],
                customdata=pd.DataFrame({
                    "score": metric_score(by_week, metric),
                    "total": by_week["weekly_volatility"],
                }).to_numpy(),
                hovertemplate=(
                    "Week %{x}<br>"
                    + METRIC_SHORT[metric]
                    + " score %{customdata[0]:.0%}, adds %{y:.0%}<br>"
                    "Weekly volatility %{customdata[1]:.0%}"
                    "<extra></extra>"
                ),
            )
        )

    volatility_fig.add_trace(
        go.Scatter(
            x=[0.5, 18.5],
            y=[1.0, 1.0],
            mode="lines",
            name="Normal week (100%)",
            line=dict(color=THEME["muted_text"], width=1.5, dash="dot"),
            hoverinfo="skip",
        )
    )

    shade_selected_weeks(volatility_fig)
    mark_highlighted_week(volatility_fig)

    highlighted_score = by_week.loc[highlighted_week, "weekly_volatility"]
    volatility_top = chart_top(by_week["weekly_volatility"])
    label_week(
        volatility_fig,
        highlighted_week,
        highlighted_score,
        f"Wk {highlighted_week} · "
        + (
            f"{highlighted_score:.0%} of normal"
            if pd.notna(highlighted_score)
            else (
                f"no score ({highlighted_status})"
                if highlighted_status.startswith("first")
                else highlighted_status
            )
        ),
        volatility_top,
        marker=False,
    )

    style_figure(volatility_fig, 380)

    volatility_fig.update_layout(
        title=dict(text="Week-to-Week Usage Volatility", x=0.02),
        barmode="stack",
        legend=dict(
            orientation="h",
            traceorder="normal",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        yaxis=dict(
            title="Weekly Volatility (% of normal)",
            tickformat=".0%",
        ),
    )

    st.plotly_chart(
        volatility_fig,
        width="stretch",
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

            previous = int(selected_row["previous_week"])
            before_row = player_weekly.loc[
                player_weekly["week"] == previous
            ].iloc[0]

            # A metric with no data that week is skipped and the other
            # weights are scaled up to 100% (see score_weeks).
            available = [
                metric
                for metric in scored_metrics
                if pd.notna(selected_row[PART_COLUMNS[metric]])
            ]
            weight_used = sum(
                POSITION_WEIGHTS[player_position][metric]
                for metric in available
            )

            terms = []
            steps = []

            for metric in available:
                weight = (
                    POSITION_WEIGHTS[player_position][metric] / weight_used
                )
                change = selected_row[CHANGE_COLUMNS[metric]]
                typical = typical_change.loc[player_position, metric]
                score = change / typical
                terms.append(f"{weight:.2f} × {score:.0%}")
                steps.append(
                    f"- **{METRIC_SHORT[metric]} score {score:.0%}:** "
                    f"{METRIC_LABELS[metric].lower()} "
                    + explain_change(
                        metric,
                        before_row[metric],
                        selected_row[metric],
                        change,
                        typical,
                        player_position,
                    )
                )

            if len(available) < len(scored_metrics):
                steps.append(
                    "- No data for the other metrics this week, so "
                    "these weights are scaled up to 100%."
                )

            with st.expander(
                f"How Week {highlighted_week}'s volatility "
                f"({selected_score:.0%}) is calculated"
            ):
                st.markdown(
                    f"**Week {highlighted_week} volatility: "
                    f"{selected_score:.0%} of a normal week** (compared "
                    f"with Week {previous}) = " + " + ".join(terms)
                    + f"\n\nWeekly volatility = {score_formula(player_position)}. "
                    "Each score is the change ÷ a normal week's change:"
                    "\n\n" + "\n".join(steps)
                )

        else:

            st.caption(
                f"Week {highlighted_week} has no volatility score: it is "
                "the player's first game, or he missed the game before "
                "it."
            )

    else:

        st.caption(
            f"No player record for Week {highlighted_week}."
        )

    # Tables list the selected weeks only.
    window_weekly = player_weekly.loc[
        player_weekly["week"].between(first_week, last_week)
    ]

    with st.expander("View Weekly Player Data"):

        weekly_data = window_weekly[
            ["week", *METRICS, "targets", "carries", "weekly_volatility"]
        ].copy()
        weekly_data[[*METRICS, "weekly_volatility"]] = (
            weekly_data[[*METRICS, "weekly_volatility"]] * 100
        )

        st.dataframe(
            weekly_data,
            hide_index=True,
            width="stretch",
            column_config={
                "week": st.column_config.NumberColumn("Week", format="%d"),
                **{
                    metric: st.column_config.NumberColumn(
                        METRIC_LABELS[metric], format=PERCENT_FORMAT
                    )
                    for metric in METRICS
                },
                "targets": st.column_config.NumberColumn(
                    "Targets", format="%d"
                ),
                "carries": st.column_config.NumberColumn(
                    "Carries", format="%d"
                ),
                "weekly_volatility": st.column_config.NumberColumn(
                    "Weekly Volatility",
                    format=VOLATILITY_FORMAT,
                    help="% of a normal week for the position.",
                ),
            },
        )

    with st.expander("View Weekly Volatility Components"):

        st.caption(
            f"Weekly volatility = {score_formula(player_position)}. "
            "Each score is the change since the previous game ÷ a "
            f"normal {player_position} week's change ("
            + ", ".join(
                f"{METRIC_SHORT[metric].lower()} "
                + format_change(
                    metric, typical_change.loc[player_position, metric]
                )
                for metric in scored_metrics
            )
            + "), so 100% = normal. Changes are signed (+ up, − down); "
            "the score counts the size either way. Only changes between "
            "two games inside the selected weeks are listed: the ones "
            "his average uses."
        )

        # Same weeks as the average (season_scores in src/volatility.py).
        compared = window_weekly.loc[
            window_weekly["previous_week"] >= first_week
        ]
        components = compared[["week", "previous_week"]].copy()
        component_config = {
            "week": st.column_config.NumberColumn("Week", format="%d"),
            "previous_week": st.column_config.NumberColumn(
                "Compared With Week",
                format="%d",
            ),
        }

        # Each week's values from the game it's compared with.
        before_values = (
            player_weekly.set_index("week")[METRICS]
            .reindex(compared["previous_week"])
            .set_axis(compared.index)
        )

        for metric in scored_metrics:
            points = compared[metric] - before_values[metric]
            components[f"{metric}_points"] = points * 100
            component_config[f"{metric}_points"] = (
                st.column_config.NumberColumn(
                    f"{METRIC_SHORT[metric]} Change",
                    format=CHANGE_FORMAT,
                    help="This game minus the previous game, in "
                    "percentage points.",
                )
            )
            if is_relative(metric):
                average = (
                    (compared[metric] + before_values[metric]) / 2
                ).clip(lower=RELATIVE_FLOOR)
                components[f"{metric}_relative"] = points / average * 100
                component_config[f"{metric}_relative"] = (
                    st.column_config.NumberColumn(
                        f"{METRIC_SHORT[metric]} Change vs His Share",
                        format="%+.0f%%",
                        help="The change ÷ his average share across the "
                        "two games. This is the change the score uses.",
                    )
                )
            components[f"{metric}_score"] = (
                metric_score(compared, metric) * 100
            )
            component_config[f"{metric}_score"] = (
                st.column_config.NumberColumn(
                    f"{METRIC_SHORT[metric]} Score",
                    format=VOLATILITY_FORMAT,
                    help="Change ÷ a normal week's change. 100% = normal.",
                )
            )

        components["weekly_volatility"] = (
            compared["weekly_volatility"] * 100
        )
        component_config["weekly_volatility"] = (
            st.column_config.NumberColumn(
                "Weekly Volatility", format=VOLATILITY_FORMAT
            )
        )

        st.dataframe(
            components,
            hide_index=True,
            width="stretch",
            column_config=component_config,
        )


# ============================================================
# 12. PLAYER VOLATILITY RANKINGS
# ============================================================

st.divider()

st.header("📊 Player Volatility Rankings")

st.write(
    "Rankings use average week-to-week volatility (100% = a normal "
    "week). The Most Consistent leaderboard also rewards meaningful "
    "playing time and offensive opportunities. Each list has its own "
    "minimum snap % slider: players below it drop out and the next "
    "players move up."
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
        "because each position has its own normal "
        "weekly change."
    ),
)

st.caption(f"{season} · {range_label}")

ranked = ranked_players(season_volatility, ranking_position)

if ranked.empty:

    st.info(
        "No volatility scores are available "
        "for this position and these weeks."
    )

else:

    thresholds = CONSISTENCY_THRESHOLDS[
        ranking_position
    ]

    eligible_players = quality_pool(ranked)

    average_volatility = ranked[
        "volatility_score"
    ].mean()

    r1, r2, r3, r4 = st.columns(4)

    r1.metric(
        "Players Scored",
        f"{len(ranked):,}",
        help=(
            f"{ranking_position}s averaging {MIN_ROLE_SNAP_PCT:.0%}+ of "
            f"snaps who played {MIN_RANKED_GAMES_SHARE:.0%}+ of their "
            "team's games in these weeks."
        ),
    )

    r2.metric(
        "Avg Weekly Volatility",
        f"{average_volatility:.0%}",
        help="100% = a normal week for the position.",
    )

    r3.metric(
        "Highest Avg Volatility",
        f"{ranked['volatility_score'].max():.0%}",
    )

    r4.metric(
        "Consistency Eligible",
        f"{len(eligible_players):,}",
    )

    # One card per leaderboard, each with its own snap % slider.
    ranking_left = st.container(border=True)
    ranking_right = st.container(border=True)

    games_column = st.column_config.NumberColumn(
        "Games Played",
        format="%.0f%%",
        help="Share of his team's games in these weeks (byes don't "
        "count).",
    )

    # --------------------------------------------------------
    # MOST VOLATILE PLAYERS
    # --------------------------------------------------------

    with ranking_left:

        st.subheader("📈 Most Volatile Players")

        role_floor = round(MIN_ROLE_SNAP_PCT * 100)

        slider_column, _ = st.columns([1, 2])

        volatile_min = slider_column.slider(
            "Minimum average snap %",
            min_value=role_floor,
            max_value=100,
            value=role_floor,
            step=5,
            format="%d%%",
            key=f"volatile_min_snap_{ranking_position}",
            help=(
                f"Players under {role_floor}% of snaps are never "
                "ranked: their few snaps swing too much to compare."
            ),
        )

        volatile_shown = ranked.loc[
            ranked["avg_snap_pct"] >= volatile_min / 100
        ]

        if volatile_shown.empty:

            st.info(
                f"No {ranking_position} averages {volatile_min}%+ "
                "of snaps in these weeks. Lower the minimum to see "
                "more players."
            )

        else:

            most_volatile = volatile_shown.head(TOP_N).copy()
            most_volatile.insert(
                0, "rank", range(1, len(most_volatile) + 1)
            )
            for column in [
                "games_share", "avg_snap_pct", "volatility_score"
            ]:
                most_volatile[column] = most_volatile[column] * 100

            st.dataframe(
                most_volatile[
                    [
                        "rank",
                        "player_name",
                        "team",
                        "games_share",
                        "avg_snap_pct",
                        "volatility_score",
                    ]
                ],
                hide_index=True,
                width="stretch",
                height=table_height(most_volatile),
                column_config={
                    "rank": st.column_config.NumberColumn(
                        "#", format="%d", width=40
                    ),
                    "player_name": "Player",
                    "team": st.column_config.TextColumn(
                        "Team", width="small"
                    ),
                    "games_share": games_column,
                    "avg_snap_pct": st.column_config.NumberColumn(
                        "Avg Snap %", format=PERCENT_FORMAT
                    ),
                    "volatility_score": st.column_config.NumberColumn(
                        "Avg Volatility",
                        format=VOLATILITY_FORMAT,
                        help="% of a normal week for the position.",
                    ),
                },
            )

            st.caption(
                f"Top {len(most_volatile)} of {len(volatile_shown)} "
                f"{ranking_position}s averaging {volatile_min}%+ of "
                f"snaps who played {MIN_RANKED_GAMES_SHARE:.0%}+ of "
                "their team's games in these weeks."
            )

    # --------------------------------------------------------
    # MOST CONSISTENT PLAYERS
    # --------------------------------------------------------

    with ranking_right:

        st.subheader("📉 Most Consistent Players")

        position_floor = round(thresholds["min_snap_pct"] * 100)

        slider_column, _ = st.columns([1, 2])

        consistent_min = slider_column.slider(
            "Minimum average snap %",
            min_value=position_floor,
            max_value=100,
            value=position_floor,
            step=5,
            format="%d%%",
            key=f"consistent_min_snap_{ranking_position}",
            help=(
                f"{ranking_position}s need {position_floor}%+ of snaps "
                "to be eligible. Raising the minimum hides players "
                "without changing anyone's quality score."
            ),
        )

        quality_formula = " + ".join(
            f"{QUALITY_WEIGHTS[part]:.2f} × {QUALITY_LABELS[part]}"
            for part in QUALITY_COLUMNS
        )

        example_row = None

        if eligible_players.empty:

            st.info(
                "No players meet the minimum playing-time "
                "and opportunity thresholds."
            )

        else:

            consistent_shown = eligible_players.loc[
                eligible_players["avg_snap_pct"] >= consistent_min / 100
            ]

            if consistent_shown.empty:

                st.info(
                    f"No eligible {ranking_position} averages "
                    f"{consistent_min}%+ of snaps. Lower the minimum "
                    "to see more players."
                )

            else:

                most_consistent = consistent_shown.head(TOP_N).copy()
                most_consistent.insert(
                    0, "rank", range(1, len(most_consistent) + 1)
                )
                for column in ["avg_snap_pct", "volatility_score"]:
                    most_consistent[column] = most_consistent[column] * 100

                percentile_help = {
                    "low_volatility": "low volatility (100 = steadiest)",
                    "snap_pct": "average snap %",
                    "opportunities": "carries + targets per game",
                }

                st.dataframe(
                    most_consistent[
                        [
                            "rank",
                            "player_name",
                            "team",
                            "avg_snap_pct",
                            "avg_opportunities_per_week",
                            "volatility_score",
                            *QUALITY_COLUMNS.values(),
                            "quality_score",
                        ]
                    ],
                    hide_index=True,
                    width="stretch",
                    height=table_height(most_consistent),
                    column_config={
                        "rank": st.column_config.NumberColumn(
                            "#", format="%d", width=40
                        ),
                        "player_name": "Player",
                        "team": st.column_config.TextColumn(
                            "Team", width="small"
                        ),
                        "avg_snap_pct": st.column_config.NumberColumn(
                            "Avg Snap %", format=PERCENT_FORMAT
                        ),
                        "avg_opportunities_per_week": (
                            st.column_config.NumberColumn(
                                "Opp / Game",
                                format="%.1f",
                                help="Carries + targets per game.",
                            )
                        ),
                        "volatility_score": st.column_config.NumberColumn(
                            "Avg Volatility",
                            format=VOLATILITY_FORMAT,
                            help="% of a normal week for the position.",
                        ),
                        **{
                            column: st.column_config.ProgressColumn(
                                QUALITY_LABELS[part],
                                min_value=0,
                                max_value=100,
                                format="%.0f",
                                width=110,
                                color="#4ADE80",
                                help=(
                                    "Percentile among the "
                                    f"{len(eligible_players)} eligible "
                                    f"{ranking_position}s for "
                                    f"{percentile_help[part]}. Counts "
                                    f"{QUALITY_WEIGHTS[part]:.0%} of "
                                    "the quality score."
                                ),
                            )
                            for part, column in QUALITY_COLUMNS.items()
                        },
                        "quality_score": st.column_config.NumberColumn(
                            "Quality",
                            format="%.1f",
                            help=f"0-100: {quality_formula}.",
                        ),
                    },
                )

                example_row = most_consistent.iloc[0]

        with st.expander("How the quality score and requirements work"):

            st.markdown(
                f"**Quality = {quality_formula}.** Each part is a "
                "percentile among all "
                f"{len(eligible_players)} eligible {ranking_position}s "
                "(100 = best), so the slider doesn't change it."
                + (
                    f" Example, {example_row['player_name']}: "
                    f"{quality_breakdown(example_row)}."
                    if example_row is not None
                    else ""
                )
            )

            st.markdown(
                "**Requirements:** played "
                f"{MIN_GAMES_SHARE:.0%}+ of his team's games in these weeks, "
                f"{thresholds['min_snap_pct']:.0%}+ average snaps, and "
                f"{thresholds['min_opportunities_per_week']}+ carries + "
                "targets per game."
            )


# ============================================================
# 13. FOOTER
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
