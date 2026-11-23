
# RosterWatch 🏈
NFL Usage Analytics & Fantasy Football Waiver Intelligence

RosterWatch is an interactive NFL analytics dashboard built with Python and Streamlit. It analyzes player usage patterns, fantasy football production, and weekly trends to help users understand player consistency and identify potential waiver-wire opportunities.

The dashboard focuses on three offensive positions: running backs (RB), wide receivers (WR), and tight ends (TE), using historical NFL data from the 2022–2025 seasons.

## 1. Problem Statement & Motivation

Fantasy football managers often make roster decisions based on recent fantasy points alone. However, fantasy production does not always reflect a player's underlying offensive involvement.

A player might score many fantasy points despite limited opportunities, while another player might receive increasing snaps and touches without immediately producing strong fantasy results.

RosterWatch addresses this problem by combining offensive usage analytics with fantasy production to provide a more comprehensive view of player performance.

The project aims to answer two questions:

1. How consistent is a player's offensive role throughout the NFL season?
2. Which players show promising or declining trends that could inform fantasy waiver-wire decisions?

## 2. Key Features

### Model 1: Weekly Usage Volatility
- Measures week-to-week changes in offensive involvement.
- Analyzes snap share, opportunity share, passing-down share, and red-zone share.
- Uses position-specific weights for RBs, WRs, and TEs.
- Identifies consistent and volatile players.
- Includes eligibility thresholds to exclude players with insufficient participation.
- Provides interactive weekly charts and player comparisons.

### Model 2: Waiver Opportunity Score
- Evaluates fantasy production and offensive involvement.
- Combines five weighted analytical components.
- Accounts for recent performance, usage stability, and trend confirmation.
- Produces a waiver opportunity score on a 0–100 scale.
- Categorizes players into waiver recommendation groups.
- Uses only data available through the selected NFL week.

### Interactive Dashboard
- Season and position filters.
- Adjustable Model 1 week range.
- Player exploration tools.
- Interactive Plotly visualizations.
- Consistency and volatility leaderboards.
- Waiver-wire rankings.
- Fantasy production and usage trend charts.

## 3. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Data processing and model implementation |
| Pandas | Data manipulation and aggregation |
| NumPy | Numerical calculations |
| nflreadpy | NFL snap counts, player stats, play-by-play and player IDs |
| Requests | Calls to the Sleeper and ESPN APIs |
| Streamlit | Interactive dashboard |
| Plotly | Data visualizations |
| Parquet / CSV | Historical data storage |
| Git / GitHub | Version control and collaboration |

### Libraries

All libraries are listed in `requirements.txt`.

| Library | Used in | Link |
|---|---|---|
| pandas | notebook, dashboard, models, tests | https://pandas.pydata.org/ |
| NumPy | notebook, dashboard | https://numpy.org/ |
| nflreadpy | notebook (data pull) | https://github.com/nflverse/nflreadpy |
| Requests | notebook (Sleeper API) | https://requests.readthedocs.io/ |
| PyArrow | reading and writing Parquet files | https://arrow.apache.org/docs/python/ |
| Matplotlib | notebook (exploratory plots) | https://matplotlib.org/ |
| Streamlit | dashboard | https://streamlit.io/ |
| Plotly | dashboard charts | https://plotly.com/python/ |
| JupyterLab | running the notebook | https://jupyter.org/ |
| pytest | tests | https://docs.pytest.org/ |
| seaborn | listed in requirements, not used yet | https://seaborn.pydata.org/ |
| scikit-learn | listed in requirements, not used yet | https://scikit-learn.org/ |
| python-dotenv | listed in requirements, not used yet | https://github.com/theskumar/python-dotenv |
| google-cloud-bigquery | listed in requirements, not used yet | https://cloud.google.com/python/docs/reference/bigquery/latest |
| db-dtypes | listed in requirements, not used yet | https://github.com/googleapis/python-db-dtypes-pandas |

The notebook also uses Python's standard library (`json`, `math`, `datetime`, `pathlib`, `urllib.request`) for the ESPN request.

## 4. System Architecture

RosterWatch follows a data-processing and analytics workflow:

```text
Historical NFL Player Data
            |
            v
     Data Loading
            |
            v
 Data Cleaning & Validation
            |
            v
    Weekly Usage Dataset
            |
      +-----+-----+
      |           |
      v           v
   Model 1      Model 2
  Volatility    Waiver Score
      |           |
      v           v
 Consistency   Waiver Rankings
  Rankings     & Player Trends
      |           |
      +-----+-----+
            |
            v
     Streamlit Dashboard
            |
            v
 Interactive Charts & Tables
```

### Architecture Design Decisions

The application separates its analytical responsibilities into two models.

Model 1 focuses on offensive role stability without directly incorporating fantasy points. This makes it possible to evaluate usage consistency independently of fantasy scoring outcomes.

Model 2 incorporates fantasy points alongside offensive usage, stability, and recent trends to support waiver-wire analysis.

The dashboard uses historical data through the selected week to avoid incorporating future game results into that week's analysis.

## 5. Data Sources & Processing

RosterWatch uses historical weekly NFL player usage data for the 2022–2025 regular seasons (weeks 1–18), built by `notebooks/weekly_usage.ipynb`.

### Data Sources

| Source | What it provides | How it is accessed |
|---|---|---|
| nflverse (https://github.com/nflverse/nflverse-data) | Weekly offensive snap counts (originally from Pro Football Reference), player stats, play-by-play, and player information | nflreadpy: `load_snap_counts`, `load_player_stats`, `load_pbp`, `load_players` |
| DynastyProcess player IDs (https://github.com/dynastyprocess/data) | Cross-reference between NFL (`gsis_id`), Sleeper and ESPN player IDs | nflreadpy: `load_ff_playerids` |
| Sleeper API (https://docs.sleeper.com/) | Weekly PPR fantasy points (`fantasy_points_ppr`) | `https://api.sleeper.com/stats/nfl/{season}/{week}` (a public stats endpoint not covered in the official docs) |
| ESPN Fantasy API (unofficial, no public documentation) | Current-season % rostered, % started and injury status | `https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{season}/segments/0/leaguedefaults/3?view=kona_player_info` |

None of these sources needs an API key.

### Data Files

| File | Contents |
|---|---|
| `notebooks/weekly_usage.parquet` | One row per player-game (RB, WR, TE), 2022–2025 |
| `notebooks/weekly_usage.csv` | The same table as CSV |
| `notebooks/espn_ownership.csv` | ESPN % rostered snapshot for the 2026 season, with the time it was collected (`collected_at_utc`); not used by the dashboard yet |

### Important Dataset Fields

| Field | Description |
|---|---|
| season | NFL season |
| week | NFL week |
| player_name | Player name |
| team | NFL team |
| position | Player position |
| snap_pct | Offensive snap share |
| targets | Number of targets |
| carries | Number of rushing attempts |
| opp_share | Share of team opportunities |
| passdown_share | Passing-down involvement |
| redzone_share | Red-zone involvement |
| fantasy_points_ppr | Weekly PPR fantasy points |

### Data Preparation

The application:
1. Loads weekly player data from supported Parquet or CSV files.
2. Filters to supported seasons, weeks, and positions.
3. Converts relevant metrics to numerical values.
4. Handles missing or invalid values where appropriate.
5. Removes duplicate player-week records.
6. Calculates opportunities using targets plus carries.
7. Prepares player histories for model calculations.

### Data Location

The notebook saves its output to `notebooks/`, which is where the data is stored in the repository.

The dashboard reads the data from `data/processed/`. It uses `weekly_usage.parquet` if present, otherwise `weekly_usage.csv`:

```text
data/processed/weekly_usage.parquet
data/processed/weekly_usage.csv
```

`data/` is listed in `.gitignore`, so this folder is not part of the repository. Copy the file there before launching the dashboard (see Section 9).

## 6. Model 1: Weekly Usage Volatility

### Objective

Model 1 measures how much a player's offensive involvement changes between consecutive NFL weeks.

A lower volatility score indicates a more stable offensive role.

### Equation

```math
V_t =
w_S |S_t-S_{t-1}|
+
w_O |O_t-O_{t-1}|
+
w_P |P_t-P_{t-1}|
+
w_R |R_t-R_{t-1}|
```

Where:

- $`V_t`$: Weekly usage volatility
- $`S_t`$: Snap share
- $`O_t`$: Opportunity share
- $`P_t`$: Passing-down share
- $`R_t`$: Red-zone share
- $`w_S,w_O,w_P,w_R`$: Position-specific weights

### Position-Specific Weights

| Position | Snap Share | Opportunity Share | Passing-Down | Red Zone |
|---|---:|---:|---:|---:|
| RB | 0.30 | 0.45 | 0.15 | 0.10 |
| WR | 0.35 | 0.35 | 0.20 | 0.10 |
| TE | 0.40 | 0.30 | 0.20 | 0.10 |

These weights reflect differences in offensive responsibilities across positions.

For example, opportunity share receives the greatest weight for running backs because rushing attempts and targets are important indicators of their offensive involvement.

### Season-Level Volatility

A player's overall volatility is calculated by averaging valid consecutive-week volatility measurements:

```math
V_{\text{season}}=\frac{1}{m}\sum_{t=1}^{m}V_t
```

Where $`m`$ is the number of valid consecutive-week comparisons.

Nonconsecutive weeks are not directly compared.

### Consistency Eligibility

To prevent players with very limited participation from appearing artificially consistent, Model 1 uses minimum participation thresholds.

| Position | Minimum Weeks | Minimum Avg. Snap Share | Minimum Opportunities/Game |
|---|---:|---:|---:|
| RB | 10 | 50% | 10 |
| WR | 10 | 65% | 5 |
| TE | 10 | 60% | 4 |

Players must also have at least eight valid consecutive-week comparisons.

### Consistency Quality Score

```math
Q=100\left[
0.50(1-R_V)+0.25R_S+0.25R_O
\right]
```

Where:

- $`R_V`$: Volatility percentile rank
- $`R_S`$: Average snap-share percentile rank
- $`R_O`$: Average opportunity percentile rank

Ranks are calculated among players in the same season and position.

A higher quality score rewards low volatility combined with meaningful offensive participation.

## 7. Model 2: Waiver Opportunity Score

### Objective

Model 2 evaluates potential waiver-wire opportunities by combining fantasy production, offensive usage, stability, and recent trends.

Unlike Model 1, Model 2 directly incorporates fantasy football performance.

### Overall Equation

```math
W=100(
0.40F+
0.20O+
0.15S+
0.15V+
0.10T
)
```

Where:

- $`W`$: Base waiver opportunity score
- $`F`$: Fantasy performance score
- $`O`$: Opportunity share score
- $`S`$: Snap share score
- $`V`$: Stability score
- $`T`$: Trend confirmation score

Each component is normalized to a 0–1 scale.

The implementation may additionally adjust the base score based on available data coverage.

### Model 2 Component Weights

| Component | Weight | Purpose |
|---|---:|---|
| Fantasy PPR Performance | 40% | Measures fantasy production |
| Opportunity Share | 20% | Measures involvement in offensive opportunities |
| Snap Share | 15% | Measures playing time |
| Volatility & Stability | 15% | Measures consistency |
| Trend Confirmation | 10% | Evaluates recent improvement or decline |

### Fantasy Point Stability

Fantasy stability measures how consistently a player produces PPR fantasy points.

```math
F_{\text{stability}}=
\frac{1}{
1+\frac{\sigma_{\text{PPR}}}
{\max(|\mu_{\text{PPR}}|,2)}
}
```

Where:

- $`\sigma_{\text{PPR}}`$: Standard deviation of fantasy points
- $`\mu_{\text{PPR}}`$: Average fantasy points

Lower fantasy-point variation generally produces higher stability.

### Offensive Usage Stability

Usage stability incorporates the weekly volatility calculated by Model 1.

```math
V_{\text{combined}}=
0.65V_{\text{recent}}+
0.35V_{\text{season}}
```

```math
U_{\text{stability}}=
\frac{1}{
1+\frac{V_{\text{combined}}}{0.12}
}
```

Recent volatility receives greater weight because recent offensive usage is more relevant to current roster decisions.

### Combined Stability Score

```math
V=
0.60F_{\text{stability}}+
0.40U_{\text{stability}}
```

This score contributes 15% of the overall waiver opportunity score.

### Waiver Categories

Model 2 assigns players to recommendation categories based on their scores and trends.

- Must Pick Up
- Slow Riser
- Slow Drop
- Must Drop
- Stable / No Clear Signal
- Insufficient Data

These classifications are analytical indicators, not guaranteed predictions of future performance.

## 8. Repository Structure

```text
RosterWatch/
├── docs/
│   ├── AGENTS.md                  # guidelines for AI coding agents
│   ├── DesignDoc.md               # design document template
│   └── codeReviewer.md            # code review guidelines
├── frontend/
│   ├── streamlit_app/
│   │   └── app.py                 # Streamlit dashboard (Model 1 and Model 2)
│   └── README.md
├── infra/
│   └── README.md                  # placeholder for deployment
├── models/
│   ├── .gitkeep
│   └── usage_models.py            # earlier usage scoring prototype
├── notebooks/
│   ├── README.md
│   ├── espn_ownership.csv         # ESPN % rostered snapshot
│   ├── weekly_usage.csv           # weekly usage data (CSV)
│   ├── weekly_usage.ipynb         # data pull and cleaning
│   └── weekly_usage.parquet       # weekly usage data (Parquet)
├── pipelines/
│   └── score_players.py           # runs the prototype from the command line
├── results/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   └── example.py                 # project template example
├── tests/
│   ├── test_example.py
│   └── test_usage_models.py
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

`data/processed/` is created locally during setup (Section 9) and is not tracked by Git.

## 9. Installation & Local Setup

### Prerequisites

- Python 3.10 or newer
- Git
- pip
- Access to the required weekly NFL usage dataset

### Clone the Repository

```bash
git clone https://github.com/KylanHuynh7/RosterWatch.git
cd RosterWatch
```

All commands below are run from the repository root on the `main` branch.

### Create a Virtual Environment

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Install Dependencies

```bash
python -m pip install -r requirements.txt
```

This installs everything in Section 3, including Streamlit, Plotly, nflreadpy and PyArrow (for Parquet files).

### Environment Configuration

The dashboard and the notebook do not need any environment variables or API keys.

`.env.example` is a template from the project starter. If a future pipeline needs settings, copy it to `.env` and fill in your own values. `.env` is listed in `.gitignore`; never commit passwords, API keys, or other secrets.

### Prepare the Dataset

Copy the data from `notebooks/` into `data/processed/`, where the dashboard looks for it.

macOS/Linux:

```bash
mkdir -p data/processed
cp notebooks/weekly_usage.parquet data/processed/
```

Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force data/processed
Copy-Item notebooks/weekly_usage.parquet data/processed/
```

### Refresh the Data (Optional)

The repository already includes the data, so this step is only needed to pull newer games or a new ESPN snapshot. Open `notebooks/weekly_usage.ipynb` in JupyterLab and run all cells from the `notebooks/` folder:

```bash
jupyter lab notebooks/weekly_usage.ipynb
```

The notebook writes `weekly_usage.csv`, `weekly_usage.parquet` and `espn_ownership.csv` into `notebooks/`. Then repeat the copy step above.

### Launch the Dashboard

```bash
streamlit run frontend/streamlit_app/app.py
```

Streamlit will display a local URL in the terminal, typically:

```text
http://localhost:8501
```

Open that address in your browser.

## 10. Using the Dashboard

### Model 1

1. Select the NFL season and position.
2. Choose the desired week range.
3. Explore player usage charts.
4. Review volatility and consistency rankings.
5. Compare how player involvement changes over time.

### Model 2

1. Select the NFL season and position.
2. Choose the evaluation week.
3. Review waiver opportunity scores.
4. Filter players by recommendation category.
5. Inspect fantasy production, usage trends, and model components.

## 11. Results & Evaluation

RosterWatch produces interpretable player-level analytics, including:

- Weekly usage volatility scores
- Season-level consistency scores
- Offensive usage trends
- Fantasy scoring trends
- Waiver opportunity rankings
- Recommendation categories

These outputs are descriptive and heuristic.

Before presenting claims about predictive accuracy, the models should be evaluated against historical outcomes using appropriate time-based validation.

Recommended evaluation approaches include:

- Comparing waiver rankings against subsequent fantasy production.
- Measuring whether rising-player classifications correspond to future increases in usage.
- Comparing the waiver model against a fantasy-points-only baseline.
- Testing the sensitivity of rankings to model weights and thresholds.

Actual evaluation results, charts, and case studies should be added after running these analyses.

## 12. Testing

Run the test suite from the repository root:

```bash
python -m pytest
```

Current tests:

- `tests/test_usage_models.py`: checks the earlier usage scoring prototype in `models/usage_models.py`.
- `tests/test_example.py`: checks the project template example in `src/`.

The dashboard's Model 1 and Model 2 calculations in `frontend/streamlit_app/app.py` are not covered by automated tests yet.

Recommended additional test coverage includes:

- Data loading and validation
- Duplicate handling
- Consecutive-week volatility calculations
- Position-specific weights
- Consistency eligibility thresholds
- Fantasy stability calculations
- Waiver score normalization
- Missing-data behavior
- Prevention of future-week data leakage

## 13. Limitations

RosterWatch has several limitations:

1. Model weights are manually selected rather than learned through supervised training.
2. Historical usage does not guarantee future fantasy production.
3. Injuries, trades, coaching changes, and opponent matchups may affect outcomes.
4. Missing or incomplete player data can reduce scoring reliability.
5. Fantasy scoring can be influenced by touchdowns and other high-variance events.
6. Waiver recommendations do not automatically account for league-specific roster availability.
7. Model scores should not be interpreted as calibrated probabilities.

These limitations provide opportunities for further research and improvement.

## 14. Future Improvements

Potential extensions include:

- Incorporating injury reports and depth-chart changes.
- Adding matchup and defensive-strength features.
- Developing machine-learning models for future fantasy production.
- Evaluating model performance using historical backtesting.
- Supporting additional fantasy scoring formats.
- Adding customizable league settings.
- Deploying a publicly accessible dashboard.
- Improving player comparisons and interactive visualizations.

## 15. Team Contributions

RosterWatch was developed as part of the UC San Diego Data Science Student Society (DS3) Projects program.

Team members and verified contributions should be documented here, including responsibility for:

- Data collection and preprocessing
- Analytical model development
- Dashboard implementation
- Testing and documentation
- Project presentation

## 16. Project Purpose

RosterWatch is an educational sports analytics project intended to demonstrate how data science can support interpretable decision-making in fantasy football.

Its primary contribution is combining offensive role consistency with fantasy production and trend analysis in an interactive dashboard.

---

Developed for DS3 Projects at UC San Diego.
