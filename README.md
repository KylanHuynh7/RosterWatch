
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

### Player Consistency Model (weekly usage volatility)
- Measures week-to-week changes in offensive involvement.
- Analyzes snap share, opportunity share, passing-down share, and red-zone share.
- Uses position-specific weights for RBs, WRs, and TEs.
- Identifies consistent and volatile players.
- Includes eligibility thresholds to exclude players with insufficient participation.
- Provides interactive weekly charts and player comparisons.

### Waiver Wire Alerts (waiver opportunity score and flags)
- Scores only players rostered in at most 50% of ESPN leagues.
- Combines five weighted components: fantasy points, opportunity share, snap share, stability, and trend confirmation.
- Produces a waiver score on a 0–100 scale, lowered for players with few valid games.
- Gives every scored player one of four flags: Must Pick Up, Slow Riser, Slow Drop, or Must Drop.
- Uses only data available through the selected NFL week.

### Interactive Dashboard
- Season and position filters.
- Adjustable week range for the Player Consistency Model.
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
    Weekly Usage Dataset            ESPN % Rostered Snapshot
            |                                  |
      +-----+-----+                            |
      |           |                            |
      v           v                            |
 Player         Waiver Wire  <-----------------+
 Consistency    Alerts         (only players <= 50% rostered)
 Model
      |           |
      v           v
 Consistency   Waiver Rankings, Flags
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

The Player Consistency Model focuses on offensive role stability without directly incorporating fantasy points. This makes it possible to evaluate usage consistency independently of fantasy scoring outcomes.

Waiver Wire Alerts incorporate fantasy points alongside offensive usage, stability, and recent trends to support waiver-wire analysis. They reuse the weekly volatility from the Player Consistency Model as part of their stability score.

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
| `notebooks/espn_ownership.csv` | ESPN % rostered snapshot (`ownership_season`, `gsis_id`, `percentOwned`, with the time it was collected in `collected_at_utc`). Waiver Wire Alerts use it to keep only players rostered in at most 50% of leagues |

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

The dashboard reads the data from `data/processed/`. It uses `weekly_usage.parquet` if present, otherwise `weekly_usage.csv`, and it needs `espn_ownership.csv` for Waiver Wire Alerts:

```text
data/processed/weekly_usage.parquet
data/processed/weekly_usage.csv
data/processed/espn_ownership.csv
```

`data/` is listed in `.gitignore`, so this folder is not part of the repository. Copy the files there before launching the dashboard (see Section 9). If `espn_ownership.csv` is missing, the Player Consistency Model still loads but Waiver Wire Alerts stop with an error.

The dashboard keeps ownership rows for the 2022–2025 seasons only, matched to the selected season through `ownership_season`. `notebooks/espn_ownership.csv` holds one ESPN % rostered value per player per season for 2022–2025 (`ownership_scope = season_reference_not_weekly`); it is a season-level reference, not weekly ownership history.

The hosted version on Google Cloud Run reads the same files from BigQuery. See [`infra/README.md`](infra/README.md).

## 6. Player Consistency Model (Weekly Usage Volatility)

### Objective

The Player Consistency Model measures how much a player's offensive involvement changes between consecutive NFL weeks.

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
\bar{V}=\frac{1}{m}\sum_{t\in T}V_t
```

Where $`T`$ is the set of weeks in the selected range with a valid comparison to the previous week, and $`m`$ is the number of those weeks.

Nonconsecutive weeks are not directly compared.

### Consistency Eligibility

To prevent players with very limited participation from appearing artificially consistent, the Player Consistency Model uses minimum participation thresholds.

| Position | Minimum Weeks | Minimum Avg. Snap Share | Minimum Opportunities/Game |
|---|---:|---:|---:|
| RB | 10 | 50% | 10 |
| WR | 10 | 65% | 5 |
| TE | 10 | 60% | 4 |

Players must also have at least eight valid consecutive-week comparisons.

### Consistency Quality Score

```math
Q=100\left(0.50\,C+0.25\,S_{\text{rank}}+0.25\,O_{\text{rank}}\right),
\qquad C=1-\operatorname{pct}(\bar{V})
```

Where:

- $`C`$: Consistency, one minus the volatility percentile rank
- $`S_{\text{rank}}`$: Average snap-share percentile rank
- $`O_{\text{rank}}`$: Average opportunities (carries + targets) per game percentile rank

Ranks are calculated among eligible players in the same season and position.

A higher quality score rewards low volatility combined with meaningful offensive participation.

## 7. Waiver Wire Alerts (Waiver Opportunity Score)

### Objective

Waiver Wire Alerts evaluate potential waiver-wire opportunities by combining fantasy production, offensive usage, stability, and recent trends. As of a chosen week, they ask whose production and usage are rising or falling, and whether the move is backed by the player's role.

Unlike the Player Consistency Model, Waiver Wire Alerts directly incorporate fantasy football performance.

### Who Is Scored

Only players matched by `gsis_id` to the ESPN snapshot and rostered in at most 50% of leagues (`MAX_PERCENT_OWNED`) are scored. The filter is applied before scoring, so heavily rostered starters never appear. The snapshot is a single collection, not weekly ownership history.

### Windows

A valid game has snap share, opportunity share, opportunities, and a PPR score. The recent window is the last 2–4 valid games (one third of the player's valid games, at least 2 and at most 4). The previous window is every valid game before it.

### Overall Equation

```math
W=100\left(0.40F+0.20O+0.15S+0.15V+0.10T\right)\times E
```

Where:

- $`W`$: Waiver score, 0–100
- $`F`$: Fantasy performance score
- $`O`$: Opportunity share score
- $`S`$: Snap share score
- $`V`$: Stability score
- $`T`$: Trend confirmation score
- $`E`$: Evidence factor

Each component is kept on a 0–1 scale.

### Position Benchmarks

| Position | PPR points per game $`B_p`$ | Snap share $`B_s`$ | Opportunity share $`B_o`$ | PPR trend scale $`s_p`$ |
|---|---:|---:|---:|---:|
| RB | 12.0 | 50% | 18% | 6 |
| WR | 12.0 | 65% | 17% | 6 |
| TE | 8.0 | 60% | 13% | 4 |

The benchmarks are set in `MODEL2_SETTINGS` in `app.py`. A player at or above a benchmark gets full credit for that part of the score; they are not minimums.

### Trends

Each trend compares the recent window with the previous window and is clipped to −1 to +1:

```math
\tau_x=\max\!\left(-1,\ \min\!\left(1,\ \frac{\bar{x}_{\text{recent}}-\bar{x}_{\text{previous}}}{s_x}\right)\right)
```

Where $`x`$ is PPR points ($`\tau_p`$), opportunity share ($`\tau_o`$), or snap share ($`\tau_s`$), with scales $`s_p=B_p/2`$, $`s_o=0.06`$, and $`s_s=0.10`$.

### Components

| Component | Weight | Formula |
|---|---:|---|
| $`F`$ Fantasy performance | 40% | $`0.75\left[0.65\min\left(1,\frac{\bar{p}_{\text{recent}}}{B_p}\right)+0.35\min\left(1,\frac{\tilde{p}_{\text{recent}}}{B_p}\right)\right]+0.25\cdot\frac{1+\tau_p}{2}`$, where $`\tilde{p}`$ is the recent median |
| $`O`$ Opportunity share | 20% | $`0.80\min\left(1,\frac{\bar{o}_{\text{recent}}}{B_o}\right)+0.20\cdot\frac{1+\tau_o}{2}`$ |
| $`S`$ Snap share | 15% | $`0.80\min\left(1,\frac{\bar{s}_{\text{recent}}}{B_s}\right)+0.20\cdot\frac{1+\tau_s}{2}`$ |
| $`V`$ Stability | 15% | $`0.60\cdot\frac{1}{1+CV_p}+0.40\cdot\frac{1}{1+\bar{V}_{\text{mix}}/0.12}`$ |
| $`T`$ Trend confirmation | 10% | $`0.65A+0.35U`$ |
| $`E`$ Evidence factor | × | $`0.80+0.20\min\left(1,\frac{n_{\text{valid}}}{w-1}\right)`$, where $`w`$ is the selected week |

Where:

- $`CV_p=\sigma_p/\max(|\bar{p}_{\text{season}}|,2)`$: variation in PPR points across the player's valid games
- $`\bar{V}_{\text{mix}}=0.65\bar{V}_{\text{recent}}+0.35\bar{V}_{\text{season}}`$: weekly volatility from the Player Consistency Model, weighted toward recent weeks
- $`A=\frac{1+(0.5\tau_p+0.3\tau_o+0.2\tau_s)}{2}`$: how strongly the three trends agree
- $`U`$: share of recent games with PPR above the previous average and snap or opportunity share above it too

The median term keeps one big game from dominating $`F`$. Only upward confirmation adds to $`T`$, so a steady decline is not rewarded.

### Waiver Flags

Every scored player receives exactly one of four flags. Usage rising means the snap or opportunity share trend is at least +0.10 (falling: −0.10 or lower). Fantasy rising means the PPR trend is at least +0.20 (falling: −0.20 or lower).

| Flag | Qualifies when |
|---|---|
| 🟢 Must Pick Up | Fantasy and usage rising, at least 67% of recent games confirm the rise, waiver score ≥ 72, recent PPR ≥ 85% of the position benchmark, 5+ valid games |
| 🔵 Slow Riser | Fantasy and usage rising with at least 50% of recent games confirming and score ≥ 52, **or** usage rising with the PPR trend above −0.30 while recent PPR is still below the benchmark, **or** mixed signals that lean upward |
| 🟠 Slow Drop | Fantasy and usage falling with at least 50% of recent games confirming the drop, **or** mixed signals that lean downward |
| 🔴 Must Drop | Fantasy and usage falling, at least 67% of recent games confirm the drop, waiver score ≤ 43, recent PPR under 60% of the benchmark |

Mixed signals use a direction score, $`d=0.5\tau_p+0.3\tau_o+0.2\tau_s`$: Slow Riser if $`d>0.05`$, Slow Drop if $`d<-0.05`$, and otherwise Slow Riser only if recent PPR is at or above the benchmark. The rules are in `classify_waiver_scenario` and `conservative_waiver_flag` in `app.py`.

### Limited History

A full history means at least 4 valid games, covering at least 75% of the games played, with 2+ games before the recent window. Players with less are still scored, using a recent window of 1–2 games and only the metrics they have. Their evidence factor is stricter, $`(0.75+0.25\min(n/5,1))(0.80+0.20c)`$, where $`c`$ is the share of non-missing values. They never receive Must Pick Up or Must Drop.

These flags are analytical indicators, not guaranteed predictions of future performance.

## 8. Repository Structure

```text
RosterWatch/
├── docs/
│   ├── AGENTS.md                  # guidelines for AI coding agents
│   ├── DesignDoc.md               # design document template
│   └── codeReviewer.md            # code review guidelines
├── frontend/
│   ├── streamlit_app/
│   │   └── app.py                 # Streamlit dashboard (both models)
│   └── README.md
├── infra/
│   └── README.md                  # placeholder for deployment
├── models/
│   ├── .gitkeep
│   └── usage_models.py            # earlier usage scoring prototype
├── notebooks/
│   ├── README.md
│   ├── espn_ownership.csv         # ESPN % rostered snapshot (Waiver Wire Alerts filter)
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

Copy the usage data and the ESPN ownership snapshot from `notebooks/` into `data/processed/`, where the dashboard looks for them.

macOS/Linux:

```bash
mkdir -p data/processed
cp notebooks/weekly_usage.parquet notebooks/espn_ownership.csv data/processed/
```

Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force data/processed
Copy-Item notebooks/weekly_usage.parquet, notebooks/espn_ownership.csv data/processed/
```

Waiver Wire Alerts only show players for a season that has ownership rows in `espn_ownership.csv` (see Data Location in Section 5).

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

The page has three tabs: **About RosterWatch** (what the tool does and both models' equations), **Player Consistency Model** and **Waiver Wire Alerts**. The sidebar filters apply to every tab.

### Player Consistency Model

1. Select the NFL season and position in the sidebar (Control Center).
2. Open the Player Consistency Model tab and choose the desired week range.
3. Explore player usage charts in the Player Explorer.
4. Review the Most Volatile and Most Consistent rankings.
5. Compare how player involvement changes over time.

### Waiver Wire Alerts

1. Select the NFL season and position in the sidebar.
2. Open the Waiver Wire Alerts tab and choose the week to analyze through.
3. Review the Weekly Waiver Rankings: flag, waiver score, % rostered, recent and season PPR, PPR trend, and volatility.
4. Filter players by waiver flag.
5. Pick a player in the Player Waiver Trend Explorer to see their fantasy production and usage trends.

The "How players qualify for each waiver flag" panel on the Waiver Wire Alerts tab lists the flag rules from Section 7.

## 11. Results & Evaluation

RosterWatch produces interpretable player-level analytics, including:

- Weekly usage volatility scores
- Season-level consistency scores
- Offensive usage trends
- Fantasy scoring trends
- Waiver opportunity rankings
- Waiver flags

These outputs are descriptive and heuristic.

### Player Consistency Model Backtest

A backtest on 2022–2025 asked whether a player's average volatility so far (after 4+ games, 35%+ snaps) separated players who lost 15+ points of snap share in two of their next three games from those who did not. The current formula scores 0.55 on 2022–2024 and 0.53 on 2025, where 0.50 is a coin flip. Volatility describes how stable a role has been; on its own it is a weak warning of a coming role loss.

### Waiver Wire Alerts Backtest

Every player was scored as of each of Weeks 5–15 in 2022–2025 (44 analysis weeks, 15,406 player-weeks with three later games). A flag counts as a hit if the player then averaged starter-level points over their next three games: at least the per-game average of the RB24, WR36, or TE12 in 2022–2024 (9.9, 7.7, and 5.8 points in nflverse standard scoring). The waiver-pool columns keep only players scoring under 75% of that level, as a stand-in for availability, because % rostered history does not exist for past seasons.

| Flag | All players: flags/week | All players: hit rate | Waiver pool: flags/week | Waiver pool: hit rate |
|---|---:|---:|---:|---:|
| Must Pick Up | 20.3 | 47% | 5.8 | 23% |
| Slow Riser | 149.7 | 19% | 110.4 | 7% |
| Slow Drop | 158.2 | 17% | 115.0 | 5% |
| Must Drop | 22.0 | 9% | 19.8 | 6% |
| Any flag (base rate) | 350.1 | 19% | 250.9 | 7% |

Must Pick Up hits at about 2.5 times the base rate among all players and over 3 times in the waiver pool. Must Drop hits at half the base rate among all players and slightly below it in the waiver pool. Slow Riser and Slow Drop hit at about the base rate, so they describe direction rather than predict a breakout. Case studies and the full method are in the project design document.

Further evaluation could compare the waiver score against a fantasy-points-only baseline and test how sensitive the rankings are to the weights and thresholds.

## 12. Testing

Run the test suite from the repository root:

```bash
python -m pytest
```

Current tests:

- `tests/test_usage_models.py`: checks the earlier usage scoring prototype in `models/usage_models.py`.
- `tests/test_example.py`: checks the project template example in `src/`.

The dashboard's Player Consistency Model and Waiver Wire Alerts calculations in `frontend/streamlit_app/app.py` are not covered by automated tests yet.

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
6. The ownership filter uses one ESPN-wide % rostered snapshot, not your league's rosters or week-by-week ownership history.
7. Model scores should not be interpreted as calibrated probabilities.

These limitations provide opportunities for further research and improvement.

## 14. Future Improvements

Potential extensions include:

- Incorporating injury reports and depth-chart changes.
- Adding matchup and defensive-strength features.
- Developing machine-learning models for future fantasy production.
- Collecting weekly % rostered history so the ownership filter can be backtested.
- Supporting additional fantasy scoring formats.
- Adding customizable league settings.
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
