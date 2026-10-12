# Frontend

## Looker Studio dashboard (planned)

The pipeline writes one tidy table per view to `results/`. Later these go to BigQuery (see `infra/`). Until then, upload them to Google Sheets and connect Looker Studio with Add data → Google Sheets.

| View | Source table | Charts |
|---|---|---|
| Waiver opportunity | `signals_current.csv` (Waiver Riser, Slow Climb) | Flag table with `reason`; % rostered vs. shift score |
| Role instability | `signals_current.csv` (Fragile Starter, Slow Bleed, Injury Check) | Flag table; the starter shown next to his rising teammate |
| Player trends | `usage_trends_current.csv`, `cusum_current.csv` | Weekly snap / opportunity / passing-down share; CUSUM tally |
| Methodology | `backtest_thresholds.csv`, `backtest_cusum.csv`, `case_studies.csv` | Threshold vs. hit rate; 2025 case studies |

## Kickoff brief

`brief_template.html` is the one-page summary. `src/brief.py` fills it with data from `results/` and writes `results/brief.html`, which runs automatically at the end of the pipeline.
