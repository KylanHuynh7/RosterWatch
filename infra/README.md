# Infra

## BigQuery (implemented: `src/warehouse.py`)

The loader uploads 12 tables to one dataset, replacing each table on every run so re-running is always safe:

| Table | What |
|---|---|
| `weekly_usage` | Every player-game 2022–present: snap, opportunity, passing-down, red-zone share |
| `signals_current` | This week's flags with reasons |
| `shift_scores_current`, `cusum_current`, `usage_trends_current` | Scores and trends for the dashboard |
| `backtest_*`, `case_studies`, `case_study_flags` | Methodology evidence |
| `rostership_snapshots` | All weekly % rostered snapshots, rebuilt from `data/raw/rostership/` |
| `sleeper_players` | Sleeper depth chart, injury status, trending adds/drops |

### One-time setup (one person does this, then shares access)

1. **Create a Google Cloud project.** Go to https://console.cloud.google.com, open the project picker, then New project. Copy the **project ID** (for example `rosterwatch-123456`), not the display name.
2. **Choose sandbox or billing.**
   - **Sandbox (no credit card):** works right away, but **tables are deleted after 60 days**. That's fine for EDA. The pipeline can re-upload everything, and the rostership history is safe in git.
   - **Billing account:** no expiry. This project's data is tiny (under 10 MB), well inside BigQuery's free tier.
3. **Enable the BigQuery API** for the project (APIs & Services → Enable APIs → BigQuery API). It's usually on already.
4. **Choose how your laptop signs in to Google (pick one):**
   - **A. Your own Google login (simplest; no key file to leak).** Install the Google Cloud CLI (https://cloud.google.com/sdk/docs/install), then run:
     ```bash
     gcloud auth application-default login
     ```
   - **B. A service-account key.** In IAM & Admin → Service Accounts, create a service account with the roles **BigQuery Data Editor** and **BigQuery Job User**, then Keys → Add key → JSON. Save the file **outside the repo** (for example `~/keys/rosterwatch-sa.json`) and set `GOOGLE_APPLICATION_CREDENTIALS` in `.env`. Never commit it.
5. **Fill in `.env`** (copy it from `.env.example`):
   ```
   GCP_PROJECT_ID=rosterwatch-123456
   BQ_DATASET=rosterwatch
   ```
6. **Give teammates access:** IAM & Admin → IAM → Grant access → their Google email, with the roles **BigQuery Data Viewer** and **BigQuery Job User**. Use **Data Editor** for anyone who uploads.

### Run it

```bash
python -m src.warehouse --dry-run    # what would be uploaded; no Google account needed
python -m src.warehouse              # upload
python -m src.pipeline --bigquery    # full weekly run + upload
```

### Starter queries: `infra/bigquery/*.sql`

Paste these into the BigQuery console (Query editor). Each was checked against the real data.

| File | Shows |
|---|---|
| `01_data_quality.sql` | Values out of range, missing IDs (currently 71 rows lack a `gsis_id`, an EDA item) |
| `02_rolling_usage.sql` | Rolling 3-game average and standard deviation with `AVG() OVER (... ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)` |
| `03_weekly_change_by_position.sql` | Typical game-to-game swing by position, using `LAG()` |
| `04_this_weeks_flags.sql` | The current flag list |

## Planned

- **Weekly job:** a GitHub Actions workflow on a schedule (Tuesday morning, after Monday Night Football) that runs `pytest`, then `python -m src.pipeline --bigquery`. GCP credentials go in GitHub repository secrets.
- **Looker Studio:** Create → Data source → BigQuery → pick the project → dataset `rosterwatch` → a table. The dashboard then refreshes itself whenever the pipeline uploads.
- **Vertex AI (optional):** Gemini rewords each flag's `reason`. Prompts contain only our numbers, and the output is checked against them.
