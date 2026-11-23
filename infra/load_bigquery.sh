#!/usr/bin/env bash
# Reload the RosterWatch BigQuery tables from the files committed in notebooks/.
# Run from the repo root after the notebook produces new data:  bash infra/load_bigquery.sh
set -euo pipefail

PROJECT="${RW_BQ_PROJECT:-rosterwatch-ds3-fall2026}"
DATASET="${RW_BQ_DATASET:-rosterwatch}"

bq load --project_id "$PROJECT" --replace --source_format=PARQUET \
  "$PROJECT:$DATASET.weekly_usage" notebooks/weekly_usage.parquet

# Explicit schema keeps the ID columns as STRING (autodetect would turn them into integers).
bq load --project_id "$PROJECT" --replace --source_format=CSV --skip_leading_rows=1 \
  "$PROJECT:$DATASET.espn_ownership" notebooks/espn_ownership.csv \
  ownership_season:INTEGER,collected_at_utc:STRING,source:STRING,espn_id:STRING,player_name:STRING,position:STRING,espn_team_id:INTEGER,percentOwned:FLOAT,percentStarted:FLOAT,percentChange:FLOAT,roster_share:FLOAT,injury_status:STRING,ownership_date_verified:BOOLEAN,ownership_scope:STRING,gsis_id:STRING,low_rostered:BOOLEAN

echo "Loaded. Running instances keep their old copy; redeploy (see infra/README.md) to pick up the new data right away."
