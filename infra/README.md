# RosterWatch on Google Cloud

The dashboard runs on **Cloud Run** and reads its data from **BigQuery**.

```text
notebooks/*.parquet, *.csv ──(infra/load_bigquery.sh)──▶ BigQuery dataset `rosterwatch`
                                                              │
Cloud Run service `rosterwatch` ── on startup: infra/fetch_data.py ──┘
        │                          writes data/processed/
        └─ then: streamlit run frontend/streamlit_app/app.py
```

`app.py` is unchanged: it reads `data/processed/` the same way it does locally. The container just fills that folder from BigQuery first.

| Item | Value |
|---|---|
| Project | `rosterwatch-ds3-fall2026` |
| Region | `us-central1` |
| BigQuery dataset | `rosterwatch` (tables `weekly_usage`, `espn_ownership`) |
| Cloud Run service | `rosterwatch` |
| Runtime service account | `rosterwatch-run` (BigQuery read-only) |

## Files

| File | Purpose |
|---|---|
| `Dockerfile` | Python 3.12 image: fetch data, then start Streamlit on `$PORT` |
| `infra/fetch_data.py` | Copies both BigQuery tables into `data/processed/` at container start |
| `infra/requirements-app.txt` | Pinned runtime dependencies (smaller than the full `requirements.txt`) |
| `infra/load_bigquery.sh` | Reloads the BigQuery tables from `notebooks/` |
| `.dockerignore`, `.gcloudignore` | Keep data, notebooks and local files out of the upload and image |

## Updating the data

1. Run the notebook and commit the new `notebooks/weekly_usage.parquet` / `notebooks/espn_ownership.csv`.
2. Reload BigQuery from the repo root:

   ```bash
   bash infra/load_bigquery.sh
   ```

3. Each instance loads the data once, when it starts. The service scales to zero when idle, so the next cold start picks up the new data. To refresh immediately, start a new revision:

   ```bash
   gcloud run services update rosterwatch --region us-central1 \
     --update-env-vars DATA_REFRESHED_AT="$(date -u +%Y%m%dT%H%M%SZ)"
   ```

## Deploying code changes

From the repo root, on the branch you want live:

```bash
gcloud config get-value project   # must print rosterwatch-ds3-fall2026
gcloud run deploy rosterwatch --source . --region us-central1 \
  --service-account rosterwatch-run@rosterwatch-ds3-fall2026.iam.gserviceaccount.com \
  --min-instances 0 --max-instances 1 --memory 1Gi --cpu 1 \
  --set-env-vars RW_BQ_PROJECT=rosterwatch-ds3-fall2026
```

Cloud Build builds the image from the `Dockerfile` and Cloud Run rolls it out. Only the files listed by `gcloud meta list-files-for-upload .` are uploaded.

## Cost guardrails

The goal is to stay inside the free tier.

- `--min-instances 0`: no instance runs (or bills) while nobody is visiting.
- `--max-instances 1`: traffic spikes cannot scale costs up.
- The budget `rosterwatch-spend-cap` pauses Cloud Run if it reaches $2 in a month. The budget `rosterwatch-cap` emails at $3 across the project.
- The data is about 4 MB, so BigQuery storage and queries stay within the free tier.

Do not raise these limits without checking with the project owner.

## Running locally against BigQuery (optional)

The normal local setup copies files from `notebooks/` (see the main README). To test the container's data path instead:

```bash
gcloud auth application-default login
RW_BQ_PROJECT=rosterwatch-ds3-fall2026 python infra/fetch_data.py
streamlit run frontend/streamlit_app/app.py
```
