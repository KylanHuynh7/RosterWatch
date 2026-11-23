# RosterWatch on Google Cloud

The dashboard runs on **Cloud Run** and reads its data from **BigQuery**.

```text
notebooks/*.parquet, *.csv ──(infra/load_bigquery.sh)──▶ BigQuery dataset `rosterwatch`
                                                              │
Cloud Build (infra/cloudbuild.yaml)                           │
  1. fetch-data: infra/fetch_data.py ◀────────────────────────┘  writes build_data/
  2. build: Dockerfile copies build_data/ into the image's data/processed/
  3. push to Artifact Registry
  4. deploy to Cloud Run service `rosterwatch`
                │
Cloud Run starts the container ── streamlit run frontend/streamlit_app/app.py
```

`app.py` is unchanged: it reads `data/processed/` the same way it does locally. The data is baked into the image at build time, so a cold start only has to start Streamlit (about 4-5 s instead of about 20 s). If an image is built without data, the container falls back to fetching from BigQuery at startup.

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
| `infra/cloudbuild.yaml` | Build pipeline: fetch data from BigQuery, build, push, deploy |
| `Dockerfile` | Python 3.12 image with the data baked in; starts Streamlit on `$PORT` |
| `infra/fetch_data.py` | Copies both BigQuery tables into a folder (`RW_OUT_DIR`, default `data/processed/`) |
| `build_data/` | Filled by the build; only `.keep` is committed |
| `infra/requirements-app.txt` | Pinned runtime dependencies (smaller than the full `requirements.txt`) |
| `infra/load_bigquery.sh` | Reloads the BigQuery tables from `notebooks/` |
| `.dockerignore`, `.gcloudignore` | Keep data, notebooks and local files out of the upload and image |

## Updating the data

1. Run the notebook and commit the new `notebooks/weekly_usage.parquet` / `notebooks/espn_ownership.csv`.
2. Reload BigQuery from the repo root:

   ```bash
   bash infra/load_bigquery.sh
   ```

3. Rebuild and deploy (next section). The data is part of the image, so the site keeps showing the old data until a new build is deployed.

## Deploying code changes

From the repo root, on the branch you want live:

```bash
gcloud config get-value project   # must print rosterwatch-ds3-fall2026
gcloud builds submit --config infra/cloudbuild.yaml --region us-central1 \
  --gcs-source-staging-dir gs://run-sources-rosterwatch-ds3-fall2026-us-central1/cloudbuild-source
```

This takes about 3 minutes. Only the files listed by `gcloud meta list-files-for-upload .` are uploaded. Staging in the `us-central1` bucket keeps storage inside the free tier; the default staging bucket is multi-region and is not.

Do not use `gcloud run deploy --source .` any more: it skips the data step, so every cold start would go back to fetching from BigQuery.

## Cold starts and the warm-up job

With `--min-instances 0`, Cloud Run shuts the container down after a while without traffic. The next visitor then waits for a cold start, about 4-5 s with the data baked in.

The Cloud Scheduler job `rosterwatch-warmup` requests `/_stcore/health` every 10 minutes from 7:00 to 23:59 Eastern. During those hours the container is usually already running. This is not guaranteed, because Cloud Run can still stop idle instances. Each ping is a tiny request, so it stays in the free tier (3 Scheduler jobs per billing account are free).

For zero cold starts, set `--min-instances 1` in `infra/cloudbuild.yaml`. One instance then runs all the time and is billed while idle. Check the Cloud Run pricing calculator first, and raise the `rosterwatch-spend-cap` budget above that amount, or the cap will pause the site.

## Cost guardrails

The goal is to stay inside the free tier.

- `--min-instances 0`: no instance runs (or bills) while nobody is visiting.
- `--max-instances 1`: traffic spikes cannot scale costs up.
- The budget `rosterwatch-spend-cap` pauses Cloud Run if it reaches $2 in a month. The budget `rosterwatch-cap` emails at $3 across the project.
- The data is about 4 MB, so BigQuery storage and queries stay within the free tier.
- Each build adds an image to Artifact Registry (free up to 0.5 GB). A cleanup policy on the `cloud-run-source-deploy` repo keeps the 3 newest images and deletes older ones (it runs about once a day), so you can roll back up to 2 builds.

Do not raise these limits without checking with the project owner.

## Running locally against BigQuery (optional)

The normal local setup copies files from `notebooks/` (see the main README). To test the container's data path instead:

```bash
gcloud auth application-default login
RW_BQ_PROJECT=rosterwatch-ds3-fall2026 python infra/fetch_data.py
streamlit run frontend/streamlit_app/app.py
```
