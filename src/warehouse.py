"""Load RosterWatch tables into Google BigQuery.

    python -m src.warehouse --dry-run    # list what would be uploaded (no Google account needed)
    python -m src.warehouse              # upload (needs GCP_PROJECT_ID + credentials, see infra/README.md)
    python -m src.pipeline --bigquery    # run the pipeline, then upload

Every table is fully replaced on each run, so re-running is always safe. Rostership
history is rebuilt from the snapshot CSVs in data/raw/rostership/, which are committed
to git and are the source of truth.
"""
import argparse
import os

import pandas as pd

from . import config

LOCATION = os.environ.get("BQ_LOCATION", "US")


def collect_tables():
    """Return {table_name: DataFrame} for everything that should live in BigQuery."""
    tables = {}
    usage = sorted(config.PROCESSED.glob("usage_*.parquet"))
    if usage:
        tables["weekly_usage"] = pd.read_parquet(usage[-1])
    for csv in sorted(config.RESULTS.glob("*.csv")):
        tables[csv.stem] = pd.read_csv(csv)
    snaps = sorted(config.ROSTERSHIP_SNAPSHOTS.glob("rostership_snapshot_*.csv"))
    if snaps:
        tables["rostership_snapshots"] = pd.concat([pd.read_csv(f) for f in snaps], ignore_index=True)
    sleeper = sorted(config.RAW.glob("sleeper_*.parquet"))
    if sleeper:
        tables["sleeper_players"] = pd.read_parquet(sleeper[-1])
    for df in tables.values():
        for col in df.columns:
            if col.endswith("_date"):
                df[col] = pd.to_datetime(df[col]).dt.date
    return tables


def get_client():
    """BigQuery client for GCP_PROJECT_ID. Credentials come from GOOGLE_APPLICATION_CREDENTIALS
    (a service-account key file) or from `gcloud auth application-default login`."""
    from google.cloud import bigquery   # imported here so the rest of the project works without it

    project = os.environ.get("GCP_PROJECT_ID")
    if not project or project == "your-gcp-project-id":
        raise SystemExit("Set GCP_PROJECT_ID in .env first (see infra/README.md).")
    return bigquery.Client(project=project, location=LOCATION)


def upload(tables, client, dataset=None):
    from google.cloud import bigquery

    dataset = dataset or os.environ.get("BQ_DATASET", "rosterwatch")
    ds = bigquery.Dataset(f"{client.project}.{dataset}")
    ds.location = LOCATION
    client.create_dataset(ds, exists_ok=True)
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    for name, df in tables.items():
        table_id = f"{client.project}.{dataset}.{name}"
        client.load_table_from_dataframe(df, table_id, job_config=job_config).result()
        print(f"  loaded {len(df):>6} rows -> {table_id}")


def main(dry_run=False):
    tables = collect_tables()
    if not tables:
        raise SystemExit("Nothing to upload. Run `python -m src.pipeline` first.")
    if dry_run:
        print("Dry run: these tables would be uploaded")
        for name, df in tables.items():
            print(f"  {name:<28} {len(df):>6} rows  {df.shape[1]:>3} columns")
        return
    upload(tables, get_client())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="list tables without uploading")
    main(p.parse_args().dry_run)
