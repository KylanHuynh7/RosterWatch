"""Download RosterWatch tables from BigQuery into data/processed/ before the app starts.

The Streamlit app already looks in data/processed/ first, so app.py needs no changes.
Locally you can still drop the files there by hand; this script is only used in the container.

Env vars:
  RW_BQ_PROJECT  GCP project id (defaults to the client's default project)
  RW_BQ_DATASET  dataset name (default: rosterwatch)
"""
import os
import sys
from pathlib import Path

from google.cloud import bigquery

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
DATASET = os.environ.get("RW_BQ_DATASET", "rosterwatch")


def main() -> int:
    client = bigquery.Client(project=os.environ.get("RW_BQ_PROJECT") or None)
    OUT.mkdir(parents=True, exist_ok=True)

    usage = client.query(f"SELECT * FROM `{client.project}.{DATASET}.weekly_usage`").to_dataframe()
    # BigQuery returns nullable Int64; restore plain int64 where there are no nulls (matches the original parquet).
    for col in usage.select_dtypes("Int64").columns:
        if not usage[col].isna().any():
            usage[col] = usage[col].astype("int64")
    usage.to_parquet(OUT / "weekly_usage.parquet", index=False)

    ownership = client.query(f"SELECT * FROM `{client.project}.{DATASET}.espn_ownership`").to_dataframe()
    ownership.to_csv(OUT / "espn_ownership.csv", index=False)

    print(f"fetched weekly_usage={len(usage)} rows, espn_ownership={len(ownership)} rows into {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
