"""Download RosterWatch tables from BigQuery into a folder the Streamlit app reads.

Normally this runs once per build (infra/cloudbuild.yaml), and the data is baked into the image
so cold starts skip it. If the image has no data, the container runs it at startup instead.
The Streamlit app already looks in data/processed/ first, so app.py needs no changes.

Env vars:
  RW_BQ_PROJECT  GCP project id (defaults to the client's default project)
  RW_BQ_DATASET  dataset name (default: rosterwatch)
  RW_OUT_DIR     output folder (default: data/processed under the repo root)
"""
import os
import sys
from pathlib import Path

from google.cloud import bigquery

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("RW_OUT_DIR") or ROOT / "data" / "processed")
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
