# RosterWatch dashboard on Cloud Run.
# Data is pulled from BigQuery during the build (infra/cloudbuild.yaml writes build_data/) and baked
# into the image, so cold starts only need to start Streamlit. If build_data/ is empty, the container
# falls back to fetching from BigQuery at startup.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY infra/requirements-app.txt infra/requirements-app.txt
RUN pip install --no-cache-dir -r infra/requirements-app.txt

COPY .streamlit/ .streamlit/
COPY frontend/ frontend/
COPY infra/fetch_data.py infra/fetch_data.py
COPY build_data/ data/processed/

# Cloud Run sets $PORT (default 8080).
CMD if [ ! -f data/processed/weekly_usage.parquet ] || [ ! -f data/processed/espn_ownership.csv ]; then \
        python infra/fetch_data.py; \
    fi && exec streamlit run frontend/streamlit_app/app.py \
        --server.port=${PORT:-8080} --server.address=0.0.0.0 --server.headless=true \
        --server.fileWatcherType=none --browser.gatherUsageStats=false
