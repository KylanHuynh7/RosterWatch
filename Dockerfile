# RosterWatch dashboard on Cloud Run: pulls data from BigQuery at startup, then serves Streamlit.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY infra/requirements-app.txt infra/requirements-app.txt
RUN pip install --no-cache-dir -r infra/requirements-app.txt

COPY frontend/ frontend/
COPY infra/fetch_data.py infra/fetch_data.py
RUN mkdir -p data/processed

# Cloud Run sets $PORT (default 8080).
CMD python infra/fetch_data.py && exec streamlit run frontend/streamlit_app/app.py \
    --server.port=${PORT:-8080} --server.address=0.0.0.0 --server.headless=true
