# Frontend

The dashboard is a single Streamlit script: `streamlit_app/app.py`.

- **What it draws:** a sidebar (season and position filters) and three tabs: About RosterWatch (both models' equations and settings), Player Consistency Model and Waiver Wire Alerts.
- **Where the math lives:** the data loader and both models are in `src/rosterwatch/`. `app.py` imports them and caches their results with `st.cache_data`, so changing a filter does not recompute everything.
- **Data it reads:** `data/processed/weekly_usage.parquet` (or `.csv`) and `data/processed/espn_ownership.csv`. Section 9 of the main README shows how to create them.
- **Theme:** `.streamlit/config.toml` at the repository root.

Run it from the repository root:

```bash
streamlit run frontend/streamlit_app/app.py
```
