"""Paths and settings shared by every module.

Settings can be overridden with environment variables (see .env.example).
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path=ROOT / ".env"):
    """Read KEY=VALUE lines from .env into the environment (real env vars win)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()

RAW = ROOT / "data" / "raw"                # API pulls, cached as-is
PROCESSED = ROOT / "data" / "processed"    # built tables (weekly usage)
ROSTERSHIP_SNAPSHOTS = RAW / "rostership"  # ESPN only exposes today's number: keep every pull
RESULTS = ROOT / "results"                 # CSVs for the dashboard + brief.html
MODELS = ROOT / "models"                   # fitted noise calibrations
FRONTEND = ROOT / "frontend"

for _d in (RAW, PROCESSED, ROSTERSHIP_SNAPSHOTS, RESULTS, MODELS):
    _d.mkdir(parents=True, exist_ok=True)

LIVE_SEASON = int(os.environ.get("ROSTERWATCH_SEASON", 2026))
HISTORY_START = int(os.environ.get("ROSTERWATCH_HISTORY_START", 2022))
HISTORY = list(range(HISTORY_START, LIVE_SEASON))
