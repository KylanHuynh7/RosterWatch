"""Model settings shared by the data loader, both models and the dashboard.

The dashboard imports these same objects to fill its explanation tables, so a
change here changes both the model and what the About tab shows.
"""

MAX_PERCENT_OWNED = 50.0
POSITIONS = ["RB", "WR", "TE"]
VOLATILITY_METRICS = [
    "snap_pct",
    "opp_share",
    "passdown_share",
    "redzone_share",
]
POSITION_WEIGHTS = {
    "RB": [0.30, 0.45, 0.15, 0.10],
    "WR": [0.35, 0.35, 0.20, 0.10],
    "TE": [0.40, 0.30, 0.20, 0.10],
}
CONSISTENCY_THRESHOLDS = {
    "RB": (10, 0.50, 10),
    "WR": (10, 0.65, 5),
    "TE": (10, 0.60, 4),
}
MIN_COMPARISONS = 8
MODEL2_WEIGHTS = {
    "fantasy": 0.40,
    "opportunity": 0.20,
    "snap": 0.15,
    "stability": 0.15,
    "confirmation": 0.10,
}
MODEL2_SETTINGS = {
    "RB": {
        "ppr": 12.0,
        "snap": 0.50,
        "opp": 0.18,
        "volume": 10,
    },
    "WR": {
        "ppr": 12.0,
        "snap": 0.65,
        "opp": 0.17,
        "volume": 6,
    },
    "TE": {
        "ppr": 8.0,
        "snap": 0.60,
        "opp": 0.13,
        "volume": 4,
    },
}
WAIVER_FLAGS = [
    "Must Pick Up",
    "Slow Riser",
    "Slow Drop",
    "Must Drop",
]
