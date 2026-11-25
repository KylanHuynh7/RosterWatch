"""Model settings shared by the data loader, both models and the dashboard.

The dashboard imports these same objects to fill its explanation tables, so a
change here changes both the model and what the About tab shows.
"""

# Waiver Wire Alerts only score players rostered in at most this % of ESPN leagues.
MAX_PERCENT_OWNED = 50.0
# Positions both models cover.
POSITIONS = ["RB", "WR", "TE"]
# The four usage shares compared week to week by the Player Consistency Model.
VOLATILITY_METRICS = [
    "snap_pct",
    "opp_share",
    "passdown_share",
    "redzone_share",
]
# Weight of each VOLATILITY_METRICS share in weekly volatility, by position
# (same order as VOLATILITY_METRICS; each row sums to 1).
POSITION_WEIGHTS = {
    "RB": [0.30, 0.45, 0.15, 0.10],
    "WR": [0.35, 0.35, 0.20, 0.10],
    "TE": [0.40, 0.30, 0.20, 0.10],
}
# Minimums for the Most Consistent table, by position:
# (weeks played, average snap share, average opportunities per week).
CONSISTENCY_THRESHOLDS = {
    "RB": (10, 0.50, 10),
    "WR": (10, 0.65, 5),
    "TE": (10, 0.60, 4),
}
# ...plus at least this many valid week-to-week comparisons.
MIN_COMPARISONS = 8
# Weights of the five components of the waiver score W (they sum to 1).
MODEL2_WEIGHTS = {
    "fantasy": 0.40,
    "opportunity": 0.20,
    "snap": 0.15,
    "stability": 0.15,
    "confirmation": 0.10,
}
# Waiver Wire Alerts benchmarks, by position: "ppr" points per game, "snap" share
# and "opp" opportunity share. Reaching a benchmark earns full credit for that part;
# they are not minimums. "volume" (opportunities per game) is shown on the About tab
# but not used in the score.
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
# The four waiver flags. Every scored player gets exactly one.
WAIVER_FLAGS = [
    "Must Pick Up",
    "Slow Riser",
    "Slow Drop",
    "Must Drop",
]
