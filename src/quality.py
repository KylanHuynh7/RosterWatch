"""Data-quality checks: does every player link to every source?

The pipeline writes two tables (also uploaded to BigQuery):
  id_mapping_report  one row per check, with how many rows/players matched
  id_unmatched       players found only by name or not at all: the to-do list for fixing ids

A player who doesn't link to ESPN looks 0% rostered, and one who doesn't link to Sleeper
has no injury status, so these checks matter for the flags, not just for tidiness.
"""
import pandas as pd


def id_mapping_report(usage, player_ids, rostership, sleeper, live_season):
    """Match rates for each id the pipeline depends on."""
    live = usage[usage.season == live_season].drop_duplicates("pfr_player_id")
    checks = [
        ("Player-games linked to nflverse stats and play-by-play (gsis_id)",
         usage.gsis_id.notna().sum(), len(usage),
         "Unlinked player-games get 0 targets, carries and situational opportunities"),
        ("Players matched by Pro Football Reference id",
         (player_ids.id_source == "pfr_id").sum(), len(player_ids),
         "The reliable match; the rest are matched by name or not at all"),
        ("Players matched by name only (check by hand)",
         (player_ids.id_source == "name_match").sum(), len(player_ids),
         "Exact name + position match; usually right, but worth confirming"),
        (f"{live_season} players found in the latest ESPN rostership snapshot",
         live.espn_id.isin(rostership.espn_id).sum(), len(live),
         "Unmatched players look 0% rostered, so they can be wrongly flagged as waiver risers"),
        (f"{live_season} players found in Sleeper data",
         live.sleeper_id.isin(sleeper.sleeper_id).sum(), len(live),
         "Unmatched players have no injury status or depth chart on their flags"),
    ]
    rows = [{"check": c, "matched": int(m), "total": int(t), "share": round(m / t, 4) if t else None,
             "why_it_matters": w} for c, m, t, w in checks]
    return pd.DataFrame(rows)


def unmatched_players(player_ids):
    """Players not matched by PFR id, most-played first."""
    cols = ["pfr_player_id", "player", "position", "team", "first_season", "last_season", "games",
            "id_source", "gsis_id", "espn_id", "sleeper_id"]
    out = player_ids[player_ids.id_source != "pfr_id"][cols]
    return out.sort_values(["id_source", "games"], ascending=[False, False]).reset_index(drop=True)
