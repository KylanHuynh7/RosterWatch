"""Data loading for RosterWatch.

Sources (all free/public):
  - nflverse via nflreadpy: weekly snap counts (PFR) + weekly player stats
  - ESPN fantasy API (undocumented, public): current % rostered per player

API pulls are cached in data/raw and the built usage table in data/processed, so
re-runs are fast. Delete those files (or pass refresh=True) to pull fresh weekly data.
"""
import json

import nflreadpy as nfl
import pandas as pd
import requests

from .config import PROCESSED, RAW

SKILL_POSITIONS = ["RB", "WR", "TE"]


def _cached(name, loader, refresh=False, folder=RAW):
    path = folder / f"{name}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    df = loader()
    df.to_parquet(path, index=False)
    return df


def load_weekly_usage(seasons, refresh=False):
    """One row per player-game: snap %, target share, carry share, opportunity share.

    Only counts games the player actually took an offensive snap in, so injuries and
    bye weeks show up as missing rows rather than zeros (a zero would look like a
    collapse in usage when it is really an absence).
    """
    key = f"usage_v2_{min(seasons)}_{max(seasons)}"
    return _cached(key, lambda: _build_usage(seasons), refresh, folder=PROCESSED)


def _build_usage(seasons):
    snaps = nfl.load_snap_counts(seasons).to_pandas()
    stats = nfl.load_player_stats(seasons).to_pandas()
    ids = nfl.load_ff_playerids().to_pandas()

    snaps = snaps[(snaps.game_type == "REG") & snaps.position.isin(SKILL_POSITIONS)
                  & (snaps.offense_snaps > 0)]
    stats = stats[stats.season_type == "REG"]

    # Team totals per game: everyone's targets and carries, including QBs.
    team = (stats.groupby(["season", "week", "team"], as_index=False)
                 [["targets", "carries"]].sum()
                 .rename(columns={"targets": "team_targets", "carries": "team_carries"}))

    # Snap counts use PFR ids; stats use GSIS ids. The ffverse crosswalk links them.
    xwalk = ids[["pfr_id", "gsis_id", "espn_id"]].dropna(subset=["pfr_id"]).drop_duplicates("pfr_id")
    df = snaps.merge(xwalk, left_on="pfr_player_id", right_on="pfr_id", how="left")

    # Left join: a WR who played 40 snaps but saw 0 targets has no stat row. That is a 0, not missing.
    player_cols = ["player_id", "season", "week", "targets", "carries", "receptions",
                   "receiving_yards", "rushing_yards", "fantasy_points_ppr"]
    df = df.merge(stats[player_cols], left_on=["gsis_id", "season", "week"],
                  right_on=["player_id", "season", "week"], how="left")
    df[["targets", "carries", "receptions", "receiving_yards", "rushing_yards",
        "fantasy_points_ppr"]] = df[["targets", "carries", "receptions", "receiving_yards",
                                     "rushing_yards", "fantasy_points_ppr"]].fillna(0)
    df = df.merge(team, on=["season", "week", "team"], how="left")

    df["snap_pct"] = df.offense_pct
    df["target_share"] = df.targets / df.team_targets
    df["carry_share"] = df.carries / df.team_carries
    # "Opportunity share": the player's slice of all targets + carries. Works for RBs and pass-catchers alike.
    df["opp_share"] = (df.targets + df.carries) / (df.team_targets + df.team_carries)

    # "Meaningful" work: passing downs (3rd/4th down or last 2 min of a half) and red zone.
    sit = load_situational(seasons)
    df = df.merge(sit, on=["season", "week", "team", "gsis_id"], how="left")
    team_sit = sit.groupby(["season", "week", "team"], as_index=False)[["passdown_opps", "redzone_opps"]].sum()
    team_sit.columns = ["season", "week", "team", "team_passdown_opps", "team_redzone_opps"]
    df = df.merge(team_sit, on=["season", "week", "team"], how="left")
    df[["passdown_opps", "redzone_opps"]] = df[["passdown_opps", "redzone_opps"]].fillna(0)
    df["passdown_share"] = df.passdown_opps / df.team_passdown_opps
    df["redzone_share"] = df.redzone_opps / df.team_redzone_opps.where(df.team_redzone_opps > 0)

    keep = ["season", "week", "game_id", "player", "pfr_player_id", "gsis_id", "espn_id",
            "position", "team", "opponent", "offense_snaps", "snap_pct", "targets", "carries",
            "target_share", "carry_share", "opp_share", "passdown_opps", "passdown_share",
            "redzone_opps", "redzone_share", "receptions", "receiving_yards",
            "rushing_yards", "fantasy_points_ppr"]
    return df[keep].sort_values(["season", "pfr_player_id", "week"]).reset_index(drop=True)


def load_situational(seasons):
    """Per player-game counts of targets + carries in high-leverage situations, from play-by-play.

    passdown: 3rd or 4th down, or the last 2 minutes of either half (where pass-catching backs live)
    redzone : snap starts inside the opponent's 20
    """
    cols = ["season", "week", "season_type", "posteam", "down", "yardline_100",
            "half_seconds_remaining", "play_type", "rusher_player_id", "receiver_player_id",
            "qb_kneel", "qb_spike"]
    pbp = nfl.load_pbp(seasons).select(cols).to_pandas()
    pbp = pbp[(pbp.season_type == "REG") & pbp.play_type.isin(["run", "pass"])
              & (pbp.qb_kneel != 1) & (pbp.qb_spike != 1)]
    pbp["player"] = pbp.receiver_player_id.fillna(pbp.rusher_player_id)  # target or carry
    pbp = pbp.dropna(subset=["player"])
    pbp["passdown"] = (pbp.down >= 3) | (pbp.half_seconds_remaining <= 120)
    pbp["redzone"] = pbp.yardline_100 <= 20
    out = (pbp.groupby(["season", "week", "posteam", "player"], as_index=False)
              .agg(passdown_opps=("passdown", "sum"), redzone_opps=("redzone", "sum")))
    return out.rename(columns={"posteam": "team", "player": "gsis_id"})


def load_sleeper():
    """Sleeper player metadata (depth chart, injury status) + 7-day trending adds/drops.

    Sleeper has no "% rostered" field. Trending adds/drops (top 100 each) is the closest
    market signal it offers. Sleeper asks that /players/nfl is fetched at most once a day,
    so it is cached per day.
    """
    day = pd.Timestamp.today().strftime("%Y-%m-%d")

    def fetch():
        players = requests.get("https://api.sleeper.app/v1/players/nfl", timeout=60).json()
        rows = [{"sleeper_id": k, "gsis_id": v.get("gsis_id"), "sleeper_name": v.get("full_name"),
                 "depth_chart_order": v.get("depth_chart_order"),
                 "injury_status": v.get("injury_status")}
                for k, v in players.items() if v.get("position") in SKILL_POSITIONS and v.get("gsis_id")]
        df = pd.DataFrame(rows)
        for kind in ("add", "drop"):
            t = requests.get(f"https://api.sleeper.app/v1/players/nfl/trending/{kind}",
                             params={"lookback_hours": 168, "limit": 100}, timeout=30).json()
            counts = {x["player_id"]: x["count"] for x in t}
            df[f"sleeper_{kind}s_7d"] = df.sleeper_id.map(counts).fillna(0).astype(int)
        df["gsis_id"] = df.gsis_id.str.strip()
        return df

    return _cached(f"sleeper_{day}", fetch)


def load_espn_rostership(season):
    """Current % rostered in ESPN leagues for every fantasy-relevant player.

    ESPN only exposes *today's* number, so to have a rostership history the project
    needs to snapshot this weekly (save each pull with a date).
    """
    url = (f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{season}"
           "/segments/0/leaguedefaults/3?view=kona_player_info")
    flt = {"players": {"limit": 3000, "sortPercOwned": {"sortPriority": 1, "sortAsc": False},
                       "filterSlotIds": {"value": [2, 4, 6]}}}  # RB, WR, TE slots
    r = requests.get(url, headers={"X-Fantasy-Filter": json.dumps(flt),
                                   "User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    rows = []
    for p in r.json()["players"]:
        pl = p["player"]
        own = pl.get("ownership", {})
        rows.append({"espn_id": float(pl["id"]), "espn_name": pl["fullName"],
                     "pct_rostered": own.get("percentOwned"),
                     "pct_rostered_change": own.get("percentChange")})
    df = pd.DataFrame(rows)
    df["snapshot_date"] = pd.Timestamp.today().normalize()
    return df
