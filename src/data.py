"""Data loading for RosterWatch.

Sources (all free/public):
  - nflverse via nflreadpy: snap counts (PFR), weekly player stats, play-by-play, ID crosswalk
  - Sleeper API: depth chart, injury status, trending adds/drops
  - ESPN fantasy API (undocumented, public): current % rostered per player

Clean tables (also uploaded to BigQuery by warehouse.py):
  snap_counts   one row per player-game, regular season, RB/WR/TE who took a snap
  pbp_plays     regular-season run and pass plays, with situation tags (passing down, red zone)
  player_ids    one row per player: his id in every source, and how it was matched
  weekly_usage  one row per player-game: snap, target, carry, opportunity and situational shares

API pulls are cached in data/raw and built tables in data/processed, so re-runs are fast.
Delete those files (or pass refresh=True) to pull fresh weekly data.
"""
import json
import re

import nflreadpy as nfl
import numpy as np
import pandas as pd
import requests

from .config import PROCESSED, RAW

SKILL_POSITIONS = ["RB", "WR", "TE"]
ID_COLUMNS = ["gsis_id", "espn_id", "sleeper_id"]   # gsis: nflverse stats + play-by-play, espn: % rostered, sleeper: injuries


def _cached(name, loader, refresh=False, folder=RAW):
    path = folder / f"{name}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    df = loader()
    df.to_parquet(path, index=False)
    return df


def _span(seasons):
    return f"{min(seasons)}_{max(seasons)}"


# ---------------------------------------------------------------------------
# Clean source tables
# ---------------------------------------------------------------------------

def load_snap_counts(seasons, refresh=False):
    """Regular-season snap counts for RB/WR/TE, only games where the player took an offensive snap.

    Missed games are absent rows, not zeros: a zero would look like a collapse in usage
    when it is really an injury or a bye.
    """
    def build():
        s = nfl.load_snap_counts(seasons).to_pandas()
        s = s[(s.game_type == "REG") & s.position.isin(SKILL_POSITIONS) & (s.offense_snaps > 0)]
        keep = ["season", "week", "game_id", "team", "opponent", "player", "pfr_player_id",
                "position", "offense_snaps", "offense_pct"]
        return s[keep].sort_values(["season", "week", "game_id", "team", "player"]).reset_index(drop=True)

    return _cached(f"snap_counts_{_span(seasons)}", build, refresh, folder=PROCESSED)


PBP_COLUMNS = ["game_id", "play_id", "season", "week", "season_type", "posteam", "defteam",
               "qtr", "down", "ydstogo", "yardline_100", "half_seconds_remaining",
               "game_seconds_remaining", "score_differential", "wp", "play_type",
               "rusher_player_id", "receiver_player_id", "qb_kneel", "qb_spike"]


def load_pbp_plays(seasons, refresh=False):
    """Regular-season run and pass plays with the columns RosterWatch uses (see clean_pbp)."""
    def build():
        return clean_pbp(nfl.load_pbp(seasons).select(PBP_COLUMNS).to_pandas())

    return _cached(f"pbp_plays_{_span(seasons)}", build, refresh, folder=PROCESSED)


def clean_pbp(pbp):
    """Keep regular-season runs and passes (no kneel-downs or spikes) and tag each play.

    player_id : who got the opportunity, the targeted receiver or the ball carrier (gsis id).
                Empty for sacks and throwaways.
    passdown  : 3rd or 4th down, or the last 2 minutes of either half
    redzone   : snap starts inside the opponent's 20

    New situational stats go here as one more tag, e.g. a garbage-time rule built from
    qtr, score_differential and wp. load_situational() then counts them per player-game.
    """
    p = pbp[(pbp.season_type == "REG") & pbp.play_type.isin(["run", "pass"])
            & (pbp.qb_kneel != 1) & (pbp.qb_spike != 1)].copy()
    p["player_id"] = p.receiver_player_id.fillna(p.rusher_player_id)
    p["passdown"] = (p.down >= 3) | (p.half_seconds_remaining <= 120)
    p["redzone"] = p.yardline_100 <= 20
    return p.drop(columns=["season_type", "qb_kneel", "qb_spike"]).reset_index(drop=True)


def load_situational(seasons):
    """Per player-game counts of targets + carries in high-leverage situations (from pbp_plays)."""
    pbp = load_pbp_plays(seasons).dropna(subset=["player_id"])
    out = (pbp.groupby(["season", "week", "posteam", "player_id"], as_index=False)
              .agg(passdown_opps=("passdown", "sum"), redzone_opps=("redzone", "sum")))
    return out.rename(columns={"posteam": "team", "player_id": "gsis_id"})


# ---------------------------------------------------------------------------
# Player ID mapping
# ---------------------------------------------------------------------------

def load_player_ids(seasons, refresh=False):
    """One row per player in the snap counts, with his id in every source.

    id_source says how the ids were found:
      pfr_id      matched on the Pro Football Reference id (the reliable way)
      name_match  not in the crosswalk under his PFR id; matched on exact name + position,
                  only when that name is unique. Worth checking by hand.
      unmatched   no ids found; his stats, % rostered and injury status are missing
    """
    def build():
        snaps = load_snap_counts(seasons)
        players = (snaps.sort_values(["season", "week"])
                        .groupby("pfr_player_id", as_index=False)
                        .agg(player=("player", "last"), position=("position", "last"),
                             team=("team", "last"), first_season=("season", "min"),
                             last_season=("season", "max"), games=("week", "size")))
        return link_ids(players, nfl.load_ff_playerids().to_pandas())

    return _cached(f"player_ids_{_span(seasons)}", build, refresh, folder=PROCESSED)


def _norm_name(name):
    """'Kwamie Lassiter II' -> 'kwamielassiter'. Drops suffixes, punctuation and spaces."""
    name = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", str(name or "").lower())
    return re.sub(r"[^a-z]", "", name)


def link_ids(players, crosswalk):
    """Attach gsis/espn/sleeper ids to players (pfr_player_id, player, position) using the
    nflverse crosswalk: by PFR id first, then by unique name + position."""
    cw = crosswalk.copy()
    by_pfr = cw.dropna(subset=["pfr_id"]).drop_duplicates("pfr_id").set_index("pfr_id")[ID_COLUMNS]
    out = players.join(by_pfr, on="pfr_player_id")
    out["id_source"] = np.where(out[["gsis_id", "espn_id"]].notna().any(axis=1), "pfr_id", None)

    cw["_key"] = cw["name"].map(_norm_name) + "|" + cw["position"].astype(str)
    unique = cw[cw["_key"].map(cw["_key"].value_counts()) == 1].set_index("_key")[ID_COLUMNS]
    need = out.index[out.id_source.isna()]
    keys = out.loc[need, "player"].map(_norm_name) + "|" + out.loc[need, "position"].astype(str)
    found = unique.reindex(keys.values).set_axis(need)
    hit = found.index[found[["gsis_id", "espn_id"]].notna().any(axis=1)]
    out.loc[hit, ID_COLUMNS] = found.loc[hit, ID_COLUMNS]
    out.loc[hit, "id_source"] = "name_match"
    out["id_source"] = out.id_source.fillna("unmatched")

    out["espn_id"] = pd.to_numeric(out.espn_id, errors="coerce")
    out["sleeper_id"] = out.sleeper_id.map(lambda v: None if pd.isna(v) else str(int(float(v))))
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Weekly usage (the table the models run on)
# ---------------------------------------------------------------------------

def load_weekly_usage(seasons, refresh=False):
    """One row per player-game: snap %, target share, carry share, opportunity share,
    passing-down share and red-zone share."""
    return _cached(f"usage_v3_{_span(seasons)}", lambda: _build_usage(seasons), refresh, folder=PROCESSED)


def _build_usage(seasons):
    snaps = load_snap_counts(seasons)
    ids = load_player_ids(seasons)
    stats = nfl.load_player_stats(seasons).to_pandas()
    stats = stats[stats.season_type == "REG"]

    # Team totals per game: everyone's targets and carries, including QBs.
    team = (stats.groupby(["season", "week", "team"], as_index=False)
                 [["targets", "carries"]].sum()
                 .rename(columns={"targets": "team_targets", "carries": "team_carries"}))

    # Snap counts use PFR ids; stats and play-by-play use GSIS ids. player_ids links them.
    df = snaps.merge(ids[["pfr_player_id"] + ID_COLUMNS], on="pfr_player_id", how="left")

    # Left join: a WR who played 40 snaps but saw 0 targets has no stat row. That is a 0, not missing.
    player_cols = ["player_id", "season", "week", "targets", "carries", "receptions",
                   "receiving_yards", "rushing_yards", "fantasy_points_ppr"]
    counts = ["targets", "carries", "receptions", "receiving_yards", "rushing_yards", "fantasy_points_ppr"]
    df = df.merge(stats[player_cols], left_on=["gsis_id", "season", "week"],
                  right_on=["player_id", "season", "week"], how="left")
    df[counts] = df[counts].fillna(0)
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

    keep = ["season", "week", "game_id", "player", "pfr_player_id", "gsis_id", "espn_id", "sleeper_id",
            "position", "team", "opponent", "offense_snaps", "snap_pct", "targets", "carries",
            "target_share", "carry_share", "opp_share", "passdown_opps", "passdown_share",
            "redzone_opps", "redzone_share", "receptions", "receiving_yards",
            "rushing_yards", "fantasy_points_ppr"]
    return df[keep].sort_values(["season", "pfr_player_id", "week"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Market data: Sleeper and ESPN
# ---------------------------------------------------------------------------

def load_sleeper():
    """Sleeper player metadata (depth chart, injury status) + 7-day trending adds/drops.

    Linked to our players by sleeper_id from player_ids (most Sleeper players don't list a
    gsis id). Sleeper has no "% rostered" field. Sleeper asks that /players/nfl is fetched
    at most once a day, so it is cached per day.
    """
    day = pd.Timestamp.today().strftime("%Y-%m-%d")

    def fetch():
        players = requests.get("https://api.sleeper.app/v1/players/nfl", timeout=60).json()
        rows = [{"sleeper_id": k, "sleeper_name": v.get("full_name"),
                 "gsis_id": (v.get("gsis_id") or "").strip() or None,
                 "espn_id": pd.to_numeric(v.get("espn_id"), errors="coerce"),
                 "depth_chart_order": v.get("depth_chart_order"),
                 "injury_status": v.get("injury_status")}
                for k, v in players.items() if v.get("position") in SKILL_POSITIONS]
        df = pd.DataFrame(rows)
        for kind in ("add", "drop"):
            t = requests.get(f"https://api.sleeper.app/v1/players/nfl/trending/{kind}",
                             params={"lookback_hours": 168, "limit": 100}, timeout=30).json()
            counts = {x["player_id"]: x["count"] for x in t}
            df[f"sleeper_{kind}s_7d"] = df.sleeper_id.map(counts).fillna(0).astype(int)
        return df

    return _cached(f"sleeper_v2_{day}", fetch)


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
