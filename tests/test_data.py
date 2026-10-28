"""Tests for data cleaning and player-id linking, on small made-up tables (no network). Run: pytest"""
import pandas as pd

from src import data, quality


def make_pbp(rows):
    base = {"game_id": "G1", "play_id": 1.0, "season": 2025, "week": 1, "season_type": "REG",
            "posteam": "HOU", "defteam": "JAX", "qtr": 1, "down": 1.0, "ydstogo": 10.0,
            "yardline_100": 60.0, "half_seconds_remaining": 900.0, "game_seconds_remaining": 2700.0,
            "score_differential": 0.0, "wp": 0.5, "play_type": "run", "rusher_player_id": "RB1",
            "receiver_player_id": None, "qb_kneel": 0, "qb_spike": 0}
    return pd.DataFrame([{**base, **r} for r in rows])


# --- play-by-play cleaning -------------------------------------------------

def test_clean_pbp_keeps_only_regular_season_runs_and_passes():
    pbp = make_pbp([{}, {"play_type": "punt"}, {"season_type": "POST"},
                    {"qb_kneel": 1}, {"play_type": "pass", "qb_spike": 1}])
    assert len(data.clean_pbp(pbp)) == 1


def test_clean_pbp_credits_the_receiver_on_passes_and_the_runner_on_runs():
    pbp = make_pbp([{"play_type": "pass", "receiver_player_id": "WR1", "rusher_player_id": None},
                    {"play_type": "run"},
                    {"play_type": "pass", "rusher_player_id": None}])   # sack / throwaway
    ids = data.clean_pbp(pbp).player_id
    assert ids.tolist()[:2] == ["WR1", "RB1"] and pd.isna(ids.iloc[2])


def test_clean_pbp_tags_passing_downs_and_red_zone():
    pbp = make_pbp([{"down": 3.0}, {"half_seconds_remaining": 90.0}, {"yardline_100": 15.0}, {}])
    out = data.clean_pbp(pbp)
    assert out.passdown.tolist() == [True, True, False, False]
    assert out.redzone.tolist() == [False, False, True, False]


# --- player-id linking -----------------------------------------------------

CROSSWALK = pd.DataFrame([
    {"pfr_id": "ChubNi00", "name": "Nick Chubb", "position": "RB", "gsis_id": "00-1", "espn_id": 11.0, "sleeper_id": 101.0},
    {"pfr_id": None, "name": "Cody White", "position": "WR", "gsis_id": "00-2", "espn_id": 22.0, "sleeper_id": 202.0},
    {"pfr_id": None, "name": "Kwamie Lassiter", "position": "WR", "gsis_id": "00-3", "espn_id": 33.0, "sleeper_id": None},
    {"pfr_id": "SmitJo01", "name": "John Smith", "position": "TE", "gsis_id": "00-4", "espn_id": 44.0, "sleeper_id": None},
    {"pfr_id": "SmitJo02", "name": "John Smith", "position": "TE", "gsis_id": "00-5", "espn_id": 55.0, "sleeper_id": None},
])


def link(players):
    return data.link_ids(pd.DataFrame(players, columns=["pfr_player_id", "player", "position"]), CROSSWALK)


def test_link_ids_prefers_the_pfr_id():
    out = link([("ChubNi00", "Nick Chubb", "RB")])
    assert out.id_source[0] == "pfr_id" and out.gsis_id[0] == "00-1" and out.sleeper_id[0] == "101"


def test_link_ids_falls_back_to_a_unique_name_match():
    out = link([("WhitCo05", "Cody White", "WR"), ("LassKw00", "Kwamie Lassiter II", "WR")])
    assert out.id_source.tolist() == ["name_match", "name_match"]   # "II" suffix ignored
    assert out.gsis_id.tolist() == ["00-2", "00-3"]


def test_link_ids_refuses_an_ambiguous_name():
    out = link([("SmitJo99", "John Smith", "TE")])   # two John Smiths in the crosswalk
    assert out.id_source[0] == "unmatched" and pd.isna(out.gsis_id[0])


# --- id mapping report -----------------------------------------------------

def test_id_mapping_report_counts_players_missing_from_espn_and_sleeper():
    usage = pd.DataFrame({"season": [2026] * 3, "pfr_player_id": ["A", "B", "C"],
                          "gsis_id": ["g1", "g2", None], "espn_id": [1.0, 2.0, None],
                          "sleeper_id": ["s1", None, None]})
    ids = pd.DataFrame({"id_source": ["pfr_id", "pfr_id", "unmatched"]})
    report = quality.id_mapping_report(usage, ids, pd.DataFrame({"espn_id": [1.0, 2.0]}),
                                       pd.DataFrame({"sleeper_id": ["s1"]}), 2026)
    got = {c: (m, t) for c, m, t in zip(report.check, report.matched, report.total)}
    assert got["Player-games linked to nflverse stats and play-by-play (gsis_id)"] == (2, 3)
    assert got["2026 players found in the latest ESPN rostership snapshot"] == (2, 3)
    assert got["2026 players found in Sleeper data"] == (1, 3)
    assert report.share.between(0, 1).all()
