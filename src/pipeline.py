"""Run the full RosterWatch pipeline: Looker Studio-ready CSVs + brief.html in results/.

From the repo root:
    python -m src.pipeline                  # default: 2.0 SD shift threshold, CUSUM +/-3
    python -m src.pipeline --threshold 1.5
    python -m src.pipeline --bigquery       # then upload everything to BigQuery
"""
import argparse
import json

import pandas as pd

from . import backtest, brief, data, signals, trend, warehouse
from .config import HISTORY, LIVE_SEASON as LIVE, MODELS, RESULTS as OUT, ROSTERSHIP_SNAPSHOTS


def save_model(name, params):
    """The fitted 'model' here is the noise calibration; save it so results are reproducible."""
    (MODELS / f"{name}.json").write_text(json.dumps(params, indent=2, default=float))


CASE_STUDIES = [(2025, "HOU", ["Nick Chubb", "Woody Marks"]),
                (2025, "MIN", ["Jordan Mason", "Aaron Jones"])]


def main(threshold, cusum_h):
    usage = data.load_weekly_usage(HISTORY + [LIVE])

    # 1. Calibrate noise + evaluate thresholds on past seasons
    hist = signals.build_windows(usage[usage.season < LIVE], with_future=True)
    noise = signals.fit_noise(hist)
    save_model("shift_noise", noise)
    scored_hist = signals.score(hist, noise)
    table = backtest.threshold_table(scored_hist)
    slope = backtest.persistence_slope(scored_hist)
    table.to_csv(OUT / "backtest_thresholds.csv", index=False)
    print(f"Backtest {HISTORY[0]}-{HISTORY[-1]} (persistence slope {slope}):")
    print(table.to_string(index=False))
    fill = backtest.fill_in_comparison(backtest.tag_fill_in(scored_hist, usage[usage.season < LIVE]), threshold)
    fill.to_csv(OUT / "backtest_fill_in.csv", index=False)
    print(f"\nRisers >= {threshold} SD, split by whether a same-position regular was absent:")
    print(fill.to_string(index=False))

    # 1b. Slow-bleed detector (CUSUM) on the same history
    game_noise = trend.fit_game_noise(usage[usage.season < LIVE])
    save_model("cusum_game_noise", game_noise)
    cus_hist = trend.cusum_all(usage[usage.season < LIVE], game_noise)
    cbt = trend.cusum_backtest(cus_hist)
    cbt.to_csv(OUT / "backtest_cusum.csv", index=False)
    print("\nSlow bleed / climb (CUSUM) backtest:")
    print(cbt.to_string(index=False))
    case = pd.concat([cus_hist[(cus_hist.season == s) & cus_hist.player.isin(ps)] for s, _, ps in CASE_STUDIES])
    case = case.merge(usage[["season", "week", "pfr_player_id", "opp_share", "passdown_share"]],
                      on=["season", "week", "pfr_player_id"], how="left")
    case.round(3).to_csv(OUT / "case_studies.csv", index=False)
    crossings = pd.concat([trend.first_crossings(cus_hist, ps, s, cusum_h) for s, _, ps in CASE_STUDIES])
    crossings.to_csv(OUT / "case_study_flags.csv", index=False)
    print("\nCase studies, first week flagged:")
    print(crossings.to_string(index=False))

    # 2. Score the current season as of the latest week
    live = signals.score(signals.build_windows(usage[usage.season == LIVE]), noise)
    roster = data.load_espn_rostership(LIVE)
    roster.to_csv(ROSTERSHIP_SNAPSHOTS / f"rostership_snapshot_{roster.snapshot_date.iloc[0]:%Y-%m-%d}.csv", index=False)
    flags = signals.flag(live, roster, threshold, usage[usage.season == LIVE])
    cus_live = trend.cusum_all(usage[usage.season == LIVE], game_noise)
    cus_latest = cus_live.sort_values("week").groupby("pfr_player_id").tail(1)
    slow = trend.flag_cusum(cus_latest, roster, cusum_h)
    if len(slow):
        slow = slow.rename(columns={"week": "as_of_week"}).assign(context="")
        flags = pd.concat([flags, slow], ignore_index=True)
    cus_live.round(3).to_csv(OUT / "cusum_current.csv", index=False)

    # Sleeper: depth chart ("secure on paper"), injury status, 7-day add/drop counts
    sleeper = data.load_sleeper()
    flags = flags.merge(sleeper.drop(columns="sleeper_name"), on="gsis_id", how="left") \
        if "gsis_id" in flags else flags

    cols = ["signal", "player", "position", "team", "as_of_week", "pct_rostered",
            "pct_rostered_change", "shift_score", "snap_pct_base", "snap_pct_recent",
            "opp_share_base", "opp_share_recent", "depth_chart_order", "injury_status",
            "sleeper_adds_7d", "sleeper_drops_7d", "context", "reason"]
    flags = flags.reindex(columns=cols).sort_values(["signal", "shift_score"], ascending=[False, False])
    flags.round(3).to_csv(OUT / "signals_current.csv", index=False)

    # 3. Trend data for dashboard charts: every player-game this season + all scores
    usage[usage.season == LIVE].round(4).to_csv(OUT / "usage_trends_current.csv", index=False)
    live.merge(roster[["espn_id", "pct_rostered"]], on="espn_id", how="left") \
        .merge(cus_latest[["pfr_player_id", "cusum_bleed", "cusum_climb"]], on="pfr_player_id", how="left") \
        .round(3).to_csv(OUT / "shift_scores_current.csv", index=False)

    week = int(live.as_of_week.max())
    print(f"\n{LIVE} through week {week}, threshold {threshold} SD:")
    for sig, g in flags.groupby("signal"):
        print(f"\n== {sig} ({len(g)}) ==")
        for _, r in g.iterrows():
            inj = f" [Sleeper: {r.injury_status}]" if isinstance(r.injury_status, str) else ""
            print(f"  {r.player:<22} {r.position} {r.team:<3} {r.pct_rostered:5.1f}% rost | {r.reason}{inj}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--threshold", type=float, default=2.0, help="shift score threshold (SD)")
    p.add_argument("--cusum", type=float, default=3.0, help="slow bleed/climb tally threshold")
    p.add_argument("--bigquery", action="store_true", help="upload results to BigQuery afterwards")
    a = p.parse_args()
    main(a.threshold, a.cusum)
    print(f"\nwrote {brief.build()}")
    if a.bigquery:
        print("\nUploading to BigQuery:")
        warehouse.main()
