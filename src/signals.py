"""Usage-shift statistics for RosterWatch.

The core idea, in one sentence: compare a player's last 2 games to his earlier games
this season, and express the change in units of "typical week-to-week noise" for his
position. A shift score of +2 means usage rose twice as much as normal fluctuation.

Why this design:
  - Transparent: every flag traces back to two averages and one noise estimate.
  - Position-aware: RB snap shares swing more than WR1 snap shares, so each position
    gets its own noise scale (estimated from 2022-2025 history).
  - Testable: the same calculation runs on past seasons, so we can measure how often
    a flagged shift actually held up over the next 3 games (see backtest.py).
"""
import numpy as np
import pandas as pd

METRICS = ["snap_pct", "opp_share"]   # snap share + share of team targets/carries
RECENT_GAMES = 2                      # "recent" window
MIN_BASELINE_GAMES = 2                # need at least this many earlier games
FUTURE_GAMES = 3                      # backtest: how long a shift must hold
MIN_RECENT_SNAP = 0.35                # materiality floor: ignore deep backups bouncing 5% -> 20%
SUDDEN_DROP = 0.5                     # a game below half the player's usual snap share = likely in-game injury


def build_windows(usage, with_future=False):
    """For every player-game, compute baseline/recent (and optionally future) averages.

    Returns one row per (player, season, as-of game). With with_future=False only the
    latest game per player-season is kept, i.e. "as of right now".
    """
    rows = []
    for (season, pid), g in usage.groupby(["season", "pfr_player_id"], sort=False):
        g = g.sort_values("week")
        vals = g[METRICS].to_numpy()
        weeks = g.week.to_numpy()
        n = len(g)
        drop = _sudden_drops(vals[:, 0])
        start = RECENT_GAMES + MIN_BASELINE_GAMES - 1
        idxs = range(start, n) if with_future else ([n - 1] if n - 1 >= start else [])
        for i in idxs:
            base = vals[: i - RECENT_GAMES + 1]
            recent = vals[i - RECENT_GAMES + 1: i + 1]
            row = {"season": season, "pfr_player_id": pid, "as_of_week": weeks[i],
                   "games_in_baseline": len(base)}
            for j, m in enumerate(METRICS):
                row[f"{m}_base"] = base[:, j].mean()
                row[f"{m}_recent"] = recent[:, j].mean()
                healthy = vals[: i + 1, j][~drop[: i + 1]]
                row[f"{m}_vol"] = healthy.std() if len(healthy) >= 3 else np.nan
            row["last_snap_pct"] = vals[i, 0]
            row["usual_snap_pct"] = np.median(vals[:i, 0])
            row["sudden_drop"] = bool(drop[i])
            if with_future:
                fut = vals[i + 1: i + 1 + FUTURE_GAMES]
                if len(fut) < 2:
                    continue
                for j, m in enumerate(METRICS):
                    row[f"{m}_future"] = fut[:, j].mean()
            rows.append(row)
    if not rows:   # early season: nobody has enough games yet
        return pd.DataFrame(columns=["season", "pfr_player_id", "as_of_week", "player",
                                     "position", "team", "gsis_id", "espn_id"])
    w = pd.DataFrame(rows)
    info = usage.drop_duplicates(["season", "pfr_player_id"], keep="last")[
        ["season", "pfr_player_id", "player", "position", "team", "gsis_id", "espn_id"]]
    return w.merge(info, on=["season", "pfr_player_id"], how="left")


def _sudden_drops(snap):
    """Game k is a sudden drop if snap share < half the median of the player's earlier games
    (and he was a real contributor, median >= 50%). These are usually in-game injuries."""
    out = np.zeros(len(snap), dtype=bool)
    for k in range(1, len(snap)):
        usual = np.median(snap[:k])
        out[k] = usual >= 0.5 and snap[k] < SUDDEN_DROP * usual
    return out


def fit_noise(hist_windows):
    """Estimate typical (noise) size of a shift, per position and metric, from history."""
    out = {}
    for pos, g in hist_windows.groupby("position"):
        out[pos] = {m: (g[f"{m}_recent"] - g[f"{m}_base"]).std() for m in METRICS}
        z = sum((g[f"{m}_recent"] - g[f"{m}_base"]) / out[pos][m] for m in METRICS) / len(METRICS)
        out[pos]["composite"] = z.std()   # rescale so the final score has SD 1
    return out


def score(windows, noise):
    """Add z-scores per metric and a combined shift_score (SD units of normal noise)."""
    w = windows.copy()
    for m in METRICS:
        sd = w.position.map(lambda p: noise[p][m])
        w[f"{m}_z"] = (w[f"{m}_recent"] - w[f"{m}_base"]) / sd
        if f"{m}_future" in w:
            w[f"{m}_future_z"] = (w[f"{m}_future"] - w[f"{m}_base"]) / sd
    comp_sd = w.position.map(lambda p: noise[p]["composite"])
    w["shift_score"] = sum(w[f"{m}_z"] for m in METRICS) / len(METRICS) / comp_sd
    if "snap_pct_future" in w:
        w["future_shift_score"] = sum(w[f"{m}_future_z"] for m in METRICS) / len(METRICS) / comp_sd
    return w


def flag(scored, rostership, threshold, usage, rostered_cut=50.0):
    """Turn scores into signals, each with a plain-English reason.

    Waiver Riser    : < rostered_cut % rostered, shift >= threshold, now playing real snaps
    Fragile Starter : >= rostered_cut % rostered and usage falling / teammate surging / volatile
    Injury Check    : rostered player whose latest game collapsed (likely got hurt mid-game);
                      not a role change, so it is reported separately for a human to verify
    """
    w = scored.merge(rostership[["espn_id", "pct_rostered", "pct_rostered_change"]],
                     on="espn_id", how="left")
    w["pct_rostered"] = w.pct_rostered.fillna(0)
    pct = lambda x: f"{x:.0%}"

    is_riser = (w.shift_score >= threshold) & (w.snap_pct_recent >= MIN_RECENT_SNAP)
    surging = w[is_riser].groupby(["team", "position"]).player.apply(list).to_dict()
    # Teammates who may have vacated work: usage falling or a sudden-drop game
    fading = w[(w.shift_score <= -1) | w.sudden_drop].groupby(["team", "position"]).player.apply(list).to_dict()
    # ...or teammates who were regulars (>= 50% snaps) but did not play the team's latest game
    per = usage.groupby(["team", "position", "player"], as_index=False).agg(
        last_week=("week", "max"), avg_snap=("snap_pct", "mean"))
    per["team_last_week"] = per.team.map(usage.groupby("team").week.max())
    absent = per[(per.last_week < per.team_last_week) & (per.avg_snap >= 0.5)]
    for (team, pos), g in absent.groupby(["team", "position"]):
        fading.setdefault((team, pos), []).extend(f"{n} (missed wk {wk})" for n, wk in
                                                  zip(g.player, g.team_last_week))

    # Volatility (healthy games only) relative to position peers. Experimental: not backtested yet.
    w["volatility_pctile"] = w.groupby("position").snap_pct_vol.rank(pct=True)

    out = []
    for _, r in w.iterrows():
        key = (r.team, r.position)
        change = (f"snap share {pct(r.snap_pct_base)} -> {pct(r.snap_pct_recent)}, "
                  f"opportunity share {pct(r.opp_share_base)} -> {pct(r.opp_share_recent)}")
        if r.pct_rostered < rostered_cut and is_riser[_]:
            vacated = [m for m in fading.get(key, []) if m != r.player]
            ctx = (f" Context: same-position teammate(s) {', '.join(vacated)} lost usage or missed time, so "
                   "this may be fill-in work. Check their status." if vacated else
                   " Context: no same-position regular lost usage or missed time, so the role may be earned.")
            out.append({**r, "signal": "Waiver Riser", "context": "fill-in?" if vacated else "earned?",
                        "reason": f"Last {RECENT_GAMES} games vs earlier: {change} "
                                  f"(shift {r.shift_score:+.1f} SD).{ctx}"})
        elif r.pct_rostered >= rostered_cut and r.sudden_drop:
            out.append({**r, "signal": "Injury Check", "context": "",
                        "reason": f"Week {r.as_of_week} snap share {pct(r.last_snap_pct)} vs usual "
                                  f"{pct(r.usual_snap_pct)}. Likely left the game early; verify injury news."})
        elif r.pct_rostered >= rostered_cut:
            reasons = []
            if r.shift_score <= -threshold:
                reasons.append(f"Usage falling: {change} (shift {r.shift_score:+.1f} SD)")
            mates = [m for m in surging.get(key, []) if m != r.player]
            if mates and r.shift_score < 0:
                reasons.append(f"Same-position teammate surging: {', '.join(mates)}")
            if r.volatility_pctile >= 0.9 and r.snap_pct_recent < 0.8:
                reasons.append(f"Snap share among the most volatile {r.position}s "
                               f"(SD {r.snap_pct_vol:.0%} game to game) [experimental]")
            if reasons:
                out.append({**r, "signal": "Fragile Starter", "context": "", "reason": "; ".join(reasons)})
    return pd.DataFrame(out)
