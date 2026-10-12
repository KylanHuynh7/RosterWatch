"""Slow-bleed detection: a steady multi-week slide (or climb) in usage.

The shift score in signals.py catches sudden jumps (last 2 games vs earlier). A slow bleed,
like an RB2 giving up a few more passing-down snaps each week, may never jump enough in
any single window. So here we fit a straight line through the last 5 games and use its
slope: "percentage points of usage gained or lost per game."

As with the shift score, each slope is divided by the typical slope for that position
(from 2022-2025), so a trend score of -2 means the decline is twice as steep as normal
fluctuation. In-game-injury games (sudden drops) are left out so one bad game can't
fake a trend.
"""
import numpy as np
import pandas as pd

from . import signals

TREND_METRICS = ["snap_pct", "opp_share", "passdown_share"]
TREND_GAMES = 5          # fit the line through the last 5 healthy games...
MIN_TREND_GAMES = 4      # ...or 4, early in the season
FUTURE_GAMES = 3


def _slope(y):
    x = np.arange(len(y))
    return np.polyfit(x, y, 1)[0]


def build_trends(usage, with_future=False):
    rows = []
    for (season, pid), g in usage.groupby(["season", "pfr_player_id"], sort=False):
        g = g.sort_values("week")
        vals = g[TREND_METRICS].fillna(0).to_numpy()
        healthy = ~signals._sudden_drops(vals[:, 0])
        weeks = g.week.to_numpy()
        n = len(g)
        idxs = range(n) if with_future else [n - 1]
        for i in idxs:
            if not healthy[i]:
                continue
            past = np.where(healthy[: i + 1])[0][-TREND_GAMES:]
            if len(past) < MIN_TREND_GAMES:
                continue
            win = vals[past]
            row = {"season": season, "pfr_player_id": pid, "as_of_week": weeks[i],
                   "trend_games": len(past), "trend_weeks": weeks[past].tolist()}
            for j, m in enumerate(TREND_METRICS):
                row[f"{m}_slope"] = _slope(win[:, j])
                row[f"{m}_start"] = win[:2, j].mean()
                row[f"{m}_now"] = win[-2:, j].mean()
            if with_future:
                fut = vals[i + 1: i + 1 + FUTURE_GAMES]
                if len(fut) < 2:
                    continue
                for j, m in enumerate(TREND_METRICS):
                    row[f"{m}_future"] = fut[:, j].mean()
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["season", "pfr_player_id", "as_of_week", "player",
                                     "position", "team", "gsis_id", "espn_id"])
    t = pd.DataFrame(rows)
    info = usage.drop_duplicates(["season", "pfr_player_id"], keep="last")[
        ["season", "pfr_player_id", "player", "position", "team", "gsis_id", "espn_id"]]
    return t.merge(info, on=["season", "pfr_player_id"], how="left")


def fit_noise(hist):
    out = {}
    for pos, g in hist.groupby("position"):
        out[pos] = {m: g[f"{m}_slope"].std() for m in TREND_METRICS}
        z = sum(g[f"{m}_slope"] / out[pos][m] for m in TREND_METRICS) / len(TREND_METRICS)
        out[pos]["composite"] = z.std()
    return out


def score(trends, noise):
    t = trends.copy()
    comp = 0
    for m in TREND_METRICS:
        t[f"{m}_slope_z"] = t[f"{m}_slope"] / t.position.map(lambda p: noise[p][m])
        comp = comp + t[f"{m}_slope_z"]
    t["trend_score"] = comp / len(TREND_METRICS) / t.position.map(lambda p: noise[p]["composite"])
    return t


def backtest(scored, thresholds=(1.0, 1.5, 2.0, 2.5)):
    """Did a flagged slide keep going? Outcome: over the next 3 games, was snap share
    still below where the slide started (start = first 2 games of the trend window)?"""
    s = scored[scored.snap_pct_start >= 0.4]   # only players who had a real role to lose
    n_weeks = s.groupby(["season", "as_of_week"]).ngroups
    lower = s.snap_pct_future < s.snap_pct_start
    kept_falling = s.snap_pct_future < s.snap_pct_now
    rows = [{"trend_threshold": "flat (|score| < 0.5)", "flags_per_week": None,
             "still_below_start": round(lower[s.trend_score.abs() < 0.5].mean(), 3),
             "fell_further": round(kept_falling[s.trend_score.abs() < 0.5].mean(), 3),
             "flags": int((s.trend_score.abs() < 0.5).sum())}]
    for k in thresholds:
        m = s.trend_score <= -k
        rows.append({"trend_threshold": f"<= -{k}", "flags_per_week": round(m.sum() / n_weeks, 1),
                     "still_below_start": round(lower[m].mean(), 3),
                     "fell_further": round(kept_falling[m].mean(), 3), "flags": int(m.sum())})
    return pd.DataFrame(rows)


def first_flag_weeks(scored, players, season, k=1.5):
    """Case-study helper: the week each player's trend score first crossed +/-k."""
    out = []
    for p in players:
        g = scored[(scored.player == p) & (scored.season == season)].sort_values("as_of_week")
        down = g[g.trend_score <= -k].as_of_week.min()
        up = g[g.trend_score >= k].as_of_week.min()
        out.append({"player": p, "first_bleed_flag": down, "first_climb_flag": up})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# CUSUM: built for small, persistent shifts (the slow bleed)
# ---------------------------------------------------------------------------
CUSUM_METRICS = ["snap_pct", "opp_share", "passdown_share"]
ESTABLISH_GAMES = 3   # a player's "established role" = average of his first 3 games
CUSUM_K = 0.5         # allowance: games within 0.5 SD of normal don't count
CUSUM_CAP = 3.0       # one game can move the tally by at most 3 SD (limits in-game injuries)


def fit_game_noise(usage):
    """SD of a single game's deviation from the player's established level, per position."""
    devs = []
    for _, g in usage.groupby(["season", "pfr_player_id"]):
        g = g.sort_values("week")
        if len(g) <= ESTABLISH_GAMES:
            continue
        est = g[CUSUM_METRICS].iloc[:ESTABLISH_GAMES].mean()
        d = g[CUSUM_METRICS].iloc[ESTABLISH_GAMES:] - est
        d["position"] = g.position.iloc[0]
        devs.append(d)
    d = pd.concat(devs)
    return d.groupby("position")[CUSUM_METRICS].std().to_dict("index")


def cusum_path(g, noise):
    """Week-by-week lower (bleed) and upper (climb) CUSUM for one player-season."""
    g = g.sort_values("week")
    pos = g.position.iloc[0]
    x = g[CUSUM_METRICS].fillna(0)
    est = x.iloc[:ESTABLISH_GAMES].mean()
    lo = hi = 0.0
    out = []
    for i, (wk, row) in enumerate(zip(g.week, x.itertuples(index=False))):
        if i < ESTABLISH_GAMES:
            out.append((wk, 0.0, 0.0))
            continue
        z = np.mean([(v - est[m]) / noise[pos][m] for m, v in zip(CUSUM_METRICS, row)])
        z = float(np.clip(z, -CUSUM_CAP, CUSUM_CAP))
        lo = min(0.0, lo + z + CUSUM_K)
        hi = max(0.0, hi + z - CUSUM_K)
        out.append((wk, lo, hi))
    return pd.DataFrame(out, columns=["week", "cusum_bleed", "cusum_climb"])


def cusum_all(usage, noise):
    parts = []
    for (season, pid), g in usage.groupby(["season", "pfr_player_id"]):
        if len(g) <= ESTABLISH_GAMES:
            continue
        c = cusum_path(g, noise)
        c["season"], c["pfr_player_id"] = season, pid
        c = c.merge(g[["week", "snap_pct", "player", "position", "team", "gsis_id", "espn_id"]], on="week")
        parts.append(c)
    if not parts:
        return pd.DataFrame(columns=["week", "cusum_bleed", "cusum_climb", "season", "pfr_player_id",
                                     "snap_pct", "player", "position", "team", "gsis_id", "espn_id"])
    return pd.concat(parts, ignore_index=True)


def cusum_backtest(c, thresholds=(2, 3, 4, 5)):
    """Evaluate the first week a tally crosses h (bleed: <= -h, climb: >= +h).

    Outcome over the next 3 games, vs the player's established level (first 3 games):
      bleed -> snap share more than 5 points below it
      climb -> snap share more than 5 points above it
    Reference rows: player-weeks with a tally of exactly 0 in that direction.
    """
    c = c.sort_values(["season", "pfr_player_id", "week"]).copy()
    grp = c.groupby(["season", "pfr_player_id"])
    c["est_snap"] = grp.snap_pct.transform(lambda s: s.iloc[:ESTABLISH_GAMES].mean())
    c["future_snap"] = grp.snap_pct.transform(
        lambda s: s[::-1].rolling(3, min_periods=2).mean()[::-1].shift(-1))
    c = c[c.future_snap.notna()]
    n_weeks = c.groupby(["season", "week"]).ngroups
    rows = []
    for direction in ("bleed", "climb"):
        if direction == "bleed":
            d = c[c.est_snap >= 0.4]                       # had a real role to lose
            tally, hit = -d.cusum_bleed, d.future_snap < d.est_snap - 0.05
        else:
            d = c[c.est_snap < 0.6]                        # had room to grow
            tally, hit = d.cusum_climb, d.future_snap > d.est_snap + 0.05
        rows.append({"direction": direction, "threshold": "reference (tally = 0)",
                     "new_flags_per_week": None, "hit_rate": round(hit[tally == 0].mean(), 3),
                     "flags": int((tally == 0).sum())})
        for h in thresholds:
            crossed = tally >= h
            first = crossed & ~crossed.groupby([d.season, d.pfr_player_id]).shift(1, fill_value=False)
            rows.append({"direction": direction, "threshold": f"tally {'<= -' if direction == 'bleed' else '>= +'}{h}",
                         "new_flags_per_week": round(first.sum() / n_weeks, 1),
                         "hit_rate": round(hit[first].mean(), 3), "flags": int(first.sum())})
    return pd.DataFrame(rows)


def first_crossings(c, players, season, h=3):
    """Case-study helper: first week each player's bleed / climb tally crossed h."""
    out = []
    for p in players:
        g = c[(c.player == p) & (c.season == season)]
        out.append({"player": p, "season": season,
                    "first_bleed_week": g[g.cusum_bleed <= -h].week.min(),
                    "first_climb_week": g[g.cusum_climb >= h].week.min()})
    return pd.DataFrame(out)


def flag_cusum(latest, rostership, h=3.0, rostered_cut=50.0):
    """Slow Bleed: rostered player whose bleed tally <= -h. Slow Climb: low-rostered, climb >= +h."""
    w = latest.merge(rostership[["espn_id", "pct_rostered"]], on="espn_id", how="left")
    w["pct_rostered"] = w.pct_rostered.fillna(0)
    out = []
    for _, r in w.iterrows():
        if r.pct_rostered >= rostered_cut and r.cusum_bleed <= -h:
            out.append({**r, "signal": "Slow Bleed",
                        "reason": f"Usage has run below his established level game after game "
                                  f"(bleed tally {r.cusum_bleed:.1f}; flag at -{h:g})."})
        elif r.pct_rostered < rostered_cut and r.cusum_climb >= h:
            out.append({**r, "signal": "Slow Climb",
                        "reason": f"Usage has run above his established level game after game "
                                  f"(climb tally {r.cusum_climb:+.1f}; flag at +{h:g})."})
    return pd.DataFrame(out)
