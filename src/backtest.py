"""Choose the flag threshold with evidence instead of gut feel.

For every player-game in 2022-2025 we compute the shift score exactly as we would live,
then look at the NEXT 3 games and ask two questions:
  stayed_rate : did usage stay on the same side of the old baseline? (rise stayed up / fall stayed down)
  kept_half   : did the player keep at least half of the shift?
Sweeping the threshold shows the trade-off:
  low threshold  -> many flags per week, more of them noise
  high threshold -> few flags, more of them real
"""
import numpy as np
import pandas as pd

from . import signals


def threshold_table(scored, thresholds=(0.0, 1.0, 1.5, 2.0, 2.5, 3.0)):
    n_weeks = scored.groupby(["season", "as_of_week"]).ngroups
    rows = []
    for direction, sign in (("up", 1), ("down", -1)):
        for k in thresholds:
            s = scored.shift_score * sign
            fut = scored.future_shift_score * sign
            floor = scored.snap_pct_recent if sign > 0 else scored.snap_pct_base
            # k = 0 row is the reference group: players whose usage barely moved (|score| < 0.5)
            mask = (s >= k) if k else (scored.shift_score.abs() < 0.5)
            mask &= floor >= signals.MIN_RECENT_SNAP
            f_s, f_fut = s[mask], fut[mask]
            rows.append({"direction": direction,
                         "threshold_sd": k if k else "no shift (|score|<0.5)",
                         "flags": int(mask.sum()),
                         "flags_per_week": round(mask.sum() / n_weeks, 1),
                         "stayed_rate": round((f_fut > 0).mean(), 3),
                         "kept_half_rate": round((f_fut >= 0.5 * f_s).mean(), 3)})
    return pd.DataFrame(rows)


def persistence_slope(scored):
    """Regression of future shift on current shift: the average fraction of a shift that lasts."""
    s = scored[scored.snap_pct_recent >= signals.MIN_RECENT_SNAP]
    return round(float(np.polyfit(s.shift_score, s.future_shift_score, 1)[0]), 3)


def tag_fill_in(scored, usage):
    """Mark each window where a same-team, same-position regular (>= 50% snaps in his
    earlier games) missed the as-of game or the game before. Those rises are likely
    injury fill-in work, which we expect to fade when the starter returns."""
    absent = set()
    for (season, team), g in usage.groupby(["season", "team"]):
        team_weeks = sorted(g.week.unique())
        for (pos, pid), pg in g.groupby(["position", "pfr_player_id"]):
            played = dict(zip(pg.week, pg.snap_pct))
            for t in team_weeks:
                prior = [v for w, v in played.items() if w < t]
                if t not in played and len(prior) >= 2 and np.mean(prior) >= 0.5:
                    absent.add((season, team, pos, t))
    def is_fill_in(r):
        return any((r.season, r.team, r.position, w) in absent for w in (r.as_of_week, r.as_of_week - 1))
    out = scored.copy()
    out["fill_in"] = out.apply(is_fill_in, axis=1)
    return out


def fill_in_comparison(scored, threshold=2.0):
    s = scored[(scored.shift_score >= threshold) & (scored.snap_pct_recent >= signals.MIN_RECENT_SNAP)]
    return (s.groupby("fill_in")
             .apply(lambda g: pd.Series({"flags": len(g),
                                         "stayed_rate": round((g.future_shift_score > 0).mean(), 3),
                                         "kept_half_rate": round((g.future_shift_score >= 0.5 * g.shift_score).mean(), 3)}))
             .reset_index())
