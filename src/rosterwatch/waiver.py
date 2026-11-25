"""Waiver Wire Alerts: the waiver opportunity score (W) and the four waiver flags."""

import numpy as np
import pandas as pd

from .consistency import calculate_volatility
from .settings import MODEL2_SETTINGS, MODEL2_WEIGHTS, WAIVER_FLAGS


# ============================================================
# SHARED HELPERS
# ============================================================
def safe_mean(series):
    # Fast path for plain float/int columns (called thousands of times per page).
    # sum / count matches pandas' mean for NaN-free float64 data exactly.
    if isinstance(series.dtype, np.dtype) and series.dtype.kind in "fi":
        values = series.to_numpy(dtype="float64")
        values = values[~np.isnan(values)]
        if values.size == 0:
            return np.nan
        return float(values.sum() / values.size)
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()
    if values.empty:
        return np.nan
    return float(values.mean())


def clip01(value):
    if pd.isna(value):
        return 0.0
    return float(np.clip(value, 0, 1))


def normalized_change(
    recent,
    previous,
    column,
    scale,
):
    current = safe_mean(recent[column])
    baseline = safe_mean(previous[column])
    if pd.isna(current) or pd.isna(baseline):
        return 0.0
    return float(
        np.clip(
            (current - baseline) / scale,
            -1,
            1,
        )
    )


# ============================================================
# MODEL 2 — FANTASY PERFORMANCE
# ============================================================
def calculate_fantasy_score(
    player,
    recent,
    previous,
    settings,
):
    baseline = settings["ppr"]
    season_ppr = safe_mean(player["fantasy_points_ppr"])
    recent_ppr = safe_mean(recent["fantasy_points_ppr"])
    recent_median = float(recent["fantasy_points_ppr"].median())
    ppr_trend = normalized_change(
        recent,
        previous,
        "fantasy_points_ppr",
        baseline * 0.50,
    )
    scoring_level = 0.65 * clip01(recent_ppr / baseline) + 0.35 * clip01(
        recent_median / baseline
    )
    trend_component = 0.5 + 0.5 * ppr_trend
    fantasy_score = clip01(0.75 * scoring_level + 0.25 * trend_component)
    return {
        "score": fantasy_score,
        "season_ppr": season_ppr,
        "recent_ppr": recent_ppr,
        "recent_median": recent_median,
        "trend": ppr_trend,
    }


# ============================================================
# MODEL 2 — SNAP AND OPPORTUNITY
# ============================================================
def calculate_usage_scores(
    recent,
    previous,
    settings,
):
    snap_trend = normalized_change(
        recent,
        previous,
        "snap_pct",
        0.10,
    )
    opp_trend = normalized_change(
        recent,
        previous,
        "opp_share",
        0.06,
    )
    recent_snap = safe_mean(recent["snap_pct"])
    recent_opp = safe_mean(recent["opp_share"])
    snap_level = clip01(recent_snap / settings["snap"])
    opp_level = clip01(recent_opp / settings["opp"])
    snap_score = clip01(0.80 * snap_level + 0.20 * (0.5 + 0.5 * snap_trend))
    opportunity_score = clip01(0.80 * opp_level + 0.20 * (0.5 + 0.5 * opp_trend))
    return {
        "snap_score": snap_score,
        "opportunity_score": opportunity_score,
        "snap_trend": snap_trend,
        "opp_trend": opp_trend,
        "recent_snap": recent_snap,
        "recent_opp": recent_opp,
    }


# ============================================================
# MODEL 2 — VOLATILITY AND STABILITY
# ============================================================
def calculate_stability(
    player,
    recent,
    season_ppr,
):
    ppr_std = float(player["fantasy_points_ppr"].std(ddof=0))
    ppr_cv = ppr_std / max(abs(season_ppr), 2.0)
    fantasy_stability = 1 / (1 + ppr_cv)
    season_volatility = safe_mean(player["weekly_volatility"])
    recent_volatility = safe_mean(recent["weekly_volatility"])
    if pd.notna(recent_volatility) and pd.notna(season_volatility):
        combined_volatility = 0.65 * recent_volatility + 0.35 * season_volatility
    elif pd.notna(recent_volatility):
        combined_volatility = recent_volatility
    else:
        combined_volatility = season_volatility
    if pd.notna(combined_volatility):
        usage_stability = 1 / (1 + combined_volatility / 0.12)
    else:
        usage_stability = 0.50
    stability_score = clip01(0.60 * fantasy_stability + 0.40 * usage_stability)
    return {
        "score": stability_score,
        "fantasy_stability": fantasy_stability,
        "usage_stability": usage_stability,
        "volatility": combined_volatility,
    }


# ============================================================
# MODEL 2 — TREND CONFIRMATION
# ============================================================
def calculate_trend_confirmation(
    recent,
    previous,
    ppr_trend,
    opp_trend,
    snap_trend,
):
    weighted_trend = 0.50 * ppr_trend + 0.30 * opp_trend + 0.20 * snap_trend
    alignment_score = clip01(0.5 + 0.5 * weighted_trend)
    baseline_ppr = safe_mean(previous["fantasy_points_ppr"])
    baseline_opp = safe_mean(previous["opp_share"])
    baseline_snap = safe_mean(previous["snap_pct"])
    ppr_up = recent["fantasy_points_ppr"] > baseline_ppr
    opp_up = recent["opp_share"] > baseline_opp
    snap_up = recent["snap_pct"] > baseline_snap
    confirmed_up = ppr_up & (opp_up | snap_up)
    ppr_down = recent["fantasy_points_ppr"] < baseline_ppr
    opp_down = recent["opp_share"] < baseline_opp
    snap_down = recent["snap_pct"] < baseline_snap
    confirmed_down = ppr_down & (opp_down | snap_down)
    upward_confirmation = float(confirmed_up.mean())
    downward_confirmation = float(confirmed_down.mean())
    if weighted_trend >= 0:
        persistence = upward_confirmation
    else:
        persistence = downward_confirmation
    confirmation_score = clip01(0.65 * alignment_score + 0.35 * upward_confirmation)
    return {
        "score": confirmation_score,
        "weighted_trend": weighted_trend,
        "upward_confirmation": upward_confirmation,
        "downward_confirmation": downward_confirmation,
        "persistence": persistence,
    }


# ============================================================
# MODEL 2 — WAIVER CLASSIFICATION
# ============================================================
def conservative_waiver_flag(
    score,
    ppr_trend=0.0,
    opp_trend=0.0,
    snap_trend=0.0,
    recent_ppr=np.nan,
    baseline=np.nan,
):
    """Choose an existing directional flag when evidence is mixed or sparse.

    Neutral/mixed cases use Slow Riser or Slow Drop, not a strong action.
    This is a heuristic classification, not a verified trend.
    """
    signals = [(ppr_trend, 0.50), (opp_trend, 0.30), (snap_trend, 0.20)]
    available = [(float(x), weight) for x, weight in signals if pd.notna(x)]
    direction = (
        (
            sum(x * weight for x, weight in available)
            / sum(weight for _, weight in available)
        )
        if available
        else 0.0
    )
    if direction > 0.05:
        return "Slow Riser"
    if direction < -0.05:
        return "Slow Drop"
    if pd.notna(recent_ppr) and pd.notna(baseline) and baseline > 0:
        if recent_ppr >= baseline:
            return "Slow Riser"
        return "Slow Drop"
    return "Slow Riser" if pd.notna(score) and score >= 50 else "Slow Drop"


def classify_waiver_scenario(
    score,
    recent_ppr,
    recent_median,
    latest_ppr,
    ppr_trend,
    opp_trend,
    snap_trend,
    upward_confirmation,
    downward_confirmation,
    recent_volatility,
    games_analyzed,
    settings,
):
    baseline = settings["ppr"]
    usage_rising = opp_trend >= 0.10 or snap_trend >= 0.10
    usage_falling = opp_trend <= -0.10 or snap_trend <= -0.10
    fantasy_rising = ppr_trend >= 0.20
    fantasy_falling = ppr_trend <= -0.20
    scoring_spike = (
        latest_ppr
        > max(
            1.8 * recent_median,
            1.5 * baseline,
        )
        and ppr_trend > 0
    )
    high_volatility = pd.notna(recent_volatility) and recent_volatility >= 0.15
    if scoring_spike and upward_confirmation < 0.67:
        return (
            conservative_waiver_flag(
                score, ppr_trend, opp_trend, snap_trend, recent_ppr, baseline
            ),
            "Possible one-week fantasy spike",
        )
    if (
        fantasy_rising
        and usage_rising
        and upward_confirmation >= 0.67
        and score >= 72
        and recent_ppr >= baseline * 0.85
        and games_analyzed >= 5
    ):
        return (
            "Must Pick Up",
            "Confirmed fantasy breakout",
        )
    if fantasy_rising and usage_rising and upward_confirmation >= 0.50 and score >= 52:
        return (
            "Slow Riser",
            "Rising fantasy points and offensive usage",
        )
    if usage_rising and ppr_trend > -0.30 and recent_ppr < baseline:
        return (
            "Slow Riser",
            "Increasing opportunities; fantasy upside",
        )
    if (
        fantasy_falling
        and usage_falling
        and downward_confirmation >= 0.67
        and score <= 43
        and recent_ppr < baseline * 0.60
    ):
        return (
            "Must Drop",
            "Sustained fantasy and usage decline",
        )
    if fantasy_falling and usage_falling and downward_confirmation >= 0.50:
        return (
            "Slow Drop",
            "Declining fantasy production and usage",
        )
    if fantasy_rising and opp_trend <= -0.10:
        return (
            conservative_waiver_flag(
                score, ppr_trend, opp_trend, snap_trend, recent_ppr, baseline
            ),
            "Fantasy scoring rising despite fewer opportunities",
        )
    if fantasy_falling and opp_trend >= 0.10:
        return (
            conservative_waiver_flag(
                score, ppr_trend, opp_trend, snap_trend, recent_ppr, baseline
            ),
            "Opportunities increasing but scoring declining",
        )
    if high_volatility:
        return (
            conservative_waiver_flag(
                score, ppr_trend, opp_trend, snap_trend, recent_ppr, baseline
            ),
            "High weekly usage volatility",
        )
    if (
        recent_ppr >= baseline
        and score >= 65
        and abs(ppr_trend) < 0.20
        and not usage_falling
    ):
        return (
            conservative_waiver_flag(
                score, ppr_trend, opp_trend, snap_trend, recent_ppr, baseline
            ),
            "Reliable fantasy producer",
        )
    return (
        conservative_waiver_flag(
            score, ppr_trend, opp_trend, snap_trend, recent_ppr, baseline
        ),
        "Stable or mixed fantasy indicators",
    )


# ============================================================
# MODEL 2 — ADAPTIVE SCORING FOR LIMITED / PARTIAL HISTORY
# ============================================================
def calculate_adaptive_waiver(player, settings, as_of_week):
    """Score limited history using observed metrics only.

    Missing metrics are excluded and remaining weights renormalized.
    One game has no trend evidence, so it receives a neutral flag.
    Short histories never qualify for high-confidence Must Pick Up/Drop.
    """
    player = player.sort_values("week")
    metrics = {
        "fantasy_points_ppr": (
            "fantasy_score",
            MODEL2_WEIGHTS["fantasy"],
            settings["ppr"],
        ),
        "opp_share": (
            "opportunity_score",
            MODEL2_WEIGHTS["opportunity"],
            settings["opp"],
        ),
        "snap_pct": ("snap_score", MODEL2_WEIGHTS["snap"], settings["snap"]),
    }
    n = len(player)
    recent_count = min(2, max(1, n // 2)) if n > 1 else 1
    recent = player.tail(recent_count)
    previous = player.iloc[:-recent_count]
    result = {
        "fantasy_score": np.nan,
        "opportunity_score": np.nan,
        "snap_score": np.nan,
        "stability_score": np.nan,
        "confirmation_score": np.nan,
        "season_ppr": safe_mean(player["fantasy_points_ppr"]),
        "recent_ppr": safe_mean(recent["fantasy_points_ppr"]),
        "ppr_trend": 0.0,
        "snap_trend": 0.0,
        "opportunity_trend": 0.0,
        "volatility": safe_mean(player["weekly_volatility"]),
        "upward_confirmation": 0.0,
        "downward_confirmation": 0.0,
        "games_analyzed": n,
    }
    weighted = []
    trend_values = {}
    for column, (output, weight, benchmark) in metrics.items():
        current = safe_mean(recent[column])
        if not np.isfinite(current):
            continue
        level = clip01(current / benchmark)
        baseline = safe_mean(previous[column]) if not previous.empty else np.nan
        scale = {
            "fantasy_points_ppr": settings["ppr"] * 0.5,
            "opp_share": 0.06,
            "snap_pct": 0.10,
        }[column]
        trend = (
            float(np.clip((current - baseline) / scale, -1, 1))
            if np.isfinite(baseline)
            else 0.0
        )
        trend_values[column] = trend
        component = clip01(0.80 * level + 0.20 * (0.5 + 0.5 * trend))
        result[output] = round(component, 3)
        weighted.append((weight, component))
    result["ppr_trend"] = trend_values.get("fantasy_points_ppr", 0.0)
    result["snap_trend"] = trend_values.get("snap_pct", 0.0)
    result["opportunity_trend"] = trend_values.get("opp_share", 0.0)

    # Stability is neutral when only one observation exists.
    stability_values = []
    for column, floor in [
        ("fantasy_points_ppr", 2.0),
        ("opp_share", 0.03),
        ("snap_pct", 0.05),
    ]:
        values = pd.to_numeric(player[column], errors="coerce").dropna()
        if len(values) >= 2:
            cv = float(values.std(ddof=0)) / max(abs(float(values.mean())), floor)
            stability_values.append(1.0 / (1.0 + cv))
    if stability_values:
        stability = float(np.mean(stability_values))
        result["stability_score"] = round(stability, 3)
        weighted.append((MODEL2_WEIGHTS["stability"], stability))
    else:
        result["stability_score"] = 0.5

    # Confirmation is counted only where both a recent value and an
    # earlier baseline exist; otherwise it remains neutral and unweighted.
    available = [
        c
        for c in ("fantasy_points_ppr", "opp_share", "snap_pct")
        if pd.notna(safe_mean(previous[c])) and pd.notna(safe_mean(recent[c]))
    ]
    if available:
        directions = [trend_values.get(c, 0.0) for c in available]
        direction = float(np.mean(directions))
        confirmed = []
        for _, game in recent.iterrows():
            checks = []
            for column in available:
                value = game[column]
                if pd.isna(value):
                    continue
                baseline = safe_mean(previous[column])
                checks.append(value > baseline if direction >= 0 else value < baseline)
            if checks:
                confirmed.append(all(checks))
        confirmation = float(np.mean(confirmed)) if confirmed else 0.5
        result["confirmation_score"] = round(confirmation, 3)
        if confirmed:
            weighted.append((MODEL2_WEIGHTS["confirmation"], confirmation))
            if direction >= 0:
                result["upward_confirmation"] = confirmation
            else:
                result["downward_confirmation"] = confirmation
    else:
        result["confirmation_score"] = 0.5
    total_weight = sum(weight for weight, _ in weighted)
    raw_score = (
        100 * sum(w * value for w, value in weighted) / total_weight
        if total_weight
        else 0.0
    )

    # Sparse histories and missing fields receive a transparent discount.
    observed = sum(int(player[c].notna().sum()) for c in metrics)
    completeness = observed / max(3 * n, 1)
    evidence = (0.75 + 0.25 * min(n / 5, 1)) * (0.80 + 0.20 * completeness)
    score = float(np.clip(raw_score * evidence, 0, 100))

    # Never claim a strong breakout/decline without enough observations.
    usage_directions = [
        trend_values[c]
        for c in ("opp_share", "snap_pct")
        if c in trend_values and not previous.empty
    ]
    usage_direction = float(np.mean(usage_directions)) if usage_directions else 0.0
    if n >= 2 and usage_direction >= 0.10:
        flag = "Slow Riser"
    elif n >= 2 and usage_direction <= -0.10:
        flag = "Slow Drop"
    else:
        flag = conservative_waiver_flag(
            score,
            result["ppr_trend"],
            result["opportunity_trend"],
            result["snap_trend"],
            safe_mean(recent["fantasy_points_ppr"]),
            settings["ppr"],
        )
    available_names = ", ".join(c for c in metrics if player[c].notna().any())
    scenario = (
        f"Limited-history estimate ({n} recorded game(s)); "
        f"available metrics: {available_names or 'none'}. "
        "Confidence discounted; no strong waiver flag assigned."
    )
    result.update(waiver_flag=flag, waiver_score=round(score, 1), scenario=scenario)
    return result


# ============================================================
# MODEL 2 — COMPLETE WAIVER MODEL
# ============================================================
def calculate_waiver_model(
    weekly_data,
    selected_season,
    as_of_week,
    consistent_player_ids,
):
    data = weekly_data.loc[
        (weekly_data["season"] == selected_season) & (weekly_data["week"] <= as_of_week)
    ].copy()
    data, _ = calculate_volatility(data)

    # Model 2 classification is independent of Model 1 consistency.
    columns = [
        "gsis_id",
        "player_name",
        "position",
        "team",
        "waiver_flag",
        "waiver_score",
        "scenario",
        "fantasy_score",
        "opportunity_score",
        "snap_score",
        "stability_score",
        "confirmation_score",
        "season_ppr",
        "recent_ppr",
        "ppr_trend",
        "snap_trend",
        "opportunity_trend",
        "volatility",
        "upward_confirmation",
        "downward_confirmation",
        "games_analyzed",
    ]
    results = []

    def add_result(result):
        # Every scored player receives exactly one of the four waiver flags.
        # Model 1 consistency does not override Model 2 classification.
        if result.get("waiver_flag") not in WAIVER_FLAGS:
            result["waiver_flag"] = conservative_waiver_flag(
                result.get("waiver_score"),
                result.get("ppr_trend", 0.0),
                result.get("opportunity_trend", 0.0),
                result.get("snap_trend", 0.0),
                result.get("recent_ppr", np.nan),
                MODEL2_SETTINGS[result["position"]]["ppr"],
            )
        results.append(result)

    for player_id, player in data.groupby("gsis_id"):
        player = player.sort_values("week").copy()
        latest = player.iloc[-1]

        # Include every ownership-eligible player with recorded data through
        # the selected week, even if their last appearance was earlier.
        position = latest["position"]
        if position not in MODEL2_SETTINGS:
            continue
        settings = MODEL2_SETTINGS[position]
        result = {column: np.nan for column in columns}
        result.update(
            {
                "gsis_id": player_id,
                "player_name": latest["player_name"],
                "position": position,
                "team": latest["team"],
                "waiver_flag": "Slow Drop",
                "scenario": "Limited history; using available data",
                "games_analyzed": len(player),
            }
        )
        required = ["snap_pct", "opp_share", "opportunities", "fantasy_points_ppr"]
        valid_player = player.dropna(subset=required).copy()

        # Use adaptive scoring instead of returning Insufficient Data.
        # This also handles players with missing fantasy-point observations.
        if len(valid_player) < 4 or len(valid_player) / len(player) < 0.75:
            result.update(calculate_adaptive_waiver(player, settings, as_of_week))
            add_result(result)
            continue
        recent_count = min(
            4,
            max(2, len(valid_player) // 3),
        )
        recent = valid_player.tail(recent_count)
        previous = valid_player.iloc[:-recent_count]
        if len(previous) < 2:
            result.update(calculate_adaptive_waiver(player, settings, as_of_week))
            add_result(result)
            continue
        fantasy = calculate_fantasy_score(
            valid_player,
            recent,
            previous,
            settings,
        )
        usage_scores = calculate_usage_scores(
            recent,
            previous,
            settings,
        )
        stability = calculate_stability(
            valid_player,
            recent,
            fantasy["season_ppr"],
        )
        confirmation = calculate_trend_confirmation(
            recent,
            previous,
            fantasy["trend"],
            usage_scores["opp_trend"],
            usage_scores["snap_trend"],
        )
        raw_score = 100 * (
            MODEL2_WEIGHTS["fantasy"] * fantasy["score"]
            + MODEL2_WEIGHTS["opportunity"] * usage_scores["opportunity_score"]
            + MODEL2_WEIGHTS["snap"] * usage_scores["snap_score"]
            + MODEL2_WEIGHTS["stability"] * stability["score"]
            + MODEL2_WEIGHTS["confirmation"] * confirmation["score"]
        )
        coverage = min(
            1.0,
            len(valid_player) / max(as_of_week - 1, 1),
        )
        evidence_factor = 0.80 + 0.20 * coverage
        waiver_score = float(
            np.clip(
                raw_score * evidence_factor,
                0,
                100,
            )
        )
        latest_ppr = float(valid_player["fantasy_points_ppr"].iloc[-1])
        flag, scenario = classify_waiver_scenario(
            score=waiver_score,
            recent_ppr=fantasy["recent_ppr"],
            recent_median=fantasy["recent_median"],
            latest_ppr=latest_ppr,
            ppr_trend=fantasy["trend"],
            opp_trend=usage_scores["opp_trend"],
            snap_trend=usage_scores["snap_trend"],
            upward_confirmation=confirmation["upward_confirmation"],
            downward_confirmation=confirmation["downward_confirmation"],
            recent_volatility=stability["volatility"],
            games_analyzed=len(valid_player),
            settings=settings,
        )
        result.update(
            {
                "waiver_flag": flag,
                "waiver_score": round(waiver_score, 1),
                "scenario": scenario,
                "fantasy_score": round(fantasy["score"], 3),
                "opportunity_score": round(usage_scores["opportunity_score"], 3),
                "snap_score": round(usage_scores["snap_score"], 3),
                "stability_score": round(stability["score"], 3),
                "confirmation_score": round(confirmation["score"], 3),
                "season_ppr": round(fantasy["season_ppr"], 2),
                "recent_ppr": round(fantasy["recent_ppr"], 2),
                "ppr_trend": round(fantasy["trend"], 3),
                "snap_trend": round(usage_scores["snap_trend"], 3),
                "opportunity_trend": round(usage_scores["opp_trend"], 3),
                "volatility": (
                    round(stability["volatility"], 3)
                    if pd.notna(stability["volatility"])
                    else np.nan
                ),
                "upward_confirmation": round(
                    confirmation["upward_confirmation"],
                    3,
                ),
                "downward_confirmation": round(
                    confirmation["downward_confirmation"],
                    3,
                ),
                "games_analyzed": len(valid_player),
            }
        )
        add_result(result)
    return pd.DataFrame(
        results,
        columns=columns,
    )
