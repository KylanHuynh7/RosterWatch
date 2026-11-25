"""Backtest the Player Consistency Model on 2022-2025 (README section 11).

Question: does a player's average weekly volatility so far warn that he is
about to lose his role? For every player-week after his 4th game, with a
season-to-date snap share of at least 35%, the score is his average volatility
so far. Two outcomes are checked over his next three games:

- snap loss: his snap share is 15+ points below his season-to-date average
  in at least two of them;
- touch loss: his opportunity share is under two thirds of his season-to-date
  average in at least two of them.

The result is the AUC: the chance that a player who then lost his role had a
higher volatility than one who did not. 0.50 is a coin flip; higher is better.

Run from the repository root (takes a few seconds):

    python analysis/backtest_consistency.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rosterwatch.consistency import calculate_volatility  # noqa: E402
from rosterwatch.data import load_data  # noqa: E402

DATA = ROOT / "notebooks" / "weekly_usage.parquet"


def role_loss(games, column, lost):
    """True if ``lost(next value)`` holds in 2+ of the next three games; NaN without three later games."""
    hits = pd.concat(
        [
            lost(games[column].shift(-k))
            .astype(float)
            .where(games[column].shift(-k).notna())
            for k in (1, 2, 3)
        ],
        axis=1,
    )
    return (hits.sum(axis=1) >= 2).where(hits.notna().sum(axis=1) == 3)


def auc(score, outcome):
    """Area under the ROC curve via ranks (ties count half)."""
    ok = score.notna() & outcome.notna()
    score, outcome = score[ok], outcome[ok].astype(bool)
    n_pos, n_neg = outcome.sum(), (~outcome).sum()
    ranks = score.rank()
    return (ranks[outcome].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def player_weeks(usage):
    """One row per player-week with season-to-date averages and the two outcomes."""
    weekly, _ = calculate_volatility(usage)
    weekly = weekly.sort_values(["season", "gsis_id", "week"]).reset_index(drop=True)
    games = weekly.groupby(["season", "gsis_id"])
    weekly["game_number"] = games.cumcount() + 1
    weekly["snap_so_far"] = games["snap_pct"].transform(lambda s: s.expanding().mean())
    weekly["opp_so_far"] = games["opp_share"].transform(lambda s: s.expanding().mean())
    weekly["volatility_so_far"] = games["weekly_volatility"].transform(
        lambda s: s.expanding().mean()
    )
    snap_loss, touch_loss = [], []
    for _, player in weekly.groupby(["season", "gsis_id"], sort=False):
        snap_loss.append(
            role_loss(
                player, "snap_pct", lambda nxt: nxt <= player["snap_so_far"] - 0.15
            )
        )
        touch_loss.append(
            role_loss(
                player, "opp_share", lambda nxt: nxt <= player["opp_so_far"] * (2 / 3)
            )
        )
    weekly["snap_loss"] = pd.concat(snap_loss)
    weekly["touch_loss"] = pd.concat(touch_loss)
    keep = (
        (weekly["game_number"] >= 4)
        & (weekly["snap_so_far"] >= 0.35)
        & weekly["volatility_so_far"].notna()
    )
    return weekly.loc[keep]


def main():
    usage = load_data(str(DATA))
    rows = player_weeks(usage)
    print(f"Player-weeks checked (4+ games in, 35%+ snaps so far): {len(rows):,}\n")
    print("AUC of average volatility so far (0.50 = coin flip, higher = better)")
    for label, part in [
        ("2022-2024", rows.loc[rows["season"] <= 2024]),
        ("2025", rows.loc[rows["season"] == 2025]),
        ("All", rows),
    ]:
        print(
            f"  {label:9s} snap loss {auc(part['volatility_so_far'], part['snap_loss']):.2f}"
            f" | touch loss {auc(part['volatility_so_far'], part['touch_loss']):.2f}"
        )


if __name__ == "__main__":
    main()
