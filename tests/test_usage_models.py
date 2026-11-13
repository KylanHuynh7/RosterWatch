
import pandas as pd

from models.usage_models import calculate_scores


def make_data():
    rows = []

    for week in range(1, 7):
        rows.append({
            "gsis_id": "test_player",
            "player_name": "Test Player",
            "season": 2025,
            "week": week,
            "team": "TEST",
            "position": "RB",
            "snap_pct": 0.50 if week <= 3 else 0.70,
            "opp_share": 0.20 if week <= 3 else 0.30,
            "passdown_share": 0.10,
            "redzone_share": 0.20,
        })

    return pd.DataFrame(rows)


def test_scores_are_generated():
    scores = calculate_scores(make_data())

    assert not scores.empty
    assert "role_score" in scores.columns
    assert "waiver_score" in scores.columns


def test_increased_usage_produces_growth_score():
    scores = calculate_scores(make_data())

    assert scores["waiver_score"].max() > 0


def test_scores_are_nonnegative():
    scores = calculate_scores(make_data())

    assert (scores["role_score"] >= 0).all()
    assert (scores["waiver_score"] >= 0).all()


def test_no_future_data_leakage():
    data = make_data()

    original = calculate_scores(data)
    modified = data.copy()

    modified.loc[
        modified["week"] == 6, "snap_pct"
    ] = 0.01

    changed = calculate_scores(modified)

    # Changing Week 6 must not affect Week 5.
    original_week5 = original.loc[
        original["week"] == 5, "waiver_score"
    ].iloc[0]

    changed_week5 = changed.loc[
        changed["week"] == 5, "waiver_score"
    ].iloc[0]

    assert original_week5 == changed_week5


def test_missing_redzone_values():
    data = make_data()

    data.loc[
        data["week"] == 2, "redzone_share"
    ] = float("nan")

    scores = calculate_scores(data)

    assert not scores.empty
    assert scores["role_score"].notna().all()
    assert scores["waiver_score"].notna().all()
