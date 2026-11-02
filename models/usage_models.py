
import pandas as pd

METRICS = {
    "snap_pct": 0.35,
    "opp_share": 0.30,
    "passdown_share": 0.20,
    "redzone_share": 0.15,
}

def calculate_scores(df):
    results = []

    df = df.sort_values(
        ["gsis_id", "season", "week"]
    )

    for (player_id, season), group in df.groupby(
        ["gsis_id", "season"]
    ):
        group = group.sort_values("week")

        if len(group) < 5:
            continue

        for i in range(4, len(group)):
            baseline = group.iloc[i-4:i-1]
            recent = group.iloc[i-1:i+1]

            row = group.iloc[i].to_dict()

            decline = 0
            growth = 0
            total_weight = 0

            for metric, weight in METRICS.items():
                before = baseline[metric].mean()
                after = recent[metric].mean()

                if pd.isna(before) or pd.isna(after):
                    continue

                change = after - before

                row[f"{metric}_change_pp"] = round(
                    change * 100, 2
                )

                decline += weight * max(0, -change)
                growth += weight * max(0, change)
                total_weight += weight

            if total_weight == 0:
                continue

            row["role_score"] = round(
                100 * decline / total_weight, 2
            )

            row["waiver_score"] = round(
                100 * growth / total_weight, 2
            )

            results.append(row)

    return pd.DataFrame(results)
