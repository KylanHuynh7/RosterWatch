
import pandas as pd
from models.usage_models import calculate_scores

def main():
    df = pd.read_csv(
        "data/processed/weekly_usage.csv"
    )

    scores = calculate_scores(df)

    scores.to_csv(
        "data/processed/player_scores.csv",
        index=False
    )

    print("Model successfully completed!")
    print("Total scored records:", len(scores))
    print("Seasons:", sorted(scores["season"].unique()))

if __name__ == "__main__":
    main()
