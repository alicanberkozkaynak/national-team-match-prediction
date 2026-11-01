from pathlib import Path

import pandas as pd


project_dir = Path(__file__).resolve().parent
data_dir = project_dir / "data"

results = pd.read_csv(data_dir / "results.csv")

results["date"] = pd.to_datetime(
    results["date"],
    errors="coerce",
)

results["home_score"] = pd.to_numeric(
    results["home_score"],
    errors="coerce",
)

results["away_score"] = pd.to_numeric(
    results["away_score"],
    errors="coerce",
)

completed_matches = results.dropna(
    subset=[
        "date",
        "home_score",
        "away_score",
    ]
).copy()

future_or_unplayed = results[
    results["home_score"].isna()
    | results["away_score"].isna()
].copy()

print("Tüm veri içindeki ilk tarih:", results["date"].min())
print("Tüm veri içindeki son tarih:", results["date"].max())
print("Toplam satır:", len(results))

print("\nTamamlanmış maç sayısı:", len(completed_matches))
print(
    "Son tamamlanmış maç tarihi:",
    completed_matches["date"].max(),
)

print(
    "Skoru olmayan fikstür sayısı:",
    len(future_or_unplayed),
)

print("\nSon 10 tamamlanmış maç:")
print(
    completed_matches[
        [
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
        ]
    ]
    .sort_values("date")
    .tail(10)
    .to_string(index=False)
)

print("\nİlk 10 skoru olmayan fikstür:")
print(
    future_or_unplayed[
        [
            "date",
            "home_team",
            "away_team",
        ]
    ]
    .sort_values("date")
    .head(10)
    .to_string(index=False)
)