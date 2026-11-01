from pathlib import Path

import pandas as pd


project_dir = Path(__file__).resolve().parent
data_dir = project_dir / "data"

old_path = data_dir / "fifa_mens_rank.csv"
new_path = data_dir / "fifa_mens_world_ranking_2025_2026_with_points.csv"
output_path = data_dir / "fifa_mens_rank_updated.csv"


old_fifa = pd.read_csv(old_path)
new_fifa = pd.read_csv(new_path)

print("Eski kolonlar:", old_fifa.columns.tolist())
print("Yeni kolonlar:", new_fifa.columns.tolist())


# Yeni dosyayı eski dosyanın kolon isimlerine dönüştür.
new_fifa = new_fifa.rename(
    columns={
        "Year": "date",
        "Date": "ranking_date",
        "Rank": "rank",
        "Team": "team",
        "Total Points": "total.points",
    }
)

new_fifa["ranking_date"] = pd.to_datetime(
    new_fifa["ranking_date"],
    errors="raise",
)

# Eski pipeline semester kolonu bekliyorsa üret.
new_fifa["semester"] = (
    new_fifa["ranking_date"].dt.month.gt(6).astype(int) + 1
)

# Eski dosyada gerçek snapshot tarihi yoksa yıl/semester üzerinden üret.
if "ranking_date" not in old_fifa.columns:
    old_fifa["ranking_date"] = pd.to_datetime(
        {
            "year": old_fifa["date"].astype(int),
            "month": old_fifa["semester"].map({1: 1, 2: 7}),
            "day": 1,
        }
    )
else:
    old_fifa["ranking_date"] = pd.to_datetime(
        old_fifa["ranking_date"],
        errors="coerce",
    )

required_columns = [
    "date",
    "semester",
    "ranking_date",
    "team",
    "rank",
    "total.points",
]

for column in required_columns:
    if column not in old_fifa.columns:
        old_fifa[column] = pd.NA

old_fifa = old_fifa[required_columns]
new_fifa = new_fifa[required_columns]

combined = pd.concat(
    [old_fifa, new_fifa],
    ignore_index=True,
)

combined = combined.dropna(
    subset=["ranking_date", "team", "rank", "total.points"]
)

combined = combined.sort_values(
    ["team", "ranking_date"]
)

combined = combined.drop_duplicates(
    subset=["team", "ranking_date"],
    keep="last",
)

combined.to_csv(
    output_path,
    index=False,
)

print("\nBirleştirme tamamlandı.")
print("Toplam kayıt:", len(combined))
print("İlk ranking tarihi:", combined["ranking_date"].min())
print("Son ranking tarihi:", combined["ranking_date"].max())
print("Çıktı:", output_path)