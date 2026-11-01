from national_team_score_model import predict_match


home_team = input("Ev sahibi takım: ").strip()
away_team = input("Deplasman takımı: ").strip()

neutral_input = input("Tarafsız saha mı? (e/h): ").strip().lower()
neutral = neutral_input in {"e", "evet", "y", "yes"}

tournament = input("Turnuva: ").strip()

if not tournament:
    tournament = "Friendly"

try:
    result = predict_match(
        home_team=home_team,
        away_team=away_team,
        neutral=neutral,
        tournament=tournament,
        model_path="national_team_model.joblib",
    )

    print("\n" + "=" * 45)
    print(f"{home_team} - {away_team}")
    print("=" * 45)

    print(
        f"Beklenen {home_team} golü: "
        f"{result['expected_home_goals']:.2f}"
    )

    print(
        f"Beklenen {away_team} golü: "
        f"{result['expected_away_goals']:.2f}"
    )

    print("\nMaç sonucu olasılıkları")
    print("-" * 45)

    print(
        f"{home_team} kazanır: "
        f"%{result['home_win_probability'] * 100:.2f}"
    )

    print(
        f"Beraberlik: "
        f"%{result['draw_probability'] * 100:.2f}"
    )

    print(
        f"{away_team} kazanır: "
        f"%{result['away_win_probability'] * 100:.2f}"
    )

    print("\nEn olası skorlar")
    print("-" * 45)

    for item in result["top_score_probabilities"]:
        percentage = item["probability"] * 100
        print(f"{item['score']:>5} → %{percentage:.2f}")

except FileNotFoundError:
    print(
        "\nModel dosyası bulunamadı. "
        "national_team_model.joblib dosyasını kontrol et."
    )

except ValueError as error:
    print(f"\nTahmin oluşturulamadı: {error}")

except Exception as error:
    print(f"\nBeklenmeyen hata: {error}")