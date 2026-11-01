from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

RANDOM_STATE = 42

MANUAL_TEAM_MAP = {
    "Türkiye": "Turkey",
    "United States": "USA",
    "United States of America": "USA",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "IR Iran": "Iran",
    "China PR": "China",
    "Chinese Taipei": "Taiwan",
    "Côte d'Ivoire": "Ivory Coast",
    "Curaçao": "Curaçao",
    "Czechia": "Czech Republic",
    "FYR Macedonia": "North Macedonia",
    "Republic of Ireland": "Ireland",
    "Cape Verde Islands": "Cape Verde",
    "St. Kitts and Nevis": "Saint Kitts and Nevis",
    "St. Lucia": "Saint Lucia",
    "St. Vincent / Grenadines": "Saint Vincent and the Grenadines",
    "São Tomé and Príncipe": "São Tomé and Príncipe",
    "Eswatini": "Eswatini",
}

TOURNAMENT_GROUPS = {
    "Friendly": "Friendly",
    "FIFA World Cup": "World Cup",
    "UEFA Nations League": "Nations League",
    "CONCACAF Nations League": "Nations League",
}


def tournament_category(name: str) -> str:
    s = str(name).strip()
    if s in TOURNAMENT_GROUPS:
        return TOURNAMENT_GROUPS[s]
    low = s.lower()
    if "world cup" in low and ("qualification" not in low and "qualifier" not in low):
        return "World Cup"
    if "nations league" in low:
        return "Nations League"
    if any(k in low for k in ["qualification", "qualifier", "qualifying"]):
        return "Qualification"
    continental_terms = [
        "african cup of nations", "afc asian cup", "uefa euro", "copa américa",
        "copa america", "concacaf championship", "concacaf gold cup", "ofc nations cup",
        "uefa european championship", "asian cup", "arab cup"
    ]
    if any(k in low for k in continental_terms):
        return "Continental tournament"
    if "friendly" in low:
        return "Friendly"
    return "Other"


def make_name_map(former_names: pd.DataFrame) -> Dict[str, str]:
    mapping = dict(MANUAL_TEAM_MAP)
    for row in former_names.itertuples(index=False):
        mapping[str(row.former).strip()] = str(row.current).strip()
    return mapping


def normalize_name(name: str, mapping: Dict[str, str]) -> str:
    name = str(name).strip()
    seen = set()
    while name in mapping and name not in seen:
        seen.add(name)
        name = mapping[name]
    return name


def fifa_snapshot_date(year: int, semester: int) -> pd.Timestamp:
    # Dataset supplies semester snapshots, not exact publication days.
    # Jan 1 / Jul 1 is a conservative availability timestamp for backward joins.
    return pd.Timestamp(year=int(year), month=1 if int(semester) == 1 else 7, day=1)


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def elo_k_factor(tournament_group: str) -> float:
    return {
        "Friendly": 20.0,
        "Qualification": 35.0,
        "Continental tournament": 45.0,
        "World Cup": 55.0,
        "Nations League": 30.0,
        "Other": 30.0,
    }.get(tournament_group, 30.0)


def build_prematch_features(results: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    ratings = defaultdict(lambda: 1500.0)
    histories = defaultdict(lambda: deque(maxlen=10))
    rows = []

    def rolling(team: str, n: int) -> Dict[str, float]:
        games = list(histories[team])[-n:]
        prefix = f"last{n}_"
        if not games:
            return {
                prefix + "gf_avg": 0.0, prefix + "ga_avg": 0.0,
                prefix + "win_rate": 0.0, prefix + "draw_rate": 0.0,
                prefix + "loss_rate": 0.0, prefix + "goal_diff_avg": 0.0,
                prefix + "games": 0,
            }
        gf = np.array([g[0] for g in games], dtype=float)
        ga = np.array([g[1] for g in games], dtype=float)
        return {
            prefix + "gf_avg": float(gf.mean()),
            prefix + "ga_avg": float(ga.mean()),
            prefix + "win_rate": float((gf > ga).mean()),
            prefix + "draw_rate": float((gf == ga).mean()),
            prefix + "loss_rate": float((gf < ga).mean()),
            prefix + "goal_diff_avg": float((gf - ga).mean()),
            prefix + "games": len(games),
        }

    for m in results.itertuples(index=False):
        h, a = m.home_team, m.away_team
        h_elo, a_elo = ratings[h], ratings[a]
        feat = {
            "date": m.date, "home_team": h, "away_team": a,
            "home_score": float(m.home_score), "away_score": float(m.away_score),
            "neutral": bool(m.neutral), "tournament": m.tournament,
            "tournament_group": m.tournament_group,
            "home_elo": h_elo, "away_elo": a_elo, "elo_diff": h_elo - a_elo,
        }
        for n in (5, 10):
            for side, team in (("home", h), ("away", a)):
                for k, v in rolling(team, n).items():
                    feat[f"{side}_{k}"] = v
        rows.append(feat)

        hs, aws = float(m.home_score), float(m.away_score)
        actual_h = 1.0 if hs > aws else 0.5 if hs == aws else 0.0
        home_adv = 0.0 if bool(m.neutral) else 100.0
        exp_h = expected_score(h_elo + home_adv, a_elo)
        margin = abs(hs - aws)
        margin_mult = math.log1p(margin) if margin > 0 else 1.0
        k = elo_k_factor(m.tournament_group)
        delta = k * margin_mult * (actual_h - exp_h)
        ratings[h] += delta
        ratings[a] -= delta
        histories[h].append((hs, aws))
        histories[a].append((aws, hs))

    state = {
        "ratings": dict(ratings),
        "histories": {k: list(v) for k, v in histories.items()},
    }
    return pd.DataFrame(rows), state


def attach_fifa(features: pd.DataFrame, fifa: pd.DataFrame) -> pd.DataFrame:
    def side_join(df: pd.DataFrame, side: str) -> pd.DataFrame:
        left = df.sort_values(["date", f"{side}_team"]).copy()
        right = fifa[["snapshot_date", "team", "rank", "total.points"]].rename(columns={
            "snapshot_date": f"{side}_fifa_date", "team": f"{side}_team",
            "rank": f"{side}_fifa_rank", "total.points": f"{side}_fifa_points",
        }).sort_values([f"{side}_fifa_date", f"{side}_team"])
        return pd.merge_asof(
            left, right, left_on="date", right_on=f"{side}_fifa_date",
            by=f"{side}_team", direction="backward", allow_exact_matches=True
        )

    out = side_join(features, "home")
    out = side_join(out, "away")
    out["fifa_rank_diff"] = out["home_fifa_rank"] - out["away_fifa_rank"]
    out["fifa_points_diff"] = out["home_fifa_points"] - out["away_fifa_points"]
    return out.sort_values("date").reset_index(drop=True)


def load_and_prepare(data_dir: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    data_dir = Path(data_dir)
    results = pd.read_csv(data_dir / "results.csv")
    former = pd.read_csv(data_dir / "former_names.csv")
    fifa = pd.read_csv(data_dir / "fifa_mens_rank_updated.csv")

    results["date"] = pd.to_datetime(results["date"], errors="coerce")
    results = results.dropna(subset=["date", "home_score", "away_score"]).copy()
    results = results.sort_values("date", kind="stable").reset_index(drop=True)
    name_map = make_name_map(former)
    for c in ["home_team", "away_team", "country"]:
        results[c] = results[c].map(lambda x: normalize_name(x, name_map))
    results["tournament_group"] = results["tournament"].map(tournament_category)

    fifa["team"] = fifa["team"].map(
    lambda x: normalize_name(x, name_map)
)

    fifa["team"] = fifa["team"].map(
        lambda x: normalize_name(x, name_map)
    )

    if "ranking_date" in fifa.columns:
        fifa["snapshot_date"] = pd.to_datetime(
            fifa["ranking_date"],
            errors="coerce",
        )
    else:
        fifa["snapshot_date"] = [
            fifa_snapshot_date(y, s)
            for y, s in zip(
                fifa["date"],
                fifa["semester"],
            )
        ]

    fifa = fifa.dropna(
        subset=[
            "team",
            "snapshot_date",
            "rank",
            "total.points",
        ]
    )

    fifa = fifa.sort_values(
        ["team", "snapshot_date"]
    ).drop_duplicates(
        ["team", "snapshot_date"],
        keep="last",
    )

    prematch, state = build_prematch_features(results)
    full = attach_fifa(prematch, fifa)
    return full, fifa, state, name_map


def model_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    categorical = ["tournament_group"]
    excluded = {"date", "home_score", "away_score", "tournament", "home_fifa_date", "away_fifa_date", "fifa_date"}
    numeric = [c for c in df.columns if c not in excluded | set(categorical) and pd.api.types.is_numeric_dtype(df[c])]
    return categorical, numeric


def make_preprocessor(categorical: List[str], numeric: List[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=2), categorical),
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler(with_mean=False)),
        ]), numeric),
    ])


def candidate_models(categorical: List[str], numeric: List[str]) -> Dict[str, object]:
    return {
        "PoissonRegressor": Pipeline([
            ("prep", make_preprocessor(categorical, numeric)),
            ("model", PoissonRegressor(alpha=0.2, max_iter=150)),
        ]),
        "XGBRegressorPoisson": Pipeline([
            ("prep", make_preprocessor(categorical, numeric)),
            ("model", XGBRegressor(
                objective="count:poisson", n_estimators=20, learning_rate=0.1,
                max_depth=4, min_child_weight=8, subsample=0.85, colsample_bytree=0.8,
                reg_lambda=2.0, n_jobs=-1, random_state=RANDOM_STATE
            )),
        ]),
    }


def evaluate(y_true: np.ndarray, pred: np.ndarray) -> dict:
    pred = np.clip(pred, 0.03, 10.0)
    return {
        "mae": float(mean_absolute_error(y_true, pred)),
        "rmse": float(mean_squared_error(y_true, pred) ** 0.5),
        "mean_poisson_deviance": float(mean_poisson_deviance(y_true, pred)),
    }


def train_models(features: pd.DataFrame, output_dir: str | Path, fifa: pd.DataFrame, state: dict, name_map: dict) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    required = ["home_fifa_rank", "away_fifa_rank", "home_fifa_points", "away_fifa_points"]
    model_df = features.dropna(subset=required).copy()
    model_df = model_df[model_df["date"] >= pd.Timestamp("1992-01-01")].tail(5000).reset_index(drop=True)

    split_idx = int(len(model_df) * 0.80)
    split_date = model_df.iloc[split_idx]["date"]
    train = model_df[model_df["date"] < split_date]
    test = model_df[model_df["date"] >= split_date]
    categorical, numeric = model_feature_columns(model_df)
    X_train, X_test = train[categorical + numeric], test[categorical + numeric]

    leaderboard = []
    fitted = {}
    for model_name, model in candidate_models(categorical, numeric).items():
        pair = {}
        predictions = {}
        for target in ["home_score", "away_score"]:
            fitted_model = clone(model)
            fitted_model.fit(X_train, train[target])
            pred = np.clip(fitted_model.predict(X_test), 0.03, 10.0)
            pair[target] = fitted_model
            predictions[target] = pred
            metrics = evaluate(test[target].to_numpy(), pred)
            leaderboard.append({"model": model_name, "target": target, **metrics})
        pair_mae = np.mean([
            mean_absolute_error(test["home_score"], predictions["home_score"]),
            mean_absolute_error(test["away_score"], predictions["away_score"]),
        ])
        rounded_exact = np.mean(
            (np.rint(predictions["home_score"]) == test["home_score"].to_numpy()) &
            (np.rint(predictions["away_score"]) == test["away_score"].to_numpy())
        )
        fitted[model_name] = {"models": pair, "pair_mae": float(pair_mae), "rounded_exact_accuracy": float(rounded_exact)}

    best_name = min(fitted, key=lambda k: fitted[k]["pair_mae"])
    bundle = {
        "home_model": fitted[best_name]["models"]["home_score"],
        "away_model": fitted[best_name]["models"]["away_score"],
        "model_name": best_name,
        "categorical_features": categorical,
        "numeric_features": numeric,
        "feature_columns": categorical + numeric,
        "state": state,
        "fifa": fifa,
        "name_map": name_map,
        "last_match_date": features["date"].max(),
        "training_split_date": split_date,
    }
    joblib.dump(bundle, output_dir / "national_team_model.joblib", compress=3)

    report = {
        "rows_total_results": int(len(features)),
        "rows_model_eligible": int(len(model_df)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_end_exclusive": str(split_date.date()),
        "test_start": str(split_date.date()),
        "best_model": best_name,
        "best_pair_mae": fitted[best_name]["pair_mae"],
        "best_rounded_exact_score_accuracy": fitted[best_name]["rounded_exact_accuracy"],
        "leaderboard": leaderboard,
        "model_summary": {k: {kk: vv for kk, vv in v.items() if kk != "models"} for k, v in fitted.items()},
        "fifa_missing_rows": int(features[required].isna().any(axis=1).sum()),
    }
    (output_dir / "model_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _rolling_from_state(history: List[Tuple[float, float]], n: int) -> dict:
    games = history[-n:]
    prefix = f"last{n}_"
    if not games:
        return {prefix + k: v for k, v in {
            "gf_avg": 0.0, "ga_avg": 0.0, "win_rate": 0.0, "draw_rate": 0.0,
            "loss_rate": 0.0, "goal_diff_avg": 0.0, "games": 0,
        }.items()}
    gf = np.array([x[0] for x in games]); ga = np.array([x[1] for x in games])
    return {prefix + k: v for k, v in {
        "gf_avg": float(gf.mean()), "ga_avg": float(ga.mean()),
        "win_rate": float((gf > ga).mean()), "draw_rate": float((gf == ga).mean()),
        "loss_rate": float((gf < ga).mean()), "goal_diff_avg": float((gf-ga).mean()),
        "games": len(games),
    }.items()}


def _latest_fifa(fifa: pd.DataFrame, team: str, as_of_date: pd.Timestamp) -> pd.Series:
    rows = fifa[(fifa["team"] == team) & (fifa["snapshot_date"] <= as_of_date)]
    if rows.empty:
        raise ValueError(f"{team!r} için {as_of_date.date()} öncesinde FIFA ranking kaydı yok.")
    return rows.iloc[-1]


def predict_match(
    home_team: str,
    away_team: str,
    neutral: bool = False,
    tournament: str = "Friendly",
    model_path: str | Path = "/mnt/data/national_team_model.joblib",
    as_of_date: str | pd.Timestamp | None = None,
    max_goals: int = 10,
) -> dict:
    bundle = joblib.load(model_path)
    name_map = bundle["name_map"]
    home_team = normalize_name(home_team, name_map)
    away_team = normalize_name(away_team, name_map)
    as_of = pd.Timestamp(as_of_date) if as_of_date is not None else pd.Timestamp(bundle["last_match_date"])
    state, fifa = bundle["state"], bundle["fifa"]
    if home_team not in state["ratings"] or away_team not in state["ratings"]:
        missing = [t for t in [home_team, away_team] if t not in state["ratings"]]
        raise ValueError(f"Geçmiş maç verisi bulunamayan takım(lar): {missing}")
    hf, af = _latest_fifa(fifa, home_team, as_of), _latest_fifa(fifa, away_team, as_of)
    row = {
        "home_team": home_team, "away_team": away_team, "neutral": bool(neutral),
        "tournament_group": tournament_category(tournament),
        "home_elo": state["ratings"][home_team], "away_elo": state["ratings"][away_team],
        "elo_diff": state["ratings"][home_team] - state["ratings"][away_team],
        "home_fifa_rank": hf["rank"], "away_fifa_rank": af["rank"],
        "fifa_rank_diff": hf["rank"] - af["rank"],
        "home_fifa_points": hf["total.points"], "away_fifa_points": af["total.points"],
        "fifa_points_diff": hf["total.points"] - af["total.points"],
    }
    for n in (5, 10):
        for side, team in (("home", home_team), ("away", away_team)):
            for k, v in _rolling_from_state(state["histories"].get(team, []), n).items():
                row[f"{side}_{k}"] = v
    X = pd.DataFrame([row]).reindex(columns=bundle["feature_columns"])
    lam_h = float(np.clip(bundle["home_model"].predict(X)[0], 0.03, 8.0))
    lam_a = float(np.clip(bundle["away_model"].predict(X)[0], 0.03, 8.0))

    hg = np.arange(max_goals + 1)
    hp = poisson.pmf(hg, lam_h); ap = poisson.pmf(hg, lam_a)
    matrix = np.outer(hp, ap)
    # Fold omitted tail mass into the last row/column so probabilities sum to 1.
    hp[-1] += max(0.0, 1.0 - hp.sum()); ap[-1] += max(0.0, 1.0 - ap.sum())
    matrix = np.outer(hp, ap)
    home_win = float(np.tril(matrix, -1).sum())
    draw = float(np.trace(matrix))
    away_win = float(np.triu(matrix, 1).sum())
    scores = sorted(
        [((i, j), float(matrix[i, j])) for i in range(max_goals + 1) for j in range(max_goals + 1)],
        key=lambda x: x[1], reverse=True
    )[:5]
    return {
        "expected_home_goals": round(lam_h, 3),
        "expected_away_goals": round(lam_a, 3),
        "home_win_probability": round(home_win, 4),
        "draw_probability": round(draw, 4),
        "away_win_probability": round(away_win, 4),
        "top_score_probabilities": [
            {"score": f"{i}-{j}", "probability": round(p, 4)} for (i, j), p in scores
        ],
        "model": bundle["model_name"],
        "as_of_date": str(as_of.date()),
        "fifa_snapshot_home": str(pd.Timestamp(hf["snapshot_date"]).date()),
        "fifa_snapshot_away": str(pd.Timestamp(af["snapshot_date"]).date()),
    }


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"

    features, fifa, state, name_map = load_and_prepare(data_dir)

    features.tail(5000).to_csv(
        project_dir / "match_features_sample.csv.gz",
        index=False,
        compression="gzip",
    )

    report = train_models(
        features=features,
        output_dir=project_dir,
        fifa=fifa,
        state=state,
        name_map=name_map,
    )

    example = predict_match(
        home_team="Turkey",
        away_team="Germany",
        neutral=False,
        tournament="Friendly",
        model_path=project_dir / "national_team_model.joblib",
    )

    (project_dir / "example_prediction.json").write_text(
        json.dumps(example, indent=2),
        encoding="utf-8",
    )

    print(json.dumps({
        "report": report,
        "example": example,
    }, indent=2))


if __name__ == "__main__":
    main()
