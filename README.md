# International Football Match Prediction

A Python machine-learning project that estimates international football match outcomes, expected goals, and likely final scores through a local web interface.

> **Project status:** Experimental / portfolio project. Predictions are probabilistic and are not guarantees. The model and interface have not been independently validated for production use.

## Features

- Choose two national teams and a competition type.
- Toggle neutral venue conditions.
- See estimated goals, home-win/draw/away-win probabilities, and the five most probable scorelines.
- Run locally via a Python HTTP server; no external web API is required for predictions.
- Train and compare Poisson regression and XGBoost Poisson regression models.

## Technology

Python, pandas, NumPy, SciPy, scikit-learn, XGBoost, joblib, HTML, CSS, and JavaScript. The local web server uses Python's built-in `http.server`.

## Run locally

Python 3.12 is the environment previously used for development; package compatibility can vary by platform and Python version.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:8000** in your browser. Windows users can alternatively run `start_web.bat` after installing dependencies.

The included `national_team_model.joblib` is needed to run predictions. Load `.joblib` files only from trusted sources.

## Train the model

The training script expects datasets under `data/`:

```powershell
python national_team_score_model.py
```

This writes the fitted model, `model_report.json`, and `example_prediction.json`. The `merge_fifa_rankings.py` utility regenerates the updated FIFA ranking file from its inputs.

## Reported evaluation

The bundled `model_report.json` reports a chronological training/test split:

| Metric | Reported value |
| --- | ---: |
| Total records in prepared results | 49,495 |
| Eligible matches selected for evaluation | 5,000 |
| Training matches | 3,975 |
| Test matches | 1,025 |
| Test period begins | 2025-06-10 |
| Selected model | PoissonRegressor |
| Mean absolute error, averaged across both teams | 0.9332 goals |
| Rounded exact-score accuracy | 10.34% |

These are test-set metrics, not promises of future performance. The source training script uses the latest 5,000 eligible matches for its evaluation subset.

## Repository layout

```text
app.py                            # Local web server and JSON prediction endpoints
static/                           # Browser interface
national_team_score_model.py      # Features, training, evaluation, prediction
predict.py                        # Command-line prediction example
merge_fifa_rankings.py            # Combine FIFA ranking data
check_data.py                     # Inspect match data
national_team_model.joblib        # Pre-trained model artifact
model_report.json                 # Model evaluation results
example_prediction.json           # Example output
requirements.txt                  # Python dependencies
data/                              # Match and ranking CSV files
```

## Known limitations

- **Neutral venue:** The two separately trained home/away goal models are not guaranteed to produce symmetric estimates when the teams are swapped, even if neutral venue is enabled. This is an open issue to resolve and test.
- **Data recency:** By default, the prediction method uses the last completed match date in the bundled model, rather than automatically downloading current results or rankings. Refreshing data and retraining are manual steps.
- **Dataset licensing:** Verify original dataset licenses and attribution before redistributing the CSV files. Add links to the exact data sources when known.
- **Scope:** Probabilities are model estimates and have not been demonstrated to be calibrated.

## Next steps

- Add automated tests for neutral-venue symmetry and probability totals.
- Document data provenance and permitted reuse.
- Add interface screenshots and a reproducible model training environment.
- Evaluate calibration and performance against baseline models.

## License

No software license has been selected yet. Add a license only after deciding how others may reuse the code and confirming applicable data rights.
