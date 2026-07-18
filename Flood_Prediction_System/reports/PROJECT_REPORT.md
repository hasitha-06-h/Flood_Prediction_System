# Project Report

## AI Powered Flood Prediction System Using Machine Learning

---

## 1. Abstract

Floods remain one of the most frequent and destructive natural disasters, and the lack of
accessible, timely early-warning tools continues to amplify their impact on vulnerable
communities. This project builds a machine learning based flood prediction system trained on
historical meteorological data - annual rainfall, cloud visibility, temperature, humidity and
seasonal rainfall - and compares four supervised classification algorithms (Decision Tree,
Random Forest, K-Nearest Neighbours and XGBoost) to identify the strongest performer for
deployment. The selected model is served through a Flask web application that lets a user
enter live weather readings and receive an instant flood risk verdict, complete with a
probability score, a risk classification, and a plain-language recommendation.

## 2. Problem Statement

Conventional flood forecasting typically depends on hydrological simulation models that require
dense sensor networks, river-gauge telemetry and specialist interpretation - infrastructure many
smaller districts and municipalities don't have consistent access to. This leaves a practical gap:
basic weather station readings (rainfall, temperature, humidity, cloud cover) are collected
routinely almost everywhere, but are rarely turned into an actionable, easy-to-read risk signal
for the people who need to act on it.

## 3. Objectives

- Build a clean, reproducible data pipeline for a historical flood dataset.
- Train and fairly benchmark four different classification algorithms on identical data.
- Automatically select the best-performing model based on accuracy, precision, recall and F1.
- Package the winning model behind a usable, non-technical web interface.
- Persist every prediction made through the interface for later review (prediction history).
- Ship the whole system in a form that can be containerised and deployed without modification.

## 4. Existing System

Existing flood-warning infrastructure in most regions relies on:

- Manual monitoring of river and reservoir water levels by field staff.
- Regional meteorological department bulletins issued on a fixed schedule rather than on demand.
- Hydrological models (e.g. rainfall-runoff simulations) that require specialist software and
  detailed catchment data that isn't available for every district.

These approaches are valuable but slow to reach smaller or resource-constrained areas, and they
generally don't offer an on-demand, self-serve way for a local officer to check risk for a
specific reading right now.

## 5. Proposed System

The proposed system takes five commonly available weather readings and returns an immediate,
model-backed flood risk assessment through a web form. It consists of:

1. A **training pipeline** (`train_model.py`) that cleans historical data, trains four
   classifiers, benchmarks them, and saves the best one.
2. A **Flask web application** (`app.py`) that loads the saved model once at startup and serves
   predictions with sub-50ms latency per request.
3. A **SQLite-backed history log** so every prediction is auditable after the fact.

## 6. Advantages of the Proposed System

- No specialist hydrological expertise required to operate - the interface accepts plain
  numeric readings.
- Model selection is automatic and benchmark-driven rather than a fixed choice, so retraining
  on a larger dataset in the future can change which algorithm gets deployed without any code
  changes.
- Every prediction is logged, giving a lightweight audit trail of past readings and verdicts.
- Deployable as a single Docker container or directly to a platform like Render with no
  additional infrastructure.

## 7. System Modules

| Module | Responsibility |
|---|---|
| `preprocessing/data_prep.py` | Loading, cleaning, outlier capping, scaling, train/test split |
| `training/decision_tree.py` | Decision Tree train / predict / evaluate |
| `training/random_forest.py` | Random Forest train / predict / evaluate |
| `training/knn.py` | K-Nearest Neighbours train / predict / evaluate |
| `training/xgboost_model.py` | XGBoost train / predict / evaluate |
| `evaluation/metrics.py` | Shared accuracy / precision / recall / F1 / confusion matrix helpers |
| `compare_models.py` | Runs all four models, builds the comparison table and chart |
| `train_model.py` | End-to-end orchestrator; saves the winning model + scaler |
| `prediction.py` | Loads the saved model/scaler and runs inference for the web app |
| `database.py` | SQLite schema and read/write helpers |
| `app.py` | Flask routes and page rendering |
| `utils.py` | Form validation, risk classification, recommendation text |

## 8. Methodology

The project followed an Agile-style breakdown across four epics:

- **Epic 1 - Data Collection**: loading the dataset, inspecting shape, columns, summary
  statistics, missing values, duplicates and class balance.
- **Epic 2 - Data Analysis**: univariate and multivariate exploratory analysis (histograms,
  box plots, count plots, correlation heatmap, pair plot, feature importance) - see
  `Flood_Analysis.ipynb`.
- **Epic 3 - Data Preprocessing**: missing value imputation, IQR-based outlier capping,
  an 80/20 train-test split, and feature scaling with `StandardScaler` (persisted via Joblib).
- **Epic 4 - Model Building**: four independent training modules sharing a common
  train/predict/evaluate interface, compared on identical data splits.

## 9. Algorithms Used

- **Decision Tree** - a single interpretable tree, depth-limited to avoid memorising a
  115-row dataset.
- **Random Forest** - an ensemble of 200 depth-limited trees trained on bootstrapped samples.
- **K-Nearest Neighbours** - distance-weighted voting over the 5 nearest scaled neighbours.
- **XGBoost** - gradient boosted trees; selected for deployment due to its regularisation and
  strong generalisation properties on tabular data.

## 10. Testing

- Every Flask route (`/`, `/about`, `/predict`, `/history`, `/alert`, `/safe`) was smoke-tested
  with both GET and POST requests.
- Form validation was tested against missing fields, non-numeric input and out-of-range values.
- The prediction pipeline was verified to return a consistent probability, risk label and
  recommendation for a known input.
- 404 and 500 error handlers were verified to render the custom error pages rather than a raw
  stack trace.

## 11. Screenshots

*(Insert screenshots of the Home, Predict, Result, Flood Alert, Safe, About and History pages
here before final submission - run the app locally with `python app.py` and capture each page.)*

## 12. Conclusion

The project demonstrates that classical machine learning models, trained purely on routinely
available meteorological readings, can produce a reasonably accurate flood risk signal without
requiring hydrological simulation infrastructure. Benchmarking four algorithms side by side
rather than committing to one upfront made the final model choice defensible and reproducible,
and packaging the result behind a simple web form makes the system usable by non-specialists.

## 13. Future Scope

- Incorporate real-time weather API feeds instead of manual form entry.
- Add authentication so the `users` table becomes fully functional and predictions can be tied
  to specific field officers or districts.
- Extend the dataset with more recent years and additional regions to improve generalisation.
- Add SMS/email alerting when a Severe risk prediction is logged.
- Track multiple deployed model versions side by side (the `ML_MODEL` entity in the ER diagram
  anticipates this).

## 14. References

1. Scikit-learn documentation - https://scikit-learn.org/stable/
2. XGBoost documentation - https://xgboost.readthedocs.io/
3. Flask documentation - https://flask.palletsprojects.com/
4. Bootstrap 5 documentation - https://getbootstrap.com/docs/5.3/

## 15. License

This project is released under the MIT License - see `LICENSE` for details.
