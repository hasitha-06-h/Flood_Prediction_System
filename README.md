# AquaGuard - AI Powered Flood Prediction System

A machine learning based flood early-warning system. Historical weather data trains four
classification models (Decision Tree, Random Forest, KNN, XGBoost); the strongest performer is
selected automatically and served through a Flask web application.

---

## Overview

AquaGuard takes five weather readings - annual rainfall, cloud visibility, temperature, humidity
and seasonal (Jun-Sep) rainfall - and returns a flood risk verdict with a probability score, a
risk band, and a plain-language recommendation. Every prediction is logged to SQLite for later
review.

## Objectives

- Build a reproducible pipeline from raw dataset to a benchmarked, deployable model.
- Compare four classifiers fairly on identical preprocessing and splits.
- Serve predictions through a clean, non-technical web interface.
- Ship the whole system as a single Docker image, deployable to Render with zero extra setup.

## Architecture

```
                 ┌───────────────────────┐
                 │   flood_dataset.xlsx  │
                 └──────────┬────────────┘
                            │
                 ┌──────────▼────────────┐
                 │ preprocessing/        │  clean, impute, cap outliers,
                 │  data_prep.py         │  scale, train/test split
                 └──────────┬────────────┘
                            │
        ┌───────────────────┼────────────────────┐
        │                   │                    │
┌───────▼──────┐   ┌────────▼───────┐   ┌────────▼───────┐   ┌───────────────┐
│ decision_tree │   │ random_forest  │   │      knn       │   │ xgboost_model │
└───────┬──────┘   └────────┬───────┘   └────────┬───────┘   └───────┬───────┘
        └───────────────────┴────────────────────┴───────────────────┘
                            │
                 ┌──────────▼────────────┐
                 │  compare_models.py    │  benchmark + select best
                 └──────────┬────────────┘
                            │
                 ┌──────────▼────────────┐
                 │ models/floods.save    │  serialized winning model
                 │ models/transform.save │  serialized scaler
                 └──────────┬────────────┘
                            │
                 ┌──────────▼────────────┐
                 │      app.py (Flask)   │──▶ SQLite (database.py)
                 └───────────────────────┘
```

## Folder Structure

```
Flood_Prediction_System/
├── app.py                     Flask routes
├── config.py                  Central configuration
├── database.py                SQLite schema + helpers
├── train_model.py             Full training pipeline entry point
├── compare_models.py           Runs + benchmarks all four models
├── prediction.py               Loads saved model, runs inference
├── utils.py                    Validation, risk classification, helpers
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── render.yaml
├── Flood_Analysis.ipynb        EDA notebook
├── dataset/
│   └── flood_dataset.xlsx
├── models/                     floods.save, transform.save, metadata.json
├── training/
│   ├── decision_tree.py
│   ├── random_forest.py
│   ├── knn.py
│   └── xgboost_model.py
├── preprocessing/
│   └── data_prep.py
├── evaluation/
│   └── metrics.py
├── database/                   flood_system.db (generated)
├── reports/                     charts, training summary, ER diagram, project report
├── templates/                   Jinja HTML pages
│   ├── base.html, home.html, predict.html, about.html
│   ├── result.html, chance.html, no_chance.html, history.html
│   └── 404.html, 500.html
└── static/
    ├── css/ (style.css, animations.css)
    ├── js/ (app.js, validation.js)
    └── images/
```

## Installation (Local)

```bash
git clone <this-repo>
cd Flood_Prediction_System

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Train the model (writes models/floods.save + models/transform.save)
python train_model.py

# Start the app
python app.py
```

Visit **http://127.0.0.1:5000**.

## Running with Docker

```bash
docker build -t flood-prediction-system .
docker run -p 5000:5000 flood-prediction-system
```

or with Docker Compose (recommended - persists the SQLite history across restarts):

```bash
docker compose up --build
```

Visit **http://localhost:5000**.

## Deploying to Render

1. Push this repository to GitHub/GitLab.
2. In Render, choose **New +  Blueprint** and point it at the repo - `render.yaml` configures
   the build (`pip install -r requirements.txt && python train_model.py`) and start command
   (`gunicorn ... app:app`) automatically.
3. Alternatively, create a Web Service manually with:
   - **Build Command**: `pip install -r requirements.txt && python train_model.py`
   - **Start Command**: `gunicorn --bind 0.0.0.0:$PORT app:app`
4. Render injects `$PORT` automatically; `config.py` reads it, so no changes are needed.

## Requirements

See `requirements.txt`. Core dependencies: Flask, gunicorn, NumPy, Pandas, scikit-learn,
XGBoost, Matplotlib, Seaborn, Joblib, openpyxl.

## Retraining on New Data

Replace `dataset/flood_dataset.xlsx` (keeping the same column names) and re-run:

```bash
python train_model.py
```

This regenerates `models/floods.save`, `models/transform.save`, `models/model_metadata.json`,
`reports/model_comparison.png` and `reports/training_summary.txt` - the About and Home pages
read `model_metadata.json` directly, so the site reflects the new numbers immediately after a
restart.

## Screenshots

<img width="1831" height="811" alt="image" src="https://github.com/user-attachments/assets/ef12da70-6a0e-479d-8d1a-266a40edcdd8" />
<img width="1861" height="897" alt="image" src="https://github.com/user-attachments/assets/1d436cfe-a806-41c0-a2d8-6ba82add8ddd" />
<img width="1817" height="887" alt="image" src="https://github.com/user-attachments/assets/a0a7e66b-5490-4a85-b84e-1e974f45467f" />




## Future Scope

- Live weather API integration instead of manual form entry.
- User authentication tied to the existing `users` table.
- SMS/email alerting on Severe-risk predictions.
- Multi-model-version tracking (see `reports/ER_DIAGRAM.md`).

## License

MIT - see `LICENSE`.
