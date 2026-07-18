"""
config.py

Centralised configuration so app.py doesn't hardcode paths and secrets
inline. Reads sensitive values from environment variables where possible,
falling back to sane local-dev defaults.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Config:
    SECRET_KEY = os.environ.get("FLOOD_APP_SECRET_KEY", "dev-key-change-this-in-production")

    MODELS_DIR = os.path.join(BASE_DIR, "models")
    MODEL_PATH = os.path.join(MODELS_DIR, "floods.save")
    SCALER_PATH = os.path.join(MODELS_DIR, "transform.save")
    METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")

    DATABASE_DIR = os.path.join(BASE_DIR, "database")
    DATABASE_PATH = os.environ.get(
        "FLOOD_APP_DB_PATH", os.path.join(DATABASE_DIR, "flood_system.db")
    )

    REPORTS_DIR = os.path.join(BASE_DIR, "reports")

    # Threshold above which a prediction is treated as "flood likely" when
    # the underlying model exposes predict_proba. Kept configurable in case
    # a future retrain wants a more conservative / aggressive cutoff.
    FLOOD_PROBABILITY_THRESHOLD = 0.5

    DEBUG = os.environ.get("FLASK_DEBUG", "0") == "1"
    HOST = os.environ.get("FLOOD_APP_HOST", "0.0.0.0")
    PORT = int(os.environ.get("PORT", 5000))
