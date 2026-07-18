"""
prediction.py

Wraps the saved model + scaler behind a single `predict_flood()` call so
app.py never has to know whether the deployed model exposes predict_proba
or not (KNN/DecisionTree/RandomForest/XGBoost all do, but this keeps the
route handler defensive anyway).

The model/scaler are loaded once at import time rather than per-request -
joblib.load isn't free, and there's no reason to pay that cost on every
single form submission.
"""

import os
import joblib
import pandas as pd

from config import Config
from preprocessing.data_prep import FEATURE_COLUMNS
from utils import classify_risk, recommendation_for_risk

_model = None
_scaler = None


class ModelNotReadyError(Exception):
    pass


def _ensure_loaded():
    global _model, _scaler
    if _model is None or _scaler is None:
        if not (os.path.exists(Config.MODEL_PATH) and os.path.exists(Config.SCALER_PATH)):
            raise ModelNotReadyError(
                "Model artifacts not found. Run 'python train_model.py' first."
            )
        _model = joblib.load(Config.MODEL_PATH)
        _scaler = joblib.load(Config.SCALER_PATH)
    return _model, _scaler


def predict_flood(model_reading):
    """
    model_reading: dict with keys matching FEATURE_COLUMNS
    (ANNUAL, Cloud Cover, Temp, Humidity, Jun-Sep)

    Returns a dict with the prediction, probability (if available),
    a risk band label and a plain-language recommendation.
    """
    model, scaler = _ensure_loaded()

    ordered_df = pd.DataFrame([[model_reading[col] for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
    scaled_input = scaler.transform(ordered_df)

    prediction = int(model.predict(scaled_input)[0])

    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(scaled_input)[0][1])

    # If the model doesn't expose probabilities for some reason, fall back
    # to a binary confidence so the risk meter on the result page still
    # has something sensible to display.
    effective_probability = probability if probability is not None else float(prediction)
    risk_level = classify_risk(effective_probability)

    return {
        "prediction": prediction,
        "probability": probability,
        "probability_percent": round(effective_probability * 100, 1),
        "risk_level": risk_level,
        "recommendation": recommendation_for_risk(risk_level),
    }


def get_model_instance():
    """Exposed for the About page, in case it wants to show the model's
    class name (e.g. 'XGBClassifier') alongside the metadata JSON."""
    model, _ = _ensure_loaded()
    return model
