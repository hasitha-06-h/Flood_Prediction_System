"""
utils.py

Grab-bag of small helpers used by app.py and prediction.py that didn't
deserve their own module - form validation, risk labelling and metadata
loading.
"""

import json
import os
from config import Config


class ValidationError(Exception):
    """Raised when a submitted weather reading fails a sanity check."""
    pass


# (field_name, human label, min, max) - bounds are generous on purpose,
# they exist to catch typos (like a stray extra zero) rather than to
# enforce strict climatology.
FIELD_BOUNDS = [
    ("annual_rainfall", "Annual Rainfall", 0, 10000),
    ("cloud_cover", "Cloud Visibility", 0, 100),
    ("temperature", "Temperature", -10, 60),
    ("humidity", "Humidity", 0, 100),
    ("seasonal_rainfall", "Seasonal Rainfall", 0, 5000),
]


def parse_and_validate_form(form):
    """
    Pulls the five expected fields out of a Flask request.form object,
    casts them to float and checks they fall in a believable range.
    Raises ValidationError with a user-friendly message on the first
    field that fails.
    """
    parsed = {}
    for field_name, label, low, high in FIELD_BOUNDS:
        raw_value = form.get(field_name, "").strip()
        if raw_value == "":
            raise ValidationError(f"{label} is required.")
        try:
            value = float(raw_value)
        except ValueError:
            raise ValidationError(f"{label} must be a number.")
        if not (low <= value <= high):
            raise ValidationError(f"{label} should be between {low} and {high}.")
        parsed[field_name] = value

    return parsed


def to_model_reading(parsed_form):
    """Maps the friendlier form field names onto the column names the
    trained model actually expects (see preprocessing.FEATURE_COLUMNS)."""
    return {
        "ANNUAL": parsed_form["annual_rainfall"],
        "Cloud Cover": parsed_form["cloud_cover"],
        "Temp": parsed_form["temperature"],
        "Humidity": parsed_form["humidity"],
        "Jun-Sep": parsed_form["seasonal_rainfall"],
    }


def classify_risk(probability):
    """Turns a raw flood probability into a human risk band used across
    the result / alert / safe pages."""
    if probability is None:
        return "Unknown"
    if probability >= 0.75:
        return "Severe"
    if probability >= 0.5:
        return "High"
    if probability >= 0.25:
        return "Moderate"
    return "Low"


def recommendation_for_risk(risk_level):
    messages = {
        "Severe": "Immediate evacuation planning is advised. Notify local disaster "
                  "management authorities and move essential belongings to higher ground.",
        "High": "Flooding is likely in the coming days. Prepare an emergency kit, "
                "monitor official alerts closely, and avoid low-lying routes.",
        "Moderate": "Conditions are borderline. Keep monitoring rainfall updates and "
                    "avoid unnecessary travel through drainage-prone areas.",
        "Low": "No immediate flood risk detected. Continue routine monitoring of "
               "seasonal weather bulletins.",
        "Unknown": "Insufficient data to generate a recommendation.",
    }
    return messages.get(risk_level, messages["Unknown"])


def load_model_metadata():
    """Reads models/model_metadata.json written by train_model.py. Falls
    back to a generic placeholder if training hasn't been run yet, so the
    About page never crashes on a fresh checkout."""
    if os.path.exists(Config.METADATA_PATH):
        with open(Config.METADATA_PATH) as f:
            return json.load(f)
    return {
        "model_name": "Not trained yet",
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1_score": None,
        "trained_on_rows": 0,
        "trained_at": None,
    }
