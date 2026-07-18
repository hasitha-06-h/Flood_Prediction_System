"""
preprocessing/data_prep.py

Handles everything that happens between "raw dataset on disk" and
"clean numpy arrays ready for a model". Kept separate from the training
scripts on purpose so both train_model.py and the Flask app (prediction.py)
can reuse the exact same cleaning logic - if this drifts between training
and inference, predictions silently go wrong, which is the #1 bug I ran
into while building this the first time around.
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Columns the model was actually trained on. avgjune/sub/Oct-Dec exist in the
# raw sheet but the prediction form only collects five values, so we keep
# a short list here that both training and the web app agree on.
FEATURE_COLUMNS = ["ANNUAL", "Cloud Cover", "Temp", "Humidity", "Jun-Sep"]
TARGET_COLUMN = "flood"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "flood_dataset.xlsx")
MODELS_DIR = os.path.join(BASE_DIR, "models")
SCALER_PATH = os.path.join(MODELS_DIR, "transform.save")


def load_raw_dataset(path=DATASET_PATH):
    """Reads the source spreadsheet, tolerating either .csv or .xlsx."""
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    # column names in the original sheet sometimes carry stray whitespace
    df.columns = [c.strip() for c in df.columns]
    return df


def cap_outliers_iqr(series, k=1.5):
    """
    Classic IQR capping - anything beyond Q1-k*IQR / Q3+k*IQR gets clamped
    to the boundary instead of dropped, since the dataset is tiny (115 rows)
    and we can't afford to lose rows.
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr
    return series.clip(lower=lower_bound, upper=upper_bound)


def clean_dataset(df):
    """Fills gaps, caps outliers on the numeric feature columns, drops dupes."""
    df = df.copy()

    # Missing value handling - median is safer than mean here because a
    # couple of extreme rainfall years would otherwise drag the mean up.
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    df = df.drop_duplicates().reset_index(drop=True)

    for col in FEATURE_COLUMNS:
        if col in df.columns:
            df[col] = cap_outliers_iqr(df[col])

    return df


def get_feature_target_split(df):
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y


def build_train_test_sets(test_size=0.2, random_state=42, save_scaler=True):
    """
    End to end: load -> clean -> split -> scale.
    Returns X_train_scaled, X_test_scaled, y_train, y_test, scaler
    """
    raw = load_raw_dataset()
    cleaned = clean_dataset(raw)
    X, y = get_feature_target_split(cleaned)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if save_scaler:
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)

    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, scaler


def load_scaler():
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            "transform.save not found - run train_model.py before starting the app."
        )
    return joblib.load(SCALER_PATH)


def scale_single_reading(reading_dict, scaler=None):
    """
    Takes the raw form values coming from the Flask prediction page and turns
    them into the scaled feature vector the model expects. `reading_dict`
    keys must match FEATURE_COLUMNS.
    """
    if scaler is None:
        scaler = load_scaler()

    ordered_values = [[reading_dict[col] for col in FEATURE_COLUMNS]]
    df_row = pd.DataFrame(ordered_values, columns=FEATURE_COLUMNS)
    return scaler.transform(df_row)
