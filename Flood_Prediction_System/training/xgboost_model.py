"""
training/xgboost_model.py

Gradient boosted trees via XGBoost. This is the model that ends up
getting deployed (see compare_models.py) - boosting tends to generalise
a little better than a single Random Forest run and it's become the de
facto standard for tabular classification problems like this one.
"""

from xgboost import XGBClassifier
from evaluation.metrics import summarize_predictions


def train(X_train, y_train, n_estimators=150, max_depth=4, learning_rate=0.1, random_state=42):
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def predict(model, X):
    return model.predict(X)


def evaluate(model, X_test, y_test):
    predictions = predict(model, X_test)
    return summarize_predictions(y_test, predictions, model_label="XGBoost")
