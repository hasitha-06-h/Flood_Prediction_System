"""
training/random_forest.py

Random Forest is the ensemble version of decision_tree.py - averaging many
shallow trees over bootstrapped samples. Tends to be the most stable of
the four on small, noisy datasets like this one.
"""

from sklearn.ensemble import RandomForestClassifier
from evaluation.metrics import summarize_predictions


def train(X_train, y_train, n_estimators=200, max_depth=6, random_state=42):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def predict(model, X):
    return model.predict(X)


def evaluate(model, X_test, y_test):
    predictions = predict(model, X_test)
    return summarize_predictions(y_test, predictions, model_label="Random Forest")
